#include <memory>
#include <type_traits>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// Brings the (complete) generic CopyToDevice<PortableHostCollection<...>> specialization required by
// the alpaka ESProducer::setWhatProduced to register the host->device copy of the produced payload.
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/Record/interface/CAGeometryRecord.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

#include "CAGeometryBuild.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // EventSetup producer for the geometry-only blocks of the CA geometry SoA (per-module surface
  // frames + per-CA-layer geometry classification + the layer-pair graph + per-layer fishbone
  // cut). It runs the once-per-conditions module/layer fill (buildCAGeometryHost, lifted from the
  // CAHitNtuplet globalBeginRun) so the walk/refit consumers can read the geometry from the
  // EventSetup instead of a per-event event product. The doublet/triplet/ntuplet CUT blocks are
  // per-producer configuration and are intentionally left default in this product.
  //
  // Produces the host collection; the framework copies it to the device once per IOV (generic
  // CopyToDevice for the portable collection) when a consumer requests the device handle via a
  // device::ESGetToken<reco::CAGeometrySoACollection, CAGeometryRecord>. Templated on TrackerTraits
  // and registered per topology (mirroring CAHitNtupletAlpaka<TrackerTraits>); the topology is a
  // template parameter, so buildCAGeometryHost<TrackerTraits> takes the correct `if constexpr`
  // branch (Phase2OTStubs uses the StackedModuleGeometry OT ordering; Phase2OT collects OT-barrel
  // P-in-PS modules and needs no stacked geometry; the pixel-only topologies collect pixel modules
  // only). The StackedModuleGeometry record is consumed only for Phase2OTStubs.
  template <typename TrackerTraits>
  class CAGeometryESProducer : public ESProducer {
  public:
    CAGeometryESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig),
          layerPairs_(
              iConfig.getParameter<edm::ParameterSet>("geometry").getParameter<std::vector<unsigned int>>("pairGraph")),
          skipsLayers_(iConfig.getParameter<edm::ParameterSet>("geometry")
                           .getParameter<std::vector<unsigned int>>("skipsLayers")),
          fishboneCuts_(
              iConfig.getParameter<edm::ParameterSet>("geometry").getParameter<std::vector<double>>("fishboneCuts")) {
      const std::vector<unsigned int> startingPairs =
          iConfig.getParameter<edm::ParameterSet>("geometry").getParameter<std::vector<unsigned int>>("startingPairs");

      // Size validation on the geometry-only surface, mirroring the CA producer that shares this
      // geometry (CAHitNtupletAlpaka): the graph must be a non-empty list of layer pairs, everything
      // per layer pair must agree with the pair count, and every layer id must exist. Here the layer
      // count is declared by fishboneCuts, the only per-layer member of this producer. Without these
      // checks an empty or inconsistent `geometry` PSet reaches buildCAGeometryHost, whose per-pair
      // asserts and *std::max_element over the pair graph are undefined on an empty range.
      const size_t nPairs = layerPairs_.size() / 2;
      const size_t nLayers = fishboneCuts_.size();
      if (layerPairs_.empty())
        throw cms::Exception("Configuration") << "CAGeometryESProducer: geometry.pairGraph is empty.";
      if (layerPairs_.size() % 2 != 0)
        throw cms::Exception("Configuration")
            << "CAGeometryESProducer: geometry.pairGraph has an odd number of entries (" << layerPairs_.size() << ").";
      if (skipsLayers_.size() != nPairs)
        throw cms::Exception("Configuration")
            << "CAGeometryESProducer: geometry.skipsLayers has " << skipsLayers_.size()
            << " entries but the CA graph has " << nPairs << " layer pairs.";
      for (const unsigned int& i : startingPairs)
        if (i >= nPairs)
          throw cms::Exception("Configuration") << "CAGeometryESProducer: geometry.startingPairs contains the pair id "
                                                << i << " but the CA graph only has " << nPairs << " layer pairs.";
      for (size_t i = 0; i < layerPairs_.size(); ++i)
        if (layerPairs_[i] >= nLayers)
          throw cms::Exception("Configuration")
              << "CAGeometryESProducer: geometry.pairGraph refers to the layer " << layerPairs_[i]
              << " but geometry.fishboneCuts only declares " << nLayers << " layers.";

      // The `geometry` PSet is upstream's CA surface: `startingPairs` is a list of pair IDs, while the
      // geometry builder wants one flag per pair.
      startingPair_.assign(nPairs, 0u);
      for (unsigned int id : startingPairs)
        startingPair_[id] = 1u;
      auto c = setWhatProduced(this);
      geomToken_ = c.consumes();
      topoToken_ = c.consumes();
      // Only the OT-stubs build path reads the StackedModuleGeometry (CA-ordered OT module frames);
      // for the other topologies the record is neither needed nor required to exist.
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        stackedToken_ = c.consumes();
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      // The four geometry-only members of upstream's `geometry` PSet (same names, same shapes, so a
      // CA producer's geometry PSet can be copied field by field).
      edm::ParameterSetDescription geometryDesc;
      geometryDesc.add<std::vector<unsigned int>>("pairGraph", {});
      geometryDesc.add<std::vector<unsigned int>>("startingPairs", {});
      geometryDesc.add<std::vector<unsigned int>>("skipsLayers", {});
      geometryDesc.add<std::vector<double>>("fishboneCuts", {});
      desc.add<edm::ParameterSetDescription>("geometry", geometryDesc);

      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<reco::CAGeometryHost> produce(CAGeometryRecord const& iRecord) {
      auto const& trackerGeometry = iRecord.get(geomToken_);
      auto const& trackerTopology = iRecord.get(topoToken_);

      reco::StackedModuleGeometryHost const* stackedGeometry = nullptr;
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        stackedGeometry = &iRecord.get(stackedToken_);
      }

      return std::make_unique<reco::CAGeometryHost>(reco::buildCAGeometryHost<TrackerTraits>(
          trackerGeometry, trackerTopology, stackedGeometry, layerPairs_, startingPair_, skipsLayers_, fishboneCuts_));
    }

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
    edm::ESGetToken<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> stackedToken_;

    const std::vector<unsigned int> layerPairs_;
    std::vector<unsigned int> startingPair_;  // per-pair flags, built from the `startingPairs` id list
    const std::vector<unsigned int> skipsLayers_;
    const std::vector<double> fishboneCuts_;
  };

  // Register per topology, mirroring CAHitNtupletAlpaka. Only Phase2OTStubs and Phase2OT
  // (the Phase-2 OT tracking chain) are registered; pixel-only topologies are supported
  // by buildCAGeometryHost but not part of this OT chain.
  using CAGeometryESProducerPhase2OT = CAGeometryESProducer<pixelTopology::Phase2OT>;
  using CAGeometryESProducerPhase2OTStubs = CAGeometryESProducer<pixelTopology::Phase2OTStubs>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(CAGeometryESProducerPhase2OT);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(CAGeometryESProducerPhase2OTStubs);
