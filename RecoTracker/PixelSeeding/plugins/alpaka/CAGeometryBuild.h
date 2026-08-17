#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAGeometryBuild_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAGeometryBuild_h

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

namespace reco {

  // Build the GEOMETRY-ONLY blocks of the CA geometry SoA once at conditions time. This is a
  // faithful lift of the module/layer/graph fill in CAHitNtuplet.cc globalBeginRun (the geometry,
  // per-CA-layer classification, and layer-pair graph). It intentionally does NOT fill the
  // doublet/triplet/ntuplet cut blocks: those are per-producer configuration, not conditions, and
  // are left default in the returned product (only `layers.fishboneCut` and the `graph` block --
  // which are pure per-run geometry/topology inputs -- are written here alongside the module and
  // layer geometry).
  //
  //   trackerGeometry / trackerTopology : the per-run tracker geometry + topology records.
  //   stackedGeometry                   : the CA-ordered OT stacked-module geometry (non-null only
  //                                       for the Phase2OTStubs topology); provides the OT module
  //                                       ordering and per-sensor frames.
  //   layerPairs / startingPair / skipsLayers : the layer-pair graph configuration (per producer;
  //                                       identical across producers sharing a geometry).
  //   fishboneCuts                      : per-CA-layer fishbone merge threshold (sizes n_layers).
  template <typename TrackerTraits>
  reco::CAGeometryHost buildCAGeometryHost(TrackerGeometry const& trackerGeometry,
                                           TrackerTopology const& trackerTopology,
                                           reco::StackedModuleGeometryHost const* stackedGeometry,
                                           std::vector<unsigned int> const& layerPairs,
                                           std::vector<unsigned int> const& startingPair,
                                           std::vector<unsigned int> const& skipsLayers,
                                           std::vector<double> const& fishboneCuts) {
    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

    int n_layers = fishboneCuts.size();
    int n_pairs = layerPairs.size() / 2;
    int n_modules = 0;

    assert(n_pairs == int(startingPair.size()));
    assert(n_pairs == int(skipsLayers.size()));
    assert(int(*std::max_element(layerPairs.begin(), layerPairs.end())) < n_layers);

    auto const& dets = trackerGeometry.dets();

    auto oldLayer = 0u;
    auto layerCount = 0u;

    std::vector<bool> layerIsBarrel(n_layers);
    std::vector<bool> layerIsOT(n_layers);
    std::vector<bool> layerIsSS(n_layers);
    std::vector<int> layerStarts(n_layers + 1);
    //^ why n_layers + 1? This is a cumulative sum of the number
    // of modules each layer has. And we need the  extra spot
    // at the end to hold the total number of modules.

    std::vector<int> moduleToindexInDets;

    auto isPinPSinOTBarrel = [&](DetId detId) {
      // Select only P-hits from the OT barrel
      return (trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
              detId.subdetId() == StripSubdetector::TOB);
    };
    auto isPixel = [&](DetId detId) {
      auto subId = detId.subdetId();
      return (subId == PixelSubdetector::PixelBarrel || subId == PixelSubdetector::PixelEndcap);
    };
    auto isBarrel = [&](DetId detId) {
      auto subId = detId.subdetId();
      auto subDetector = trackerGeometry.geomDetSubDetector(subId);
      return GeomDetEnumerators::isBarrel(subDetector);
    };

    // loop over all detector modules and build the CA layers
    int counter = 0;
    for (auto& det : dets) {
      DetId detid = det->geographicalId();
      auto layer = trackerTopology.layer(detid);
      // Logic:
      // - if we are not inside pixels, we need to ignore anything **but** the OT.
      // - for the time being, this is assuming that the CA extension will
      //   only cover the OT barrel part, and will ignore the OT forward.

      // Modules of the pixel layers
      if (isPixel(detid)) {
        if (layer != oldLayer) {
          layerIsBarrel[layerCount] = isBarrel(detid);
          layerIsOT[layerCount] = false;  // we are in the loop over pixel dets, so these are not OT layers
          layerIsSS[layerCount] = false;  // we are in the loop over pixel dets, so these are not SS layers
          layerStarts[layerCount++] = n_modules;
          if (layerCount >= layerStarts.size())
            break;
          oldLayer = layer;
        }
        moduleToindexInDets.push_back(counter);
        n_modules++;
      }

      // if we are using the CA extension for Phase-2,
      // we also have to collect the modules from the considered OT layers
      if constexpr (std::is_same_v<pixelTopology::Phase2OT, TrackerTraits>) {
        auto const& detUnits = det->components();
        for (auto& detUnit : detUnits) {
          DetId unitDetId(detUnit->geographicalId());
          // Modules of the considered OT layers
          if (isPinPSinOTBarrel(unitDetId)) {
            if (layer != oldLayer) {
              layerIsBarrel[layerCount] = isBarrel(detid);
              layerIsOT[layerCount] = true;   // we are in the loop over PS modules, so these are all OT layers
              layerIsSS[layerCount] = false;  // we are in the loop over PS modules, so these are not SS layers
              layerStarts[layerCount++] = n_modules;
              if (layerCount >= layerStarts.size())
                break;
              oldLayer = layer;
            }
            moduleToindexInDets.push_back(counter);
            n_modules++;
          }
        }
      }
      counter++;
    }

    // Process OT stacked modules for Phase-2 with stubs
    // CA layers follow inside-out ordering:
    // - CA layers 28-33: Barrel (layers 1-6)
    // - CA layers 34-43: disks 1-5 at z > 0, each disk as a PS layer (even id) then a 2S layer (odd id)
    // - CA layers 44-53: disks 1-5 at z < 0, same PS/2S alternation
    //
    // IMPORTANT: StackedModuleGeometry is ALREADY sorted in CA order by StackedModuleGeometryESProducer
    // (barrel by layer -> z > 0 disks by layer -> z < 0 disks by layer, PS before 2S) using stable_sort.
    // We iterate through it in index order WITHOUT re-sorting to ensure the frame array
    // index matches the detectorIndex assigned to hits/stubs (detectorIndex = nPixelModules + geomIndex).
    // Number of pixel modules already processed; OT modules start at this CA module index.
    // Used below when populating CAModulesSoA::innerSensorFrame for OT modules.
    const int nPixelModulesInCA = n_modules;

    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      if (stackedGeometry != nullptr) {
        auto stackedView = stackedGeometry->view();
        const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());

        bool firstModule = true;
        uint8_t prevLayer = 0;
        bool prevIsPS = false;
        int prevCategory = -1;  // 0=barrel, 1=z < 0 endcap, 2=z > 0 endcap

        // Iterate through StackedModuleGeometry in index order (already sorted in CA order)
        for (uint32_t i = 0; i < nStackedModules; ++i) {
          bool isBarrelMod = stackedView.isBarrel()[i];
          bool isFwdEndcap = stackedView.isFwdEndcap()[i];
          bool isPS = stackedView.isPS()[i];
          uint8_t otLayer = stackedView.layer()[i];
          DetId stackedDetId(stackedView.stackedDetId()[i]);

          // Determine category: 0=barrel, 1=z < 0 endcap, 2=z > 0 endcap
          int category = isBarrelMod ? 0 : (isFwdEndcap ? 2 : 1);

          // Check if we've transitioned to a new CA layer
          // A new layer starts when category changes OR layer number changes within same category
          if (firstModule || category != prevCategory || otLayer != prevLayer || isPS != prevIsPS) {
            // Start new CA layer
            if (layerCount < layerStarts.size()) {
              layerIsBarrel[layerCount] = isBarrelMod;
              layerIsOT[layerCount] =
                  true;  // we are in the loop over StackedModuleGeometry, so these are all OT layers
              layerIsSS[layerCount] = !isPS;
              layerStarts[layerCount++] = n_modules;
            }
            prevCategory = category;
            prevLayer = otLayer;
            prevIsPS = isPS;
            firstModule = false;
          }

          // Find this module in TrackerGeometry dets list
          bool found = false;
          for (int detIdx = 0; detIdx < static_cast<int>(dets.size()); ++detIdx) {
            if (dets[detIdx]->geographicalId() == stackedDetId) {
              moduleToindexInDets.push_back(detIdx);
              n_modules++;
              found = true;
              break;
            }
          }
          if (!found) {
            edm::LogWarning("CAGeometryBuild")
                << "Could not find stacked module " << stackedDetId.rawId() << " in TrackerGeometry";
          }
        }
      }
    }

    layerStarts[n_layers] = n_modules;

    reco::CAGeometryHost product{
        cms::alpakatools::host(), n_layers + 1, n_pairs, n_pairs, n_pairs, n_layers, n_modules};

    auto layerSoA = product.view().layers();
    auto graphSoA = product.view().graph();
    auto modulesSoA = product.view().modules();

    // For Phase2OTStubs, stackedView is needed below to pick the right per-sensor
    // frame for each OT module's `innerSensorFrame` per the contract:
    //   - PSP (moduleType==0): inner = lower sensor (P-side).
    //   - PSS (moduleType==1): inner = upper sensor (P-side).
    //   - SS:                   inner = physically-inner = lower if !isFlipped, else upper.
    auto pickInnerFrame = [&](int iCA) -> Frame {
      // Default fallback: identity-handling via detFrame for non-OT or missing stacked info.
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        if (stackedGeometry != nullptr && iCA >= nPixelModulesInCA) {
          auto stackedView = stackedGeometry->view();
          const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());
          const uint32_t iGeom = static_cast<uint32_t>(iCA - nPixelModulesInCA);
          if (iGeom < nStackedModules) {
            uint8_t mt = stackedView.moduleType()[iGeom];
            bool isFlipped = stackedView.isFlipped()[iGeom];
            bool useUpper;
            if (mt == 0)
              useUpper = false;  // PSP: pixel is lower
            else if (mt == 1)
              useUpper = true;  // PSS: pixel is upper
            else
              useUpper = isFlipped;  // SS: inner = lower if !flipped, else upper
            return useUpper ? Frame(stackedView[iGeom].upperSensorFrame())
                            : Frame(stackedView[iGeom].lowerSensorFrame());
          }
        }
      }
      // Pixel modules (or non-OTStubs topology): innerSensorFrame == detFrame.
      return modulesSoA[iCA].detFrame();
    };

    for (int i = 0; i < n_modules; ++i) {
      auto idx = moduleToindexInDets[i];
      auto det = dets[idx];
      auto vv = det->surface().position();
      auto rr = Rotation(det->surface().rotation());
      modulesSoA[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
      modulesSoA[i].innerSensorFrame() = pickInnerFrame(i);
    }

    for (int i = 0; i < n_layers; ++i) {
      layerSoA.fishboneCut()[i] = fishboneCuts[i];
      layerSoA.layerStarts()[i] = layerStarts[i];
      layerSoA.isBarrel()[i] = layerIsBarrel[i];
      layerSoA.isOT()[i] = layerIsOT[i];
      layerSoA.isSS()[i] = layerIsSS[i];
    }

    layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

    for (int i = 0; i < n_pairs; ++i) {
      graphSoA.layerPair()[i] = {{uint32_t(layerPairs[2 * i]), uint32_t(layerPairs[2 * i + 1])}};
      graphSoA.skipsLayers()[i] = uint16_t(bool(skipsLayers[i]));
      graphSoA.startingPair()[i] = startingPair[i];
    }

    return product;
  }

}  // namespace reco

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAGeometryBuild_h
