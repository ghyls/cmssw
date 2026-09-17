#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "CAHitNtupletGenerator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelTracksMaskingSoA : public global::EDProducer<> {
    using Algo = CAHitMaskingAndMerger;

  public:
    explicit PixelTracksMaskingSoA(const edm::ParameterSet& iConfig);
    ~PixelTracksMaskingSoA() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    uint32_t const iterationIndex_;
    pixelTrack::Quality const minQuality_;
    bool const applyMasking_;
    // When true, hits attached by the extension stage are masked for the next iteration as well;
    // otherwise only the hits the CA itself found are masked.
    bool const maskAttachedHits_;

    // Optional starting mask. An empty "recHitsMaskSoASrc" tag means "start all open": the chain's
    // first masking stage has no earlier mask to inherit, and the mask is then sized from the two hit
    // collections the global hit index space is made of (pixel rechits + stubs), read for their row
    // counts only.
    const bool hasInputMask_;
    device::EDGetToken<reco::TrackingRecHitsMaskingCollection> inputRecHitsMaskToken_;
    device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelRecHitToken_;
    device::EDGetToken<reco::StubsSoACollection> stubsToken_;
    const device::EDGetToken<reco::TracksSoACollection> inputTrackSoAToken_;

    const device::EDPutToken<reco::TrackingRecHitsMaskingCollection> outputRecHitsMaskToken_;

    Algo deviceAlgo_;
  };

  PixelTracksMaskingSoA::PixelTracksMaskingSoA(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        iterationIndex_(iConfig.getParameter<uint32_t>("iterationIndex")),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        applyMasking_(iConfig.getParameter<bool>("applyMasking")),
        maskAttachedHits_(iConfig.getParameter<bool>("maskAttachedHits")),
        hasInputMask_(not iConfig.getParameter<edm::InputTag>("recHitsMaskSoASrc").label().empty()),
        inputTrackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("tracksSoASrc"))),
        outputRecHitsMaskToken_(produces()) {
    if (hasInputMask_) {
      inputRecHitsMaskToken_ = consumes(iConfig.getParameter<edm::InputTag>("recHitsMaskSoASrc"));
    } else {
      pixelRecHitToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
      stubsToken_ = consumes(iConfig.getParameter<edm::InputTag>("stubsSrc"));
    }
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
    // 0 is the "hit is free" value of the mask, so it cannot be used as an iteration tag: a module
    // configured with it would mask nothing and say nothing.
    if (iterationIndex_ == 0u) {
      throw cms::Exception("PixelTrackConfiguration")
          << "iterationIndex 0 means 'not masked': use a distinct non-zero bit per masking stage.";
    }
  }

  void PixelTracksMaskingSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("recHitsMaskSoASrc", edm::InputTag(""))
        ->setComment(
            "Starting mask of this stage. Empty (the default) means the chain starts all open and the "
            "mask is seeded here, sized from pixelRecHitSrc + stubsSrc; set it to an earlier masking "
            "stage in a multi-iteration chain.");
    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltPhase2SiPixelRecHitsSoA"))
        ->setComment("Pixel rechits, read for their row count only. Used only when recHitsMaskSoASrc is empty.");
    desc.add<edm::InputTag>("stubsSrc", edm::InputTag("hltOTStubProducer"))
        ->setComment("Outer-tracker stubs, read for their row count only. Used only when recHitsMaskSoASrc is empty.");
    desc.add<edm::InputTag>("tracksSoASrc",
                            edm::InputTag("pixelTracksHighPtAlpaka"));  // has to be changed for each iteration
    desc.add<std::string>("minQuality", "highPurity");
    desc.add<uint32_t>("iterationIndex", 1)
        ->setComment(
            "Non-zero tag this stage ORs into the mask of every hit it takes; 0 is the 'hit is free' value and is "
            "refused. Give each masking stage its own bit (1, 2, 4, ...) so the tags stay readable when several "
            "stages run.");
    desc.add<bool>("maskAttachedHits", false)
        ->setComment("also mask hits attached by the in-fit extension (default: original-hits policy)");
    desc.add<bool>("applyMasking", true)
        ->setComment(
            "If false, pass through the input mask unchanged (no hits masked) so the next iteration sees all hits; the "
            "module stays in the chain.");

    descriptions.addWithDefaultLabel(desc);
  }

  void PixelTracksMaskingSoA::produce(edm::StreamID streamID,
                                      device::Event& iEvent,
                                      const device::EventSetup& es) const {
    if (hasInputMask_) {
      // Reading the mask layout first puts the module on the queue that produced it, which orders its
      // reads before the early release of that product. Keep this get() first.
      const auto& inpMaskColl = iEvent.get(inputRecHitsMaskToken_);
      const auto& inpTkColl = iEvent.get(inputTrackSoAToken_);
      iEvent.emplace(
          outputRecHitsMaskToken_,
          deviceAlgo_.makeMaskingAsync(
              inpMaskColl, inpTkColl, minQuality_, iterationIndex_, iEvent.queue(), applyMasking_, maskAttachedHits_));
    } else {
      // Seed the chain: all-open over the global hit index space, [0, nPixelHits + nStubs).
      const auto& pixColl = iEvent.get(pixelRecHitToken_);
      const auto& stubColl = iEvent.get(stubsToken_);
      const auto& inpTkColl = iEvent.get(inputTrackSoAToken_);
      const uint32_t nHits = pixColl.nHits() + stubColl.nStubs();
      iEvent.emplace(
          outputRecHitsMaskToken_,
          deviceAlgo_.makeMaskingAsync(
              nHits, inpTkColl, minQuality_, iterationIndex_, iEvent.queue(), applyMasking_, maskAttachedHits_));
    }
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTracksMaskingSoA);
