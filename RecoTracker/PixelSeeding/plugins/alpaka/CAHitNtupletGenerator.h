#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h

#include <alpaka/alpaka.hpp>

#include <optional>
#include <memory>

#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/StackedModuleGeometrySoACollection.h"

#include "CACell.h"
#include "CAExtensionKernels.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CATripletDumpMacro.h"  // CA_TRIPLET_DUMP toggle for the #ifdef'd dump member/accessor below
#include "HelixFit.h"

namespace edm {
  class ParameterSetDescription;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletGenerator {
  public:
    using HitsView = ::reco::TrackingRecHitView;
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using HitsOnHost = ::reco::TrackingRecHitHost;

    using HitsOnDeviceRefProdVector = edm::RefProdVector<HitsOnDevice>;

    // Everything the CA build needs from the event's hit collections, assembled by the producer: the
    // topology's hit view and module-start view (see CAStructures.h) plus the two scalars.
    using HitsInput = caStructures::HitsInputT<TrackerTraits>;
    using MapToHitConstView = ::reco::TrackingRecHitsMaskingConstView;

    using TkSoADevice = reco::TracksSoACollection;
    using Quality = ::pixelTrack::Quality;

    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

    using CAGeometryOnDevice = reco::CAGeometrySoACollection;

  public:
    CAHitNtupletGenerator(const edm::ParameterSet& cfg);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

    // NOTE: beginJob and endJob were meant to be used
    // to fill the statistics. This is still not implemented in Alpaka
    // since we are missing the begin/endJob functionality for the Alpaka
    // producers.
    //
    // void beginJob();
    // void endJob();

    // The producer builds the CA ntuplets, runs the broken-line fit and selects high-purity tracks.
    // It does not consume the outer-tracker or stacked-sensor products.
    //
    // Acquire/produce split (stream::SynchronizingEDProducer): the producer calls beginTuplesAsync
    // in acquire() -- hit prep, doublets, CA kernels, and one async D2H of the tuple-multiplicity
    // per-N-bin offsets -- and finishTuplesAsync in produce() -- both fit passes (consuming the
    // landed offsets host-side, see HelixFit::setHostTupleMultiplicityOffsets), tuple
    // classification and the product. The framework's acquire->produce boundary waits for the copy
    // in its async callback instead of on a TBB thread. State crossing the boundary lives in
    // PendingTuples, a member of the producer.
    struct PendingTuples {
      // unique_ptr, not optional: the kernels object is not move-constructible (reference +
      // const members), and PendingTuples must move across the acquire->produce seam.
      std::unique_ptr<CAHitNtupletGeneratorKernels<TrackerTraits>> kernels;
      std::optional<TkSoADevice> tracks;
      // Pinned host mirror of the tuple-multiplicity per-N-bin offsets; filled by one async D2H
      // enqueued at the end of beginTuplesAsync. The framework seam guarantees it has landed
      // before finishTuplesAsync runs.
      std::optional<cms::alpakatools::host_buffer<uint32_t[]>> offsetsHost;
      // Pinned host mirror of the 5-word extraStorage counter block, enqueued async in
      // beginTuplesAsync when delayAllocations is on and landed by the same seam guarantee.
      // Word [0] is the tuple AtomicPairCounter; finishTuplesAsync decodes it to size the
      // hit->track storage without any host wait.
      std::optional<cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>> countsHost;
      uint32_t maxTuples = 0;
      uint32_t maxDoublets = 0;
      float bfield = 0.f;
      const float* rhoMapDevice = nullptr;
      // Normalized (Bz,Br) r-z field map (BLBFieldMap EventSetup condition, device-resident). Carried
      // across the acquire->produce seam beside the material map because the fit consumes them together
      // (see HelixFit::setBFieldMap); null is the scalar-field fallback.
      const float* bMapDevice = nullptr;
      bool built = false;  // false = early-out (too few hits): tracks holds the empty collection
    };

    PendingTuples beginTuplesAsync(HitsInput const& hits,
                                   CAGeometryOnDevice const& params_d,
                                   float bfield,
                                   uint32_t maxDoublets,
                                   uint32_t maxTuples,
                                   // Optional per-iteration hit mask. A default-constructed (null,
                                   // zero-row) view means "no masking": the doublet kernels skip
                                   // every mask lookup and no product is read.
                                   MapToHitConstView const& maskView,
                                   Queue& queue,
                                   const float* rhoMapDevice,
                                   const float* bMapDevice) const;

    TkSoADevice finishTuplesAsync(PendingTuples&& pending,
                                  HitsInput const& hits,
                                  CAGeometryOnDevice const& params_d,
                                  Queue& queue) const;

    // Always-on overflow reporting, independent of doStats_: the capacity guards in the build
    // kernels truncate silently, so without this a shortfall leaves no trace in production.
    // Per-stream persistent device accumulator (kOvfWords words, layout in
    // Kernel_overflowSentinel; slots 0-5 in use, 6-7 reserved), armed into each event's kernels
    // object, plus a pinned host mirror refreshed by an async D2H enqueued after classifyTuples
    // (where the sentinel is launched). No wait: by endStream every event queue has drained, so
    // the mirror holds the final totals. The mirror has two slots and each event writes the one
    // its parity selects: consecutive events of a stream run on different queues, so their copies
    // can be in flight at the same time, and alternating slots keeps them from writing the same
    // bytes. reportOverflows() takes the element-wise maximum of the two slots, which is exact
    // because the device counters only ever grow. Mutable because the *TuplesAsync methods are const.
    static constexpr uint32_t kOvfWords = 8u;
    mutable std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> ovfAccum_;
    mutable std::optional<cms::alpakatools::host_buffer<uint32_t[]>> ovfHost_;
    mutable uint32_t ovfSlot_ = 0;
    void reportOverflows(std::string const& moduleLabel) const;

#ifdef CA_TRIPLET_DUMP
    // Per-built-triplet training-dataset capture surfaced out of finishTuplesAsync: the kernels
    // object that owns the buffer lives in PendingTuples and is destroyed with it, so
    // finishTuplesAsync moves the kernels' device_tripletDump_ into here before returning; the
    // producer then moves it out and emplaces it as the 'Triplet' nano product. Mutable because
    // finishTuplesAsync is const. Empty/zero footprint when CA_TRIPLET_DUMP is off (the whole
    // member is compiled out).
    mutable std::optional<TripletDumpSoACollection> device_tripletDump_;
    std::optional<TripletDumpSoACollection>& tripletDumpBuffer() const { return device_tripletDump_; }
#endif

  private:
    Params m_params;
    // One-shot post-fit device dump of the first fitted tracks (HelixFit::setVerboseDump). Config: verboseBLFit.
    bool m_verboseBLDump;
  };

  class CAHitMaskingAndMerger {
  public:
    using MapToHit = reco::TrackingRecHitsMaskingCollection;
    using TkSoADevice = reco::TracksSoACollection;

  public:
    CAHitMaskingAndMerger() = default;
    ~CAHitMaskingAndMerger() = default;

    CAHitMaskingAndMerger(const CAHitMaskingAndMerger&) = delete;
    CAHitMaskingAndMerger(CAHitMaskingAndMerger&&) = delete;
    CAHitMaskingAndMerger& operator=(const CAHitMaskingAndMerger&) = delete;
    CAHitMaskingAndMerger& operator=(CAHitMaskingAndMerger&&) = delete;

    MapToHit makeMaskingAsync(MapToHit const& mask_d,
                              TkSoADevice const& tracks_d,
                              const pixelTrack::Quality minQuality,
                              uint32_t const& iterationIndex,
                              Queue& queue,
                              bool applyMasking = true,
                              bool maskAttachedHits = false) const;

    // Device-side gather/compact. Reads each input's actual nTracks() and hitOffsets() on device
    // and compacts all track columns + trackHits columns (id, detId, attached) into a dense
    // merged layout, writing the merged nTracks scalar and the shifted hitOffsets on device.
    // No host readback is involved in the sizing or in the copy.
    void mergeGather(TkSoADevice& outTracks,
                     TkSoADevice const& inp0Tracks,
                     TkSoADevice const& inp1Tracks,
                     int nInputs,
                     int32_t* armBuf,
                     const int32_t arm0,
                     const int32_t arm1,
                     Queue& queue) const;

    TkSoADevice makeFilteredTracks(int const& nTracks,
                                   int const& nHits,
                                   TkSoADevice const& inpTracks,
                                   pixelTrack::Quality const& minQuality,
                                   double const& matchFraction,
                                   Queue& queue,
                                   bool twinMerge = false,
                                   const int32_t* armOfTrack = nullptr,
                                   float twinMergeDeltaEta = 0.03f,
                                   float twinMergeDeltaPhi = 0.03f,
                                   int twinMergeMinSharedHits = 1,
                                   bool twinMergeTier2 = false,
                                   float twinMergeTier2DeltaEta = 0.01f,
                                   float twinMergeTier2DeltaPhi = 0.01f,
                                   float twinMergeNSigma2 = -1.f,
                                   int twinMergeMinSharedFwd = 1,
                                   bool twinMergeRefit = false,
                                   bool refitAllTracks = false,
                                   int32_t* unitedWinnerMask = nullptr,
                                   // pocket gate: per-track arm (input order in, merged-SoA order out), carried
                                   // through the filterTracks compaction. Both null when no arm is tracked.
                                   const uint8_t* pocketArmIn = nullptr,
                                   uint8_t* pocketArmIdOut = nullptr) const;

    // Full merger-side GBL refit of the twin-united winners (twinMergeRefit path). Given the merged
    // tracks (state overwritten in place), the per-output-track united-winner mask filled by
    // filterTracks (>=0 for winners), the CA-ordered module geometry, the shared rechit SoA, the raw
    // OT source (otHits + stacked-sensor frames; null => pixel-only) and the BL material map + bfield,
    // re-fits each united winner's post-union hit list (pixel + absorbed OT extras) via
    // HelixFit::refitMergedTwins and overwrites its state/cov/chi2/ndof, so united tracks carry
    // fitted parameters in the final collection.
    void refitUnitedTracks(TkSoADevice& tracks,
                           const int32_t* unitedWinnerMask,
                           reco::CAGeometrySoACollection const& geometry,
                           reco::TrackingRecHitsSoACollection const& hits,
                           reco::OTRecHitsSoACollection const* otHits,
                           reco::StackedModuleGeometrySoACollection const* stackedGeom,
                           const float* rhoMapDevice,
                           // Normalized (Bz,Br) r-z field-map device buffer (EventSetup condition), or null
                           // to use the scalar field at the origin. Forwarded to the GBL refit.
                           const float* bFieldMapDevice,
                           float bfield,
                           Queue& queue,
                           // When true, removes the refit's single-outlier drop from the emitted hit list
                           // (nHits == nMeasFit); when false the dropped node stays on the track. The
                           // merger passes it true only for the FINAL refit (not the pass-2 sharpen).
                           bool dropOutlierFromHitList = false,
                           // ABSTAIN from dropping anything this round when the largest-pull measured
                           // node is an original pixel-core hit; when false the largest-pull node is
                           // dropped whatever its provenance. The merger passes it true for the final
                           // refit only.
                           bool outlierCoreProtect = false,
                           // Fit-consistent curvature->pT conversion field: the GBL's effective bending
                           // field comes from the fit's own curvature weights; when false it is the
                           // hit-count average of B_bend. Needs bFieldMapDevice, so it is inert without it.
                           bool fieldKernelWeights = false,
                           // Charge-symmetric corrections: the signed arc of the node-0 -> PCA step, and
                           // the bending-field profile deterministic offset.
                           bool chargeSymmetric = false,
                           // Reference-trajectory corrections: the node-0 path length, the arc->azimuth
                           // sign of a measurement-less node, and the field's B_r lambda row.
                           bool trajectoryCorrections = false,
                           // Evaluate Highland's log at the track's TOTAL declared material instead of gap
                           // by gap: theta0^2 does not add over a chain, so the form's one logarithm
                           // belongs at the accumulated total and its variance is apportioned to the gaps
                           // by thickness.
                           bool scatteringLogAtTotal = false,
                           // Cumulative-column typical-loss law: the Landau family is stable under
                           // convolution, so the typical (median) loss of the charged column is the
                           // single-column law at the accumulated thickness; per-node increments keep
                           // every lump placement untouched. When false each lump is charged its own MPV.
                           bool cumulativeEloss = false) const;

    // Final post-refit de-dup (mergerExtend path): given the REFINED merged tracks (attached
    // + refit), remove the same-particle duplicate pairs the pre-attach dedup missed -- dominantly the
    // cross-arm prompt+displaced twins that shared < 2 hits before the merge-stage attach -- via the
    // cov-dedup (shared-hit co-occurrence pairing + covariance-scaled 3-param gate + always-on 0-shared
    // forward fallback), keeping the higher-quality member. Returns a fresh compacted SoA (survivors
    // only); the input is unchanged. nHits/nOTHits size the hit-id key space.
    TkSoADevice finalDedupTracks(TkSoADevice const& refinedTracks,
                                 uint32_t nHits,
                                 uint32_t nOTHits,
                                 Queue& queue,
                                 // Merger GBL-refit inputs, forwarded to finalDedup for the 0-shared
                                 // merge-or-keep-both confirm. Null => the confirm is inert.
                                 const MergerDedupConfirmInputs* confirm = nullptr) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
