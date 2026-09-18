// The masking and merger steps of the two-iteration chain: the launchers of CAHitMaskingAndMergerKernels, a
// class that is not a template, so they live in one translation unit while the topology-templated CA kernels
// are instantiated per topology by CAHitNtupletGeneratorKernels_<Topology>.dev.cc.
#include "CAHitNtupletGeneratorKernelsImpl.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void CAHitMaskingAndMergerKernels::updateMasking(::reco::TrackingRecHitsMaskingView &mask_view,
                                                   const ::reco::TrackSoAConstView &trackd_view,
                                                   const ::reco::TrackHitSoAConstView &trackhitd_view,
                                                   const pixelTrack::Quality minQuality,
                                                   uint32_t const &iterationIndex,
                                                   Queue &queue,
                                                   bool maskAttachedHits) {
    using namespace caHitNtupletGeneratorKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::updateMasking" << std::endl;
#endif

    int threadsPerBlock = 128;
    // max(1, ...) guards against a 0-block launch (invalid CUDA launch configuration)
    // when the track collection is empty (e.g. an empty event in a no-PU sample).
    int blocks = std::max(1, int((trackd_view.metadata().size() + threadsPerBlock - 1) / threadsPerBlock));
    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
    // Parallel updateMasking: fills the already-sized ceil(nTracks/128) grid one thread per track
    // (pure idempotent same-value writes; see Kernel_updateMaskingParallel in Impl.h).
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_updateMaskingParallel{},
                        mask_view,
                        trackd_view,
                        trackhitd_view,
                        minQuality,
                        iterationIndex,
                        maskAttachedHits);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_updateMasking -> done!" << std::endl;
#endif
  }

  void CAHitMaskingAndMergerKernels::filterTracks(::reco::TrackSoAView &track_view,
                                                  ::reco::TrackHitSoAView &trackHit_view,
                                                  const ::reco::TrackSoAConstView &inpTrack_view,
                                                  const ::reco::TrackHitSoAConstView &inpTrackHit_view,
                                                  const pixelTrack::Quality minQuality,
                                                  Queue &queue,
                                                  const int32_t *loserOf,
                                                  const int32_t *isLoser,
                                                  const bool twinMergeRefit,
                                                  const bool refitAllTracks,
                                                  int32_t *unitedMaskOut) {
    using namespace caHitNtupletGeneratorKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::filterTracks" << std::endl;
#endif

    // Parallel Mark -> prefix-sum -> Scatter filter. nIn == 0 (empty event) leaves the output count to the
    // caller, which never reaches this point with an empty input (PixelTracksSoAMerger early-returns).
    const int32_t nIn = int32_t(inpTrack_view.metadata().size());

    if (nIn > 0) {
      const int threadsPerBlock = 128;
      const int blocks = cms::alpakatools::divide_up_by(nIn, threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      // Scratch (freed stream-ordered by the caching allocator after the launches below complete).
      // keep[]/outHitCnt[] are fully written by the Mark kernel over [0,nIn) -> no memset needed.
      auto keep = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);
      auto outHitCnt = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);
      auto tkOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);
      auto hitOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_filterTracksMark{},
                          inpTrack_view,
                          inpTrackHit_view,
                          minQuality,
                          loserOf,
                          isLoser,
                          keep.data(),
                          outHitCnt.data());

      cms::alpakatools::iterativePrefixScan<Acc1D>(keep.data(), tkOff.data(), uint32_t(nIn), queue);
      cms::alpakatools::iterativePrefixScan<Acc1D>(outHitCnt.data(), hitOff.data(), uint32_t(nIn), queue);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_filterTracksScatter{},
                          track_view,
                          trackHit_view,
                          inpTrack_view,
                          inpTrackHit_view,
                          loserOf,
                          twinMergeRefit,
                          refitAllTracks,
                          unitedMaskOut,
                          keep.data(),
                          outHitCnt.data(),
                          tkOff.data(),
                          hitOff.data(),
                          nIn);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "filterTracks -> done!" << std::endl;
#endif
  }

  void CAHitMaskingAndMergerKernels::twinMerge(const ::reco::TrackSoAConstView &inpTrack_view,
                                               const ::reco::TrackHitSoAConstView &inpTrackHit_view,
                                               const int32_t *armOfTrack,
                                               const float qGate3,
                                               const pixelTrack::Quality minQuality,
                                               int32_t *bestTwin,
                                               int32_t *loserOf,
                                               int32_t *isLoser,
                                               int const &nTracks,
                                               Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;
    if (nTracks <= 0)
      return;
    // isLoser must already be zero-initialised by the caller: it is written cross-thread in
    // Kernel_twinConfirm. makeFilteredTracks memsets it before calling this.

    const int threadsPerBlock = 128;
    const int blocks = cms::alpakatools::divide_up_by(nTracks, threadsPerBlock);
    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    // (eta, phi) -> track pre-filter binner over this collection, so twinFindBest visits only the bins
    // overlapping the gate-derived phi AND eta windows instead of all N tracks; a twin has this track's
    // eta to within its own sigma, which is the cheapest cut there is. The trackBinKey clamp keeps
    // every key in range so the binner cannot overflow. nItems = -1 makes the binner iterate the
    // device-side nTracks() instead of the capacity, so the tail, whose eta/phi are uninitialised, is
    // never binned; a candidate there would be rejected at Kernel_twinFindBest's quality gate anyway.
    const int32_t nBin = int32_t(inpTrack_view.metadata().size());
    const uint32_t nKeys = uint32_t(kTwinPhiBins * kTwinEtaSlabs);
    auto phiBinnerBuf = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    auto phiOffBuf = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nKeys + 1);
    auto phiStoreBuf = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, uint32_t(nBin));
    auto phiOvf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1);
    alpaka::memset(queue, phiOvf, 0);
    GenericContainerView view{phiBinnerBuf.data(), phiOffBuf.data(), phiStoreBuf.data(), nKeys + 1, uint32_t(nBin)};
    GenericContainer::template launchZero<Acc1D>(view, queue);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_trackBinCount{},
                        inpTrack_view,
                        int32_t(-1),
                        phiBinnerBuf.data(),
                        kTwinPhiBins,
                        kTwinEtaSlabs,
                        kTwinEtaMax,
                        nKeys,
                        phiOvf.data());
    finalizeAssocOffsets(view, queue);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_trackBinFill{},
                        inpTrack_view,
                        int32_t(-1),
                        phiBinnerBuf.data(),
                        kTwinPhiBins,
                        kTwinEtaSlabs,
                        kTwinEtaMax,
                        nKeys,
                        phiOvf.data());

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_twinFindBest{},
                        inpTrack_view,
                        inpTrackHit_view,
                        armOfTrack,
                        minQuality,
                        qGate3,
                        phiBinnerBuf.data(),
                        kTwinPhiBins,
                        kTwinEtaSlabs,
                        kTwinEtaMax,
                        bestTwin);
    // twinFindBest and twinConfirm cannot be fused: twinConfirm thread i reads bestTwin[j] with
    // j = bestTwin[i], an arbitrary opposite-arm track index, so it needs the whole bestTwin[]
    // finalized, a grid-wide producer to consumer barrier. The inter-kernel queue boundary provides
    // it; alpaka::syncBlockThreads inside a fused kernel is only block-local.
    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_twinConfirm{}, inpTrack_view, bestTwin, loserOf, isLoser);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "CAHitMaskingAndMergerKernels::twinMerge -> done!" << std::endl;
#endif
  }

  void CAHitMaskingAndMergerKernels::finalDedup(::reco::TrackSoAView &out_view,
                                                ::reco::TrackHitSoAView &outHit_view,
                                                const ::reco::TrackSoAConstView &tracks_view,
                                                const ::reco::TrackHitSoAConstView &trackHit_view,
                                                int const &nTracksCap,
                                                uint32_t nHits,
                                                uint32_t nOTHits,
                                                Queue &queue,
                                                const MergerDedupConfirmInputs *confirm) {
    using namespace caHitNtupletGeneratorKernels;
    if (nTracksCap <= 0)
      return;

    // drop[] flags (1 = this refined track is the duplicate loser). Zero-init: the mark kernel only
    // ever sets 0/1 over [0,nTracks), but the compaction reads it, so clear the whole capacity.
    auto drop = cms::alpakatools::make_device_buffer<uint8_t[]>(queue, nTracksCap);
    alpaka::memset(queue, drop, 0);

    // The dedup kernels can tally a per-reason breakdown of the dropped losers into an 18-word device
    // buffer. It is left unarmed: reading it means a device-to-host copy the host consumes at once,
    // i.e. a full stream drain every event, and nothing downstream uses the numbers. A null pointer
    // switches those tallies off at the source.
    uint32_t *const diagPtr = nullptr;

    const int threadsPerBlock = 128;
    const int blocks = cms::alpakatools::divide_up_by(nTracksCap, threadsPerBlock);
    const auto markDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    // Cov-dedup: shared-hit co-occurrence pairing plus a covariance-scaled 3-parameter gate.
    // Hit-id -> refined-track co-occurrence histogram over the key space [0, nHits + nOTHits): merged
    // pixel/strip ids bin on the id, bit30-tagged OT extras compress to nHits + otIdx. The content is
    // the trackHit CSR capacity, so the fill never overruns, and the count-and-clamp guard flags any
    // out-of-range key in ovf[0] instead of writing out of bounds.
    const uint32_t nKeys = nHits + nOTHits;
    const uint32_t nContent = uint32_t(trackHit_view.metadata().size());  // input trackHit CSR capacity
    auto ovf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1);
    alpaka::memset(queue, ovf, 0);

    auto hitAssoc = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    auto hitOff = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nKeys + 1);
    auto hitStore = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nContent);
    GenericContainerView hitView{hitAssoc.data(), hitOff.data(), hitStore.data(), nKeys + 1, nContent};
    GenericContainer::template launchZero<Acc1D>(hitView, queue);
    alpaka::exec<Acc1D>(
        queue, markDiv, Kernel_dedupHitCount{}, tracks_view, trackHit_view, hitAssoc.data(), nHits, nKeys, ovf.data());
    finalizeAssocOffsets(hitView, queue);
    alpaka::exec<Acc1D>(
        queue, markDiv, Kernel_dedupHitFill{}, tracks_view, trackHit_view, hitAssoc.data(), nHits, nKeys, ovf.data());

    // (iii) 0-shared forward fallback eta-phi track binner (always built; the mark kernel drops the
    // in-bound 0-shared forward losers and diag-counts the out-of-bound ones).
    const uint32_t nFbKeys = uint32_t(kDedupFbPhiBins) * uint32_t(kDedupFbEtaSlabs);
    auto fbAssoc = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    auto fbOff = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nFbKeys + 1);
    auto fbStore = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, uint32_t(nTracksCap));
    GenericContainerView fbView{fbAssoc.data(), fbOff.data(), fbStore.data(), nFbKeys + 1, uint32_t(nTracksCap)};
    GenericContainer::template launchZero<Acc1D>(fbView, queue);
    // nItems = -1 => the binner iterates the device-side nTracks() (the refined collection's real
    // count), matching Kernel_dedupCovMark's range so the tail capacity slots are never binned.
    alpaka::exec<Acc1D>(queue,
                        markDiv,
                        Kernel_trackBinCount{},
                        tracks_view,
                        int32_t(-1),
                        fbAssoc.data(),
                        kDedupFbPhiBins,
                        kDedupFbEtaSlabs,
                        kDedupFbEtaMax,
                        nFbKeys,
                        ovf.data());
    finalizeAssocOffsets(fbView, queue);
    alpaka::exec<Acc1D>(queue,
                        markDiv,
                        Kernel_trackBinFill{},
                        tracks_view,
                        int32_t(-1),
                        fbAssoc.data(),
                        kDedupFbPhiBins,
                        kDedupFbEtaSlabs,
                        kDedupFbEtaMax,
                        nFbKeys,
                        ovf.data());

    // Fallback neighbourhood reach, as passed to the dedup kernel: s_scanFbEtaReach = 1 (eta-slab reach de in
    // [-r, r]) and s_scanFbPhiReach = 1 (phi-bin reach dp2 in [-r, r], wrap kept; 2r+1 <= kDedupFbPhiBins).
    // Both are kept runtime-opaque on purpose: constant-folding them on a single-TU backend would unroll the
    // neighbourhood walk and change the float accumulation order. Do not make them constexpr.
    static const int s_scanFbEtaReach = [] {
      volatile int v = 1;
      int r = v;
      if (r < 0)
        r = 0;
      if (r > kDedupFbEtaSlabs)
        r = kDedupFbEtaSlabs;
      return r;
    }();
    static const int s_scanFbPhiReach = [] {
      volatile int v = 1;
      int r = v;
      if (r < 0)
        r = 0;
      if (r > kDedupFbPhiBins / 2)
        r = kDedupFbPhiBins / 2;
      return r;
    }();
    // The duplicate criterion and its |eta| reach come from the merger; without a confirm struct the
    // compile-time defaults apply. The compatibility threshold is not a cfi number: two tracks are
    // kept apart only when their five fitted parameters disagree at 5 sigma (ExtDerivedTables.h).
    const float qGate5 = float(extDerivedTables::kDedupRejectChi2_5);
    const float fbDropBound = (confirm != nullptr) ? confirm->dropAbsEtaMax : kDedupFbDropAbsEtaMax;
    // Hit and stub views for the length and shared counts: the confirm struct's (empty when absent,
    // which makes a stub count as one published rechit keyed on its own id).
    const caStructures::CAHitsView dedupHitView = (confirm != nullptr) ? confirm->hv : caStructures::CAHitsView{};
    const ::reco::StubsConstView dedupStubView = (confirm != nullptr) ? confirm->sv : ::reco::StubsConstView{};
    // The raw outer-tracker rechits, for the sensor kind and the position of a shared published rechit.
    const ::reco::OTRecHitsConstView dedupOTView = (confirm != nullptr) ? confirm->ov : ::reco::OTRecHitsConstView{};

    // One pass: a track is a loser as soon as some better partner is compatible with it. That is a
    // statement about the pair alone, so it needs no iteration and does not depend on whether the
    // killer is itself someone else's loser. The order is a strict total order, so the best member of
    // every duplicate group always survives.
    alpaka::exec<Acc1D>(queue,
                        markDiv,
                        Kernel_dedupCovMark{},
                        tracks_view,
                        trackHit_view,
                        hitAssoc.data(),
                        fbAssoc.data(),
                        nHits,
                        nKeys,
                        drop.data(),
                        diagPtr,
                        qGate5,
                        fbDropBound,
                        s_scanFbEtaReach,
                        s_scanFbPhiReach,
                        dedupHitView,
                        dedupStubView,
                        dedupOTView);

    // Surface any count-and-clamp overflow; never fatal, since clamped writes were skipped and
    // unregistered contested pairs are kept both. The two counters are consumed on device by a
    // one-thread reporter kernel: reading them back would serialize the host against everything
    // queued ahead of the copy, for a diagnostic.
    const auto reportDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, reportDiv, Kernel_dedupOverflowReport{}, ovf.data(), nullptr);

    // Parallel Counts -> prefix-sum -> Scatter compaction. WHICH tracks are dropped is decided in
    // Kernel_dedupCovMark.
    {
      // keep and hitCnt share one 2*nTracksCap allocation with two pointers into it: one allocate/free
      // pair and one memset instead of two of each.
      auto keepAndHitCnt = cms::alpakatools::make_device_buffer<int32_t[]>(queue, 2 * std::size_t(nTracksCap));
      int32_t *keep = keepAndHitCnt.data();
      int32_t *hitCnt = keepAndHitCnt.data() + nTracksCap;
      auto tkOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracksCap);
      auto hitOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracksCap);
      // Trailing entries [nTracks, nTracksCap) are not written by Counts -> zero them so the scans
      // stay constant past the last real track (tkOff[cap-1] == total kept).
      alpaka::memset(queue, keepAndHitCnt, 0);

      alpaka::exec<Acc1D>(queue, markDiv, Kernel_finalDedupCounts{}, tracks_view, drop.data(), keep, hitCnt);

      cms::alpakatools::iterativePrefixScan<Acc1D>(keep, tkOff.data(), uint32_t(nTracksCap), queue);
      cms::alpakatools::iterativePrefixScan<Acc1D>(hitCnt, hitOff.data(), uint32_t(nTracksCap), queue);

      alpaka::exec<Acc1D>(queue,
                          markDiv,
                          Kernel_finalDedupScatter{},
                          out_view,
                          outHit_view,
                          tracks_view,
                          trackHit_view,
                          keep,
                          tkOff.data(),
                          hitOff.data(),
                          nTracksCap);
    }

    // No host wait: the dedup histograms, drop flags and compaction scratch are function-scope caching
    // allocator buffers, and the allocator only re-hands a freed block once the event recorded on its
    // queue at free time has completed, so the queued launches that still read them are safe.
  }

  void CAHitMaskingAndMergerKernels::mergeGather(::reco::TrackSoAView &outTrack_view,
                                                 ::reco::TrackHitSoAView &outHit_view,
                                                 const ::reco::TrackSoAConstView &inp0Track_view,
                                                 const ::reco::TrackHitSoAConstView &inp0Hit_view,
                                                 const ::reco::TrackSoAConstView &inp1Track_view,
                                                 const ::reco::TrackHitSoAConstView &inp1Hit_view,
                                                 int nInputs,
                                                 int32_t *armBuf,
                                                 const int32_t arm0,
                                                 const int32_t arm1,
                                                 Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

    // Grid-stride kernel: every phase is thread-independent, so a multi-block grid is safe (each thread
    // recomputes the per-input offsets from the device-side scalars, and the merged nTracks scalar is
    // written by grid thread 0 and never read inside the kernel). The grid is sized from the dominant
    // copy range, the hit capacity, clamped so tiny events do not launch empty blocks.
    const uint32_t threadsPerBlock = 256;
    const uint32_t hitCap = uint32_t(outHit_view.metadata().size());
    const uint32_t blocks = std::clamp(cms::alpakatools::divide_up_by(std::max(hitCap, 1u), threadsPerBlock), 1u, 128u);
    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_mergeGather{},
                        outTrack_view,
                        outHit_view,
                        inp0Track_view,
                        inp0Hit_view,
                        inp1Track_view,
                        inp1Hit_view,
                        nInputs,
                        armBuf,
                        arm0,
                        arm1);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
