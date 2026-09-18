#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS
// Per-track printf of the fitted chi2 and its inputs; keep off in timed or high-occupancy runs.
// #define CA_CHI2_DUMP

// C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

// Alpaka includes
#include <alpaka/alpaka.hpp>

// CMSSW includes
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"
// Type defined unconditionally (zero memory); the kernel argument and the writes are #ifdef'd.
#include "RecoTracker/PixelSeeding/interface/TripletDumpSoA.h"
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "RecoTracker/PixelSeeding/interface/CATrackFeatures.h"
#include "RecoTracker/PixelSeeding/interface/CAStubMS.h"
#include "CAFitHitSelection.h"

// local includes
#include "CACell.h"
#include "CAExtensionKernels.h"  // caExtension::isOTId / otIdx (OT-tagged hit branches below)
#include "CAHitNtupletGeneratorKernels.h"
#include "CAStructures.h"
#include "CADnnBank.h"
#include "CATrackDNN.h"
#include "CATripletCuts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels {

  using namespace ::caStructures;

  constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  // Gate width of the two-parameter (1/pT, cot(theta)) compatibility check used by the Phase-1
  // specializations of the duplicate removers.
  constexpr float nSigma2Phase1 = 25.f;
  // The five-parameter gate width is the runtime cfi parameter fastDupNSigma2.
  constexpr int nTrackParameters = 5;
  // Per-track hit capacity of the twin-merge union step; matches the in-fit extension's merged-hit cap.
  constexpr int kTwinMaxMergedHits = 32;
  // pi constants (M_PI is not guaranteed on all alpaka device backends)
  constexpr float kTwinPi = 3.14159265358979323846f;
  constexpr float kTwinTwoPi = 2.f * kTwinPi;
  // map: index of a track parameter -> index of its covariance
  HOST_DEVICE_CONSTANT std::array<uint8_t, nTrackParameters> iParam2iCov = {0u, 5u, 9u, 12u, 14u};
  // cotTheta's own variance in that packed layout (parameter 3).
  constexpr int kCovCotCot = 12;

  // (eta, phi) bins of the twinFindBest pre-filter. The bins are candidate generation only -- the
  // covariance gate is the physics -- and the sweep below sizes its window from that gate, so the
  // binning only has to be fine enough to be worth traversing: 2pi/128 ~ 0.049 rad in phi and 0.18 in
  // eta, against gate windows of a few times the track's own sigma.
  constexpr int kTwinPhiBins = 128;
  constexpr int kTwinEtaSlabs = 50;
  constexpr float kTwinEtaMax = 4.5f;
  // eta-phi binner of the 0-shared forward fallback; candidate generation only, the cov gate is the physics gate.
  constexpr int kDedupFbPhiBins = 128;
  constexpr int kDedupFbEtaSlabs = 50;
  constexpr float kDedupFbEtaMax = 4.0f;
  // Fallback drop authority is bounded to |eta| <= this: beyond it the covariance is at its widest, so
  // dropping costs efficiency. Candidates beyond the bound are counted in the diagnostics, never dropped.
  constexpr float kDedupFbDropAbsEtaMax = 2.5f;
  // Post-refit cov-dedup nSigma^2 gate; acts on the refitted covariance, unlike the twin gate.
  constexpr float kDedupNSigma2Default = 25.f;
  // Shared-hit fraction above which two tracks are the same track: CMS reconstruction's own duplicate
  // convention, ShareFrac = 0.19 in TrackListMerger (RecoTracker/FinalTrackSelectors), the pixel cleaner
  // being the same rule at small hit counts. Not a knob. As in TrackListMerger (allowFirstHitShare), the
  // innermost shared hit is taken out of both sides of the fraction.
  constexpr float kDedupShareFrac = 0.19f;
  // yerrLocal is a VARIANCE, and it separates the outer tracker's two sensor kinds by itself: a 5 cm
  // strip leaves its along-strip coordinate with a variance of 5^2/12 = 2.1 cm^2, a 1.5 mm macro-pixel
  // with 0.0019 cm^2. Above this bound the rechit measures one coordinate and leaves the other free.
  constexpr float kStripYVarMin = 0.1f;
  // |eta| boundary of the central/forward dedup diagnostics; not a physics gate.
  constexpr float kDedupFwdEta = 1.3f;
  // A fit that does not describe its own hits cannot be trusted to report the covariance it reports,
  // so it may not win a duplicate comparison on that covariance. The bound is the same 5-parameter
  // 5-sigma rejection the duplicate test itself uses (ExtDerivedTables.h kDedupRejectChi2_5 = 37.09)
  // per degree of freedom; chi2() is already chi2/ndof everywhere in this SoA.
  constexpr float kDedupMaxChi2Ndof = 37.0948f / 5.f;
  // Capacity of the merge-or-keep-both contested-pair list: at most one pair per loser track. Pairs
  // beyond the cap are counted and kept both.
  constexpr uint32_t kDedupConfirmMaxPairs = 1024u;

  // Bin a track by (eta, phi). nEtaSlabs <= 1 gives pure phi binning; otherwise the key is
  // etaSlab * nPhiBins + phiBin. Shared by the fill and the mark/scan kernels.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int trackBinKey(float eta, float phi, int nPhiBins, int nEtaSlabs, float etaMax) {
    float f = (phi + kTwinPi) * (float(nPhiBins) / kTwinTwoPi);
    int pb = int(f);
    if (pb < 0)
      pb = 0;
    if (pb >= nPhiBins)
      pb = nPhiBins - 1;
    if (nEtaSlabs <= 1)
      return pb;
    float g = (eta + etaMax) * (float(nEtaSlabs) / (2.f * etaMax));
    int eb = int(g);
    if (eb < 0)
      eb = 0;
    if (eb >= nEtaSlabs)
      eb = nEtaSlabs - 1;
    return eb * nPhiBins + pb;
  }

  // all of these below are mostly to avoid carrying around the relative namespace

  using Quality = ::pixelTrack::Quality;
  using TkSoAView = ::reco::TrackSoAView;
  using TkHitSoAView = ::reco::TrackHitSoAView;

  template <typename TrackerTraits>
  using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

  using Counters = caHitNtupletGenerator::Counters;
  using HitToTuple = caStructures::GenericContainer;
  using HitContainer = caStructures::SequentialContainer;
  using TupleMultiplicity = caStructures::GenericContainer;
  using HitToCell = caStructures::GenericContainer;
  using CellToCell = caStructures::GenericContainer;
  using CellToTrack = caStructures::GenericContainer;

  using namespace cms::alpakatools;

  // Templated on the module-start view: the ModulesMultiView, or the CAHitsView facade, which
  // answers moduleStartOf() out of the pixel and stub module blocks.
  template <typename ModulesView>
  class SetHitsLayerStart {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ModulesView &mm,
                                  const reco::CALayersSoAConstView &ll,
                                  uint32_t *__restrict__ hitsLayerStart) const {
      ALPAKA_ASSERT_ACC(0 == caStructures::moduleStartOf(mm, 0));

      for (int32_t i : cms::alpakatools::uniform_elements(acc, ll.metadata().size())) {
        hitsLayerStart[i] = caStructures::moduleStartOf(mm, ll.layerStarts()[i]);
#ifdef GPU_DEBUG
        int old = i == 0 ? 0 : caStructures::moduleStartOf(mm, ll.layerStarts()[i - 1]);
        printf("LayerStart %d/%d at module %d: %d - %d\n",
               i,
               ll.metadata().size() - 1,
               ll.layerStarts()[i],
               hitsLayerStart[i],
               hitsLayerStart[i] - old);
#endif
      }
    }
  };

  template <typename HitsView>
  class Kernel_printSizes {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsView hh,
                                  TkSoAView tt,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks) const {
      if (cms::alpakatools::once_per_grid(acc))
        printf(
            "nSizes: hh.size() %d; hh.size() - hh.view(0).offsetBPIX2() %d; nCells %d; nTrips %d; "
            "nCellTracks %d; nTracks %d; tt.metadata().size() %d\n",
            static_cast<int>(hh.size()),
            static_cast<int>(hh.size()) - hh.view(0).offsetBPIX2(),
            *nCells,
            *nTrips,
            *nCellTracks,
            tt.nTracks(),
            tt.metadata().size());
    }
  };

  template <typename TrackerTraits>
  class Kernel_checkOverflows {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  HitToTuple const *hitToTuple,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks,
                                  caStructures::CAPairSoAConstView cellCell,
                                  caStructures::CAPairSoAConstView cellTrack,
                                  int32_t nHits,
                                  uint32_t maxNumberOfDoublets,
                                  AlgoParams const &params,
                                  Counters *counters) const {
      auto &c = *counters;
      // counters once per event
      if (cms::alpakatools::once_per_grid(acc)) {
        alpaka::atomicAdd(acc, &c.nEvents, 1ull, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &c.nHits, static_cast<unsigned long long>(nHits), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &c.nCells, static_cast<unsigned long long>(*nCells), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(
            acc, &c.nTuples, static_cast<unsigned long long>(apc->get().first), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc,
                          &c.nFitTracks,
                          static_cast<unsigned long long>(tupleMultiplicity->size()),
                          alpaka::hierarchy::Blocks{});
      }

#ifdef NTUPLE_DEBUGS
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("number of found cells %d \n found tuples %d with total hits %d out of %d\n",
               *nCells,
               apc->get().first,
               apc->get().second,
               nHits);
        if (apc->get().first < tracks_view.metadata().size()) {
          ALPAKA_ASSERT_ACC(foundNtuplets->size(apc->get().first) == 0);
          ALPAKA_ASSERT_ACC(foundNtuplets->size() == apc->get().second);
        }
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) > TrackerTraits::maxHitsOnTrack)  // current real limit
          printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
        ALPAKA_ASSERT_ACC(foundNtuplets->size(idx) <= TrackerTraits::maxHitsOnTrack);
        for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
          ALPAKA_ASSERT_ACC(int(*ih) < nHits);
      }
#endif

      if (cms::alpakatools::once_per_grid(acc)) {
        // Non-corrupting: the build kernels already clamp; these counters surface the magnitude.
        if (apc->get().first >= uint32_t(tracks_view.metadata().size())) {
          printf("Tuples overflow\n");
          alpaka::atomicAdd(acc, &c.nTupleOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nCells >= maxNumberOfDoublets) {
          printf("Cells overflow\n");
          alpaka::atomicAdd(acc, &c.nCellOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nTrips >= uint32_t(cellCell.metadata().size())) {
          printf("Triplets overflow\n");
          alpaka::atomicAdd(acc, &c.nTripletOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nCellTracks >= uint32_t(cellTrack.metadata().size())) {
          printf("TracksToCell overflow\n");
          alpaka::atomicAdd(acc, &c.nCellTrackOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (thisCell.hasFishbone() && !thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nFishCells, 1ull, alpaka::hierarchy::Blocks{});
        if (thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nKilledCells, 1ull, alpaka::hierarchy::Blocks{});
        if (!thisCell.unused())
          alpaka::atomicAdd(acc, &c.nEmptyCells, 1ull, alpaka::hierarchy::Blocks{});
        if ((0 == hitToTuple->size(thisCell.inner_hit_id())) && (0 == hitToTuple->size(thisCell.outer_hit_id())))
          alpaka::atomicAdd(acc, &c.nZeroTrackCells, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Always-on overflow sentinel: the capacity guards in the build kernels truncate silently, so this kernel
  // tests the same conditions once per event into a per-stream 8-word buffer reported at endStream.
  //   accum[0] = tuple-count overflow          (apc.first   >= tracks capacity)
  //   accum[1] = doublet/cell overflow         (nCells      >= maxNumberOfDoublets)
  //   accum[2] = cellToCell overflow           (nTriplets   >= cellCell capacity)
  //   accum[3] = cellToTrack overflow          (nCellTracks >= cellTrack capacity)
  //   accum[4] = hitContainer content overflow (apc.second  >  content slots)
  //   accum[5] = hitToTuple content overflow   (apc.second  >  its storage extent; UINT32_MAX disables it)
  class Kernel_overflowSentinel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  cms::alpakatools::AtomicPairCounter const *apc,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks,
                                  uint32_t tracksCap,
                                  uint32_t maxNumberOfDoublets,
                                  uint32_t cellCellCap,
                                  uint32_t cellTrackCap,
                                  uint32_t hitContentCap,
                                  uint32_t hitToTupleContentCap,
                                  uint32_t *__restrict__ accum) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        if (apc->get().first >= tracksCap)
          alpaka::atomicAdd(acc, &accum[0], 1u, alpaka::hierarchy::Blocks{});
        if (*nCells >= maxNumberOfDoublets)
          alpaka::atomicAdd(acc, &accum[1], 1u, alpaka::hierarchy::Blocks{});
        if (*nTrips >= cellCellCap)
          alpaka::atomicAdd(acc, &accum[2], 1u, alpaka::hierarchy::Blocks{});
        if (*nCellTracks >= cellTrackCap)
          alpaka::atomicAdd(acc, &accum[3], 1u, alpaka::hierarchy::Blocks{});
        if (apc->get().second > hitContentCap)
          alpaka::atomicAdd(acc, &accum[4], 1u, alpaka::hierarchy::Blocks{});
        if (hitToTupleContentCap != 0xFFFFFFFFu && apc->get().second > hitToTupleContentCap)
          alpaka::atomicAdd(acc, &accum[5], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fishboneCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view) const {
      constexpr auto reject = Quality::dup;

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (!thisCell.isKilled())
          continue;

        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);
        for (auto i = 0u; i < cellTracksHisto->size(idx); i++)
          tracks_view[tracksOfCell[i]].quality() = reject;
      }
    }
  };

  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  // Work division: Acc2D with Y indexing cells and X indexing warp lanes
  // (warpSize threads per cell). All lanes of a warp cooperate on a single cell
  template <typename TrackerTraits>
  class Kernel_earlyDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const &acc,
                                  CACell<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      constexpr auto reject = Quality::edup;  /// cannot be loose
      ALPAKA_ASSERT_ACC(nCells);

      const int32_t warpSize = alpaka::warp::getSize(acc);
      const int32_t laneId = static_cast<int32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u]);

      for (uint32_t idx : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
#ifdef CA_SIZES
        if (laneId == 0)
          printf("cellTracksSizes;%d;%d;%d\n", idx, cT.size(), cT.capacity());
#endif
        const int ntr = static_cast<int>(cellTracksHisto->size(idx));
        if (ntr < 2)
          continue;

        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);

        // Warp-reduce maxNl over the cell's tracks.
        // Lanes scan a strided subset of the cell's tracks and hold a local maxNl in register
        int32_t localMax = 0;
        for (int k = laneId; k < ntr; k += warpSize) {
          const int32_t nl = tracks_view[tracksOfCell[k]].nLayers();
          if (nl > localMax)
            localMax = nl;
        }
        // Warp-reduce to find the maxNl across all lanes. The result is uniform across the warp.
        // Idle lanes start with 0 and do not influence the result.
        // All lanes must be active for the shuffle to work: no branching or return early here.
        for (int32_t off = 1; off < warpSize; off <<= 1) {
          const int32_t y = alpaka::warp::shfl_xor(acc, localMax, off);
          if (y > localMax)
            localMax = y;
        }
        const int32_t maxNl = localMax;

        // Process tracks sequentially using warps
        for (int i = 0; i < ntr; ++i) {
          const auto it = tracksOfCell[i];
          const int32_t nli = tracks_view[it].nLayers();
          // Same nli and maxNl across lanes, so uniform check and no early return here to keep all lanes active.
          if (nli >= maxNl) {
            continue;
          }

          // Look for compatible tracks in the same cell with fewer layers and similar curvature
          // tracks_view[].pt() holds the PRE-FIT CURVATURE here, not a pT, and the demotion is terminal
          // (Kernel_fillMultiplicity skips Quality::edup), so the window is topology dependent: Phase2OTStubs
          // uses a window RELATIVE to the compared curvatures, since an absolute |dcurv| window accepts every
          // pair at high pT and would delete the shorter track of every collimated pair; the other topologies
          // keep the absolute window.
          constexpr bool kHasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
          constexpr float kEarlyDupRelCurv = 0.05f;       // relative window, stub topology
          constexpr float kEarlyDupAbsCurv2 = 0.000001f;  // absolute |dcurv|^2 window, upstream
          // Same value as CACell<TrackerTraits>::kUninitializeCurvature for every topology.
          constexpr float kUninitCurv = std::numeric_limits<float>::max();
          const float curvi = tracks_view[it].pt();
          bool foundCompatible = false;
          // Parallelize inner loop across lanes
          for (int j = laneId; j < ntr; j += warpSize) {
            const auto jt = tracksOfCell[j];
            if (tracks_view[jt].nLayers() <= nli)
              continue;  // need a strictly longer companion
            const float curvj = tracks_view[jt].pt();
            const float dcurv = curvi - curvj;
            if constexpr (kHasStubs) {
              // An uninitialised pre-fit curvature must never be compatible with anything.
              if (curvi == kUninitCurv || curvj == kUninitCurv)
                continue;
              const float thr = kEarlyDupRelCurv * (std::abs(curvi) + std::abs(curvj));
              if (dcurv * dcurv <= thr * thr) {
                foundCompatible = true;
                break;
              }
            } else {
              if (dcurv * dcurv <= kEarlyDupAbsCurv2) {
                foundCompatible = true;
                break;
              }
            }
          }
          // All lanes converge here to check if any foundCompatible is true, and if so, mark track as duplicate.
          if (alpaka::warp::any(acc, static_cast<int32_t>(foundCompatible))) {
            // One thread assigns warp-wide decision
            if (laneId == 0) {
              tracks_view[it].quality() = reject;
            }
          }
        }
      }
    }
  };

  // Specialization for Phase-1 to keep the same behavior as before.
  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  class Kernel_earlyDuplicateRemoverPhase1 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<pixelTopology::Phase1> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      constexpr auto reject = Quality::edup;  /// cannot be loose
      ALPAKA_ASSERT_ACC(nCells);
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef CA_SIZES
        printf("cellTracksSizes;%d;%d;%d\n", idx, cT.size(), cT.capacity());
#endif
        if (cellTracksHisto->size(idx) < 2)
          continue;

        int8_t maxNl = 0;
        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);

        // find maxNl
        for (auto i = 0u; i < cellTracksHisto->size(idx); i++) {
          if (int(tracksOfCell[i]) > tracks_view.metadata().size())
            printf(">WARNING: %d %d %d %d\n", idx, i, int(tracksOfCell[i]), tracks_view.metadata().size());
          auto nl = tracks_view[tracksOfCell[i]].nLayers();
          maxNl = std::max(nl, maxNl);
        }

        // if (maxNl<4) continue;
        // quad pass through (leave it here for tests)
        //  maxNl = std::min(4, maxNl);

        for (auto i = 0u; i < cellTracksHisto->size(idx); i++) {
          auto it = tracksOfCell[i];

          if (int(it) > tracks_view.metadata().size())
            printf(">WARNING: %d %d %d\n", i, it, tracks_view.metadata().size());
          if (tracks_view[it].nLayers() < maxNl)
            tracks_view[it].quality() = reject;  // no race: simple assignment of the same constant
        }
      }
    }
  };

  // Order/backend-independent duplicate removal: a track's final quality must not depend on the order
  // concurrent threads run in. Two disciplines share the int32 quality scratch (device_qualityScratch_)
  // and the two helpers below:
  //   - Kernel_fastDuplicateRemover is cell-parallel (several threads may demote the same shared track):
  //     it reads quality(), accumulates demotions into the scratch via atomicMin, and Kernel_applyQuality
  //     copies the scratch back into quality()
  //   - The hit-based removers (rejectDuplicate, sharedHitCleaner, triplet/simpleTripletCleaner) are
  //     track-parallel single-writers: Kernel_snapshotQuality freezes quality() into the scratch, then
  //     each thread reads that snapshot and writes only its own track's quality() (no atomics, no copy-back)
  class Kernel_snapshotQuality {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t *__restrict__ qualityScratch) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes()))
        qualityScratch[i] = static_cast<int32_t>(tracks_view[i].quality());
    }
  };

  class Kernel_applyQuality {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes()))
        tracks_view[i].quality() = static_cast<Quality>(qualityScratch[i]);
    }
  };

  // Two-tier work division for Kernel_fastDuplicateRemover: the kernel does O(ntr^2) work per cell and ntr has
  // a long tail, so tier 1 (ntr <= kDupCoopMinTracks) gives one thread to a cell while tier 2 gives the whole
  // warp to one heavy cell at a time (one ballot per grid step). The file is built with -Ofast, so the two
  // tiers stay bit-identical only by evaluating the same floating-point expressions in the same order:
  // fastDupRemoverCell re-reads score(it) inside the loop and runs the maxQual / min-chi2 passes redundantly on
  // every lane, so no reduction or shuffle touches a float. The collectives are convergent: the block size is
  // a multiple of the warp size and the grid-stride loop runs up to round_up_by(*nCells, warpSize).
  inline constexpr int kDupCoopMinTracks = 64;

  // Per-cell body of Kernel_fastDuplicateRemover.
  //   Coop == false: (iFirst, iStep) = (0, 1)              -> serial loops, one thread per cell
  //   Coop == true : (iFirst, iStep) = (laneId, warpSize)  -> one warp per cell
  // Redistributing the i's cannot change the result: the kernel only reads tracks_view (the values frozen
  // by Kernel_snapshotQuality) and its only writes are atomicMin on qualityScratch, commutative and
  // idempotent; the `break` after a demotion is an early exit only.
  template <bool Coop>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fastDupRemoverCell(Acc1D const &acc,
                                                         CellToTrack const *__restrict__ cellTracksHisto,
                                                         TkSoAView tracks_view,
                                                         int32_t *__restrict__ qualityScratch,
                                                         uint32_t cellIdx,
                                                         int ntr,
                                                         int lane,
                                                         int stride,
                                                         Quality reject,
                                                         float fastDupNSigma2) {
    constexpr auto loose = Quality::loose;

    auto score = [&](uint32_t it) { return tracks_view[it].chi2(); };
    auto demote = [&](uint32_t it, Quality q) {
      alpaka::atomicMin(acc, &qualityScratch[it], static_cast<int32_t>(q), alpaka::hierarchy::Blocks{});
    };

    auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(cellIdx);

    // The i's this thread owns; compile-time constants for tier 1.
    const int iFirst = Coop ? lane : 0;
    const int iStep = Coop ? stride : 1;

    // Demote any track dominated by a compatible, strictly better one (higher quality, or equal
    // quality and lower chi2); each track tests all others and exact ties keep both
    for (int i = iFirst; i < ntr; i += iStep) {
      auto it = thisCellTracks[i];
      auto qi = tracks_view[it].quality();
      if (qi <= reject)
        continue;

      // get track parameters and covariances
      float iParams[nTrackParameters];
      float iCovs[nTrackParameters];
      for (int p{0}; p < nTrackParameters; ++p) {
        iParams[p] = tracks_view[it].state()(p);
        iCovs[p] = tracks_view[it].covariance()(iParam2iCov[p]);
      }
      // function that compares the five track parameters of tracks it and jt
      auto incompatibleTrackParams = [&](uint32_t jt) -> bool {
        // comparing phi, tip, 1/pT, cotan(theta) and zip
        for (int p{0}; p < nTrackParameters; ++p) {
          const auto dpij = iParams[p] - tracks_view[jt].state()(p);
          const auto e2dpij = fastDupNSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
          if (dpij * dpij > e2dpij)
            return true;  // incompatible param found
        }
        return false;  // all params compatible
      };

      for (int j = 0; j < ntr; ++j) {
        if (j == i)
          continue;
        auto jt = thisCellTracks[j];
        auto qj = tracks_view[jt].quality();
        if (qj <= reject)
          continue;
        if (incompatibleTrackParams(jt))
          continue;
        if ((qj > qi) || (qj == qi && score(jt) < score(it))) {
          demote(it, reject);
          break;
        }
      }
    }

    // Run whole by every lane: no reduction, so maxQual comes out of the same code on every lane.
    auto maxQual = reject;  // no duplicate!
    for (int i = 0; i < ntr; i++) {
      auto q = tracks_view[thisCellTracks[i]].quality();
      if (q > maxQual)
        maxQual = q;
    }

    if (maxQual <= loose)
      return;  // warp-uniform when Coop: every lane ran the same loop over the same data

    // min chi2 among the best-quality tracks (read from the unmodified quality, which the dup-marking
    // above does not affect for the max-quality min-chi2 track)
    // run whole by every lane, so mc is bit-for-bit the same on every lane
    float mc = maxScore;
    for (int i = 0; i < ntr; i++) {
      auto it = thisCellTracks[i];
      if (tracks_view[it].quality() == maxQual && score(it) < mc)
        mc = score(it);
    }

    // mark all other duplicates (keep them loose); same test on every lane, only the writes are distributed
    for (int i = iFirst; i < ntr; i += iStep) {
      auto it = thisCellTracks[i];
      if (tracks_view[it].quality() > loose && score(it) > mc)
        demote(it, loose);
    }
  }

  // assume the above (so, short tracks already removed)
  // Work division: one cell per thread, with the whole warp ganging up on the cells whose track list is
  // longer than kDupCoopMinTracks.
  template <typename TrackerTraits>
  class Kernel_fastDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  int32_t *__restrict__ qualityScratch,
                                  bool dupPassThrough,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      ALPAKA_ASSERT_ACC(nCells);
      const uint32_t ntNCells = (*nCells);

      const int warpSize = static_cast<int>(alpaka::warp::getSize(acc));
      const int laneId = static_cast<int>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] % uint32_t(warpSize));
      // Invariant (a): the launcher must use a block size that is a multiple of the warp size.
      ALPAKA_ASSERT_ACC((0u == alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] % uint32_t(warpSize)));
      // Invariant (b): lane-aligned extent, so a warp's lanes share their trip count.
      const uint32_t extent = cms::alpakatools::round_up_by(ntNCells, uint32_t(warpSize));

      for (auto idx : cms::alpakatools::uniform_elements(acc, extent)) {
        const bool inRange = (idx < ntNCells);
        const int ntr = inRange ? static_cast<int>(cellTracksHisto->size(idx)) : 0;

        // tier 2: hand the heavy cells of this warp to the whole warp, one at a time. The mask is
        // warp-uniform, so this loop and the collectives inside it are convergent.
        auto heavyMask = alpaka::warp::ballot(acc, (ntr > kDupCoopMinTracks) ? 1 : 0);
        using MaskT = decltype(heavyMask);
        if (heavyMask) {
          for (int l = 0; l < warpSize; ++l) {
            if (MaskT{0} == ((heavyMask >> l) & MaskT{1}))
              continue;
            const uint32_t cell = static_cast<uint32_t>(alpaka::warp::shfl(acc, static_cast<int32_t>(idx), l));
            const int n = alpaka::warp::shfl(acc, ntr, l);
            fastDupRemoverCell<true>(
                acc, cellTracksHisto, tracks_view, qualityScratch, cell, n, laneId, warpSize, reject, fastDupNSigma2);
          }
        }

        // tier 1: one cell per thread
        if (inRange && ntr >= 2 && ntr <= kDupCoopMinTracks)
          fastDupRemoverCell<false>(
              acc, cellTracksHisto, tracks_view, qualityScratch, idx, ntr, 0, 1, reject, fastDupNSigma2);
      }
    }
  };

  // Phase-1 specialization
  template <>
  class Kernel_fastDuplicateRemover<pixelTopology::Phase1> {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<pixelTopology::Phase1> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  int32_t *__restrict__ qualityScratch,
                                  bool dupPassThrough,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      constexpr auto loose = Quality::loose;

      ALPAKA_ASSERT_ACC(nCells);
      const auto ntNCells = (*nCells);

      auto score = [&](uint32_t it) { return std::abs(reco::tip(tracks_view, it)); };
      auto demote = [&](uint32_t it, Quality q) {
        alpaka::atomicMin(acc, &qualityScratch[it], static_cast<int32_t>(q), alpaka::hierarchy::Blocks{});
      };

      for (auto idx : cms::alpakatools::uniform_elements(acc, ntNCells)) {
        int ntr = cellTracksHisto->size(idx);
        if (ntr < 2)
          continue;

        auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(idx);

        // Mark as duplicate any track dominated by a compatible, strictly better one
        // (order-independent; lower track index breaks exact ties)
        for (int i = 0; i < ntr; ++i) {
          auto it = thisCellTracks[i];
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;
          auto opi = tracks_view[it].state()(2);
          auto e2opi = tracks_view[it].covariance()(9);
          auto cti = tracks_view[it].state()(3);
          auto e2cti = tracks_view[it].covariance()(12);
          for (int j = 0; j < ntr; ++j) {
            if (j == i)
              continue;
            auto jt = thisCellTracks[j];
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            auto opj = tracks_view[jt].state()(2);
            auto ctj = tracks_view[jt].state()(3);
            auto dct = nSigma2Phase1 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto dop = nSigma2Phase1 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            if ((qj > qi) || (qj == qi && (score(jt) < score(it) || (score(jt) == score(it) && jt < it)))) {
              demote(it, reject);
              break;
            }
          }
        }

        // find maxQual
        auto maxQual = reject;  // no duplicate!
        for (int i = 0; i < ntr; i++) {
          auto q = tracks_view[thisCellTracks[i]].quality();
          if (q > maxQual)
            maxQual = q;
        }

        if (maxQual <= loose)
          continue;

        // keep the single best-quality, min-score track (lower index breaks ties); demote the rest
        float mc = maxScore;
        uint32_t im = tkNotFound;
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() == maxQual) {
            auto s = score(it);
            if (s < mc || (s == mc && it < im)) {
              mc = s;
              im = it;
            }
          }
        }

        if (tkNotFound == im)
          continue;

        // mark all other duplicates (keep them loose)
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() > loose && it != im)
            demote(it, loose);
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_connect {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc2D const &acc,
                                  cms::alpakatools::AtomicPairCounter *apc,  // just to zero them
                                  HitsMultiView hh,
                                  reco::CAGraphSoAConstView cc,
                                  reco::CATripletCutsSoAConstView tripletCuts,
                                  bool useTripletDNN,
                                  float tripletDNNThreshold,
                                  DnnBank tripletBank,  // per-iteration compile-time weight bank
#ifdef CA_TRIPLET_DUMP
                                  caStructures::TripletDumpSoAView tripletDump,  // per-triplet feature capture
                                  int dumpIteration,  // dump builds only; see CATripletCuts.h
#endif
                                  caStructures::CAPairSoAView cn,
                                  CACell<TrackerTraits> *cells,
                                  uint32_t const *nCells,
                                  uint32_t *nTrips,
                                  HitToCell const *__restrict__ outerHitHisto,
                                  CellToCell *cellNeighborsHisto,
                                  uint32_t *__restrict__ pipelineCounters) const {
      using Cell = CACell<TrackerTraits>;
      uint32_t maxTriplets = cn.metadata().size();

      if (cms::alpakatools::once_per_grid(acc)) {
        *apc = 0;
      }  // ready for next kernel

      // loop on outer cells
      for (uint32_t oCellIndex : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
        auto &outerCell = cells[oCellIndex];
        auto middleHitId = outerCell.inner_hit_id() - hh.view(0).offsetBPIX2();

        if (int(middleHitId) < 0)
          continue;

        auto const *__restrict__ outerHitCells = outerHitHisto->begin(middleHitId);
        auto const numberOfPossibleNeighbors = outerHitHisto->size(middleHitId);

        // Per-layer-pair triplet cut rows: the RZ-alignment tolerance and the stub columns are anchored at
        // the outer pair (L2,L3), whose inner layer is the triplet's middle layer, while the beam-spot
        // (DCA, floorDCA) cut is anchored at the inner pair (L1,L2). See TripletCuts::accept.
        auto tripletVectorCutsCol = tripletCuts[outerCell.layerPairId()];
        auto skips = cc[outerCell.layerPairId()].skipsLayers();

#ifdef CA_DEBUG
        printf("numberOfPossibleFromHisto;%d;%d;%d;%d;%d\n",
               *nCells,
               middleHitId,
               oCellIndex,
               outerCell.innerLayer(cc),
               numberOfPossibleNeighbors);
#endif

        // loop on inner cells
        for (uint32_t j : cms::alpakatools::independent_group_elements_x(acc, numberOfPossibleNeighbors)) {
          auto iCellIndex = outerHitCells[j];
          auto &innerCell = cells[iCellIndex];
          float curvature = 0.f;

          // cc (the CA layer-pair graph) supplies the per-hit CA layer ids for the DNN layer-gap features
#ifdef CA_TRIPLET_DUMP
          float dumpFeat[18] =
              {};  // accept() fills 18 BASE DNN features; written to SoA below (zero-init defense-in-depth)
          float dumpScore = -1.f;  // accept() fills the in-kernel DNN score (consistency check)
#endif
          if (TripletCuts<TrackerTraits>::accept(acc,
                                                 innerCell,
                                                 outerCell,
                                                 curvature,
                                                 hh,
                                                 tripletCuts,
                                                 tripletVectorCutsCol,
                                                 tripletCuts[innerCell.layerPairId()],
                                                 cc,
                                                 useTripletDNN,
                                                 tripletDNNThreshold,
                                                 tripletBank,
#ifdef CA_TRIPLET_DUMP
                                                 dumpFeat,
                                                 &dumpScore,
                                                 dumpIteration,
#endif
                                                 pipelineCounters)) {
            auto t_ind = alpaka::atomicAdd(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});

#ifdef CA_DEBUG
            printf("Triplet no. %d %.5f %.5f (%d %d) - %d %d -> (%d, %d, %d, %d) \n",
                   t_ind,
                   thetaCut,
                   dcaCut,
                   outerCell.layerPairId(),
                   innerCell.layerPairId(),
                   iCellIndex,
                   oCellIndex,
                   outerCell.inner_hit_id(),
                   outerCell.outer_hit_id(),
                   innerCell.inner_hit_id(),
                   innerCell.outer_hit_id());
            printf("filling cell no. %d %d: %d -> %d\n", t_ind, cellNeighborsHisto->size(), iCellIndex, oCellIndex);
#endif

            if (t_ind >= maxTriplets) {
#ifdef CA_WARNINGS
              printf("Warning!!!! Too many cell->cell (triplets) associations (limit = %d)!\n", cn.metadata().size());
#endif
              alpaka::atomicSub(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});
              break;
            }

#ifdef CA_TRIPLET_DUMP
            // Per-built-triplet training row: 18 base features + the three merged-hit indices (truth join
            // key) + CA layers. t_ind < maxTriplets is guaranteed by the guard above.
            {
              auto row = tripletDump[t_ind];
              row.absCurvature() = dumpFeat[0];
              row.tipTimesCurvature() = dumpFeat[1];
              row.dca() = dumpFeat[2];
              row.curvatureStubs() = dumpFeat[3];
              row.curvatureStubsErrSquared() = dumpFeat[4];
              row.curvature13() = dumpFeat[5];
              row.dPhi12() = dumpFeat[6];
              row.dPhi13() = dumpFeat[7];
              row.dPhi23() = dumpFeat[8];
              row.dr12() = dumpFeat[9];
              row.dr13() = dumpFeat[10];
              row.r1() = dumpFeat[11];
              row.r2() = dumpFeat[12];
              row.r3() = dumpFeat[13];
              row.z1() = dumpFeat[14];
              row.z2() = dumpFeat[15];
              row.z3() = dumpFeat[16];
              row.nStubs() = dumpFeat[17];
              row.curvature() = curvature;  // SIGNED (Kernel_connect local, by-ref from accept); for derived feats
              row.lay1() = int32_t(innerCell.innerLayer(cc));
              row.lay2() = int32_t(outerCell.innerLayer(cc));
              row.lay3() = int32_t(outerCell.outerLayer(cc));
              row.h1() = uint32_t(innerCell.inner_hit_id());
              row.h2() = uint32_t(outerCell.inner_hit_id());
              row.h3() = uint32_t(outerCell.outer_hit_id());
              row.iter() = dumpIteration;
              row.inKernelScore() = dumpScore;
            }
#endif

            // One bin per cell (bin = iCellIndex). The non-layer-skipping vs
            // layer-skipping distinction is encoded in bit 31 of the stored
            // outer-cell index:
            //   bit 31 = 0 -> non-layer-skipping neighbor
            //   bit 31 = 1 -> layer-skipping neighbor
            // Key-range guard: iCellIndex is below the cell count the histogram was sized from, so a sizing
            // mismatch drops the association instead of writing outside off[].
            if (iCellIndex < cellNeighborsHisto->nOnes())
              cellNeighborsHisto->count(acc, iCellIndex);

            cn[t_ind].inner() = iCellIndex;
            cn[t_ind].outer() = oCellIndex | (skips ? caStructures::kSkipsLayerFlag : 0u);
            outerCell.setStatusBits(Cell::StatusBit::kUsed);
            innerCell.setStatusBits(Cell::StatusBit::kUsed);

            // Pipeline stage counters: classify triplet by hit types
            if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
              if (pipelineCounters) {
                using PC = caHitNtupletGenerator::PipelineCounter;
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsTotal], 1u, alpaka::hierarchy::Blocks{});
                auto hit1 = innerCell.inner_hit_id();
                auto hit2 = outerCell.inner_hit_id();
                auto hit3 = outerCell.outer_hit_id();
                int nStubs = (isStub(hh, hit1) ? 1 : 0) + (isStub(hh, hit2) ? 1 : 0) + (isStub(hh, hit3) ? 1 : 0);
                if (nStubs == 0)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixPix], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 1)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixOT], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 2)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixOTOT], 1u, alpaka::hierarchy::Blocks{});
                else {
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOTOTOT], 1u, alpaka::hierarchy::Blocks{});
                  auto layer1 = innerCell.innerLayer(cc);  // innermost
                  auto layer2 = outerCell.innerLayer(cc);  // middle
                  auto layer3 = outerCell.outerLayer(cc);  // outermost
                  bool l1Brl = (layer1 >= 28 && layer1 <= 33);
                  bool l2Brl = (layer2 >= 28 && layer2 <= 33);
                  bool l3Brl = (layer3 >= 28 && layer3 <= 33);
                  bool l1Fwd = (layer1 >= 34 && layer1 <= 43);  // disks at z > 0
                  bool l2Fwd = (layer2 >= 34 && layer2 <= 43);
                  bool l3Fwd = (layer3 >= 34 && layer3 <= 43);
                  bool l1Bwd = (layer1 >= 44 && layer1 <= 53);  // disks at z < 0
                  bool l2Bwd = (layer2 >= 44 && layer2 <= 53);
                  bool l3Bwd = (layer3 >= 44 && layer3 <= 53);
                  if (l1Brl && l2Brl && l3Brl)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_barrel], 1u, alpaka::hierarchy::Blocks{});
                  else if (l1Bwd && l2Bwd && l3Bwd)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_bwd], 1u, alpaka::hierarchy::Blocks{});
                  else if (l1Fwd && l2Fwd && l3Fwd)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_fwd], 1u, alpaka::hierarchy::Blocks{});
                  else if ((l1Brl || l2Brl) && (l2Bwd || l3Bwd))
                    alpaka::atomicAdd(
                        acc, &pipelineCounters[PC::kTripletsOOO_brlToBwd], 1u, alpaka::hierarchy::Blocks{});
                  else if ((l1Brl || l2Brl) && (l2Fwd || l3Fwd))
                    alpaka::atomicAdd(
                        acc, &pipelineCounters[PC::kTripletsOOO_brlToFwd], 1u, alpaka::hierarchy::Blocks{});
                  else
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_other], 1u, alpaka::hierarchy::Blocks{});
                }
              }
            }
          }
        }  // loop on inner cells
      }  // loop on outer cells
    }
  };

  template <typename TrackerTraits>
  class FillDoubletsHisto {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t *nCells,
                                  uint32_t offsetBPIX2,
                                  HitToCell *outerHitHisto,
                                  Counters *counters) const {
      const auto nKeys = outerHitHisto->nOnes();
      for (auto cellIndex : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef DOUBLETS_DEBUG
        printf("outerHitHisto;%d;%d\n", cellIndex, cells[cellIndex].outer_hit_id());
#endif
        auto const key = cells[cellIndex].outer_hit_id() - offsetBPIX2;
        // Key-range guard: a key past nOnes means the hit->cell offsets were sized for a smaller hit count
        // than the cells reference, so drop the association instead of writing outside off[]. The matching
        // count pass (CAPixelDoubletsAlgos.h) skips exactly the same keys.
        if (key < nKeys)
          outerHitHisto->fill(acc, key, cellIndex);
        else
          alpaka::atomicAdd(acc, &counters->nHitToCellOverflow, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename CAPairView, typename Container>
  class Kernel_fillGenericPair {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CAPairView cn,
                                  uint32_t const *nElements,
                                  Container *genericHisto) const {
      const auto nKeys = genericHisto->nOnes();
      for (uint32_t index : cms::alpakatools::uniform_elements(acc, *nElements)) {
        auto const key = cn[index].inner();
        // Key-range guard mirroring the count pass: the key is a cell index below the cell count the
        // histogram was sized from, so only a sizing mismatch can drop an entry here.
        if (key < nKeys)
          genericHisto->fill(acc, key, cn[index].outer());
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_find_ntuplets {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  const ::reco::CAGraphSoAConstView &cc,
                                  const ::reco::CANtupletCutsSoAConstView &ntupletCuts,
                                  TkSoAView tracks_view,
                                  HitContainer *foundNtuplets,
                                  CellToCell const *__restrict__ cellNeighborsHisto,
                                  CellToTrack *cellTracksHisto,
                                  caStructures::CAPairSoAView ct,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  uint32_t *nCellTracks,
                                  uint32_t const *nTriplets,
                                  uint32_t const *nCells,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  AlgoParams const &params) const {
      using Cell = CACell<TrackerTraits>;

#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc))
        printf("starting producing ntuplets from %d cells and %d triplets \n", *nCells, *nTriplets);
#endif

      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto const &thisCell = cells[idx];

        // cut by earlyFishbone
        if (thisCell.isKilled())
          continue;

        // we require at least three hits
        if (cellNeighborsHisto->size(idx) == 0)
          continue;

        // check if the layer pair of the cell is among the set of starting pairs
        auto pid = thisCell.layerPairId();
        bool doit = cc[pid].startingPair();

        // check if the most inner hit does not fulfill the starting requirement
        auto lid = thisCell.innerLayer(cc);
        if (thisCell.inner_r(hh) > ntupletCuts[lid].startMaxInnerR())
          doit = false;

        constexpr uint32_t maxDepth = TrackerTraits::maxLayersPerTrack - 1;
#ifdef CA_DEBUG
        printf(
            "LayerPairId %d and inner layer %d doit ? %d From cell %d with nNeighbors = %d and innerR=%f < "
            "maxInnerR=%f ?\n",
            pid,
            lid,
            doit,
            idx,
            cellNeighborsHisto->size(idx),
            thisCell.inner_r(hh),
            ntupletCuts[lid].startMaxInnerR());
#endif

        if (doit) {
          typename Cell::TmpTuple stack;
          // Per-thread buffer that find_ntuplets fills when it saves an ntuplet. Declared here, not inside the
          // recursive (inlined) find_ntuplets, so the stack holds one copy per thread and not one per depth.
          typename Cell::hindex_type hits[TrackerTraits::maxHitsOnTrack];

          stack.reset();
          thisCell.template find_ntuplets<maxDepth>(acc,
                                                    hh,
                                                    ntupletCuts,
                                                    cc,
                                                    cells,
                                                    *foundNtuplets,
                                                    cellNeighborsHisto,
                                                    cellTracksHisto,
                                                    nCellTracks,
                                                    ct,
                                                    *apc,
                                                    tracks_view.quality().data(),
                                                    tracks_view.nLayers().data(),
                                                    tracks_view.pt().data(),
                                                    stack,
                                                    hits,
                                                    params.minHitsPerNtuplet_);
          ALPAKA_ASSERT_ACC(stack.empty());
        }
      }
    }
  };
#ifdef CA_PIPELINE_COUNTERS
  // Pipeline counter: classify n-tuplets by OT hit content
  template <typename TrackerTraits>
  class Kernel_pipelineNtupletCount {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  cms::alpakatools::AtomicPairCounter const *apc,
                                  uint32_t maxTuples,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      using PC = caHitNtupletGenerator::PipelineCounter;
      // Clamp to container capacity -- apc may exceed maxTuples on overflow
      auto ntracks = std::min<uint32_t>(apc->get().first, maxTuples);
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        auto nh = foundNtuplets->size(idx);
        if (nh < 3)
          continue;
        alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsTotal], 1u, alpaka::hierarchy::Blocks{});
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          auto nHits = hh.size();
          int nOT = 0;
          for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
            if (*h >= static_cast<unsigned int>(nHits))
              break;  // content buffer corruption from overflow
            if (isStub(hh, *h))
              ++nOT;
          }
          if (nOT >= 1)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsWithOT], 1u, alpaka::hierarchy::Blocks{});
          if (nOT >= 3)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsOT3Plus], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Count cell status after all kill phases (reachability + fishbone)
  template <typename TrackerTraits>
  class Kernel_pipelineCellStatus {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *nCells,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      using PC = ::caHitNtupletGenerator::PipelineCounter;
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &cell = cells[idx];
        if (!cell.unused())  // kUsed bit is set
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsUsedInTriplet], 1u, alpaka::hierarchy::Blocks{});
        if (cell.isKilled())
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsKilledTotal], 1u, alpaka::hierarchy::Blocks{});
        else
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsAlive], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  class Kernel_pipelineCopyCellTrackCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  uint32_t const *nCellTracks,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      if (cms::alpakatools::once_per_grid(acc))
        pipelineCounters[::caHitNtupletGenerator::kCellTrackPairs] = *nCellTracks;
    }
  };
#endif  // CA_PIPELINE_COUNTERS

  template <typename TrackerTraits>
  class Kernel_mark_used {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  uint32_t const *nCells) const {
      using Cell = CACell<TrackerTraits>;
      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto &thisCell = cells[idx];
        if (cellTracksHisto->size(idx) > 0)
          thisCell.setStatusBits(Cell::StatusBit::kInTrack);
      }
    }
  };

  // Count the hits the fit will actually use, given the FitHitSelection mode (== nhits in the default All
  // mode); kept consistent with the fit's own selection in BrokenLineFit.dev.cc.
  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t nSelectedHits(HitContainer const *__restrict__ foundNtuplets,
                                                        uint32_t it,
                                                        caStructures::HitsViewT<TrackerTraits> hh) {
    // hasStubs enables the OT-stub hit treatment (kMode filtering and the same-layer pixel overlap merge).
    // It must match the fit, which keys off the runtime offsetStubs: with no stubs, offsetStubs is the
    // unsigned sentinel and every hit is a plain pixel hit.
    const bool hasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits> &&
                          (static_cast<int32_t>(caStructures::offsetStubsOf(hh)) >= 0);
    return caFitHitSel::dedupWalk(foundNtuplets, it, hh, hasStubs, /*k=*/-1);
  }

  template <typename TrackerTraits>
  class Kernel_countMultiplicity {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);
        // printf("it: %d nhits: %d \n",it,nhits);
        if (nhits < 3)
          continue;
        if (tracks_view[it].quality() == Quality::edup)
          continue;
        // On hitContainer overflow bulkFill returns kOverflow and the quality stamp below is skipped,
        // so the slot keeps its pre-init value (0, or garbage from a caching allocator). Skip such slots:
        // the tuple was already dropped by the lossy truncation.
        if (tracks_view[it].quality() != Quality::bad)
          continue;
        // On content-buffer overflow the offset is plugged (the size is correct) but the content is
        // unwritten, so nhits can read garbage; more than maxHitsOnTrack hits is an overflow artifact.
        if (nhits > TrackerTraits::maxHitsOnTrack)
          continue;
        auto const nsel = nSelectedHits<TrackerTraits>(foundNtuplets, it, hh);
        if (nsel < 3)
          continue;  // too few selected hits to fit
        tupleMultiplicity->count(acc, nsel);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillMultiplicity {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);

        if (nhits < 3)
          continue;
        if (tracks_view[it].quality() == Quality::edup)
          continue;
        // Skip overflow tuples (see Kernel_countMultiplicity for rationale).
        if (tracks_view[it].quality() != Quality::bad)
          continue;
        if (nhits > TrackerTraits::maxHitsOnTrack)
          continue;
        auto const nsel = nSelectedHits<TrackerTraits>(foundNtuplets, it, hh);
        if (nsel < 3)
          continue;
        tupleMultiplicity->fill(acc, nsel, it);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_classifyTracks {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsMultiView hh,
                                  QualityCuts<TrackerTraits> cuts,
                                  bool useTrackDNN,
                                  float trackDNNThreshold,
                                  DnnBank trackBank,  // per-iteration compile-time weight bank
                                  // Raw OT-rechit view for the feature walk. nOTHits == 0 means merged
                                  // hits only, and the view is then unused.
                                  ::reco::OTRecHitsConstView otHits,
                                  uint32_t nOTHits) const {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
      uint32_t nTracks = 0;
      uint32_t nFitted = 0;
      uint32_t nNaN = 0;
      uint32_t nDoublets = 0;
      uint32_t nDuplicates = 0;
#endif

      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);
        if (nhits == 0)
          break;  // guard

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
        nTracks++;
#endif

        // if duplicate: not even fit
        if (tracks_view[it].quality() == Quality::edup) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nDuplicates++;
#endif
          continue;
        }

        // Skip overflow tuples (see Kernel_countMultiplicity for rationale).
        if (tracks_view[it].quality() != Quality::bad)
          continue;

        // mark doublets as bad
        if (nhits < 3) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nDoublets++;
#endif
          continue;
        }

        // if the fit has any invalid parameters, mark it as bad
        bool isNaN = false;
        for (int i = 0; i < 5; ++i) {
          isNaN |= edm::isNotFinite(tracks_view[it].state()(i));
        }
        // A non-finite chi2 is a failed fit and must never be promoted. The test is explicit because the
        // promotion gates downstream are written in the rejecting sense (strictCut returns chi2 >= maxChi2,
        // the stub-curvature walk tests chi2Stub > cut) and a comparison with a NaN operand is false.
        // edm::isNotFinite is a bit-pattern test, so it keeps working under -ffinite-math-only.
        isNaN |= edm::isNotFinite(tracks_view[it].chi2());
        // state(2) is the (finite) inverse pt: an exactly-zero value from a straight-line or
        // numerically-degenerate fit maps to an infinite momentum in the host local-to-global
        // transform, so treat it as bad here too and never promote such a track
        isNaN |= (tracks_view[it].state()(2) == 0.f);
        if (isNaN) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nNaN++;
          printf("FIT_DEBUG: Track %d has NaN - nhits=%d chi2=%f pt=%f eta=%f\n",
                 it,
                 nhits,
                 tracks_view[it].chi2(),
                 tracks_view[it].pt(),
                 tracks_view[it].eta());
#endif
          continue;
        }

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
        nFitted++;
        if (nFitted <= 10) {
          printf("FIT_DEBUG: Track %d FITTED - nhits=%d pt=%.3f eta=%.3f phi=%.3f chi2=%.3f tip=%.4f zip=%.4f\n",
                 it,
                 nhits,
                 tracks_view[it].pt(),
                 tracks_view[it].eta(),
                 tracks_view[it].state()(0),  // phi is state[0]
                 tracks_view[it].chi2(),
                 tracks_view[it].state()(1),   // tip is state[1]
                 tracks_view[it].state()(4));  // zip is state[4]
        }
#endif

        tracks_view[it].quality() = Quality::strict;

        bool failChi2 = cuts.strictCut(tracks_view, nhits, it);
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          const auto nHitsTot = hh.size();

          // When enabled, the MLP score replaces the chi2-based strict->tight decision (both the strictCut
          // fit-chi2 gate and the stub-consistency demotion below), while the fit chi2 and chi2Stub stay
          // inputs of the network (feat[0], feat[8]). Feature order is documented in CATrackDNNWeights.h.
          bool dnnHandled = false;
          if (useTrackDNN) {
            // Single-source feature fill (RecoTracker/PixelSeeding/interface/CATrackFeatures.h), producing
            // values identical to the host-side nano table producer's. fill() returns false on a corrupt or
            // short hit list -> fall through to the chi2-based path.
            float feat[caTrackFeatures::kNFeat];
            static_assert(caTrackFeatures::kNFeat == caTrackDNN::kNFeat, "feature ABI mismatch");
            // Tagged extras resolve their global position through the OT view (nullptr when no tagged ids
            // can be present).
            const ::reco::OTRecHitsConstView *otViewPtr = (nOTHits > 0u) ? &otHits : nullptr;
            const bool featOk = caTrackFeatures::fill(foundNtuplets->begin(it),
                                                      foundNtuplets->end(it),
                                                      hh,
                                                      nHitsTot,
                                                      float(tracks_view[it].nLayers()),
                                                      tracks_view[it].chi2(),
                                                      feat,
                                                      /*rzKappaOut=*/nullptr,
                                                      otViewPtr);
            // The finiteness of the network inputs is established before the network is evaluated: a
            // non-finite input propagates through the MLP and the resulting score, compared the wrong way
            // round, would promote the track. A track with any non-finite feature stays Quality::bad
            // (quality() was optimistically set to strict above, so it is written back explicitly).
            bool featFinite = featOk;
            for (int k = 0; featFinite && k < int(caTrackFeatures::kNFeat); ++k)
              featFinite = !edm::isNotFinite(feat[k]);
            if (featOk && !featFinite) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
              nNaN++;
#endif
              tracks_view[it].quality() = Quality::bad;
              continue;
            }
            if (featOk) {
              // Stage-1 high-recall loose->tight selector: a single threshold. Dedicated displaced fake
              // rejection is the post-reco displaced high-purity selector.
              const float defThr = (trackBank == DnnBank::kPrompt) ? caTrackDNN_prompt::kDefaultThreshold
                                                                   : caTrackDNN_displaced::kDefaultThreshold;
              const float dnnThr = (trackDNNThreshold < 0.f) ? defThr : trackDNNThreshold;
              const float dnnScore = (trackBank == DnnBank::kPrompt)
                                         ? caTrackDNN_eval::score<DnnBank::kPrompt>(feat)
                                         : caTrackDNN_eval::score<DnnBank::kDisplaced>(feat);
              // Promoting form on purpose: `score >= threshold` is the decision to promote and the rejection
              // is its negation. Under -ffinite-math-only a rejecting predicate may be rewritten into its
              // finite-arithmetic complement, which would let a NaN score take the promoting branch; in this
              // form anything the comparison cannot decide stays rejected.
              const bool dnnPromote = (dnnScore >= dnnThr);
              failChi2 = !dnnPromote;
              dnnHandled = true;
            }
          }

          // Ntuplet-wide stub-curvature consistency. A stub at transverse distance d0 from the beam line measures
          // kappa/2 + d0/r^2, so the per-stub curvatures of one track are linear in 1/r^2: that line is fitted by
          // weighted least squares (errors from the precision-only bend column) and the reduced chi2 of its residuals
          // (ndof = nStubs - 2) is the statistic; a chain admitted by the relaxed DCA still fails it and is demoted
          // below `tight`. Skipped when the DNN already decided.
#ifdef CA_CHI2_DUMP
          const bool computeStubChi2 = true;
#else
          const bool computeStubChi2 = !dnnHandled;
#endif
          if (computeStubChi2) {
            int nStubK = 0;
            float sumW = 0.f, sumWX = 0.f, sumWXX = 0.f, sumWK = 0.f, sumWKX = 0.f, sumWKK = 0.f;
            for (auto h = foundNtuplets->begin(it); h != foundNtuplets->end(it); ++h) {
              // A tagged raw-OT extra indexes the OT source, not hh, and is never a stub, so it contributes
              // nothing to the stub-curvature consistency.
              if (caExtension::isOTId(*h))
                continue;
              if (*h >= static_cast<unsigned int>(nHitsTot))
                break;  // content buffer corruption from overflow
              if (!isStub(hh, *h))
                continue;  // pixel hit
              auto const stub = hh.stub(int32_t(*h));
              const float s = stub.dPhiDrErrorPrec();
              if (s > 0.f) {
                const float d = stub.dPhiDr();
                const float xg = hh[*h].xGlobal();
                const float yg = hh[*h].yGlobal();
                const float rg2 = xg * xg + yg * yg;
                if (!(rg2 > 0.f))
                  continue;
                float den, w;  // same shared kappa formula as CATrackFeatures::fill
                caTrackFeatures::stubDenWeight(rg2, d, s, den, w);
                const float k = d / std::sqrt(den);  // stub curvature
                const float x = 1.f / rg2;           // the d0 term enters linearly in 1/r^2
                // hit precision plus multiple scattering, as in the doublet and triplet cuts
                const float sMS = caStubMS::kThetaPerCurv * 2.f * std::abs(k) / std::sqrt(rg2);
                w = 1.f / (1.f / w + sMS * sMS);
                sumW += w;
                sumWX += w * x;
                sumWXX += w * x * x;
                sumWK += w * k;
                sumWKX += w * k * x;
                sumWKK += w * k * k;
                ++nStubK;
              }
            }
            // chi2Stub < 0 => not enough stubs to judge consistency.
            float chi2Stub = -1.f;
            if (nStubK >= 3 && sumW > 0.f) {
              const float det = sumW * sumWXX - sumWX * sumWX;
              if (std::abs(det) > 0.f) {
                // k = a + b/r^2, a = kappa/2 and b = d0
                const float a = (sumWXX * sumWK - sumWX * sumWKX) / det;
                const float b = (sumW * sumWKX - sumWX * sumWK) / det;
                chi2Stub = (sumWKK - a * sumWK - b * sumWKX) / float(nStubK - 2);
              }
              if (chi2Stub >= 0.f && !dnnHandled && cuts.maxNtupletStubChi2 >= 0.f) {
                // The keep decision is the positive comparison (chi2Stub <= cut), so a non-finite chi2Stub
                // from a degenerate stub set falls to the demoting side; the explicit isNotFinite keeps that
                // true under -Ofast.
                const bool stubConsistent = !edm::isNotFinite(chi2Stub) && (chi2Stub <= cuts.maxNtupletStubChi2);
                if (!stubConsistent)
                  failChi2 = true;
              }
            }
#ifdef CA_CHI2_DUMP
            // Per-track calibration dump: fit chi2 vs ntuplet-wide stub consistency.
            printf("[Chi2Dump] nhits=%d nStubK=%d chi2=%.4f chi2Stub=%.4f pt=%.4f eta=%.4f\n",
                   nhits,
                   nStubK,
                   tracks_view[it].chi2(),
                   chi2Stub,
                   tracks_view[it].pt(),
                   tracks_view[it].eta());
#endif
          }
        }
        if (failChi2)
          continue;

        tracks_view[it].quality() = Quality::tight;

        if (cuts.isHP(tracks_view, nhits, it))
          tracks_view[it].quality() = Quality::highPurity;
      }

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("FIT_DEBUG SUMMARY: total=%d fitted=%d NaN=%d doublets=%d duplicates=%d\n",
               nTracks,
               nFitted,
               nNaN,
               nDoublets,
               nDuplicates);
      }
#endif
    }
  };

  class Kernel_assignIteration {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  pixelTrack::Iteration iterationName) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        tracks_view[it].iteration() = iterationName;
      }
      // Tail: the CA output SoA is allocated at capacity and only [0, nOnes) is filled.
      for (auto it : cms::alpakatools::uniform_elements(acc, uint32_t(tracks_view.metadata().size())))
        if (it >= foundNtuplets->nOnes())
          tracks_view[it].iteration() = pixelTrack::Iteration::notIteration;
    }
  };

  // updateMasking: one thread per track. The mask is cumulative: each masking stage ORs its own
  // iterationIndex bit into mask_view[hid].recHitMask(), so an earlier stage's mask survives (the
  // consumers only test != 0, CAPixelDoubletsAlgos.h). The read-modify-write needs no atomics: every
  // writer of a given hit id ORs the same constant, and the bits already in the mask were copied in
  // before the kernel started, so any interleaving gives the same result.
  class Kernel_updateMaskingParallel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackingRecHitsMaskingView mask_view,
                                  const ::reco::TrackSoAConstView &trackd_view,
                                  const ::reco::TrackHitSoAConstView &trackhitd_view,
                                  const pixelTrack::Quality minQuality,
                                  uint32_t const &iterationIndex,
                                  bool maskAttachedHits) const {
      for (auto j : cms::alpakatools::uniform_elements(acc, uint32_t(trackd_view.nTracks()))) {
        if (trackd_view[j].quality() < minQuality)
          continue;
        const uint32_t end = trackd_view[j].hitOffsets();
        const uint32_t start = (j == 0u) ? 0u : trackd_view[j - 1].hitOffsets();
        for (uint32_t p = start; p < end; ++p) {
          // In-fit-extension attachments stay available to the next iteration unless maskAttachedHits;
          // tagged raw-OT extras index the OT source, not the merged mask domain, so they are skipped.
          if (!maskAttachedHits && trackhitd_view[p].attached() != 0)
            continue;
          const uint32_t hid = trackhitd_view[p].id();
          if (caExtension::isOTId(hid))
            continue;
          mask_view[hid].recHitMask() |= iterationIndex;
        }
      }
    }
  };

  // Duplicate-removal compatibility test, shared by the cross-arm twin merge and the final dedup: two tracks
  // are compatible when the Mahalanobis distance of their parameter difference, against the sum of their full
  // covariances, is below the 5-sigma rejection quantile. n = 3 uses the arm-invariant triple (phi, q/pT,
  // cotTheta), usable before the refit; n = 5 adds tip and zip after the common refit, where both members
  // share one convention. Off-diagonals are kept: phi and q/pT are strongly correlated in a helix fit.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool dedupCompatible(
      const ::reco::TrackSoAConstView &tv, int32_t i, int32_t j, int n, float qGate) {
    constexpr int off[5] = {0, 4, 7, 9, 10};
    auto cidx = [&](int a, int b) {
      const int lo = a < b ? a : b;
      const int hi = a < b ? b : a;
      return off[lo] + hi;
    };
    // n == 3 selects the arm-invariant parameters (phi, q/pT, cotTheta) out of the 5.
    const int par3[3] = {0, 2, 3};
    float d[5];
    float A[5][5];
    for (int a = 0; a < n; ++a) {
      const int pa = (n == 3) ? par3[a] : a;
      float dp = tv[i].state()[pa] - tv[j].state()[pa];
      if (pa == 0) {  // phi: wrap to [-pi, pi]
        while (dp > kTwinPi)
          dp -= kTwinTwoPi;
        while (dp < -kTwinPi)
          dp += kTwinTwoPi;
      }
      d[a] = dp;
      for (int b = 0; b < n; ++b) {
        const int pb = (n == 3) ? par3[b] : b;
        A[a][b] = tv[i].covariance()[cidx(pa, pb)] + tv[j].covariance()[cidx(pa, pb)];
      }
    }
    // Cholesky solve of A x = d, then chi2 = d^T x. A is a sum of two covariances, so it is positive
    // definite unless the fit produced a degenerate one; a failed factorisation means "cannot judge",
    // which is answered with "not compatible" so nothing is dropped on a broken covariance.
    float L[5][5] = {{0.f}};
    for (int a = 0; a < n; ++a) {
      for (int b = 0; b <= a; ++b) {
        float sum = A[a][b];
        for (int k = 0; k < b; ++k)
          sum -= L[a][k] * L[b][k];
        if (a == b) {
          if (!(sum > 0.f))
            return false;
          L[a][a] = std::sqrt(sum);
        } else {
          L[a][b] = sum / L[b][b];
        }
      }
    }
    float y[5];
    for (int a = 0; a < n; ++a) {
      float sum = d[a];
      for (int k = 0; k < a; ++k)
        sum -= L[a][k] * y[k];
      y[a] = sum / L[a][a];
    }
    float chi2 = 0.f;
    for (int a = 0; a < n; ++a)
      chi2 += y[a] * y[a];
    return chi2 < qGate;
  }

  // How much a fit knows about its track: the generalized variance |C| of the 5x5 perigee covariance
  // (the D-optimality measure; |C| = 1 / |Fisher information|). It is the volume of the error
  // ellipsoid, so it uses the correlations the helix fit really has instead of five separate errors,
  // and since both tracks live in the same parameter space the two determinants carry identical units
  // and their comparison needs no scale. Returned as ln|C| through the Cholesky factor, because
  // |C| ~ 1e-25 underflows in float. A covariance that is not positive definite cannot be judged.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool logDetCov5(const ::reco::TrackSoAConstView &tv, int32_t t, float &lnDet) {
    constexpr int off[5] = {0, 4, 7, 9, 10};
    float L[5][5] = {{0.f}};
    float s = 0.f;
    for (int a = 0; a < 5; ++a) {
      for (int b = 0; b <= a; ++b) {
        const int lo = b, hi = a;
        float sum = tv[t].covariance()[off[lo] + hi];
        for (int k = 0; k < b; ++k)
          sum -= L[a][k] * L[b][k];
        if (a == b) {
          if (!(sum > 0.f))
            return false;
          L[a][a] = std::sqrt(sum);
          s += std::log(L[a][a]);
        } else {
          L[a][b] = sum / L[b][b];
        }
      }
    }
    lnDet = 2.f * s;
    return true;
  }

  // Which member of a duplicate pair carries more information, +1 = x, -1 = y, 0 = undecided (the
  // caller then falls through to its length/quality tie-breaks). A fit past the chi2/ndof bound, or
  // one whose covariance is not positive definite, cannot be judged and loses to one that can.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int dedupInfoOrder(const ::reco::TrackSoAConstView &tv, int32_t x, int32_t y) {
    float lx = 0.f, ly = 0.f;
    const bool okX = (tv[x].chi2() < kDedupMaxChi2Ndof) && logDetCov5(tv, x, lx);
    const bool okY = (tv[y].chi2() < kDedupMaxChi2Ndof) && logDetCov5(tv, y, ly);
    if (okX != okY)
      return okX ? 1 : -1;
    if (!okX)
      return 0;
    if (lx != ly)
      return lx < ly ? 1 : -1;  // smaller error volume = more information
    return 0;
  }

  // Strict cross-arm twin merge (PixelTracksSoAMerger twinMerge=true). The masking chain can reconstruct a
  // particle twice, as a pixel-rich prompt-arm track and an OT-rich displaced-arm track, largely disjoint in
  // hits; twin-merge pairs them by trajectory and shared-hit evidence and unites their hit lists onto the winner.
  // Kernel_twinFindBest: for every track, the single best opposite-arm, same-charge partner passing the
  // arm-invariant covariance compatibility at 5 sigma; best = largest shared-hit fraction, then smallest dR,
  // then lowest index.
  class Kernel_twinFindBest {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const ::reco::TrackHitSoAConstView inpTrackHit_view,
                                  const int32_t *__restrict__ armOfTrack,
                                  const pixelTrack::Quality minQuality,
                                  const float qGate3,  // chi2_3 at the 5-sigma duplicate rejection
                                  // (eta,phi)->track OneToManyAssoc over this same collection, plus its
                                  // binning. A window wider than the ring wraps over every phi bin once;
                                  // either way the swept set contains every pair the gate can accept.
                                  HitToTuple const *__restrict__ etaPhiBinner,
                                  const int nPhiBins,
                                  const int nEtaSlabs,
                                  const float etaMax,
                                  int32_t *__restrict__ bestTwin) const {
      const int32_t nT = inpTrack_view.metadata().size();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        bestTwin[i] = -1;
        if (inpTrack_view[i].quality() < minQuality)
          continue;
        const int32_t armI = armOfTrack[i];
        const float phiI = ::reco::phi(inpTrack_view, i);
        const float etaI = inpTrack_view[i].eta();
        const float chgI = ::reco::charge(inpTrack_view, i);
        const uint32_t iBeg = (i == 0) ? 0u : inpTrack_view[i - 1].hitOffsets();
        const uint32_t iEnd = inpTrack_view[i].hitOffsets();
        const int nHitsI = int(iEnd - iBeg);

        const float vPhiI = inpTrack_view[i].covariance()[0];
        const float vCotI = inpTrack_view[i].covariance()[kCovCotCot];
        const float cotI = inpTrack_view[i].state()[3];
        const float dPhiWin = std::sqrt(qGate3 * 4.f * (vPhiI > 0.f ? vPhiI : 0.f));
        const float dCotWin = std::sqrt(qGate3 * 4.f * (vCotI > 0.f ? vCotI : 0.f));
        const float cotEdge = std::abs(cotI) - dCotWin;
        const float cotMin = cotEdge > 0.f ? cotEdge : 0.f;
        const float dEtaWin = dCotWin / std::sqrt(1.f + cotMin * cotMin);

        float bestFrac = -1.f;
        float bestDR2 = 1e30f;
        int32_t bestJ = -1;

        // Winner update is a strict total order over (fraction desc, dr2 asc, j asc), so bestJ does not
        // depend on the order in which j is visited and the binned sweep reproduces an exhaustive scan.
        auto considerJ = [&](int32_t j) {
          if (j == i)
            return;
          if (armOfTrack[j] == armI)  // opposite arm only
            return;
          if (inpTrack_view[j].quality() < minQuality)
            return;
          if (::reco::charge(inpTrack_view, j) != chgI)  // same charge
            return;
          // Each component of a Mahalanobis distance is bounded by the distance itself, so these two
          // one-line tests are implied by the gate and reject almost every binned candidate before the
          // 3x3 factorisation runs.
          float dPhi = phiI - ::reco::phi(inpTrack_view, j);
          while (dPhi > kTwinPi)
            dPhi -= kTwinTwoPi;
          while (dPhi < -kTwinPi)
            dPhi += kTwinTwoPi;
          if (dPhi * dPhi >= qGate3 * (vPhiI + inpTrack_view[j].covariance()[0]))
            return;
          const float dCot = cotI - inpTrack_view[j].state()[3];
          if (dCot * dCot >= qGate3 * (vCotI + inpTrack_view[j].covariance()[kCovCotCot]))
            return;
          if (!dedupCompatible(inpTrack_view, i, j, 3, qGate3))
            return;
          // Shared-hit evidence as a fraction of the SHORTER track's list: it does not decide the pair,
          // it only orders the candidates when a track has more than one compatible partner.
          const uint32_t jBeg = (j == 0) ? 0u : inpTrack_view[j - 1].hitOffsets();
          const uint32_t jEnd = inpTrack_view[j].hitOffsets();
          const int nHitsJ = int(jEnd - jBeg);
          int shared = 0;
          for (uint32_t a = iBeg; a < iEnd; ++a) {
            const uint32_t ida = inpTrackHit_view[a].id();
            for (uint32_t b = jBeg; b < jEnd; ++b) {
              if (inpTrackHit_view[b].id() == ida) {
                ++shared;
                break;
              }
            }
          }
          const int nShort = nHitsI < nHitsJ ? nHitsI : nHitsJ;
          if (nShort <= 0)
            return;
          // No shared-hit requirement: the two arms often rebuild the same particle from disjoint hit
          // ids (a stub on one side, its two raw rechits on the other, or a different pixel subset), and
          // those twins are exactly the pairs nothing downstream can pair either. The covariance gate
          // above, on the opposite arm and the same charge, is the criterion.
          const float frac = float(shared) / float(nShort);
          const float dEta = etaI - inpTrack_view[j].eta();
          const float dr2 = dEta * dEta + dPhi * dPhi;
          if (frac > bestFrac || (frac == bestFrac && (dr2 < bestDR2 || (dr2 == bestDR2 && j < bestJ)))) {
            bestFrac = frac;
            bestDR2 = dr2;
            bestJ = j;
          }
        };

        // Both windows come from the gate itself, not from window parameters: the test can only accept
        // |dphi| <= sqrt(qGate3 * (V_i + V_j)) and |dcot| <= sqrt(qGate3 * (V_i + V_j)), with V_j
        // bounded by assuming a partner no more than 3x less well measured than this track. The cot
        // window becomes an eta window through deta = dcot / sqrt(1 + cot^2), evaluated at the smallest
        // |cot| the window reaches, where eta moves fastest per unit of cot -- so the eta span is an
        // over-estimate and no accepted pair can fall outside it. Two phi bins of margin; a window
        // wider than the ring wraps over every phi bin once instead of scanning the whole collection.
        const float binW = kTwinTwoPi / float(nPhiBins);
        const int half = int(dPhiWin / binW) + 2;
        const int spansRing = (2 * half + 1 >= nPhiBins);
        const int dLo = spansRing ? 0 : -half;
        const int dHi = spansRing ? nPhiBins - 1 : half;
        const int b0 = trackBinKey(0.f, phiI, nPhiBins, 1, 0.f);
        auto slabOf = [&](float eta) { return (trackBinKey(eta, phiI, nPhiBins, nEtaSlabs, etaMax) - b0) / nPhiBins; };
        const int ebLo = slabOf(etaI - dEtaWin);  // trackBinKey clamps, so both ends are in range
        const int ebHi = slabOf(etaI + dEtaWin);
        for (int eb = ebLo; eb <= ebHi; ++eb) {
          for (int d = dLo; d <= dHi; ++d) {
            const int b = (b0 + d + nPhiBins) % nPhiBins;
            const uint32_t bin = uint32_t(eb * nPhiBins + b);
            for (auto p = etaPhiBinner->begin(bin); p != etaPhiBinner->end(bin); ++p)
              considerJ(int32_t(*p));
          }
        }
        bestTwin[i] = bestJ;
      }
    }
  };

  // Kernel_twinConfirm: keep only mutual-best pairs (bestTwin[i]==j && bestTwin[j]==i), so every track takes
  // part in at most one merge with no atomics. The winner is chosen by the strict total ordering (most
  // information -> max nLayers -> max total hits -> max quality -> min chi2 -> min index) and records
  // loserOf[winner] and isLoser[loser]; isLoser must be zero-initialised by the caller.
  class Kernel_twinConfirm {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const int32_t *__restrict__ bestTwin,
                                  int32_t *__restrict__ loserOf,
                                  int32_t *__restrict__ isLoser) const {
      const int32_t nT = inpTrack_view.metadata().size();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        loserOf[i] = -1;  // single-writer (this thread owns slot i)
        const int32_t j = bestTwin[i];
        if (j < 0 || j == i)
          continue;
        if (bestTwin[j] != i)  // mutual-best match only
          continue;
        // Winner ordering: the more informative fit keeps the united hit list (same rule as the final
        // dedup), with the length keys as tie-breaks.
        const int o = dedupInfoOrder(inpTrack_view, i, j);
        const int nli = inpTrack_view[i].nLayers();
        const int nlj = inpTrack_view[j].nLayers();
        const int nhi = ::reco::nHits(inpTrack_view, i);
        const int nhj = ::reco::nHits(inpTrack_view, j);
        const auto qi = inpTrack_view[i].quality();
        const auto qj = inpTrack_view[j].quality();
        const float ci = inpTrack_view[i].chi2();
        const float cj = inpTrack_view[j].chi2();
        bool iWins;
        if (o != 0)
          iWins = o > 0;
        else if (nli != nlj)
          iWins = nli > nlj;
        else if (nhi != nhj)
          iWins = nhi > nhj;
        else if (qi != qj)
          iWins = qi > qj;
        else if (ci != cj)
          iWins = ci < cj;
        else
          iWins = i < j;
        if (iWins) {
          loserOf[i] = j;
          isLoser[j] = 1;  // unique writer: mutual match => only i claims j
        }
      }
    }
  };

  // One thread per input track. A track is dropped iff quality < minQuality, or isLoser (absorbed into its
  // twin winner), or fewer than 3 hits; every term is a function of the inputs and of loserOf/isLoser only, so
  // the marking is order-independent. The winner's hit list is its own hits followed by the loser's
  // non-duplicate hits (dedup by id, capped at kTwinMaxMergedHits), written per track; the compaction uses
  // exclusive placements from inclusive prefix sums of keep[] and the united-hit counts, preserving order.
  class Kernel_filterTracksMark {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const ::reco::TrackHitSoAConstView inpTrackHit_view,
                                  const pixelTrack::Quality minQuality,
                                  const int32_t *__restrict__ loserOf,
                                  const int32_t *__restrict__ isLoser,
                                  int32_t *__restrict__ keep,
                                  int32_t *__restrict__ outHitCnt) const {
      const int32_t nIn = int32_t(inpTrack_view.metadata().size());
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nIn)) {
        keep[i] = 0;
        outHitCnt[i] = 0;
        if (inpTrack_view[i].quality() < minQuality)
          continue;
        // twin-merge: this track was absorbed into its twin winner -> drop it.
        if (isLoser && isLoser[i])
          continue;

        const int32_t nhI = ::reco::nHits(inpTrack_view, i);
        // never forward a slot with fewer than 3 hits (a truncated/unfilled slot): downstream
        // (the SoA->legacy converter) asserts nHits >= 3 on every track it is handed.
        if (nhI < 3)
          continue;
        const uint32_t iBeg = (i == 0) ? 0u : inpTrack_view[i - 1].hitOffsets();

        // kept. Count the united hit block = own hits + twin-loser union (dedup by id, capped). Mirrors
        // Kernel_filterTracksScatter's write loop exactly, so the count matches what the scatter writes.
        keep[i] = 1;
        uint32_t nUnited = uint32_t(nhI);
        if (loserOf) {
          const int32_t loser = loserOf[i];
          if (loser >= 0 && nUnited < uint32_t(kTwinMaxMergedHits)) {
            uint32_t ids[kTwinMaxMergedHits];  // own count < cap here -> fits
            for (uint32_t k = 0; k < nUnited; ++k)
              ids[k] = inpTrackHit_view[iBeg + k].id();
            const uint32_t lBeg = (loser == 0) ? 0u : inpTrack_view[loser - 1].hitOffsets();
            const uint32_t lEnd = inpTrack_view[loser].hitOffsets();
            for (uint32_t k = lBeg; k < lEnd; ++k) {
              if (nUnited >= uint32_t(kTwinMaxMergedHits))
                break;
              const uint32_t lid = inpTrackHit_view[k].id();
              bool present = false;
              for (uint32_t m = 0; m < nUnited; ++m)
                if (ids[m] == lid) {
                  present = true;
                  break;
                }
              if (present)
                continue;
              ids[nUnited++] = lid;
            }
          }
        }
        outHitCnt[i] = int32_t(nUnited);
      }
    }
  };

  // Scatter phase: places each kept winner at its compacted output index (from the inclusive prefix
  // sums), writes its united hit block (own hits, then the twin loser's non-duplicate hits), then
  // applies the twin-merge refit ndof/chi2 recompute and the unitedMaskOut stamping.
  class Kernel_filterTracksScatter {
  public:
    static_assert(::reco::TrackSoA::Descriptor::num_cols == 11,
                  "reco::TrackLayout column count changed: update this compaction's column "
                  "enumeration (quality, chi2, nLayers, eta, pt, state[5], covariance[15], nTracks, "
                  "hitOffsets, iteration, ndof)");
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackSoAView track_view,
                                  ::reco::TrackHitSoAView trackHit_view,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const ::reco::TrackHitSoAConstView inpTrackHit_view,
                                  const int32_t *__restrict__ loserOf,
                                  const bool twinMergeRefit,
                                  const bool refitAllTracks,
                                  int32_t *__restrict__ unitedMaskOut,
                                  const int32_t *__restrict__ keep,
                                  const int32_t *__restrict__ outHitCnt,
                                  const int32_t *__restrict__ tkOff,   // inclusive scan of keep[]
                                  const int32_t *__restrict__ hitOff,  // inclusive scan of outHitCnt[]
                                  const int32_t nScanSize) const {
      // authoritative output count = total kept = last inclusive-scan value.
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0)
        track_view.nTracks() = tkOff[nScanSize - 1];

      const int32_t nIn = int32_t(inpTrack_view.metadata().size());
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nIn)) {
        if (!keep[i])
          continue;
        const uint32_t outTk = uint32_t(tkOff[i] - 1);                               // 0-based compacted track slot
        const uint32_t writtenBegin = uint32_t(hitOff[i]) - uint32_t(outHitCnt[i]);  // hit block base

        track_view[outTk].quality() = inpTrack_view[i].quality();
        track_view[outTk].chi2() = inpTrack_view[i].chi2();
        track_view[outTk].ndof() = inpTrack_view[i].ndof();
        // Provenance: carry the producing iteration through the compaction, alongside ndof.
        track_view[outTk].iteration() = inpTrack_view[i].iteration();
        track_view[outTk].nLayers() = inpTrack_view[i].nLayers();
        track_view[outTk].eta() = inpTrack_view[i].eta();
        track_view[outTk].pt() = inpTrack_view[i].pt();
        for (uint32_t k = 0; k < 5; ++k)
          track_view[outTk].state()[k] = inpTrack_view[i].state()[k];
        for (uint32_t k = 0; k < 15; ++k)
          track_view[outTk].covariance()[k] = inpTrack_view[i].covariance()[k];

        const uint32_t iBeg = (i == 0) ? 0u : inpTrack_view[i - 1].hitOffsets();
        const uint32_t iEnd = inpTrack_view[i].hitOffsets();
        uint32_t w = writtenBegin;
        for (uint32_t k = iBeg; k < iEnd; ++k) {
          trackHit_view[w].id() = inpTrackHit_view[k].id();
          trackHit_view[w].detId() = inpTrackHit_view[k].detId();
          trackHit_view[w].attached() = inpTrackHit_view[k].attached();
          ++w;
        }

        // twin-merge union (dedup by id against the already-written own+appended hits, capped); the
        // present-check reads only this track's output block, so it is per-track-local.
        if (loserOf) {
          const int32_t loser = loserOf[i];
          if (loser >= 0) {
            const uint32_t lBeg = (loser == 0) ? 0u : inpTrack_view[loser - 1].hitOffsets();
            const uint32_t lEnd = inpTrack_view[loser].hitOffsets();
            for (uint32_t k = lBeg; k < lEnd; ++k) {
              if ((w - writtenBegin) >= uint32_t(kTwinMaxMergedHits))
                break;
              const uint32_t lid = inpTrackHit_view[k].id();
              bool present = false;
              for (uint32_t m = writtenBegin; m < w; ++m) {
                if (trackHit_view[m].id() == lid) {
                  present = true;
                  break;
                }
              }
              if (present)
                continue;
              trackHit_view[w].id() = lid;
              trackHit_view[w].detId() = inpTrackHit_view[k].detId();
              trackHit_view[w].attached() = inpTrackHit_view[k].attached();
              ++w;
            }
          }
        }
        track_view[outTk].hitOffsets() = w;  // == hitOff[i] by construction

        if (twinMergeRefit && loserOf && loserOf[i] >= 0) {
          const uint32_t nUnited = w - writtenBegin;  // own + appended (CSR extent)
          constexpr uint32_t kTwinRefitNdofCap = 12;  // keep in sync with HelixFit::kRefitMaxN
          const uint32_t nFit = nUnited < kTwinRefitNdofCap ? nUnited : kTwinRefitNdofCap;
          int ndofUnited = 2 * int(nFit) - 5;
          const int ndofWinner = int(inpTrack_view[i].ndof());
          if (ndofUnited < ndofWinner)
            ndofUnited = ndofWinner;
          if (ndofUnited < 1)
            ndofUnited = 1;
          const float rawChi2 = inpTrack_view[i].chi2() * float(ndofWinner > 0 ? ndofWinner : 1);
          track_view[outTk].ndof() = int8_t(ndofUnited);
          track_view[outTk].chi2() = rawChi2 / float(ndofUnited);
          if (unitedMaskOut)
            unitedMaskOut[outTk] = int32_t(outTk);
        }
        if (refitAllTracks && unitedMaskOut)
          unitedMaskOut[outTk] = int32_t(outTk);
        // pocket gate: scatter the per-track arm to the same compacted slot the track went to (outTk), so
        // launchMergerAttach reads armId in the merged-SoA order. Null leaves it untouched.
      }
      // Tail hygiene: stamp the unused output capacity so the iteration column is never allocator garbage.
      // Index-disjoint from the kept slots written above (outTk = tkOff[i]-1 < nOut), so no barrier is needed.
      const uint32_t nOut = uint32_t(tkOff[nScanSize - 1]);
      const uint32_t hitEndTail = uint32_t(hitOff[nScanSize - 1]);  // CSR end of the last kept track
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, uint32_t(track_view.metadata().size())))
        if (k >= nOut) {
          track_view[k].iteration() = pixelTrack::Iteration::notIteration;
          track_view[k].hitOffsets() = hitEndTail;  // nHits() reads zero past the last track
        }
    }
  };

  // Final post-refit de-dup: compaction of the refined merged tracks into the output SoA, skipping the tracks
  // flagged by Kernel_dedupCovMark (rows and hit CSR copied verbatim; drop[i] == 0 means kept). keep[]/hitCnt[]
  // are memset to 0 over the scan capacity, so the inclusive scans are constant past the last real track and
  // tkOff[cap-1] is the total kept.
  // One-thread reporter for the two finalDedup diagnostic counters, consumed on the device to avoid a D2H read.
  class Kernel_dedupOverflowReport {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const uint32_t *__restrict__ binnerOvf,
                                  const uint32_t *__restrict__ contestedOvf) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        if (*binnerOvf != 0)
          printf("PixelTracksSoAMerger finalDedup: hit/track binner key overflow (clamped, not fatal): %u entries\n",
                 *binnerOvf);
        if (contestedOvf != nullptr && *contestedOvf != 0)
          printf(
              "PixelTracksSoAMerger finalDedup: contested-pair list overflow (kept both, not fatal; raise "
              "kDedupConfirmMaxPairs): %u pairs\n",
              *contestedOvf);
      }
    }
  };

  class Kernel_finalDedupCounts {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const uint8_t *__restrict__ drop,
                                  int32_t *__restrict__ keep,
                                  int32_t *__restrict__ hitCnt) const {
      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const int32_t k = drop[i] ? 0 : 1;
        keep[i] = k;
        const uint32_t beg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t end = tracks_view[i].hitOffsets();
        hitCnt[i] = k ? int32_t(end - beg) : 0;
      }
    }
  };

  // Scatter: copies each kept track's fields and hit CSR verbatim (no union, no re-fit) to the
  // compacted indices from the inclusive prefix sums. Preserves the input order of the kept tracks.
  class Kernel_finalDedupScatter {
  public:
    static_assert(::reco::TrackSoA::Descriptor::num_cols == 11,
                  "reco::TrackLayout column count changed: update this compaction's column "
                  "enumeration (quality, chi2, nLayers, eta, pt, state[5], covariance[15], nTracks, "
                  "hitOffsets, iteration, ndof)");
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackSoAView out_view,
                                  ::reco::TrackHitSoAView outHit_view,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const ::reco::TrackHitSoAConstView trackHit_view,
                                  const int32_t *__restrict__ keep,
                                  const int32_t *__restrict__ tkOff,   // inclusive scan of keep[]
                                  const int32_t *__restrict__ hitOff,  // inclusive scan of hitCnt[]
                                  const int32_t nScanSize) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0)
        out_view.nTracks() = tkOff[nScanSize - 1];

      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        if (!keep[i])
          continue;
        const uint32_t outTk = uint32_t(tkOff[i] - 1);
        const uint32_t begin = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t end = tracks_view[i].hitOffsets();
        const uint32_t outHitEnd = uint32_t(hitOff[i]);
        uint32_t outHit = outHitEnd - (end - begin);

        out_view[outTk].quality() = tracks_view[i].quality();
        out_view[outTk].chi2() = tracks_view[i].chi2();
        out_view[outTk].ndof() = tracks_view[i].ndof();
        out_view[outTk].iteration() = tracks_view[i].iteration();
        out_view[outTk].nLayers() = tracks_view[i].nLayers();
        out_view[outTk].eta() = tracks_view[i].eta();
        out_view[outTk].pt() = tracks_view[i].pt();
        for (uint32_t k = 0; k < 5; ++k)
          out_view[outTk].state()[k] = tracks_view[i].state()[k];
        for (uint32_t k = 0; k < 15; ++k)
          out_view[outTk].covariance()[k] = tracks_view[i].covariance()[k];
        for (uint32_t k = begin; k < end; ++k) {
          outHit_view[outHit].id() = trackHit_view[k].id();
          outHit_view[outHit].detId() = trackHit_view[k].detId();
          outHit_view[outHit].attached() = trackHit_view[k].attached();
          ++outHit;
        }
        out_view[outTk].hitOffsets() = outHit;  // == outHitEnd by construction
      }
      // Tail hygiene (see Kernel_filterTracksScatter): unused output capacity stamped notIteration.
      const uint32_t nOutTk = uint32_t(tkOff[nScanSize - 1]);
      const uint32_t hitEndTail = uint32_t(hitOff[nScanSize - 1]);  // CSR end of the last kept track
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, uint32_t(out_view.metadata().size())))
        if (k >= nOutTk) {
          out_view[k].iteration() = pixelTrack::Iteration::notIteration;
          out_view[k].hitOffsets() = hitEndTail;  // nHits() reads zero past the last track
        }
    }
  };

  template <typename TrackerTraits>
  class Kernel_doStatsForTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  Counters *counters) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  //guard
        if (tracks_view[idx].quality() < Quality::loose)
          continue;
        alpaka::atomicAdd(acc, &(counters->nLooseTracks), 1ull, alpaka::hierarchy::Blocks{});
        if (tracks_view[idx].quality() < Quality::strict)
          continue;
        alpaka::atomicAdd(acc, &(counters->nGoodTracks), 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Final quality distribution counter: counts tracks at each quality level after classification, fishbone
  // and duplicate removal.
#ifdef CA_PIPELINE_COUNTERS
  template <typename TrackerTraits>
  class Kernel_countFinalQuality {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsMultiView hh,
                                  uint32_t *__restrict__ pipelineCounters) const {
      using Quality = pixelTrack::Quality;
      using PC = caHitNtupletGenerator::PipelineCounter;

      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(idx);
        if (nhits == 0)
          break;  // guard

        alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTotal], 1u, alpaka::hierarchy::Blocks{});

        auto q = tracks_view[idx].quality();
        if (q == Quality::bad) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualBad], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::edup) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualEdup], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::dup) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualDup], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::loose) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualLoose], 1u, alpaka::hierarchy::Blocks{});
        } else {
          // strict, tight, or highPurity -- check OT once for all levels
          bool hasOT = false;
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            auto nHits = hh.size();
            for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
              if (*h >= static_cast<unsigned int>(nHits))
                break;  // content buffer corruption from overflow
              if (isStub(hh, *h)) {
                hasOT = true;
                break;
              }
            }
          }
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict], 1u, alpaka::hierarchy::Blocks{});
          if (hasOT)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrictWithOT], 1u, alpaka::hierarchy::Blocks{});
          if (q >= Quality::tight) {
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight], 1u, alpaka::hierarchy::Blocks{});
            if (hasOT)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTightWithOT], 1u, alpaka::hierarchy::Blocks{});
          }
          if (q >= Quality::highPurity) {
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP], 1u, alpaka::hierarchy::Blocks{});
            if (hasOT)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHPWithOT], 1u, alpaka::hierarchy::Blocks{});
          }

          // Per-nhits quality breakdown
          float chi2 = tracks_view[idx].chi2();
          if (nhits <= 4) {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict34], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight34], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP34], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 0.9f && chi2 < 1.1f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary34], 1u, alpaka::hierarchy::Blocks{});
          } else if (nhits == 5) {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict5], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight5], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP5], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 2.7f && chi2 < 3.3f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary5], 1u, alpaka::hierarchy::Blocks{});
          } else {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict6p], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight6p], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP6p], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 4.5f && chi2 < 5.5f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary6p], 1u, alpaka::hierarchy::Blocks{});
          }

          // Fishbone-hit multiplicity per track. The hit container does not mark fishbone hits and
          // the cell count is not available here, so every track lands in the 0-fishbone bucket.
          uint32_t nFishbone = 0;
          nFishbone = 0;
          if (nFishbone == 0)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone0], 1u, alpaka::hierarchy::Blocks{});
          else if (nFishbone == 1)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone1], 1u, alpaka::hierarchy::Blocks{});
          else
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone2p], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };
#endif  // CA_PIPELINE_COUNTERS

  // Map a track-hit id to its hit->tuple bin: merged hits key on the id directly; tagged OT extras key on
  // the extended domain slot nHits + otIdx (the container is sized nHits + nOTHits when the OT source is
  // active).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t hitToTupleKey(uint32_t id, uint32_t nHits) {
    return caExtension::isOTId(id) ? nHits + caExtension::otIdx(id) : id;
  }

  // The kernels below build the OneToManyAssoc candidate-generation structures behind the merger dedup,
  // following the CA's own count -> launchFinalize -> fill sequence. All fills are count-and-clamped: a key
  // outside [0, nKeys) is skipped and flagged in overflow[0] instead of written out of bounds, an unguarded
  // OneToManyAssoc overflow raising cudaErrorIllegalAddress at an allocation-dependent point.

  // Count pass for a track (eta,phi) binner (one thread per track). Used for the twinFindBest phi
  // pre-filter (nEtaSlabs=1 => pure phi) AND the 0-shared fallback eta-phi binner (nEtaSlabs>1).
  class Kernel_trackBinCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const int32_t nItems,  // consumer's iteration range; <0 => device nTracks()
                                  HitToTuple *__restrict__ assoc,
                                  const int nPhiBins,
                                  const int nEtaSlabs,
                                  const float etaMax,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = (nItems < 0) ? tracks_view.nTracks() : nItems;
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t key =
            uint32_t(trackBinKey(tracks_view[i].eta(), ::reco::phi(tracks_view, i), nPhiBins, nEtaSlabs, etaMax));
        if (key < nKeys)
          assoc->count(acc, key);
        else
          alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Fill pass for the (eta,phi) track binner (mirrors the count pass exactly).
  class Kernel_trackBinFill {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const int32_t nItems,
                                  HitToTuple *__restrict__ assoc,
                                  const int nPhiBins,
                                  const int nEtaSlabs,
                                  const float etaMax,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = (nItems < 0) ? tracks_view.nTracks() : nItems;
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t key =
            uint32_t(trackBinKey(tracks_view[i].eta(), ::reco::phi(tracks_view, i), nPhiBins, nEtaSlabs, etaMax));
        if (key < nKeys)
          assoc->fill(acc, key, i);
        else
          alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Count pass for the hit-id -> refined-track co-occurrence histogram (one thread per track, over the
  // track's CSR hit list). Merged pixel/strip ids bin on the id directly, bit30-tagged OT extras at
  // nHits+otIdx via hitToTupleKey, so the key space is [0, nHits + nOTHits) and the total fills, the sum of
  // the per-track hit counts, never exceed the trackHit CSR capacity the content is sized at.
  class Kernel_dedupHitCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const ::reco::TrackHitSoAConstView trackHit_view,
                                  HitToTuple *__restrict__ assoc,
                                  const uint32_t nHits,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t iBeg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t iEnd = tracks_view[i].hitOffsets();
        for (uint32_t a = iBeg; a < iEnd; ++a) {
          const uint32_t key = hitToTupleKey(trackHit_view[a].id(), nHits);
          if (key < nKeys)
            assoc->count(acc, key);
          else
            alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Fill pass for the hit-id co-occurrence histogram (mirrors the count pass exactly).
  class Kernel_dedupHitFill {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const ::reco::TrackHitSoAConstView trackHit_view,
                                  HitToTuple *__restrict__ assoc,
                                  const uint32_t nHits,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t iBeg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t iEnd = tracks_view[i].hitOffsets();
        for (uint32_t a = iBeg; a < iEnd; ++a) {
          const uint32_t key = hitToTupleKey(trackHit_view[a].id(), nHits);
          if (key < nKeys)
            assoc->fill(acc, key, i);
          else
            alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Kernel_dedupCovMark: one thread per refined merged track i. i is a duplicate loser iff a better
  // track j (strict total order: most information first, hit count as tie-break) is its duplicate:
  //   * pairs sharing a track-hit id: i loses when more than ShareFrac of its own published rechits
  //     go to a better track (no covariance test needed);
  //   * id-disjoint pairs (from an eta-phi neighbourhood sweep): full 5x5 covariance compatibility at
  //     the 5-sigma rejection, after the common refit so tip and zip are comparable.
  // Candidates come from the shared-hit co-occurrence histogram plus the sweep; the drop authority
  // covers the |eta| range the extension walk reaches. The order is strict and total, so the best
  // member of a group always survives in one pass.
  // diag (optional, 18 uint32): [region*3 + bucket] for region {0=central,1=forward} (split at
  // kDedupFwdEta) x shared bucket {0, 1, 2+}, [6+region] drop totals, [8+region] survivors whose
  // better partners all stayed under the shared fraction. Must be zero-initialised by the caller.
  class Kernel_dedupCovMark {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const ::reco::TrackHitSoAConstView trackHit_view,
                                  HitToTuple const *__restrict__ hitAssoc,
                                  HitToTuple const *__restrict__ etaPhiAssoc,
                                  const uint32_t nHits,
                                  const uint32_t nKeys,
                                  uint8_t *__restrict__ drop,  // 1 = this track is the duplicate loser
                                  uint32_t *__restrict__ diag,
                                  const float qGate5,         // chi2_5 at the 5-sigma duplicate rejection
                                  const float dropAbsEtaMax,  // = the walk's own |eta| reach
                                  const int fbEtaReach,
                                  const int fbPhiReach,
                                  const caStructures::CAHitsView hh,            // for caStructures::isStub
                                  const ::reco::StubsConstView sv,              // stub -> its two OT rows
                                  const ::reco::OTRecHitsConstView ov) const {  // sensor kind and position
      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const float etaI = tracks_view[i].eta();
        const auto qI = tracks_view[i].quality();
        const float c2I = tracks_view[i].chi2();
        const uint32_t iBeg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t iEnd = tracks_view[i].hitOffsets();

        // Length in PUBLISHED RECHITS, not in hit ids: the converter publishes a stub as the two
        // rechits it was built from and a raw-OT extra or a pixel hit as one, and that is the unit the
        // validation's 75 % matching fraction is a fraction of.
        auto lengthClusters = [&](int32_t t) -> int {
          const uint32_t b = (t == 0) ? 0u : tracks_view[t - 1].hitOffsets();
          const uint32_t e = tracks_view[t].hitOffsets();
          int c = 0;
          for (uint32_t a = b; a < e; ++a) {
            const uint32_t id = trackHit_view[a].id();
            c += caExtension::isOTId(id) ? 1 : (isStub(hh, int32_t(id)) ? 2 : 1);
          }
          return c;
        };
        const int nClI = lengthClusters(i);

        // Strict total order: is x a better member than y? The survivor of a duplicate pair is the
        // track whose FIT knows more -- the smaller error volume |C| -- because that is what the pair
        // is kept for; hit count only breaks a tie. Counting hits first hands every cross-arm pair to
        // its longer stub-seeded member even when the shorter pixel-seeded one measures the particle
        // better. Being total, the order names a unique loser, so drop[] needs no atomics.
        auto beats = [&](int32_t x, int32_t y) -> bool {
          if (const int o = dedupInfoOrder(tracks_view, x, y); o != 0)
            return o > 0;
          const int cx = lengthClusters(x), cy = lengthClusters(y);
          if (cx != cy)
            return cx > cy;
          const auto qx = tracks_view[x].quality(), qy = tracks_view[y].quality();
          if (qx != qy)
            return qx > qy;
          const float c2x = tracks_view[x].chi2(), c2y = tracks_view[y].chi2();
          if (c2x != c2y)
            return c2x < c2y;
          return x < y;
        };
        auto jBeatsI = [&](int32_t j) -> bool {
          if (const int o = dedupInfoOrder(tracks_view, j, i); o != 0)
            return o > 0;
          const int cj = lengthClusters(j);
          if (cj != nClI)
            return cj > nClI;
          const auto qj = tracks_view[j].quality();
          if (qj != qI)
            return qj > qI;
          const float c2j = tracks_view[j].chi2();
          if (c2j != c2I)
            return c2j < c2I;
          return j < i;
        };
        // The published rechits one track hit resolves to, in one key space: a pixel hit keys on its
        // own id, an outer-tracker rechit on nPix + its row in the OT SoA. A stub publishes the two
        // rechits it was built from, so it has two keys -- and that is why the shared count cannot be
        // taken on track-hit ids: a stub and a raw-OT extra of the same cluster, or two stubs sharing
        // a sensor hit, are different ids and the same published rechit, which is what the validation
        // sees. k = 0, 1 selects the key; false means this hit has no k-th one. Without the stub view
        // (the CA's own call) a stub keys on its id alone.
        const uint32_t nPix = uint32_t(hh.size() > 0 ? hh.offsetStubs() : 0u);
        const bool haveStubs = sv.metadata().size() > 0;
        auto pubKey = [&](uint32_t id, int k, uint32_t &key) -> bool {
          if (caExtension::isOTId(id))
            return k == 0 && (key = nPix + caExtension::otIdx(id), true);
          if (id < nPix || !haveStubs)
            return k == 0 && (key = id, true);
          const uint32_t st = id - nPix;
          if (k == 0)
            return (key = nPix + sv[int32_t(st)].lowerHitIdx(), true);
          if (!isStub(sv, int32_t(st)))
            return false;
          key = nPix + sv[int32_t(st)].upperHitIdx();
          return true;
        };
        // Which of i's published rechits track j also holds, as a bit per rechit of i (brute force;
        // lists <= kTwinMaxMergedHits, so at most 64 published rechits and one machine word). Bits are
        // returned rather than a count so that several better tracks can be OR-ed together: the rule
        // below is about the UNION of what i gives away, not about any single pair.
        const bool haveOTHits = ov.metadata().size() > 0;
        // Transverse radius of a published rechit, and whether it measures only one coordinate.
        auto pubRadius2 = [&](uint32_t key) -> float {
          float xg = 0.f, yg = 0.f;
          if (key < nPix) {
            xg = hh[int32_t(key)].xGlobal();
            yg = hh[int32_t(key)].yGlobal();
          } else if (haveOTHits && int32_t(key - nPix) < ov.metadata().size()) {
            xg = ov[int32_t(key - nPix)].xGlobal();
            yg = ov[int32_t(key - nPix)].yGlobal();
          }
          return xg * xg + yg * yg;
        };
        auto isStripRechit = [&](uint32_t key) -> bool {
          return haveOTHits && key >= nPix && int32_t(key - nPix) < ov.metadata().size() &&
                 ov[int32_t(key - nPix)].yerrLocal() > kStripYVarMin;
        };
        // The published rechit a track reaches first, by transverse radius, and its slot in i's own
        // numbering. This is the hit the share convention forgives when two tracks have the same one.
        auto innermostKey = [&](int32_t t, int &slotOut) -> uint32_t {
          const uint32_t tBeg = (t == 0) ? 0u : tracks_view[t - 1].hitOffsets();
          const uint32_t tEnd = tracks_view[t].hitOffsets();
          uint32_t best = 0xffffffffu;
          float bestR2 = 1e30f;
          int slot = 0;
          slotOut = -1;
          for (uint32_t a = tBeg; a < tEnd; ++a)
            for (int ka = 0; ka < 2; ++ka) {
              uint32_t key;
              if (!pubKey(trackHit_view[a].id(), ka, key))
                continue;
              const float r2 = pubRadius2(key);
              if (r2 < bestR2) {
                bestR2 = r2;
                best = key;
                slotOut = slot;
              }
              ++slot;
            }
          return best;
        };
        int innerSlotI = -1;
        const uint32_t innerKeyI = innermostKey(i, innerSlotI);
        // Where a track is along the beam line at a transverse radius, and how well that is known.
        // z(r) = zip + r cot(theta) from the perigee state; the arc-length correction to r is second
        // order in the curvature and is common to two tracks that could be the same particle, so it
        // cancels from the difference this is used for. The variance is the (zip, cotTheta) block of
        // the perigee covariance, correlation included.
        constexpr int kCovZipZip = 14, kCovCotZip = 13;
        auto zAtR = [&](int32_t t, float r, float &z, float &var) {
          z = tracks_view[t].state()[4] + r * tracks_view[t].state()[3];
          var = tracks_view[t].covariance()[kCovZipZip] + r * r * tracks_view[t].covariance()[kCovCotCot] +
                2.f * r * tracks_view[t].covariance()[kCovCotZip];
        };
        // Two tracks holding the same strip rechit are not thereby placed together: the strip measures
        // one coordinate and leaves the other free over its whole 5 cm support, which is wider than a
        // jet core is at the outer tracker. The strip is evidence that they are one track only if they
        // are also unresolved along it. When their own fits put them apart there at the same 5-sigma
        // rejection the duplicate test uses -- one degree of freedom, because one coordinate is at
        // stake -- the shared strip is one measurement handed to two trajectories, not two tracks of
        // one particle. A covariance that cannot be read resolves nothing, so the rechit counts.
        auto resolvedAlongStrip = [&](int32_t j, float r) -> bool {
          float zi, vi, zj, vj;
          zAtR(i, r, zi, vi);
          zAtR(j, r, zj, vj);
          const float v = vi + vj;
          if (!(v > 0.f))
            return false;
          const float d = zi - zj;
          return d * d > float(extDerivedTables::kDedupRejectChi2_1) * v;
        };
        auto sharedPublishedMask = [&](int32_t j) -> uint64_t {
          const uint32_t jBeg = (j == 0) ? 0u : tracks_view[j - 1].hitOffsets();
          const uint32_t jEnd = tracks_view[j].hitOffsets();
          uint64_t m = 0;
          int slot = 0;
          for (uint32_t a = iBeg; a < iEnd; ++a)
            for (int ka = 0; ka < 2; ++ka) {
              uint32_t keyA;
              if (!pubKey(trackHit_view[a].id(), ka, keyA))
                continue;
              if (slot >= 64)
                return m;  // structurally unreachable at this cap; never write outside the word
              for (uint32_t b = jBeg; b < jEnd; ++b)
                for (int kb = 0; kb < 2; ++kb) {
                  uint32_t keyB;
                  if (pubKey(trackHit_view[b].id(), kb, keyB) && keyB == keyA) {
                    if (!isStripRechit(keyA) || !resolvedAlongStrip(j, std::sqrt(pubRadius2(keyA))))
                      m |= uint64_t(1) << slot;
                    b = jEnd;  // this published rechit of i is matched; go to the next one
                    break;
                  }
                }
              ++slot;
            }
          return m;
        };
        auto popcount64 = [](uint64_t m) -> int {
          int c = 0;
          while (m) {
            m &= m - 1;
            ++c;
          }
          return c;
        };
        // Do the two lists share a track-hit id? That is what the co-occurrence histogram pairs on, so
        // it is the line between its candidates and the neighbourhood sweep's.
        auto sharesAnId = [&](int32_t j) -> bool {
          const uint32_t jBeg = (j == 0) ? 0u : tracks_view[j - 1].hitOffsets();
          const uint32_t jEnd = tracks_view[j].hitOffsets();
          for (uint32_t a = iBeg; a < iEnd; ++a)
            for (uint32_t b = jBeg; b < jEnd; ++b)
              if (trackHit_view[b].id() == trackHit_view[a].id())
                return true;
          return false;
        };
        // Gives too much away to be a track of its own: more than ShareFrac of i's own published
        // rechits sit on SOME better track. The test is on the union over all better tracks and not
        // pair by pair: a track that hands a fifth of itself to one better track and another fifth to
        // a second is just as much a duplicate as one that hands two fifths to either alone.
        // The rechit both tracks reach first is taken out of both sides, as the share convention does.
        // In a jet core that is the one a pair of real tracks is most likely to share: they are closest
        // to each other on the layer they are first seen on, and one cluster there serves both.
        const float shareLimit = kDedupShareFrac * float(nClI);
        const float shareLimitFirstShared = kDedupShareFrac * float(nClI - 1);

        const bool inDropRegion = (etaI <= dropAbsEtaMax) && (etaI >= -dropAbsEtaMax);
        bool loser = false;
        bool firstShared = false;
        bool sawShareMiss = false;
        int32_t bestPartner = -1;

        // shared-hit path: co-occurring candidates from the hit histogram. Every better co-occurring
        // track contributes its bits to one union mask; i loses when that union passes the fraction.
        uint64_t givenAway = 0;
        for (uint32_t a = iBeg; a < iEnd && !loser; ++a) {
          const uint32_t key = hitToTupleKey(trackHit_view[a].id(), nHits);
          if (key >= nKeys)
            continue;  // guarded (already counted in overflow during the build)
          for (auto p = hitAssoc->begin(key); p != hitAssoc->end(key); ++p) {
            const int32_t j = int32_t(*p);
            if (j == i)
              continue;
            if (!jBeatsI(j))
              continue;
            const uint64_t mj = sharedPublishedMask(j);
            if (mj == 0)
              continue;
            givenAway |= mj;
            if (bestPartner < 0 || beats(j, bestPartner))
              bestPartner = j;
            if (innerSlotI >= 0 && innerSlotI < 64 && innerKeyI != 0xffffffffu) {
              int innerSlotJ = -1;
              if (innermostKey(j, innerSlotJ) == innerKeyI)
                firstShared = true;
            }
            const int nGiven =
                popcount64(givenAway) - ((firstShared && ((givenAway >> innerSlotI) & uint64_t(1))) ? 1 : 0);
            if (nClI > 0 && float(nGiven) > (firstShared ? shareLimitFirstShared : shareLimit)) {
              loser = true;
              break;
            }
          }
        }
        if (!loser && givenAway != 0)
          sawShareMiss = true;  // better partners exist, their union stayed under the shared fraction
        // 0-shared neighbourhood sweep: an id-disjoint twin co-occurs in no hit bucket at all, so the
        // histogram above can never pair it. The two paths partition the candidates -- this one judges
        // only pairs that share nothing -- so it runs whenever the binner is there.
        if (!loser && etaPhiAssoc != nullptr) {
          const float phiI = ::reco::phi(tracks_view, i);
          const int pb = trackBinKey(0.f, phiI, kDedupFbPhiBins, 1, 0.f);
          const int eb =
              (trackBinKey(etaI, phiI, kDedupFbPhiBins, kDedupFbEtaSlabs, kDedupFbEtaMax) - pb) / kDedupFbPhiBins;
          for (int de = -fbEtaReach; de <= fbEtaReach && !loser; ++de) {
            const int e = eb + de;
            if (e < 0 || e >= kDedupFbEtaSlabs)
              continue;
            for (int dp2 = -fbPhiReach; dp2 <= fbPhiReach && !loser; ++dp2) {
              const int b = (pb + dp2 + kDedupFbPhiBins) % kDedupFbPhiBins;
              const uint32_t bin = uint32_t(e * kDedupFbPhiBins + b);
              for (auto p = etaPhiAssoc->begin(bin); p != etaPhiAssoc->end(bin); ++p) {
                const int32_t j = int32_t(*p);
                if (j == i)
                  continue;
                if (!jBeatsI(j))
                  continue;
                if (!dedupCompatible(tracks_view, i, j, 5, qGate5))
                  continue;
                if (sharesAnId(j))
                  continue;  // pairs the histogram already pairs belong to the path above
                loser = true;
                if (bestPartner < 0 || beats(j, bestPartner))
                  bestPartner = j;
              }
            }
          }
        }

        const uint8_t now = (loser && inDropRegion) ? uint8_t(1) : uint8_t(0);
        drop[i] = now;
        if (diag) {
          if (now) {
            const bool fwd =
                (std::abs(etaI) > kDedupFwdEta) || (std::abs(tracks_view[bestPartner].eta()) > kDedupFwdEta);
            const int region = fwd ? 1 : 0;
            const int shared = popcount64(givenAway);
            const int bucket = (shared <= 0) ? 0 : (shared == 1 ? 1 : 2);
            alpaka::atomicAdd(acc, &diag[region * 3 + bucket], 1u, alpaka::hierarchy::Blocks{});
            alpaka::atomicAdd(acc, &diag[6 + region], 1u, alpaka::hierarchy::Blocks{});
          } else if (sawShareMiss) {
            // Survived only because every better co-occurring partner stayed under the shared fraction.
            const int missRegion = (std::abs(etaI) > kDedupFwdEta) ? 1 : 0;
            alpaka::atomicAdd(acc, &diag[8 + missRegion], 1u, alpaka::hierarchy::Blocks{});
          }
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_countHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple,
                                  uint32_t nHits) const {  // OT extras bin at nHits + otIdx
      const auto nKeys = hitToTuple->nOnes();
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
          auto const key = hitToTupleKey(*h, nHits);
          // Key-range guard: a hitContainer content overflow leaves unwritten hit ids in the CSR, so the
          // key can land outside [0, nOnes). Drop instead of writing outside off[]; the fill pass below
          // skips exactly the same keys and counts the drop.
          if (key < nKeys)
            hitToTuple->count(acc, key);
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple,
                                  uint32_t nHits,
                                  Counters *counters) const {  // OT extras bin at nHits + otIdx
      const auto nKeys = hitToTuple->nOnes();
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
          auto const key = hitToTupleKey(*h, nHits);
          // Key-range guard mirroring the count pass; the drop is counted here, once per lost association.
          if (key < nKeys)
            hitToTuple->fill(acc, key, idx);
          else
            alpaka::atomicAdd(acc, &counters->nHitToTupleOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Content-buffer overflow repair, paired with the truncating bulkFill in OneToManyAssoc.h: when a tuple's hit
  // block does not fit the container, bulkFill plugs the offset but writes no content. Run right after
  // bulkFinalize, these kernels clamp every offset to the start of the first overflowed tuple k, so that tuple
  // and all later ones become empty. No-op when nothing overflowed. clampInfo[0] = off[k], clampInfo[1] = k,
  // which Kernel_fillHitDetIndices uses to cut nTracks so that no empty slot is published.
  class Kernel_findTupleContentOverflow {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  uint32_t *__restrict__ clampInfo) const {
      const uint32_t cap = uint32_t(foundNtuplets->capacity());
      const uint32_t nOff = uint32_t(foundNtuplets->totOnes());  // off[] has nOnes()+1 entries
      for (auto k : cms::alpakatools::uniform_elements(acc, nOff - 1)) {
        if (foundNtuplets->off[k] <= cap && foundNtuplets->off[k + 1] > cap) {
          clampInfo[0] = foundNtuplets->off[k];
          clampInfo[1] = uint32_t(k);
        }
      }
    }
  };

  class Kernel_clampTupleContentOverflow {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitContainer *__restrict__ foundNtuplets,
                                  uint32_t const *__restrict__ clampInfo) const {
      const uint32_t bound = clampInfo[0];
      for (auto j : cms::alpakatools::uniform_elements(acc, uint32_t(foundNtuplets->totOnes()))) {
        if (foundNtuplets->off[j] > bound)
          foundNtuplets->off[j] = bound;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitDetIndices {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  TkHitSoAView track_hits_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsMultiView hh,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  uint32_t const *__restrict__ tupleClampInfo) const {
      // clamp the number of tracks to the capacity of the SoA
      auto ntracks = std::min<int>(apc->get().first, tracks_view.metadata().size() - 1);
      // ... and to the first tuple dropped by a content-buffer overflow: the slots from that tuple on are
      // empty after the repair and must not be published (0xFFFFFFFF = nothing overflowed, no cut).
      if (tupleClampInfo[1] < uint32_t(ntracks))
        ntracks = int(tupleClampInfo[1]);
      if (cms::alpakatools::once_per_grid(acc))
        tracks_view.nTracks() = ntracks;

      // copy offsets, clamped to the hit SoA capacity: on a content-buffer overflow the raw offset can
      // exceed what the copy loop below writes, and a CSR end past the copied region would make downstream
      // hit walks read unwritten rows (the offset for track 0 is always 0).
      const uint32_t hitRowCap = uint32_t(track_hits_view.metadata().size());
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        tracks_view[idx].hitOffsets() = std::min(foundNtuplets->off[idx + 1], hitRowCap);
        tracks_view[idx].ndof() = 0;  // stamped by the fit for fitted tuples
      }
      // Tail: the slots past the last track carry its CSR end offset, so nHits() reads zero there and a
      // reader that walks the SoA up to the first empty slot stops at the right place.
      const uint32_t hitEndTail = std::min(foundNtuplets->off[ntracks], hitRowCap);
      for (auto idx : cms::alpakatools::uniform_elements(acc, uint32_t(tracks_view.metadata().size())))
        if (int(idx) >= ntracks)
          tracks_view[idx].hitOffsets() = hitEndTail;
      // fill hit indices, clamped to the hit SoA capacity: foundNtuplets->size() is the AtomicPairCounter's
      // hits-in-tracks total, which on a tuple overflow exceeds what was actually written, so an unclamped
      // loop would read the container beyond its filled region.
      const uint32_t nHitsInTracks = std::min<uint32_t>(foundNtuplets->size(), track_hits_view.metadata().size());
      for (auto idx : cms::alpakatools::uniform_elements(acc, nHitsInTracks)) {
        // On content-buffer overflow the content is unwritten, so the hit index can be out of range; the
        // hit was already dropped and writing a garbage detId would corrupt.
        if (foundNtuplets->content[idx] >= (uint32_t)hh.size())
          continue;
        track_hits_view[idx].id() = foundNtuplets->content[idx];
        track_hits_view[idx].detId() = hh[foundNtuplets->content[idx]].detectorIndex();
        track_hits_view[idx].attached() = 0;  // hit found by the CA
#ifdef CA_DEBUG
        printf("Kernel_fillHitDetIndices %d %d %d \n",
               idx,
               foundNtuplets->content[idx],
               track_hits_view.metadata().size());
#endif
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_doStatsForHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitToTuple const *__restrict__ hitToTuple,
                                  Counters *counters) const {
      auto &c = *counters;
      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple->nOnes())) {
        if (hitToTuple->size(idx) == 0)
          continue;  // SHALL NOT BE break
        alpaka::atomicAdd(acc, &c.nUsedHits, 1ull, alpaka::hierarchy::Blocks{});
        if (hitToTuple->size(idx) > 1)
          alpaka::atomicAdd(acc, &c.nDupHits, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_countSharedHit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  int *__restrict__ nshared,
                                  HitContainer const *__restrict__ ptuples,
                                  Quality const *__restrict__ quality,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      constexpr auto loose = Quality::loose;

      auto &hitToTuple = *phitToTuple;
      auto const &foundNtuplets = *ptuples;
      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        int nt = 0;

        // count "good" tracks
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (quality[*it] < loose)
            continue;
          ++nt;
        }

        if (nt < 2)
          continue;

        // now mark  each track triplet as sharing a hit
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (foundNtuplets.size(*it) > 3)
            continue;
          alpaka::atomicAdd(acc, &nshared[*it], 1, alpaka::hierarchy::Blocks{});
        }

      }  //  hit loop
    }
  };

  template <typename TrackerTraits>
  class Kernel_markSharedHit {
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  int const *__restrict__ nshared,
                                  HitContainer const *__restrict__ tuples,
                                  Quality *__restrict__ quality,
                                  bool dupPassThrough) const {
      // constexpr auto bad = Quality::bad;
      constexpr auto dup = Quality::dup;
      constexpr auto loose = Quality::loose;
      // constexpr auto strict = Quality::strict;

      // quality to mark rejected
      auto const reject = dupPassThrough ? loose : dup;
      for (auto idx : cms::alpakatools::uniform_elements(acc, tuples->nOnes())) {
        if (tuples->size(idx) == 0)
          break;  //guard
        if (quality[idx] <= reject)
          continue;
        if (nshared[idx] > 2)
          quality[idx] = reject;
      }
    }
  };

  // Track-parallel single-writer shared-hit removers: each thread owns one track, inspects the hit
  // buckets it belongs to (hitToTuple), reads every quality from the frozen scratch snapshot, and writes
  // only its own track's quality()
  template <typename TrackerTraits>
  class Kernel_rejectDuplicate {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      auto score = [&](uint32_t it) { return tracks_view[it].chi2(); };

      // A track is rejected iff some compatible track sharing one of its hits is strictly better by
      // the total order (more layers, then higher quality, then lower chi2, then lower track index)
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        auto const qi = qual(it);
        if (qi <= reject)
          continue;
        auto const nli = tracks_view[it].nLayers();
        // Phase2OTStubs only: the duplicate winner ordering inserts the total hit count as a tie-break
        // between nLayers and quality (max nLayers -> max total hits -> max quality -> min chi2 -> min
        // index). reco::nHits() is the track's full CSR hit extent, so it separates duplicates that tie on
        // nLayers, where otherwise a pixel-rich prompt track could beat its OT-rich displaced twin on the
        // chi2 tie-break and lose its TID hits. Off on every other topology.
        constexpr bool kUseHitCountTieBreak = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
        const uint32_t nhi = kUseHitCountTieBreak ? ::reco::nHits(tracks_view, it) : 0u;

        // get track parameters and covariances
        float iParams[nTrackParameters];
        float iCovs[nTrackParameters];
        for (int p{0}; p < nTrackParameters; ++p) {
          iParams[p] = tracks_view[it].state()(p);
          iCovs[p] = tracks_view[it].covariance()(iParam2iCov[p]);
        }
        auto incompatibleTrackParams = [&](uint32_t jt) -> bool {
          for (int p{0}; p < nTrackParameters; ++p) {
            const auto dpij = iParams[p] - tracks_view[jt].state()(p);
            const auto e2dpij = fastDupNSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
            if (dpij * dpij > e2dpij)
              return true;
          }
          return false;
        };

        bool dominated = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !dominated; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (jt == it)
              continue;
            auto const qj = qual(jt);
            if (qj <= reject)
              continue;
            if (incompatibleTrackParams(jt))
              continue;
            auto const nlj = tracks_view[jt].nLayers();
            // jt dominates it by the total order (nLayers, [total hits], quality, score, then track index).
            // The score test stays a strict order even for a NaN score, so exactly one of a pair is demoted
            bool jBetterTail =
                (qj > qi || (qj == qi && (score(jt) < score(it) || (!(score(it) < score(jt)) && jt < it))));
            bool jBetter;
            if constexpr (kUseHitCountTieBreak) {
              const uint32_t nhj = ::reco::nHits(tracks_view, jt);
              jBetter = (nlj > nli) || (nlj == nli && (nhj > nhi || (nhj == nhi && jBetterTail)));
            } else {
              jBetter = (nlj > nli) || (nlj == nli && jBetterTail);
            }
            if (jBetter) {
              dominated = true;
              break;
            }
          }
        }
        if (dominated)
          tracks_view[it].quality() = reject;
      }
    }
  };

  // Phase-1 specialization (very forward triplets)
  template <>
  class Kernel_rejectDuplicate<pixelTopology::Phase1> {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      auto score = [&](uint32_t it) { return std::abs(reco::tip(tracks_view, it)); };

      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        auto const qi = qual(it);
        if (qi <= reject)
          continue;
        auto const opi = tracks_view[it].state()(2);
        auto const e2opi = tracks_view[it].covariance()(9);
        auto const cti = tracks_view[it].state()(3);
        auto const e2cti = tracks_view[it].covariance()(12);
        auto const nli = tracks_view[it].nLayers();

        bool dominated = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !dominated; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (jt == it)
              continue;
            auto const qj = qual(jt);
            if (qj <= reject)
              continue;
            auto const opj = tracks_view[jt].state()(2);
            auto const ctj = tracks_view[jt].state()(3);
            auto const dct = nSigma2Phase1 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto const dop = nSigma2Phase1 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            auto const nlj = tracks_view[jt].nLayers();
            // jt dominates it by the total order (nLayers, quality, score, then track index). The score
            // test stays a strict order even for a non-finite score (NaN), so exactly one of a duplicate
            // pair is always demoted
            bool jBetter =
                (nlj > nli) ||
                (nlj == nli &&
                 (qj > qi || (qj == qi && (score(jt) < score(it) || (!(score(it) < score(jt)) && jt < it)))));
            if (jBetter) {
              dominated = true;
              break;
            }
          }
        }
        if (dominated)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_sharedHitCleaner {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  uint32_t const *__restrict__ layerStarts,
                                  TkSoAView tracks_view,
                                  int nmin,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      // quality of longest track
      auto const longTqual = Quality::highPurity;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      uint32_t l1end = layerStarts[1];

      // Short track `it` (nLayers <= nmin) is killed if it shares a non-bpix1 hit with a longer track
      // (nLayers == maxNl >= 4 among the highPurity tracks of that hit). maxNl is a reduction over the
      // frozen snapshot, so this is order-independent
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject)
          continue;
        auto const nlit = tracks_view[it].nLayers();
        if (nlit > nmin)
          continue;  // only short tracks are cleaned here

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h < l1end)
            continue;  // shared hit on bpix1
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          int8_t maxNl = 0;
          if (hitToTuple.size(h) >= 2) {
            for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
              if (qual(*jp) < longTqual)
                continue;
              maxNl = std::max(tracks_view[*jp].nLayers(), maxNl);
            }
          }

          // For Phase2OTStubs: several stubs sharing a lowerHitIdx have different hit indices but stand for
          // the same physical measurement, so for cleaning purposes they count as the same shared hit.
          // The merge is applied to PS stacks only: a 2S stub carries a lower cluster id as well, but
          // merging 2S stubs here kills short tracks whose long partner does not replace them -- measured,
          // 4.5 points of prompt barrel efficiency on ttbar PU200.
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            if (h < static_cast<uint32_t>(hh.size()) && isStub(hh, h) &&
                ::reco::StubFlags::isPS(hh.stub(int32_t(h)).flags())) {
              auto const lowerHitIdx = hh.stub(int32_t(h)).lowerHitIdx();
              if (lowerHitIdx != std::numeric_limits<uint32_t>::max()) {
                auto const offsetStubs = caStructures::offsetStubsOf(hh);
                auto const nHits = static_cast<uint32_t>(hh.size());
                for (uint32_t otherIdx = offsetStubs; otherIdx < nHits; ++otherIdx) {
                  if (otherIdx == h)
                    continue;
                  if (otherIdx >= hitToTuple.nOnes())
                    continue;
                  if (!isStub(hh, otherIdx) || !::reco::StubFlags::isPS(hh.stub(int32_t(otherIdx)).flags()))
                    continue;
                  if (hh.stub(int32_t(otherIdx)).lowerHitIdx() != lowerHitIdx)
                    continue;
                  for (auto jp = hitToTuple.begin(otherIdx); jp != hitToTuple.end(otherIdx); ++jp) {
                    if (qual(*jp) < longTqual)
                      continue;
                    maxNl = std::max(tracks_view[*jp].nLayers(), maxNl);
                  }
                }
              }
            }
          }

          if (maxNl >= 4 && nlit < maxNl)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_tripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::strict;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };

      // Track `it` is rejected if, on one of its shared hits whose good-quality tracks are all
      // triplets, it is not the best-tip survivor (lower track index breaks ties)
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject)
          continue;

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          if (hitToTuple.size(h) < 2)
            continue;
          bool onlyTriplets = true;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            if (qual(*jp) <= good)
              continue;
            onlyTriplets &= reco::isTriplet(tracks_view, *jp);
            if (!onlyTriplets)
              break;
          }
          if (!onlyTriplets)
            continue;
          float mc = maxScore;
          uint32_t im = tkNotFound;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (qual(jt) >= good) {
              auto const t = std::abs(reco::tip(tracks_view, jt));
              if (t < mc || (t == mc && jt < im)) {
                mc = t;
                im = jt;
              }
            }
          }
          if (im != tkNotFound && it != im)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_simpleTripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::loose;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };

      // Triplet `it` is rejected if, on one of its shared hits, it is not the best-tip survivor
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject || !reco::isTriplet(tracks_view, it))
          continue;

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          if (hitToTuple.size(h) < 2)
            continue;
          float mc = maxScore;
          uint32_t im = tkNotFound;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (qual(jt) >= good) {
              auto const t = std::abs(reco::tip(tracks_view, jt));
              if (t < mc || (t == mc && jt < im)) {
                mc = t;
                im = jt;
              }
            }
          }
          if (im != tkNotFound && it != im)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_print_found_ntuplets {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsMultiView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  uint32_t firstPrint,
                                  uint32_t lastPrint,
                                  int iev) const {
      constexpr auto loose = Quality::loose;

      for (auto i : cms::alpakatools::uniform_elements(acc, firstPrint, std::min(lastPrint, foundNtuplets->nOnes()))) {
        auto nh = foundNtuplets->size(i);
        if (nh < 3)
          continue;
        if (tracks_view[i].quality() < loose)
          continue;
        printf("TK: %d %d %d %d %f %f %f %f %f %f %f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
               10000 * iev + i,
               int(tracks_view[i].quality()),
               nh,
               tracks_view[i].nLayers(),
               reco::charge(tracks_view, i),
               tracks_view[i].pt(),
               tracks_view[i].eta(),
               reco::phi(tracks_view, i),
               reco::tip(tracks_view, i),
               reco::zip(tracks_view, i),
               tracks_view[i].chi2(),
               hh[*foundNtuplets->begin(i)].zGlobal(),
               hh[*(foundNtuplets->begin(i) + 1)].zGlobal(),
               hh[*(foundNtuplets->begin(i) + 2)].zGlobal(),
               nh > 3 ? hh[int(*(foundNtuplets->begin(i) + 3))].zGlobal() : 0,
               nh > 4 ? hh[int(*(foundNtuplets->begin(i) + 4))].zGlobal() : 0,
               nh > 5 ? hh[int(*(foundNtuplets->begin(i) + 5))].zGlobal() : 0,
               nh > 6 ? hh[int(*(foundNtuplets->begin(i) + nh - 1))].zGlobal() : 0);
      }
    }
  };

  class Kernel_printCounters {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc, Counters const *counters) const {
      auto const &c = *counters;
      printf(
          "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nLooseTracks  |  nGoodTracks | nUsedHits | "
          "nDupHits | nFishCells | nKilledCells | nUsedCells | nZeroTrackCells ||\n");
      printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
             c.nEvents,
             c.nHits,
             c.nCells,
             c.nTuples,
             c.nFitTracks,
             c.nLooseTracks,
             c.nGoodTracks,
             c.nUsedHits,
             c.nDupHits,
             c.nFishCells,
             c.nKilledCells,
             c.nEmptyCells,
             c.nZeroTrackCells);
      printf(
          "Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f|  %.3f|  "
          "%.3f||\n",
          c.nEvents,
          c.nHits / double(c.nEvents),
          c.nCells / double(c.nEvents),
          c.nTuples / double(c.nEvents),
          c.nFitTracks / double(c.nEvents),
          c.nLooseTracks / double(c.nEvents),
          c.nGoodTracks / double(c.nEvents),
          c.nUsedHits / double(c.nEvents),
          c.nDupHits / double(c.nEvents),
          c.nFishCells / double(c.nCells),
          c.nKilledCells / double(c.nCells),
          c.nEmptyCells / double(c.nCells),
          c.nZeroTrackCells / double(c.nCells));
      printf(
          "Counters Overflow %lld ||  tupleOvf=%lld  cellOvf=%lld  tripletOvf=%lld  "
          "cellTrackOvf=%lld  hitToTupleOvf=%lld  hitToCellOvf=%lld ||\n",
          c.nEvents,
          c.nTupleOverflow,
          c.nCellOverflow,
          c.nTripletOverflow,
          c.nCellTrackOverflow,
          c.nHitToTupleOverflow,
          c.nHitToCellOverflow);
    }
  };
  // Merger gather/compact kernel: reads each input's nTracks() and last hitOffsets() on the device and compacts
  // all track and trackHits columns (id, detId, attached) of every input into a dense merged layout, shifting
  // hitOffsets by the cumulative hit count of the inputs before it; every thread recomputes the per-input
  // offsets. The arm labels are passed as two scalars. The output tail [mergedNTracks, capacity) is stamped
  // Quality::bad over quality(), which makes the downstream filterTracks kernel skip every tail slot.
  class Kernel_mergeGather {
  public:
    // Tie the eigen column element counts used by the copy below (5 for state, 15 for covariance) to the
    // layout, so resizing either column breaks the build here instead of mis-striding the copy.
    static_assert(::reco::Vector5f::RowsAtCompileTime == 5 && ::reco::Vector15f::RowsAtCompileTime == 15,
                  "the eigen columns of reco::TrackLayout changed size: the element count this copy "
                  "starts from and the step it adds after each eigen column must be updated together");
    // Tie the hardcoded column set below to the SoA layouts: adding or removing a column fails the build
    // here instead of leaving the new column silently uncopied.
    // TrackLayout (Phase 2 copy): quality, chi2, nLayers, eta, pt, state[5], covariance[15], hitOffsets,
    //   iteration, ndof = 10 columns, plus the nTracks scalar written in Phase 1.
    // TrackHitsLayout (Phase 3 copy): id, detId, attached = 3 columns.
    static_assert(::reco::TrackSoA::Descriptor::num_cols == 11,
                  "reco::TrackLayout column count changed: update the gather kernel's track column "
                  "enumeration in Kernel_mergeGather (quality, chi2, nLayers, eta, pt, state[5], "
                  "covariance[15], nTracks, hitOffsets, iteration, ndof)");
    static_assert(::reco::TrackHitSoA::Descriptor::num_cols == 3,
                  "reco::TrackHitsLayout column count changed: update the gather kernel's trackHits "
                  "column enumeration in Kernel_mergeGather (id, detId, attached)");

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackSoAView outTrack_view,
                                  ::reco::TrackHitSoAView outHit_view,
                                  const ::reco::TrackSoAConstView inp0Track_view,
                                  const ::reco::TrackHitSoAConstView inp0Hit_view,
                                  const ::reco::TrackSoAConstView inp1Track_view,
                                  const ::reco::TrackHitSoAConstView inp1Hit_view,
                                  const int nInputs,
                                  int32_t *armBuf,
                                  const int32_t arm0,
                                  const int32_t arm1) const {
      // Phase 1: all threads read device-side nTracks and hitOffsets and compute the cumulative offsets
      // independently; grid thread 0 also writes the merged nTracks scalar.
      uint32_t nTks[2] = {0, 0};
      uint32_t cumulTks[3] = {0, 0, 0};
      uint32_t cumulHits[3] = {0, 0, 0};

      // The two inputs are passed as separate view arguments and the arrays above are sized for exactly two;
      // the host refuses more (PixelTracksSoAMerger throws) and the clamp keeps the loops inside the arrays.
      const int nInp = (nInputs < 2) ? nInputs : 2;

      for (int s = 0; s < nInp; ++s) {
        const uint32_t ntk = (s == 0) ? uint32_t(inp0Track_view.nTracks()) : uint32_t(inp1Track_view.nTracks());
        nTks[s] = ntk;
        cumulTks[s + 1] = cumulTks[s] + ntk;
        // Total hits for this input = last filled hitOffsets (CSR cumulative hit-end); 0 when ntk == 0.
        uint32_t totHitsS = 0;
        if (ntk > 0) {
          totHitsS = (s == 0) ? uint32_t(inp0Track_view[ntk - 1].hitOffsets())
                              : uint32_t(inp1Track_view[ntk - 1].hitOffsets());
        }
        cumulHits[s + 1] = cumulHits[s] + totHitsS;
      }

      const uint32_t outCap = uint32_t(outTrack_view.metadata().size());
      const uint32_t outHitCap = uint32_t(outHit_view.metadata().size());
      // Clamp the merged count to the output track capacity, the track-side twin of the hit-side clamp in
      // Phase 3. The output covers the sum of the inputs, so this cannot bind; it exists so that a capacity
      // mismatch degrades by dropping the excess instead of writing past the allocation.
      const uint32_t mergedNTracks = (cumulTks[nInp] < outCap) ? cumulTks[nInp] : outCap;

      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0) {
        outTrack_view.nTracks() = mergedNTracks;
      }

      // Phase 2: all threads cooperate on a grid-stride copy of every track column.
      // The per-input copy range is [cumulTks[s], cumulTks[s+1]) in the output, [0, nTks[s]) in the input.
      for (int s = 0; s < nInp; ++s) {
        auto inpTrack_view = (s == 0) ? inp0Track_view : inp1Track_view;
        const uint32_t ntk = nTks[s];
        const uint32_t outBase = cumulTks[s];
        const uint32_t hitShift = cumulHits[s];
        const int32_t armLabel = (s == 0) ? arm0 : arm1;
        // Same clamp as above, applied to this input's slice of the output.
        const uint32_t ntkCopy = (outBase >= outCap) ? 0u : ((ntk < outCap - outBase) ? ntk : (outCap - outBase));

        for (uint32_t i : cms::alpakatools::uniform_elements(acc, ntkCopy)) {
          const uint32_t outIdx = outBase + i;
          // Copy every track column enumerated in the static_asserts above.
          outTrack_view[outIdx].quality() = inpTrack_view[i].quality();
          outTrack_view[outIdx].chi2() = inpTrack_view[i].chi2();
          outTrack_view[outIdx].ndof() = inpTrack_view[i].ndof();
          outTrack_view[outIdx].nLayers() = inpTrack_view[i].nLayers();
          outTrack_view[outIdx].eta() = inpTrack_view[i].eta();
          outTrack_view[outIdx].pt() = inpTrack_view[i].pt();
          for (uint32_t k = 0; k < 5; ++k)
            outTrack_view[outIdx].state()[k] = inpTrack_view[i].state()[k];
          for (uint32_t k = 0; k < 15; ++k)
            outTrack_view[outIdx].covariance()[k] = inpTrack_view[i].covariance()[k];
          outTrack_view[outIdx].iteration() = inpTrack_view[i].iteration();
          // Shift hitOffsets by the cumulative hit count of the previous inputs, clamped to the output hit
          // capacity so the CSR end offsets stay inside the block the hit copy below is clamped to.
          const uint32_t shifted = uint32_t(inpTrack_view[i].hitOffsets()) + hitShift;
          outTrack_view[outIdx].hitOffsets() = (shifted < outHitCap) ? shifted : outHitCap;
          // arm-label buffer, in the dense merged-SoA ordering of the track copy
          if (armBuf)
            armBuf[outIdx] = armLabel;
        }
      }

      // Phase 3: copy trackHits columns (id, detId, attached) for each input.
      for (int s = 0; s < nInp; ++s) {
        auto inpHit_view = (s == 0) ? inp0Hit_view : inp1Hit_view;
        auto inpTrack_view = (s == 0) ? inp0Track_view : inp1Track_view;
        const uint32_t ntk = nTks[s];
        const uint32_t outHitBase = cumulHits[s];
        // Total hits for this input (recomputed from the CSR end, same as Phase 1).
        uint32_t totHitsS = 0;
        if (ntk > 0)
          totHitsS = uint32_t(inpTrack_view[ntk - 1].hitOffsets());
        // Truncate the copy at the output hit capacity. The output is sized to the sum of the inputs'
        // hit-block capacities, so this binds only if an input's own hit total ran past its block.
        if (outHitBase >= outHitCap) {
          if (cms::alpakatools::once_per_grid(acc) && totHitsS > 0)
            printf("Warning!!!! mergeGather: input %d hit copy dropped entirely (base %u >= capacity %u)!\n",
                   s,
                   outHitBase,
                   outHitCap);
          totHitsS = 0;
        } else if (outHitBase + totHitsS > outHitCap) {
          if (cms::alpakatools::once_per_grid(acc))
            printf("Warning!!!! mergeGather: input %d hit copy truncated (%u of %u hits kept)!\n",
                   s,
                   outHitCap - outHitBase,
                   totHitsS);
          totHitsS = outHitCap - outHitBase;
        }

        for (uint32_t h : cms::alpakatools::uniform_elements(acc, totHitsS)) {
          outHit_view[outHitBase + h].id() = inpHit_view[h].id();
          outHit_view[outHitBase + h].detId() = inpHit_view[h].detId();
          outHit_view[outHitBase + h].attached() = inpHit_view[h].attached();
        }
      }

      // Phase 4: tail [mergedNTracks, outCap). quality() is stamped bad so the downstream filterTracks
      // kernel skips every tail slot; the arm label is stamped -1 and the iteration label notIteration so a
      // consumer that reads either column before testing quality() sees a defined value.
      const uint32_t hitEndTail = (cumulHits[nInp] < outHitCap) ? cumulHits[nInp] : outHitCap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, outCap)) {
        if (i >= mergedNTracks) {
          outTrack_view[i].quality() = pixelTrack::Quality::bad;
          outTrack_view[i].iteration() = pixelTrack::Iteration::notIteration;
          outTrack_view[i].hitOffsets() = hitEndTail;  // nHits() reads zero past the last track
          if (armBuf)
            armBuf[i] = -1;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels

// The host side of CAHitNtupletGeneratorKernels: the launchers and the buffer management. Instantiated per
// topology by the CAHitNtupletGeneratorKernels_<Topology>.dev.cc translation units.
// C++ headers
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// local headers
#include "CAFishbone.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "HelixFit.h"
#include "ExtDerivedTables.h"  // the analytic chi2 quantile the duplicate test is taken at

//#define GPU_DEBUG
// #define NTUPLE_DEBUG
//#define CA_STATS

namespace ALPAKA_ACCELERATOR_NAMESPACE {

// The sizing accumulator serves both the CA_STATS report and the GPU_DEBUG allocation report.
#if defined(CA_STATS) || defined(GPU_DEBUG)
  // Thread- and event-safe accumulator for sizing-parameter recommendations.
  // Fixed-avg parameters (size = max_keys * avg):
  //   index 0 -> avgHitsPerTrack
  //   index 1 -> avgCellsPerHit
  //   index 2 -> avgCellsPerCell
  //   index 3 -> avgTracksPerCell
  // Scaling-with-nHits parameters (size = slope * nHits):
  //   index 0 -> maxNumberOfDoublets (slope = nCells  / nHits)
  //   index 1 -> maxNumberOfTuples   (slope = nTuples / nHits)
  // maxNHits is tracked too, so f(nHits) can be evaluated at the observed worst case.
  namespace {
    struct SizingAccumulator {
      std::mutex mtx;
      uint64_t n = 0;
      double maxReq[4] = {0., 0., 0., 0.};
      double sumReq[4] = {0., 0., 0., 0.};
      double maxPhy[4] = {0., 0., 0., 0.};
      double sumPhy[4] = {0., 0., 0., 0.};
      double maxRatio[2] = {0., 0.};
      double sumRatio[2] = {0., 0.};
      uint32_t maxNHits = 0u;
      uint64_t allocN = 0;
      uint64_t allocBytesSum = 0;
      uint64_t allocBytesMax = 0;
    };
    // One slot per pixelTrack::Iteration so several CA instances running
    // in the same job do not pollute each other's recommendations.
    inline SizingAccumulator &sizingAccumulator(::pixelTrack::Iteration it) {
      static std::array<SizingAccumulator, ::pixelTrack::iterationSize> s;
      return s[static_cast<uint8_t>(it)];
    }
  }  // namespace
#endif  // CA_STATS || GPU_DEBUG

  // Inclusive prefix scan of a OneToManyAssoc's offsets through the multi-level iterativePrefixScan.
  // The generic OneToManyAssoc::launchFinalize uses one multiBlockPrefixScan launch whose dynamic
  // shared memory grows with the number of 1024-entry blocks and so exceeds the 48 kB launch limit
  // for the large CA containers. Same numbers; on the serial backend a plain sequential scan.
  template <typename TView>
  inline void finalizeAssocOffsets(TView const &view, Queue &queue) {
    cms::alpakatools::iterativePrefixScan<Acc1D>(view.offStorage, view.offStorage, view.offSize, queue);
  }

  // Sizing rule of the cells + cell->track-offsets arena (see the member declaration in the header).
  // Returns the arena's extent in SimpleCell units, or 0 when one allocation is not cheaper than two.
  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::cellArenaExtent(uint32_t cellBound, uint32_t maxDoublets) {
    // The allocator's bin: the smallest power of two >= bytes, floored at its 256 B minimum bin
    // (binGrowth 2, minBin 8, maxBin 30).
    auto bin = [](std::size_t bytes) {
      std::size_t b = 256;
      while (b < bytes)
        b *= 2;
      return b;
    };
    constexpr std::size_t kCell = sizeof(SimpleCell);
    // 64 cells = 64*12 = 768 B, a multiple of 256, so the offsets region starts 256 B aligned
    // whatever the cap is.
    const std::size_t cells = ((std::size_t(cellBound) + 63u) / 64u) * 64u;
    const std::size_t offsetBytes = (std::size_t(maxDoublets) + 1u) * sizeof(GenericContainerOffsets);
    const std::size_t extra = (offsetBytes + kCell - 1u) / kCell;
    const std::size_t separate = bin(std::size_t(cellBound) * kCell) + bin(offsetBytes);
    const std::size_t together = bin((cells + extra) * kCell);
    return (together < separate) ? uint32_t(cells + extra) : 0u;
  }

  // Which of the three packings of the four cell-scale buffers is cheapest for this event's bounds.
  // The allocator's bins are powers of two, so the winner depends on where 12*cellBound falls inside
  // its bin; all three are evaluated and the smallest total is taken. The minimum is over the bytes
  // allocated, not the bytes resident at every instant: kThreeCellIndexArrays shares a buffer with
  // the cell->track offsets, which live to the end of the event, so releaseBuildScratch() cannot
  // hand those two back after launchKernels.
  template <typename TrackerTraits>
  typename CAHitNtupletGeneratorKernels<TrackerTraits>::CellLayout
  CAHitNtupletGeneratorKernels<TrackerTraits>::chooseCellLayout(uint32_t cellBound, uint32_t maxDoublets) {
    auto bin = [](std::size_t bytes) {
      std::size_t b = 256;
      while (b < bytes)
        b *= 2;
      return b;
    };
    auto pad = [](std::size_t b) { return (b + 255u) / 256u * 256u; };
    const std::size_t cells = std::size_t(cellBound) * sizeof(SimpleCell);
    const std::size_t store = std::size_t(std::max(cellBound, 1u)) * sizeof(GenericContainerStorage);
    const std::size_t off = (std::size_t(maxDoublets) + 1u) * sizeof(GenericContainerOffsets);
    const std::size_t sep = bin(cells) + bin(store) + 2u * bin(off);
    const std::size_t a = bin(pad(cells) + off) + bin(store) + bin(off);
    const std::size_t b3 = bin(cells) + bin(pad(store) + pad(off) + off);
    if (b3 < sep && b3 <= a)
      return CellLayout::kThreeCellIndexArrays;
    if (a < sep)
      return CellLayout::kCellsWithTrackOffsets;
    return CellLayout::kSeparate;
  }

  template <typename TrackerTraits>
  CAHitNtupletGeneratorKernels<TrackerTraits>::CAHitNtupletGeneratorKernels(Params const &params,
                                                                            uint32_t nHits,
                                                                            uint32_t nOTHits,
                                                                            uint32_t offsetBPIX2,
                                                                            uint32_t maxDoublets,
                                                                            uint32_t maxTuples,
                                                                            uint16_t nLayers,
                                                                            Queue &queue)
      : m_params(params), nOTHits_(nOTHits) {
    //////////////////////////////////////////////////////////
    // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
    //////////////////////////////////////////////////////////

    counters_ = cms::alpakatools::make_device_buffer<Counters>(queue);
    // Here we define the OneToMany maps and the histograms
    // allocating the buffers and defining the views.
    // For each map/histo, we need:
    // - a buffer for the offsets sized as the number of ones + 1
    //   (with the last bin holding the total number of ones)
    // - a buffer fot the content/storage itself sized as the number of many

    auto const &algoParams = m_params.algoParams_;
    // Allocation strategy (see fillDescriptions). When both delayAllocations_ and countDoubletsFirst_
    // are false every buffer is allocated here with no device->host synchronization
    // Defer cell-derived + hit->track buffers
    const bool delay = algoParams.delayAllocations_;
    // defer simpleCells/hitToCellStorage and allocate to the actual number of doublets produced
    const bool countFirst = algoParams.countDoubletsFirst_;

    // Cell bound, distinct from maxDoublets: it bounds the cell array and the hit->cell association, while
    // maxDoublets also sets the content capacities. It is capped at an allocator bin edge, because at the largest
    // maxDoublets the cell array would be served from the next power-of-two bin in one rare event and the
    // allocator never gives a block back. The content capacities are untouched.
    static constexpr uint32_t kMaxDoubletsForCellBin = (64u * 1024u * 1024u) / uint32_t(sizeof(SimpleCell));
    const uint32_t cellBound = std::min(maxDoublets, kMaxDoubletsForCellBin);
    // Packing of the four cell-scale buffers, evaluated per event (see chooseCellLayout). Only the
    // up-front allocation mode packs: the exact-allocation paths size these buffers from counts that do
    // not exist yet at this point.
    const CellLayout cellLayout =
        (!delay && !countFirst) ? chooseCellLayout(cellBound, maxDoublets) : CellLayout::kSeparate;
    if (cellLayout == CellLayout::kThreeCellIndexArrays) {
      // hit->cell storage, then the two cell-keyed offsets, each 256 B aligned inside one buffer.
      const std::size_t nStore = ((std::size_t(std::max(cellBound, 1u)) + 63u) / 64u) * 64u;
      const std::size_t nOff = ((std::size_t(maxDoublets) + 1u + 63u) / 64u) * 64u;
      device_cellIndexArena_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(
          queue, uint32_t(nStore + nOff + std::size_t(maxDoublets) + 1u));
      arenaHitToCellStorage_ = device_cellIndexArena_->data();
      arenaCellToNeighborsOffsets_ =
          reinterpret_cast<GenericContainerOffsets *>(device_cellIndexArena_->data() + nStore);
      arenaCellToTracksOffsets_ =
          reinterpret_cast<GenericContainerOffsets *>(device_cellIndexArena_->data() + nStore + nOff);
    }
    uint32_t outerHits =
        nHits - offsetBPIX2;  // the number of hits that may be used as outer hits for a cell (so not on bpix1)

    // Sizes the per-track hit storage (hitContainer/hitToTuple). Same expression as the output
    // trackHits SoA in CAHitNtupletGenerator::beginTuplesAsync, so the two capacities match.
    uint32_t nHitsToTracks = caHitNtupletGenerator::nHitRowsForTuples(maxTuples, algoParams.avgHitsPerTrack_);

#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building with: " << std::endl;
    std::cout << "- nHits          = " << nHits << std::endl;
    std::cout << "- outerHits      = " << outerHits << std::endl;
    std::cout << "- maxDoublets    = " << maxDoublets << std::endl;
    std::cout << "- maxTracks      = " << maxTuples << std::endl;
    std::cout << "- nHitsToTracks  = " << nHitsToTracks << std::endl;
#endif

    // Hits -> Track
    // The handle and the per-hit offsets (key space) are always allocated here. The storage holds one
    // entry per hit-in-track: with delayAllocations_ it is sized from the actual hits-in-tracks count
    // in allocateAfterNtuplets, otherwise here at the nHitsToTracks safety cap.
    // A non-zero nOTHits extends the hit->tuple domain to nHits + nOTHits, so tagged OT extras bin at
    // nHits + otIdx and the duplicate cleaning accounts for them.
    const uint32_t hitToTupleKeys = nHits + nOTHits_;
    device_hitToTuple_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToTupleOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, hitToTupleKeys + 1);
    if (!delay) {
      device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToTracks);
      device_hitToTupleView_ = {device_hitToTuple_->data(),
                                device_hitToTupleOffsets_->data(),
                                device_hitToTupleStorage_->data(),
                                hitToTupleKeys + 1,
                                nHitsToTracks};

      HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);
    }

    // (Outer) Hits-> Cells
    // The storage holds exactly one entry per cell: with countDoubletsFirst it is sized to the exact
    // nCells in buildDoublets, otherwise to the maxDoublets cap here, the same cap that bounds the
    // cell array itself and that the doublet finder enforces when it appends. A second, independent
    // constant would let this container truncate while maxDoublets does not: one entry per cell, one
    // bound. The handle and per-outer-hit offsets (key space) are always allocated.
    device_hitToCell_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToCellOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, outerHits + 1);
    if (!countFirst) {
      uint32_t nHitsToCells = std::max(cellBound, 1u);
      GenericContainerStorage *hitToCellStorage = arenaHitToCellStorage_;
      if (hitToCellStorage == nullptr) {
        device_hitToCellStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToCells);
        hitToCellStorage = device_hitToCellStorage_->data();
      }
      device_hitToCellView_ = {
          device_hitToCell_->data(), device_hitToCellOffsets_->data(), hitToCellStorage, outerHits + 1, nHitsToCells};

      HitToCell::template launchZero<Acc1D>(device_hitToCellView_, queue);
    }

    // Hits Phi Histograms: one histogram per layer
    device_hitPhiHist_ = cms::alpakatools::make_device_buffer<PhiBinner>(queue);
    device_phiBinnerStorage_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nHits);
    device_hitPhiView_ = {
        device_hitPhiHist_->data(), nullptr, device_phiBinnerStorage_->data(), cms::alpakatools::kDynamicSize, nHits};
    // This will hold where each layer starts in the hit soa
    device_layerStarts_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nLayers + 1);

    // Scratch quality mirror used by the (order-independent) duplicate-removal kernels
    device_qualityScratch_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, maxTuples);

    // Cell array, allocated ahead of allocateAfterDoublets because in the up-front allocation mode it
    // doubles as the arena carrying the cell->track offsets. With countDoubletsFirst it is instead
    // sized to the exact nCells in buildDoublets.
    if (!countFirst) {
      const uint32_t arenaExtent =
          (cellLayout == CellLayout::kCellsWithTrackOffsets) ? cellArenaExtent(cellBound, maxDoublets) : 0u;
      device_simpleCells_ =
          cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, arenaExtent ? arenaExtent : cellBound);
      if (arenaExtent) {
        const std::size_t cellsPadded = ((std::size_t(cellBound) + 63u) / 64u) * 64u;
        arenaCellToTracksOffsets_ =
            reinterpret_cast<GenericContainerOffsets *>(device_simpleCells_->data() + cellsPadded);
      }
    }

    // Cell -> neighbour cells and cell -> tracks (and the triplet/track-cell SoA) are sized from the
    // number of doublets: with delayAllocations_ from the actual nCells in allocateAfterDoublets,
    // otherwise here at the maxDoublets safety cap through the same function.
    if (!delay) {
      allocateAfterDoublets(maxDoublets, queue);
    }

    // Track -> Hits
    // - This is a OneToManyAssocSequential since each bin is filled
    //   in one go: all the hits forming a track are pushed together.
    device_hitContainer_ = cms::alpakatools::make_device_buffer<SequentialContainer>(queue);
    device_hitContainerStorage_ =
        cms::alpakatools::make_device_buffer<SequentialContainerStorage[]>(queue, nHitsToTracks);
    device_hitContainerOffsets_ =
        cms::alpakatools::make_device_buffer<SequentialContainerOffsets[]>(queue, maxTuples + 1);
    device_hitContainerView_ = {device_hitContainer_->data(),
                                device_hitContainerOffsets_->data(),
                                device_hitContainerStorage_->data(),
                                maxTuples + 1u,
                                nHitsToTracks};

    HitContainer::template launchZero<Acc1D>(device_hitContainerView_, queue);
    device_tupleClampBound_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u);

    // No.Hits -> Track (track multiplicity)
    device_tupleMultiplicity_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_tupleMultiplicityStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, maxTuples);
    device_tupleMultiplicityOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, TrackerTraits::maxHitsOnTrack + 2);
    device_tupleMultiplicityView_ = {
        device_tupleMultiplicity_->data(),
        device_tupleMultiplicityOffsets_->data(),
        device_tupleMultiplicityStorage_->data(),
        // this has to be +2 instead of +1 because you want all values from 0 to maxHitsOnTrack to be valid keys
        // (N+1 values) + the extra +1 for the Container definition
        TrackerTraits::maxHitsOnTrack + 2u,
        maxTuples};
    TupleMultiplicity::template launchZero<Acc1D>(device_tupleMultiplicityView_, queue);

    // device_simpleCells_ is allocated above, ahead of allocateAfterDoublets, so that it can carry the
    // cell->track offsets in its tail.
    device_extraStorage_ =
        cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 5u);
    device_hitTuple_apc_ = reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_extraStorage_->data());
    device_nCells_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 2));
    device_nTriplets_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 3));
    device_nCellTracks_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 4));

    // deviceTriplets_ and deviceTracksCells_ are sized from nCells in allocateAfterDoublets

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters for diagnostic funnel
    device_pipelineCounters_ =
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, caHitNtupletGenerator::kTotalCounters);
    alpaka::memset(queue, *device_pipelineCounters_, 0);
#endif

    //TODO: if doStats?
    alpaka::memset(queue, *counters_, 0);

    alpaka::memset(queue, *device_nCells_, 0);
    alpaka::memset(queue, *device_nTriplets_, 0);
    alpaka::memset(queue, *device_nCellTracks_, 0);

    maxNumberOfDoublets_ = cellBound;
    launchCells_ = cellBound;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Allocations for CAHitNtupletGeneratorKernels: done!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::readbackNCells(Queue &queue) {
    using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
    auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
    alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
    alpaka::wait(queue);  // Necessary wait: the value is returned to the host caller, which uses it to
                          // size the next launches.
    // device_nCells_ aliases extraStorage word [2].
    return static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[2]);
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::enqueueCountsReadback(
      cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]> &dst, Queue &queue) {
    // Async copy only -- the caller reads dst after its queue has been synchronized (the CA
    // producer's acquire->produce seam), so no host wait is needed here.
    alpaka::memcpy(queue, dst, *this->device_extraStorage_);
  }
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::readbackAllCounts(Queue &queue,
                                                                      uint32_t &nTracks,
                                                                      uint32_t &nHitsInTracks,
                                                                      uint32_t &nCells,
                                                                      uint32_t &nTriplets,
                                                                      uint32_t &nCellTracks) {
    using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
    auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
    alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
    alpaka::wait(queue);
    // Word [0] is the AtomicPairCounter: its low half is `first` (the tuple count) and its high
    // half is `second` (the hits-in-tracks total), the same halves Kernel_overflowSentinel reads.
    // Taken by position, not by size: the two are only ordered while every tuple has >= 2 hits.
    const uint64_t apc_raw = static_cast<uint64_t>(alpaka::getPtrNative(h_extra)[0]);
    nTracks = static_cast<uint32_t>(apc_raw & 0xFFFFFFFFull);
    nHitsInTracks = static_cast<uint32_t>(apc_raw >> 32);
    // word [2] = nCells, [3] = nTriplets, [4] = nCellTracks
    nCells = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[2]);
    nTriplets = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[3]);
    nCellTracks = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[4]);
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::allocateAfterDoublets(uint32_t nCells, Queue &queue) {
    auto const &algoParams = m_params.algoParams_;

    // Association-content sizes scale with the number of doublets found: the per-key averages are kept but
    // based on nCells instead of maxDoublets. Quiet-event floor: the averages are fitted at high occupancy,
    // where cells per cell is ~0.1; in a clean event the ratio rises towards 1 while nCells falls, and the
    // surplus would be dropped silently. The floor is at most kCleanEdgesPerCell edges per cell, tracked
    // linearly up to kQuietCellsCap and constant above it.
    constexpr uint32_t kQuietCellsCap = 512u;
    constexpr uint32_t kCleanEdgesPerCell = 64u;
    const uint32_t quietFloor = std::min(nCells, kQuietCellsCap) * kCleanEdgesPerCell;
    const uint32_t nCellsToCells = std::max({uint32_t(nCells * algoParams.avgCellsPerCell_), quietFloor, 1u});
    const uint32_t nCellsToTracks = std::max({uint32_t(nCells * algoParams.avgTracksPerCell_), quietFloor, 1u});
    const uint32_t nBins = nCells + 1u;  // one offset bin per cell (+1 for the total)

    // The two extents above have different natures. nBins, the offsets extent of both associators, is
    // a key-space extent: one word per cell plus the total, and every index tested against it is a
    // cell id, hence < nCells by construction. nCellsToCells and nCellsToTracks are content
    // capacities: the truncation guards compare against them, so entries past them are dropped
    // silently, and their per-cell ratios are calibrated against the basis in force here.
    // This function is called either once at construction with the maxDoublets cap, when no doublet
    // count exists yet, or after the doublet build with the exact count.

    // Cell -> Neighbor Cells. One bin per cell; bit 31 of each stored neighbor index
    // encodes layer-skipping (valid since nCells is well below 2^31).
    device_cellToNeighbors_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToNeighborsStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToCells);
    GenericContainerOffsets *cellToNeighborsOffsets = arenaCellToNeighborsOffsets_;
    if (cellToNeighborsOffsets == nullptr) {
      device_cellToNeighborsOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nBins);
      cellToNeighborsOffsets = device_cellToNeighborsOffsets_->data();
    }
    device_cellToNeighborsView_ = {device_cellToNeighbors_->data(),
                                   cellToNeighborsOffsets,
                                   device_cellToNeighborsStorage_->data(),
                                   nBins,
                                   nCellsToCells};
    CellToCell::template launchZero<Acc1D>(device_cellToNeighborsView_, queue);

    // Cell -> Tracks.
    device_cellToTracks_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToTracksStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToTracks);
    // The offsets live in the tail of the cell arena when the constructor set it up (up-front
    // allocation mode); otherwise they get their own allocation. Same extent, same contents either way.
    GenericContainerOffsets *cellToTracksOffsets = arenaCellToTracksOffsets_;
    if (cellToTracksOffsets == nullptr) {
      device_cellToTracksOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nBins);
      cellToTracksOffsets = device_cellToTracksOffsets_->data();
    }
    device_cellToTracksView_ = {
        device_cellToTracks_->data(), cellToTracksOffsets, device_cellToTracksStorage_->data(), nBins, nCellsToTracks};
    CellToTrack::template launchZero<Acc1D>(device_cellToTracksView_, queue);

    // Triplet (cell->cell) and track-cell (cell->track) SoA edge lists.
    tripletsN_ = nCellsToCells;
    tracksCellsN_ = nCellsToTracks;
    deviceTriplets_ = CAPairSoACollection(queue, tripletsN_);
    deviceTracksCells_ = CAPairSoACollection(queue, tracksCellsN_);
#ifdef CA_TRIPLET_DUMP
    device_tripletDump_ = TripletDumpSoACollection(queue, tripletsN_);
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::allocateAfterNtuplets(uint32_t nHitsInTracks, Queue &queue) {
    // Hit -> track storage holds exactly nHitsInTracks entries, the sum over surviving tracks of their
    // hit multiplicity. With delayAllocations_ it is sized here from the real count; otherwise it was
    // allocated in the constructor, nHitsInTracks is 0 and this is a no-op.
    if (m_params.algoParams_.delayAllocations_) {
      const uint32_t nStorage = std::max(nHitsInTracks, 1u);
      const uint32_t nHits = static_cast<uint32_t>(alpaka::getExtents(*device_hitToTupleOffsets_)[0u]) - 1u;
      device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nStorage);
      device_hitToTupleView_ = {device_hitToTuple_->data(),
                                device_hitToTupleOffsets_->data(),
                                device_hitToTupleStorage_->data(),
                                nHits + 1u,
                                nStorage};
      HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);
    }

#ifdef GPU_DEBUG
    // Per-buffer device allocation report. Runs here, after the last (deferred)
    // allocation, so every buffer's real extent is available
    {
      const auto coutFlags = std::cout.flags();
      const auto coutPrec = std::cout.precision();

      std::size_t total = 0;
      // Array buffers report their real element count; scalar "handle" buffers (the
      // OneToMany container objects) report 1 element when allocated.
      auto extentOf = [](auto const &optBuf) -> std::size_t {
        return optBuf ? static_cast<std::size_t>(alpaka::getExtents(*optBuf)[0u]) : 0u;
      };
      auto oneIf = [](auto const &optBuf) -> std::size_t { return optBuf ? 1u : 0u; };
      auto report = [&total](const char *name, std::size_t n_elem, std::size_t elem_size) {
        const std::size_t bytes = n_elem * elem_size;
        total += bytes;
        std::cout << "  " << std::left << std::setw(46) << name << " : " << std::right << std::setw(10) << n_elem
                  << " x " << std::setw(4) << elem_size << " B = " << std::setw(12) << bytes << " B  (" << std::fixed
                  << std::setprecision(3) << std::setw(9) << (bytes / 1024.0) << " KiB)" << std::endl;
      };
      auto reportSoA = [&total](const char *name, std::size_t n_elem, std::size_t bytes) {
        total += bytes;
        std::cout << "  " << std::left << std::setw(46) << name << " : " << std::right << std::setw(10) << n_elem
                  << " elements,  SoA layout = " << std::setw(12) << bytes << " B  (" << std::fixed
                  << std::setprecision(3) << std::setw(9) << (bytes / 1024.0) << " KiB)" << std::endl;
      };

      const auto allocIterTag = ::pixelTrack::iterationName[static_cast<uint8_t>(m_params.algoParams_.iterationName_)];
      std::cout << "================== Device allocation report [" << allocIterTag
                << "] ==================" << std::endl;
      report("counters_", oneIf(counters_), sizeof(Counters));
      report("device_hitToTuple_", oneIf(device_hitToTuple_), sizeof(GenericContainer));
      report("device_hitToTupleStorage_", extentOf(device_hitToTupleStorage_), sizeof(GenericContainerStorage));
      report("device_hitToTupleOffsets_", extentOf(device_hitToTupleOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitToCell_", oneIf(device_hitToCell_), sizeof(GenericContainer));
      report("device_hitToCellStorage_", extentOf(device_hitToCellStorage_), sizeof(GenericContainerStorage));
      report("device_hitToCellOffsets_", extentOf(device_hitToCellOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitPhiHist_", oneIf(device_hitPhiHist_), sizeof(PhiBinner));
      report("device_phiBinnerStorage_", extentOf(device_phiBinnerStorage_), sizeof(PhiBinnerStorageType));
      report("device_layerStarts_", extentOf(device_layerStarts_), sizeof(hindex_type));
      report("device_qualityScratch_", extentOf(device_qualityScratch_), sizeof(int32_t));
      report("device_cellToNeighbors_", oneIf(device_cellToNeighbors_), sizeof(GenericContainer));
      report(
          "device_cellToNeighborsStorage_", extentOf(device_cellToNeighborsStorage_), sizeof(GenericContainerStorage));
      report(
          "device_cellToNeighborsOffsets_", extentOf(device_cellToNeighborsOffsets_), sizeof(GenericContainerOffsets));
      report("device_cellToTracks_", oneIf(device_cellToTracks_), sizeof(GenericContainer));
      report("device_cellToTracksStorage_", extentOf(device_cellToTracksStorage_), sizeof(GenericContainerStorage));
      report("device_cellToTracksOffsets_", extentOf(device_cellToTracksOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitContainer_", oneIf(device_hitContainer_), sizeof(SequentialContainer));
      report("device_hitContainerStorage_", extentOf(device_hitContainerStorage_), sizeof(SequentialContainerStorage));
      report("device_hitContainerOffsets_", extentOf(device_hitContainerOffsets_), sizeof(SequentialContainerOffsets));
      report("device_tupleMultiplicity_", oneIf(device_tupleMultiplicity_), sizeof(GenericContainer));
      report("device_tupleMultiplicityStorage_",
             extentOf(device_tupleMultiplicityStorage_),
             sizeof(GenericContainerStorage));
      report("device_tupleMultiplicityOffsets_",
             extentOf(device_tupleMultiplicityOffsets_),
             sizeof(GenericContainerOffsets));
      report("device_simpleCells_", extentOf(device_simpleCells_), sizeof(SimpleCell));
      report("device_extraStorage_",
             extentOf(device_extraStorage_),
             sizeof(cms::alpakatools::AtomicPairCounter::DoubleWord));
#ifdef CA_PIPELINE_COUNTERS
      report("device_pipelineCounters_", extentOf(device_pipelineCounters_), sizeof(uint32_t));
#endif

      const std::size_t tripletsBytes = CAPairSoACollection::Layout::computeDataSize(tripletsN_);
      const std::size_t tracksCellsBytes = CAPairSoACollection::Layout::computeDataSize(tracksCellsN_);
      reportSoA("deviceTriplets_ (SoA)", tripletsN_, tripletsBytes);
      reportSoA("deviceTracksCells_ (SoA)", tracksCellsN_, tracksCellsBytes);

      std::cout << "  --------------------------------------------------------------" << std::endl;
      std::cout << "  " << std::left << std::setw(46) << "THIS EVENT'S TOTAL"
                << " : " << std::right << std::setw(12) << total << " B  (" << std::fixed << std::setprecision(3)
                << std::setw(9) << (total / (1024.0 * 1024.0)) << " MiB)" << std::endl;

      uint64_t snapAllocN = 0;
      uint64_t snapAllocSum = 0;
      uint64_t snapAllocMax = 0;
      {
        auto &agg = sizingAccumulator(this->m_params.algoParams_.iterationName_);
        std::lock_guard<std::mutex> lock(agg.mtx);
        ++agg.allocN;
        agg.allocBytesSum += static_cast<uint64_t>(total);
        if (static_cast<uint64_t>(total) > agg.allocBytesMax)
          agg.allocBytesMax = static_cast<uint64_t>(total);
        snapAllocN = agg.allocN;
        snapAllocSum = agg.allocBytesSum;
        snapAllocMax = agg.allocBytesMax;
      }
      const double meanBytes = double(snapAllocSum) / double(snapAllocN);
      std::cout << "  " << std::left << std::setw(46) << "AGGREGATE (mean over events)"
                << " : " << std::right << std::setw(12) << static_cast<uint64_t>(meanBytes) << " B  (" << std::fixed
                << std::setprecision(3) << std::setw(9) << (meanBytes / (1024.0 * 1024.0)) << " MiB)  over "
                << snapAllocN << " event" << (snapAllocN == 1u ? "" : "s") << std::endl;
      std::cout << "  " << std::left << std::setw(46) << "AGGREGATE (max over events)"
                << " : " << std::right << std::setw(12) << snapAllocMax << " B  (" << std::fixed << std::setprecision(3)
                << std::setw(9) << (snapAllocMax / (1024.0 * 1024.0)) << " MiB)" << std::endl;
      std::cout << "==============================================================" << std::endl;

      std::cout.flags(coutFlags);
      std::cout.precision(coutPrec);
    }
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::prepareHits(const HitsMultiView &hh,
                                                                const ModulesMultiView &mm,
                                                                const reco::CALayersSoAConstView &ll,
                                                                Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, ll.metadata().size() - 1);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, SetHitsLayerStart<ModulesMultiView>{}, mm, ll, this->device_layerStarts_->data());

    auto accessor_iphi = [] ALPAKA_FN_ACC(auto const &v) { return v.iphi(); };

    cms::alpakatools::fillManyFromVector<Acc1D>(device_hitPhiHist_->data(),
                                                device_hitPhiView_,
                                                TrackerTraits::numberOfLayers,  // could be ll.metadata().size() - 1
                                                hh,
                                                accessor_iphi,
                                                this->device_layerStarts_->data(),
                                                static_cast<uint32_t>(hh.size()),
                                                static_cast<uint32_t>(256),
                                                queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "CAHitNtupletGeneratorKernels -> Hits prepared (layer starts and histo) -> DONE!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::launchKernels(const HitsMultiView &hh,
                                                                  uint32_t offsetBPIX2,
                                                                  uint16_t nLayers,
                                                                  TkSoABlocksView &view,
                                                                  const reco::CAGraphSoAConstView &cc,
                                                                  const reco::CATripletCutsSoAConstView &tripletCuts,
                                                                  const reco::CANtupletCutsSoAConstView &ntupletCuts,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto tracks_view = view.tracks();
    auto tracks_hits_view = view.trackHits();

    uint32_t nhits = static_cast<uint32_t>(hh.size());
    auto const launchCells = this->launchCells_;
    auto const maxTuples = tracks_view.metadata().size();
#ifdef NTUPLE_DEBUG
    std::cout << "start tuple building. N hits " << nhits << std::endl;
    if (nhits < 2)
      std::cout << "too few hits " << nhits << std::endl;
#endif

    //
    // applying combinatoric cleaning such as fishbone at this stage is too expensive
    //

    const auto nthTot = 64;
    const auto stride = 4;
    auto blockSize = nthTot / stride;
    // The cell and tuple kernels here and below launch over the full container capacity. Their loops
    // are bounded by the true counts, so coverage never relies on the grid-stride wrap-around, and
    // with the extent at least the capacity each index is visited by exactly one thread, which keeps
    // the serial backend's visit order, and so its bit-exact output, independent of the capacity.
    // blockSize grows in multiples of 16, keeping blockSize*stride <= 1024, until the grid fits
    // CUDA's 65535-block limit.
    auto numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    while (numberOfBlocks >= 65536 && blockSize * stride < 1024) {
      blockSize += 16;
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    }
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

#ifdef CA_TRIPLET_DUMP
    // Iteration label and selection for the dump lines: this module's pixelTrack::Iteration is stamped
    // on every line, and a non-empty tripletDumpIteration makes the other iterations skip the dump
    // (dumpIteration = -1).
    const std::string &dumpIterSel = this->m_params.tripletDumpIteration_;
    const auto thisIter = this->m_params.algoParams_.iterationName_;
    const int dumpIteration =
        (dumpIterSel.empty() || ::pixelTrack::iterationName[uint8_t(thisIter)] == std::string_view(dumpIterSel))
            ? int(uint8_t(thisIter))
            : -1;
#endif
    // Per-iteration compile-time weight bank for the in-kernel triplet DNN: the prompt iteration
    // (promptHighPt) uses the prompt bank, every other iteration the displaced (default) bank.
    // The CA plugin is one compiled kernel shared by both iterations, so this is a runtime arg.
    const DnnBank tripletBank = (this->m_params.algoParams_.iterationName_ == ::pixelTrack::Iteration::promptHighPt)
                                    ? DnnBank::kPrompt
                                    : DnnBank::kDisplaced;
    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        Kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_,  // needed only to be reset, ready for next kernel
                        hh,
                        cc,
                        tripletCuts,
                        this->m_params.algoParams_.useTripletDNN_,
                        this->m_params.algoParams_.tripletDNNThreshold_,
                        tripletBank,
#ifdef CA_TRIPLET_DUMP
                        this->device_tripletDump_->view(),
                        dumpIteration,
#endif
                        this->deviceTriplets_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_hitToCell_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->pipelineCountersPtr());

    finalizeAssocOffsets(this->device_cellToNeighborsView_, queue);

#ifdef CA_TRIPLET_DUMP
    // Stamp the valid-row count into the dump SoA scalar, the number of rows Kernel_connect wrote.
    // Device-to-device copy on the same queue; the host consumer reads view().nValid() to know how
    // many of the full-capacity rows are valid.
    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(queue, this->device_tripletDump_->view().nValid()),
                   cms::alpakatools::make_device_view(queue, *this->device_nTriplets_->data()));
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_connect -> Done!" << std::endl;
#endif

    auto threadsPerBlock = 1024;
    // Floor at one block: with useExactAllocations the launch cell count can be a handful, and
    // lrint(nCells * avgCellsPerCell) then rounds to 0, i.e. a zero-block launch that fills nothing.
    auto blocks = std::max<cms::alpakatools::Idx>(
        1u,
        cms::alpakatools::divide_up_by(std::lrint(launchCells * m_params.algoParams_.avgCellsPerCell_),
                                       threadsPerBlock));
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CAPairSoAConstView, GenericContainer>{},
                        this->deviceTriplets_->view(),
                        this->device_nTriplets_->data(),
                        this->device_cellToNeighbors_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "cellToNeighbors -> Filled!" << std::endl;
#endif

    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.earlyFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto fishboneWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
      alpaka::exec<Acc2D>(queue,
                          fishboneWorkDiv,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          ntupletCuts,
                          cc,
                          this->device_simpleCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          false,
                          this->pipelineCountersPtr(),
                          this->m_params.algoParams_.onlySameLayersFishbone_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Early fishbone -> Done!" << std::endl;
#endif
    }

#ifdef CA_PIPELINE_COUNTERS
    // Count cell status after the fishbone kill phase
    {
      auto cellBlocks = cms::alpakatools::divide_up_by(launchCells, 256u);
      auto cellWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(cellBlocks, 256u);
      alpaka::exec<Acc1D>(queue,
                          cellWorkDiv,
                          Kernel_pipelineCellStatus<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->pipelineCountersPtr());
    }
#endif

    blockSize = 64;
    numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_find_ntuplets<TrackerTraits>{},
                        hh,
                        cc,
                        ntupletCuts,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->device_cellToTracks_->data(),
                        this->deviceTracksCells_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCellTracks_->data(),
                        this->device_nTriplets_->data(),
                        this->device_nCells_->data(),
                        this->device_hitTuple_apc_,
                        this->m_params.algoParams_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_find_ntuplets -> Done!" << std::endl;
#endif

#ifdef CA_PIPELINE_COUNTERS
    // Copy *nCellTracks into the pipeline counter array
    {
      auto workDiv1x1 = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
      alpaka::exec<Acc1D>(queue,
                          workDiv1x1,
                          Kernel_pipelineCopyCellTrackCount{},
                          this->device_nCellTracks_->data(),
                          this->pipelineCountersPtr());
    }
#endif

    finalizeAssocOffsets(this->device_cellToTracksView_, queue);

    // This pass fills the cell->TRACK edge list, so its grid is sized from avgTracksPerCell_.
    // The loop inside is a grid-stride uniform_elements over the true edge count, so the block
    // count is a throughput choice only and can never truncate work.
    blocks = std::max<cms::alpakatools::Idx>(
        1u,
        cms::alpakatools::divide_up_by(std::lrint(launchCells * m_params.algoParams_.avgTracksPerCell_),
                                       threadsPerBlock));
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CAPairSoAConstView, GenericContainer>{},
                        this->deviceTracksCells_->view(),
                        this->device_nCellTracks_->data(),
                        this->device_cellToTracks_->data());

    if (this->m_params.algoParams_.doStats_)
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_mark_used<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_cellToTracks_->data(),
                          this->device_nCells_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples + 1, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        typename HitContainer::finalizeBulk{},
                        this->device_hitTuple_apc_,
                        this->device_hitContainer_->data());

    // Repair the CSR after a hit-container content overflow (see Kernel_findTupleContentOverflow):
    // overflowed tuples become empty instead of describing unwritten content and nTracks is cut
    // in front of them. The memset value 0xFFFFFFFF means "no clamp" and is what the no-overflow
    // case leaves in place.
    alpaka::memset(queue, *this->device_tupleClampBound_, 0xFF);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_findTupleContentOverflow{},
                        this->device_hitContainer_->data(),
                        this->device_tupleClampBound_->data());
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_clampTupleContentOverflow{},
                        this->device_hitContainer_->data(),
                        this->device_tupleClampBound_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline counter: classify n-tuplets by OT hit content.
    // Must run AFTER finalizeBulk so that foundNtuplets offsets are valid
    // for size()/begin()/end() iteration.
    {
      auto ntupBlocks = cms::alpakatools::divide_up_by(maxTuples, 128u);
      auto ntupWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(ntupBlocks, 128u);
      alpaka::exec<Acc1D>(queue,
                          ntupWorkDiv,
                          Kernel_pipelineNtupletCount<TrackerTraits>{},
                          hh,
                          this->device_hitContainer_->data(),
                          this->device_hitTuple_apc_,
                          maxTuples,
                          this->pipelineCountersPtr());
    }
#endif

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillHitDetIndices<TrackerTraits>{},
                        tracks_view,
                        tracks_hits_view,
                        this->device_hitContainer_->data(),
                        hh,
                        this->device_hitTuple_apc_,
                        this->device_tupleClampBound_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillHitDetIndices   -> done!" << std::endl;
#endif

    // remove duplicates (tracks that share a doublet).
    if (this->m_params.algoParams_.doEarlyDuplicateRemover_) {
      if constexpr (std::is_same_v<pixelTopology::Phase1, TrackerTraits>) {
        // for Phase-1, the workdivision is simpler and therefore needs a separate call here
        blockSize = 64;
        numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_earlyDuplicateRemoverPhase1{},
                            this->device_simpleCells_->data(),
                            this->device_nCells_->data(),
                            this->device_cellToTracks_->data(),
                            tracks_view,
                            this->m_params.algoParams_.dupPassThrough_);
      } else {
        // Warp-cooperative work division: one cell per Y-thread, one warp per cell.
        const uint32_t stride = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
        const uint32_t cellsPerBlock = 8;  // 8 cells * 32 threads = 256 threads/block
        // gridDim.y is capped at 65535 by CUDA. If maxDoublets needs more, uniform_elements_y
        // strides each Y-thread across the remaining cells.
        auto earlyDupRemoverNumBlocks = cms::alpakatools::divide_up_by(launchCells, cellsPerBlock);
        // Cap at Y-dim limit
        earlyDupRemoverNumBlocks = std::min<uint32_t>(earlyDupRemoverNumBlocks, 65535u);
        const Vec2D earlyDupRemoverBlocks{earlyDupRemoverNumBlocks, 1u};
        const Vec2D earlyDupRemoverThreads{cellsPerBlock, stride};
        const auto earlyDupRemoverWorkDiv =
            cms::alpakatools::make_workdiv<Acc2D>(earlyDupRemoverBlocks, earlyDupRemoverThreads);

        alpaka::exec<Acc2D>(queue,
                            earlyDupRemoverWorkDiv,
                            Kernel_earlyDuplicateRemover<TrackerTraits>{},
                            this->device_simpleCells_->data(),
                            this->device_nCells_->data(),
                            this->device_cellToTracks_->data(),
                            tracks_view,
                            this->m_params.algoParams_.dupPassThrough_);
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_earlyDuplicateRemover   -> done!" << std::endl;
#endif
    }

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_countMultiplicity<TrackerTraits>{},
                        hh,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_tupleMultiplicity_->data());
    finalizeAssocOffsets(this->device_tupleMultiplicityView_, queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_countMultiplicity   -> done!" << std::endl;
#endif

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillMultiplicity<TrackerTraits>{},
                        hh,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_tupleMultiplicity_->data());
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillMultiplicity -> done!" << std::endl;
#endif
    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.lateFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          ntupletCuts,
                          cc,
                          this->device_simpleCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          true,
                          this->pipelineCountersPtr(),
                          this->m_params.algoParams_.onlySameLayersFishbone_);
    }

#ifdef GPU_DEBUG
    std::cout << "lateFishbone -> done!" << std::endl;
    alpaka::wait(queue);
#endif

    // Every build-only buffer has now had its last reader enqueued. Handing them back here returns tens
    // of MiB to the caching allocator half an event early, so another stream can reuse the bin instead
    // of growing the pool.
    this->releaseBuildScratch();
  }

  // No new synchronisation: CachingAllocator::free is stream-ordered, it records an event on the queue
  // the block was allocated against and only re-issues the block once that event has completed, so the
  // kernels already enqueued keep valid pointers. For each buffer released here the last reader is a
  // kernel enqueued in launchKernels or earlier. The cell->tracks container is NOT released: the
  // duplicate remover reads it in classifyTuples. Under doStats_ the triplet and track-cell SoA stay
  // alive too, since Kernel_checkOverflows reads them on device.
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::releaseBuildScratch() {
#if defined(GPU_DEBUG) || defined(CA_STATS)
    // Diagnostic builds keep everything to the end of the event: their reports read these buffers'
    // real extents after launchKernels has returned.
    return;
#else
    device_hitToCellStorage_.reset();
    device_hitToCellOffsets_.reset();
    device_hitToCell_.reset();

    device_cellToNeighborsStorage_.reset();
    device_cellToNeighborsOffsets_.reset();
    device_cellToNeighbors_.reset();

    device_phiBinnerStorage_.reset();
    device_hitPhiHist_.reset();

    if (!m_params.algoParams_.doStats_) {
      deviceTriplets_.reset();
      deviceTracksCells_.reset();
    }
#endif
  }

  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(
      const HitsMultiView &hh,
      const ::reco::CAGraphSoAConstView &cc,
      const ::reco::CALayersSoAConstView &ll,
      const ::reco::CADoubletCutsSoAConstView &doubletCuts,
      uint32_t offsetBPIX2,
      const MapToHitConstView &maskPtr,
      Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto nhits = hh.size();
    const auto maxDoublets = this->maxNumberOfDoublets_;
#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (0 == nhits)
      return 0u;  // protect against empty events

    const int stride = 4;
    int threadsPerBlock = TrackerTraits::getDoubletsFromHistoMaxBlockSize / stride;
    int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
    const Vec2D blks{blocks, 1u};
    const Vec2D thrs{threadsPerBlock, stride};
    const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

#ifdef GPU_DEBUG
    std::cout << "nActualPairs = " << cc.metadata().size() << std::endl;
    std::cout << blocks << " - " << threadsPerBlock << " - " << stride << std::endl;
#endif
    // The exact doublet count, when countDoubletsFirst_ paid a readback for it. Returned to the
    // caller so allocateAfterDoublets can be sized from it without a second D2H of the same word.
    uint32_t nCellsCounted = 0u;

    if (this->m_params.algoParams_.countDoubletsFirst_) {
      // countDoubletsFirst_ -> do a dry-run of GetDoubletsFromHisto to learn the exact nCells
      // without writing cells or the hit->cell association, then size device_simpleCells_ and device_hitToCellStorage_
      // to that count before the real fill pass
      alpaka::memset(queue, *this->device_nCells_, 0);
      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          GetDoubletsFromHisto<TrackerTraits, /*CountOnly=*/true>{},
                          maxDoublets,
                          static_cast<SimpleCell *>(nullptr),
                          this->device_nCells_->data(),
                          hh,
                          cc,
                          ll,
                          doubletCuts,
                          this->device_layerStarts_->data(),
                          this->device_hitPhiHist_->data(),
                          this->device_hitToCell_->data(),
                          this->pipelineCountersPtr(),
                          maskPtr);

      const uint32_t nCellsFound = this->readbackNCells(queue);
      nCellsCounted = nCellsFound;
      const uint32_t nCellsStorage = std::max(nCellsFound, 1u);
      this->launchCells_ = nCellsStorage;
      const uint32_t outerHits = static_cast<uint32_t>(alpaka::getExtents(*this->device_hitToCellOffsets_)[0u]) - 1u;

      this->device_simpleCells_ = cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, nCellsStorage);
      this->device_hitToCellStorage_ =
          cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsStorage);
      this->device_hitToCellView_ = {this->device_hitToCell_->data(),
                                     this->device_hitToCellOffsets_->data(),
                                     this->device_hitToCellStorage_->data(),
                                     outerHits + 1u,
                                     nCellsStorage};
      HitToCell::template launchZero<Acc1D>(this->device_hitToCellView_, queue);

      alpaka::memset(queue, *this->device_nCells_, 0);  // reset for the fill pass
    }

    // Fill the cells and the hit->cell association
    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        GetDoubletsFromHisto<TrackerTraits>{},
                        maxDoublets,
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        hh,
                        cc,
                        ll,
                        doubletCuts,
                        this->device_layerStarts_->data(),
                        this->device_hitPhiHist_->data(),
                        this->device_hitToCell_->data(),
                        this->pipelineCountersPtr(),
                        maskPtr);

    finalizeAssocOffsets(this->device_hitToCellView_, queue);

    threadsPerBlock = 512;
    blocks = cms::alpakatools::divide_up_by(this->launchCells_, threadsPerBlock);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "GetDoubletsFromHisto   -> done!" << std::endl;
#endif

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        FillDoubletsHisto<TrackerTraits>{},
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        offsetBPIX2,
                        this->device_hitToCell_->data(),
                        this->counters_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "FillDoubletsHisto   -> done!" << std::endl;
#endif

    return nCellsCounted;
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples(const HitsMultiView &hh,
                                                                   TkSoAView &tracks_view,
                                                                   Queue &queue,
                                                                   ::reco::OTRecHitsConstView const *otView) {
    using namespace caHitNtupletGeneratorKernels;
    const pixelTrack::Iteration iterationName = this->m_params.algoParams_.iterationName_;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples" << std::endl;
#endif

    const uint32_t nhits = static_cast<uint32_t>(hh.size());

    auto blockSize = 64;
    const uint32_t launchCells = this->launchCells_;
    const uint32_t maxTuples = tracks_view.metadata().size();

    // Order/backend-independent duplicate removal via the int32 quality scratch (see the kernel headers)
    // snapshotQuality() freezes quality() into the scratch; applyQuality() copies it back. The fast
    // remover needs both; the track-parallel cleaners are single-writer and need only the snapshot
    auto qScratch = this->device_qualityScratch_->data();
    auto qScratchWorkDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(maxTuples, blockSize), blockSize);
    auto snapshotQuality = [&]() {
      alpaka::exec<Acc1D>(
          queue, qScratchWorkDiv, Kernel_snapshotQuality{}, tracks_view, this->device_hitContainer_->data(), qScratch);
    };
    auto applyQuality = [&]() {
      alpaka::exec<Acc1D>(
          queue, qScratchWorkDiv, Kernel_applyQuality{}, tracks_view, this->device_hitContainer_->data(), qScratch);
    };

    // classify tracks based on kinematics
    auto numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    // Per-iteration compile-time weight bank for the in-kernel loose->tight DNN (see Kernel_connect).
    const DnnBank trackBank =
        (iterationName == ::pixelTrack::Iteration::promptHighPt) ? DnnBank::kPrompt : DnnBank::kDisplaced;
    // OT-rechit view + count for the OT-aware feature walk. When no OT source was threaded
    // (merged-hits-only), pass an empty view + 0 so the classifier never touches it.
    const ::reco::OTRecHitsConstView otHitsView = (otView != nullptr) ? *otView : ::reco::OTRecHitsConstView{};
    const uint32_t nOTForFeat = (otView != nullptr) ? this->nOTHits_ : 0u;
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_classifyTracks<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_->data(),
                        hh,
                        this->m_params.qualityCuts_,
                        this->m_params.algoParams_.useTrackDNN_,
                        this->m_params.algoParams_.trackDNNThreshold_,
                        trackBank,
                        otHitsView,
                        nOTForFeat);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_classifyTracks -> done!" << std::endl;
#endif

    alpaka::exec<Acc1D>(
        queue, workDiv1D, Kernel_assignIteration{}, tracks_view, this->device_hitContainer_->data(), iterationName);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_assignIteration -> done!" << std::endl;
#endif

    if (this->m_params.algoParams_.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fishboneCleaner<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fishboneCleaner   -> done!" << std::endl;
#endif
    if (this->m_params.algoParams_.doFastDuplicateRemover_) {
      // mark duplicates (tracks that share a doublet)
      // Two-tier work division: one cell per thread, with the whole warp ganging up on the rare cells
      // whose track list is longer than kDupCoopMinTracks. The block size must stay a multiple of the
      // warp size: the kernel issues full-mask warp collectives and its grid-stride loop is
      // lane-aligned only if the grid stride is.
      blockSize = 64;
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      snapshotQuality();
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fastDuplicateRemover<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view,
                          qScratch,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->m_params.algoParams_.fastDupNSigma2_);
      applyQuality();
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_fastDuplicateRemover   -> done!" << std::endl;
#endif
    }
    if (this->m_params.algoParams_.doSharedHitCut_ || this->m_params.algoParams_.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data(),
                          nhits);  // tagged OT extras bin at nhits + otIdx

      finalizeAssocOffsets(this->device_hitToTupleView_, queue);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data(),
                          nhits,  // tagged OT extras bin at nhits + otIdx
                          this->counters_->data());

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_countHitInTracks   -> done!" << std::endl;
#endif
    }

    if (this->m_params.algoParams_.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      // The shared-hit removers are track-parallel and single-writer: snapshotQuality() freezes the
      // quality into the scratch, the kernel reads the scratch and writes only each track's own
      // quality directly (no atomics, no copy-back)
      snapshotQuality();
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitContainer_->data(),
                          qScratch,
                          this->device_hitToTuple_->data(),
                          this->m_params.algoParams_.fastDupNSigma2_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_rejectDuplicate   -> done!" << std::endl;
#endif
      // Only run if the cut is doing something
      if (this->m_params.algoParams_.minHitsForSharingCut_ > 1) {
        snapshotQuality();
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_sharedHitCleaner<TrackerTraits>{},
                            hh,
                            this->device_layerStarts_->data(),
                            tracks_view,
                            this->m_params.algoParams_.minHitsForSharingCut_,
                            this->m_params.algoParams_.dupPassThrough_,
                            this->device_hitContainer_->data(),
                            qScratch,
                            this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "Kernel_sharedHitCleaner   -> done!" << std::endl;
#endif
      }

      if (this->m_params.algoParams_.doTripletCleaner_ && (this->m_params.algoParams_.minHitsPerNtuplet_ <= 3)) {
        if (this->m_params.algoParams_.useSimpleTripletCleaner_) {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          snapshotQuality();
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_simpleTripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitContainer_->data(),
                              qScratch,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_simpleTripletCleaner   -> done!" << std::endl;
#endif
        } else {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          snapshotQuality();
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_tripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitContainer_->data(),
                              qScratch,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_tripletCleaner   -> done!" << std::endl;
#endif
        }
      }
    }

    if (this->m_params.algoParams_.doStats_) {
      numberOfBlocks = cms::alpakatools::divide_up_by(std::max(nhits, launchCells), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_checkOverflows<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_tupleMultiplicity_->data(),
                          this->device_hitToTuple_->data(),
                          this->device_hitTuple_apc_,
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_nTriplets_->data(),
                          this->device_nCellTracks_->data(),
                          this->deviceTriplets_->view(),
                          this->deviceTracksCells_->view(),
                          nhits,
                          this->maxNumberOfDoublets_,
                          this->m_params.algoParams_,
                          this->counters_->data());
    }

    // Always-on overflow sentinel: doStats-independent surface for the capacity guards (see the
    // kernel comment). The accumulator is per-stream persistent, owned by CAHitNtupletGenerator;
    // null when no owner armed it (non-CA users of this class).
    if (this->ovfAccum_ != nullptr) {
      auto sentinelDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue,
                          sentinelDiv,
                          Kernel_overflowSentinel{},
                          this->device_hitTuple_apc_,
                          this->device_nCells_->data(),
                          this->device_nTriplets_->data(),
                          this->device_nCellTracks_->data(),
                          uint32_t(tracks_view.metadata().size()),
                          this->maxNumberOfDoublets_,
                          // The two CA pair-SoA capacities, as host-side element counts: the same
                          // numbers the views' metadata().size() returned, and they survive the
                          // build-only buffers being released at the end of launchKernels.
                          this->tripletsN_,
                          this->tracksCellsN_,
                          // hitContainer content slots (allocated in the ctor, extent host-known).
                          // This is the binding per-track-hit bound: the output trackHits SoA is
                          // sized from the same expression, so it is never the smaller of the two.
                          uint32_t(alpaka::getExtents(*this->device_hitContainerStorage_)[0u]),
                          // hitToTuple content slots. Under delayed allocation the storage is sized to
                          // exactly the demand scalar this check compares against, so the comparison
                          // carries no information and 0xFFFFFFFF disables it.
                          (this->m_params.algoParams_.delayAllocations_ || !this->device_hitToTupleStorage_)
                              ? 0xFFFFFFFFu
                              : uint32_t(alpaka::getExtents(*this->device_hitToTupleStorage_)[0u]),
                          this->ovfAccum_);
    }

#ifdef CA_STATS
    alpaka::wait(queue);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_printSizes<HitsMultiView>{},
                        hh,
                        tracks_view,
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_nCellTracks_->data());

    alpaka::wait(queue);

    // Sizing recommendations. Each avg parameter sizes a buffer as cap = max_keys * avg, which is safe
    // iff actual_fill <= cap, i.e. avg >= actual_fill / max_keys. The fills are:
    //   avgHitsPerTrack  -> Sum_t nHits(t) (= APC.n),                max_keys = maxTuples
    //   avgCellsPerHit   -> nCells (one outer hit per cell),         max_keys = outerHits
    //   avgCellsPerCell  -> nTriplets (one neighbour link each),     max_keys = maxDoublets
    //   avgTracksPerCell -> nCellTrackPairs,                         max_keys = maxDoublets
    // The counters live in device_extraStorage_: [0,1] = AtomicPairCounter (m = nTracksFound,
    // n = nHitsInTracks), [2] = nCells, [3] = nTriplets, [4] = nCellTrackPairs.
    const auto iterTag = ::pixelTrack::iterationName[static_cast<uint8_t>(this->m_params.algoParams_.iterationName_)];
    std::cout << "========== CA Tracking Summary [" << iterTag << "] ==========" << std::endl;
    {
      using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
      auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
      alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
      alpaka::wait(queue);

      DW const *const extra = alpaka::getPtrNative(h_extra);

      // Decode the AtomicPairCounter that lives in extra[0]. The struct
      // packs (m, n) into a single 64-bit word. Extract apc_lo and apc_hi
      // as the two 32-bit halves and then use the smaller as the number of tracks
      // and the larger as the number of hits in tracks, since every track has at least one hit.
      const uint64_t apc_raw = static_cast<uint64_t>(extra[0]);
      const uint32_t apc_lo = static_cast<uint32_t>(apc_raw & 0xFFFFFFFFull);
      const uint32_t apc_hi = static_cast<uint32_t>(apc_raw >> 32);
      const uint32_t nTracksFnd = std::min(apc_lo, apc_hi);
      const uint32_t nHitsInTrk = std::max(apc_lo, apc_hi);
      const uint32_t nCellsF = static_cast<uint32_t>(extra[2]);
      const uint32_t nTripletsF = static_cast<uint32_t>(extra[3]);
      const uint32_t nCTPairsF = static_cast<uint32_t>(extra[4]);

      // outerHits is not a parameter of classifyTuples and is not stored
      // on the class. Recover it from device_hitToCellOffsets_
      const uint32_t outerHitsR = static_cast<uint32_t>(alpaka::getExtents(*this->device_hitToCellOffsets_)[0u]) - 1u;
      auto const &a = this->m_params.algoParams_;

      // Required avg: this event would not have overflowed if avg >= it. Each ratio must be referenced
      // to the same basis the buffer is allocated against, and for the two cell-keyed buffers that
      // basis is nCells or maxDoublets depending on the allocation mode. Scaling the current avg by
      // the buffer's occupancy, avg_req = avg_cur * nTriplets / tripletsN_, is correct in both.
      const float req_HpT = (maxTuples > 0) ? float(nHitsInTrk) / float(maxTuples) : 0.f;
      const float req_CpH = (outerHitsR > 0) ? float(nCellsF) / float(outerHitsR) : 0.f;
      const float req_CpC = (tripletsN_ > 0u) ? a.avgCellsPerCell_ * float(nTripletsF) / float(tripletsN_) : 0.f;
      const float req_TpC = (tracksCellsN_ > 0u) ? a.avgTracksPerCell_ * float(nCTPairsF) / float(tracksCellsN_) : 0.f;

      // Physical mean per *actual* key (informative, NOT the sizing bound):
      const float phy_HpT = (nTracksFnd > 0) ? float(nHitsInTrk) / float(nTracksFnd) : 0.f;
      const float phy_CpH = req_CpH;  // each cell has exactly one outer hit
      const float phy_CpC = (nCellsF > 0) ? float(nTripletsF) / float(nCellsF) : 0.f;
      const float phy_TpC = (nCellsF > 0) ? float(nCTPairsF) / float(nCellsF) : 0.f;

      // Safety factor to provide some headroom
      constexpr float kSafety = 1.25f;

      // Pack into arrays so per-event and aggregate code can share a loop.
      const char *paramNames[4] = {"avgHitsPerTrack", "avgCellsPerHit", "avgCellsPerCell", "avgTracksPerCell"};
      const float currents[4] = {a.avgHitsPerTrack_, a.avgCellsPerHit_, a.avgCellsPerCell_, a.avgTracksPerCell_};
      const float req[4] = {req_HpT, req_CpH, req_CpC, req_TpC};
      const float phy[4] = {phy_HpT, phy_CpH, phy_CpC, phy_TpC};

      // Parameters that scale with nHits
      //   ratio[0] = nCells  / nHits  -> maxNumberOfDoublets
      //   ratio[1] = nTuples / nHits  -> maxNumberOfTuples
      const char *scaleNames[2] = {"maxNumberOfDoublets", "maxNumberOfTuples"};
      const uint32_t scaleCurrent[2] = {this->maxNumberOfDoublets_, maxTuples};
      const uint32_t scaleCount[2] = {nCellsF, nTracksFnd};
      const double ratio[2] = {(nhits > 0) ? double(nCellsF) / double(nhits) : 0.,
                               (nhits > 0) ? double(nTracksFnd) / double(nhits) : 0.};

      // Update the cross-event accumulator and take a snapshot under the
      // lock; print outside the lock to keep the critical section small.
      uint64_t snapN = 0;
      double snapMaxReq[4] = {0., 0., 0., 0.};
      double snapMeanReq[4] = {0., 0., 0., 0.};
      double snapMaxPhy[4] = {0., 0., 0., 0.};
      double snapMaxRatio[2] = {0., 0.};
      double snapMeanRatio[2] = {0., 0.};
      uint32_t snapMaxNHits = 0u;
      {
        auto &agg = sizingAccumulator(this->m_params.algoParams_.iterationName_);
        std::lock_guard<std::mutex> lock(agg.mtx);
        ++agg.n;
        for (int i = 0; i < 4; ++i) {
          if (req[i] > agg.maxReq[i])
            agg.maxReq[i] = req[i];
          if (phy[i] > agg.maxPhy[i])
            agg.maxPhy[i] = phy[i];
          agg.sumReq[i] += req[i];
          agg.sumPhy[i] += phy[i];
        }
        for (int i = 0; i < 2; ++i) {
          if (ratio[i] > agg.maxRatio[i])
            agg.maxRatio[i] = ratio[i];
          agg.sumRatio[i] += ratio[i];
        }
        if (nhits > agg.maxNHits)
          agg.maxNHits = nhits;

        snapN = agg.n;
        const double invN = 1.0 / double(snapN);
        for (int i = 0; i < 4; ++i) {
          snapMaxReq[i] = agg.maxReq[i];
          snapMeanReq[i] = agg.sumReq[i] * invN;
          snapMaxPhy[i] = agg.maxPhy[i];
        }
        for (int i = 0; i < 2; ++i) {
          snapMaxRatio[i] = agg.maxRatio[i];
          snapMeanRatio[i] = agg.sumRatio[i] * invN;
        }
        snapMaxNHits = agg.maxNHits;
      }

      const auto coutFlags = std::cout.flags();
      const auto coutPrec = std::cout.precision();

      std::cout << "============= Sizing-parameter recommendation ["
                << ::pixelTrack::iterationName[static_cast<uint8_t>(this->m_params.algoParams_.iterationName_)]
                << "] =============" << std::endl;
      std::cout << "  Fills :  nHits=" << nhits << "  outerHits=" << outerHitsR << "  nTracksFound=" << nTracksFnd
                << "  nHitsInTracks=" << nHitsInTrk << std::endl;
      std::cout << "           nCells=" << nCellsF << "  nTriplets=" << nTripletsF << "  nCellTrackPairs=" << nCTPairsF
                << std::endl;
      std::cout << "  Caps  :  maxDoublets=" << this->maxNumberOfDoublets_ << "  maxTuples=" << maxTuples << std::endl;

      // Per-event rows.
      std::cout << "  -- this event --" << std::endl;
      for (int i = 0; i < 4; ++i) {
        std::cout << "    " << std::left << std::setw(18) << paramNames[i] << " current=" << std::fixed
                  << std::setprecision(3) << std::setw(8) << currents[i] << "  phys(fill/key)=" << std::setw(8)
                  << phy[i] << "  min(fill/cap)=" << std::setw(8) << req[i] << "  recommend>= " << std::setw(8)
                  << (req[i] * kSafety) << "  headroom=" << std::setw(6)
                  << ((req[i] > 0.f) ? (currents[i] / req[i]) : 0.f) << "x" << std::endl;
      }

      // Aggregated-across-events rows
      std::cout << "  -- aggregated over " << snapN << " event" << (snapN == 1u ? "" : "s") << std::endl;
      for (int i = 0; i < 4; ++i) {
        std::cout << "    " << std::left << std::setw(18) << paramNames[i] << " current=" << std::fixed
                  << std::setprecision(3) << std::setw(8) << currents[i] << "  max(req)=" << std::setw(8)
                  << snapMaxReq[i] << "  mean(req)=" << std::setw(8) << snapMeanReq[i] << "  max(phys)=" << std::setw(8)
                  << snapMaxPhy[i] << "  recommend>= " << std::setw(8) << (snapMaxReq[i] * kSafety) << std::endl;
      }

      // Parameters that scale with nHits (doublets and tuples).
      // Safe sizing is f(nHits) = slope * nHits, where slope = max(count/nHits) across events.
      // This guarantees the observed worst-case event would have fit while
      // safety covers unobserved variation. Evaluate f at this event's nHits
      // and at the observed max nHits
      std::cout << "  -- scaling parameters f(nHits) over " << snapN << " event" << (snapN == 1u ? "" : "s") << " --"
                << std::endl;
      for (int i = 0; i < 2; ++i) {
        const double recSlope = snapMaxRatio[i] * kSafety;
        const uint64_t recAtThis = static_cast<uint64_t>(std::ceil(recSlope * double(nhits)));
        const uint64_t recAtMax = static_cast<uint64_t>(std::ceil(recSlope * double(snapMaxNHits)));
        std::cout << "    " << std::left << std::setw(20) << scaleNames[i] << " current=" << std::right << std::setw(10)
                  << scaleCurrent[i] << "  this: " << std::setw(8) << scaleCount[i] << "/" << nhits << "=" << std::fixed
                  << std::setprecision(4) << std::setw(7) << ratio[i] << "  agg max=" << std::setw(7) << snapMaxRatio[i]
                  << "  mean=" << std::setw(7) << snapMeanRatio[i] << std::endl
                  << "    " << std::setw(20) << "" << " recommend>= " << std::setprecision(4) << recSlope << " * nHits"
                  << "  (= " << recAtThis << " here,  " << recAtMax << " at observed max nHits=" << snapMaxNHits << ")"
                  << std::endl;
      }

      std::cout << "  Note: 'min'/'req' = the avg value below which the buffer would overflow." << std::endl;
      std::cout << "        Per-event 'recommend' = req * " << kSafety << " (per-event safety margin)." << std::endl;
      std::cout << "        Aggregate 'recommend' = max(req across events) * " << kSafety << std::endl;
      std::cout << "        Run a representative event sample, then read the aggregate row" << std::endl;
      std::cout << "        from the last event's report." << std::endl;
      std::cout << "===========================================================" << std::endl;

      std::cout.flags(coutFlags);
      std::cout.precision(coutPrec);
    }
#endif
    if (this->m_params.algoParams_.doStats_) {
      // counters (add flag???)

      numberOfBlocks =
          cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForHitInTracks<TrackerTraits>{},
                          this->device_hitToTuple_->data(),
                          this->counters_->data());

      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->counters_->data());

      auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_printCounters{}, this->counters_->data());
    }
#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters: count final quality distribution, copy to host, and print funnel
    {
      // Count final track quality distribution after all processing
      auto numberOfBlocksQ = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      auto workDivQ = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksQ, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDivQ,
                          Kernel_countFinalQuality<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          hh,
                          device_pipelineCounters_->data());
      alpaka::wait(queue);
      auto host_counters = cms::alpakatools::make_host_buffer<uint32_t[]>(caHitNtupletGenerator::kTotalCounters);
      alpaka::memcpy(queue, host_counters, *device_pipelineCounters_);
      alpaka::wait(queue);
      auto const *c = host_counters.data();
      using PC = caHitNtupletGenerator::PipelineCounter;
      printf("[CA Pipeline] Doublets: total=%u pix-pix=%u pix-OT=%u OT-OT=%u\n",
             c[PC::kDoubletsTotal],
             c[PC::kDoubletsPixPix],
             c[PC::kDoubletsPixOT],
             c[PC::kDoubletsOTOT]);
      printf("[CA Pipeline]   OT barrel: L28-29=%u(FF=%u FT=%u TT=%u) L29-30=%u(FF=%u FT=%u TT=%u)\n",
             c[PC::kDoubletsL28L29],
             c[PC::kDoubletsL28L29_FF],
             c[PC::kDoubletsL28L29_FT],
             c[PC::kDoubletsL28L29_TT],
             c[PC::kDoubletsL29L30],
             c[PC::kDoubletsL29L30_FF],
             c[PC::kDoubletsL29L30_FT],
             c[PC::kDoubletsL29L30_TT]);
      printf(
          "[CA Pipeline]            L30-31=%u(FF=%u FT=%u TT=%u) L31-32=%u(FF=%u FT=%u TT=%u) L32-33=%u(FF=%u FT=%u "
          "TT=%u)\n",
          c[PC::kDoubletsL30L31],
          c[PC::kDoubletsL30L31_FF],
          c[PC::kDoubletsL30L31_FT],
          c[PC::kDoubletsL30L31_TT],
          c[PC::kDoubletsL31L32],
          c[PC::kDoubletsL31L32_FF],
          c[PC::kDoubletsL31L32_FT],
          c[PC::kDoubletsL31L32_TT],
          c[PC::kDoubletsL32L33],
          c[PC::kDoubletsL32L33_FF],
          c[PC::kDoubletsL32L33_FT],
          c[PC::kDoubletsL32L33_TT]);
      printf("[CA Pipeline]   OT brl->disk1(z<0): L28=%u L29=%u L30=%u L31=%u L32=%u L33=%u\n",
             c[PC::kDoubletsL28D1B],
             c[PC::kDoubletsL29D1B],
             c[PC::kDoubletsL30D1B],
             c[PC::kDoubletsL31D1B],
             c[PC::kDoubletsL32D1B],
             c[PC::kDoubletsL33D1B]);
      printf("[CA Pipeline]   OT brl->disk1(z>0): L28=%u L29=%u L30=%u L31=%u L32=%u L33=%u\n",
             c[PC::kDoubletsL28D1F],
             c[PC::kDoubletsL29D1F],
             c[PC::kDoubletsL30D1F],
             c[PC::kDoubletsL31D1F],
             c[PC::kDoubletsL32D1F],
             c[PC::kDoubletsL33D1F]);
      printf("[CA Pipeline]   OT disk chain (z<0): D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u\n",
             c[PC::kDoubletsD1BD2B],
             c[PC::kDoubletsD2BD3B],
             c[PC::kDoubletsD3BD4B],
             c[PC::kDoubletsD4BD5B]);
      printf("[CA Pipeline]   OT disk chain (z>0): D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u other=%u\n",
             c[PC::kDoubletsD1FD2F],
             c[PC::kDoubletsD2FD3F],
             c[PC::kDoubletsD3FD4F],
             c[PC::kDoubletsD4FD5F],
             c[PC::kDoubletsOTOther]);
      // Per-cut doublet rejection counters: Total, OTEarly (L28-29), OTLate (L30-32)
      {
        using namespace caHitNtupletGenerator;
        static const char *groupNames[] = {"Total", "OTEarly(L28-29)", "OTLate(L30-32)"};
        for (int g = 0; g < 3; ++g) {
          int base = PC::kDblRejBase + g * kNCuts;
          printf(
              "[CA Pipeline] DoubletCuts %s: invalidHit=%u innerCoord=%u clusterCut=%u invalidMod=%u "
              "outerCoord=%u dzRange=%u z0=%u phi=%u zSize=%u pt=%u stubSigma=%u pixStub=%u\n",
              groupNames[g],
              c[base + kCutInvalidHit],
              c[base + kCutInnerCoord],
              c[base + kCutClusterCut],
              c[base + kCutInvalidModule],
              c[base + kCutOuterCoord],
              c[base + kCutDzRange],
              c[base + kCutZ0],
              c[base + kCutPhi],
              c[base + kCutZSize],
              c[base + kCutPt],
              c[base + kCutStubSigma],
              c[base + kCutPixStub]);
        }
      }
      printf("[CA Pipeline] Triplets: total=%u ppp=%u ppO=%u pOO=%u OOO=%u\n",
             c[PC::kTripletsTotal],
             c[PC::kTripletsPixPixPix],
             c[PC::kTripletsPixPixOT],
             c[PC::kTripletsPixOTOT],
             c[PC::kTripletsOTOTOT]);
      printf("[CA Pipeline]   OOO: barrel=%u brl->z<0=%u brl->z>0=%u z<0=%u z>0=%u other=%u\n",
             c[PC::kTripletsOOO_barrel],
             c[PC::kTripletsOOO_brlToBwd],
             c[PC::kTripletsOOO_brlToFwd],
             c[PC::kTripletsOOO_bwd],
             c[PC::kTripletsOOO_fwd],
             c[PC::kTripletsOOO_other]);
      printf("[CA Pipeline]   phiMiddle rejected: %u\n", c[PC::kTripletPhiMiddleRej]);
      printf("[CA Pipeline]   chainPhiResid rejected (early): %u\n", c[PC::kTripletChainPhiResidRej]);
      // Per-cut triplet rejection breakdown (Phase2OTStubs): how many triplet
      // candidates each TripletCuts::accept() stage rejects, in application order.
      {
        using namespace caHitNtupletGenerator;
        const int trpBase = int(kTrpRejBase);
        printf(
            "[CA Pipeline] TripletCuts rejected: alignedRZ=%u alignedXY=%u beamspotDCA=%u phiCompat=%u "
            "sameSignDPhi=%u stubGeomCurv=%u stubInnerDCurv=%u\n",
            c[trpBase + kCutAlignedRZ],
            c[trpBase + kCutAlignedXY],
            c[trpBase + kCutBeamspotCompatibleXY],
            c[trpBase + kCutPhiCompatible],
            c[trpBase + kCutSameSignDPhi],
            c[trpBase + kCutStubsCurvCompatibleWithTriplet],
            c[trpBase + kCutStubsCompatibleWithInnerDoublet]);
      }
      printf("[CA Pipeline] Fishbone killed: %u\n", c[PC::kFishboneKilled]);
      printf("[CA Pipeline] Cell status: used_in_triplet=%u killed_total=%u alive=%u\n",
             c[PC::kCellsUsedInTriplet],
             c[PC::kCellsKilledTotal],
             c[PC::kCellsAlive]);
      printf("[CA Pipeline] Cell-track pairs: %u (avgTracksPerCell=%.3f)\n",
             c[PC::kCellTrackPairs],
             c[PC::kDoubletsTotal] > 0 ? float(c[PC::kCellTrackPairs]) / float(c[PC::kDoubletsTotal]) : 0.f);
      printf("[CA Pipeline] N-tuplets: total=%u with_OT=%u with_3+OT=%u\n",
             c[PC::kNtupletsTotal],
             c[PC::kNtupletsWithOT],
             c[PC::kNtupletsOT3Plus]);
      printf("[CA Pipeline] Quality: total=%u bad=%u edup=%u dup=%u loose=%u strict=%u tight=%u HP=%u\n",
             c[PC::kQualTotal],
             c[PC::kQualBad],
             c[PC::kQualEdup],
             c[PC::kQualDup],
             c[PC::kQualLoose],
             c[PC::kQualStrict],
             c[PC::kQualTight],
             c[PC::kQualHP]);
      printf("[CA Pipeline]   with OT: strict_OT=%u tight_OT=%u HP_OT=%u\n",
             c[PC::kQualStrictWithOT],
             c[PC::kQualTightWithOT],
             c[PC::kQualHPWithOT]);
      printf("[CA Pipeline]   nhits3-4: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict34],
             c[PC::kQualTight34],
             c[PC::kQualHP34],
             c[PC::kChi2Boundary34]);
      printf("[CA Pipeline]   nhits5: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict5],
             c[PC::kQualTight5],
             c[PC::kQualHP5],
             c[PC::kChi2Boundary5]);
      printf("[CA Pipeline]   nhits6+: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict6p],
             c[PC::kQualTight6p],
             c[PC::kQualHP6p],
             c[PC::kChi2Boundary6p]);
      printf("[CA Pipeline]   fishbone: 0fb=%u 1fb=%u 2+fb=%u\n",
             c[PC::kTracksFishbone0],
             c[PC::kTracksFishbone1],
             c[PC::kTracksFishbone2p]);
      // Reset counters for next event
      alpaka::memset(queue, *device_pipelineCounters_, 0);
    }
#endif  // CA_PIPELINE_COUNTERS

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    static std::mutex lock;
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1u, 32u);
    {
      std::lock_guard<std::mutex> guard(lock);
      ++iev;
      for (uint32_t k = 0; k < 20000; k += 500) {
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_print_found_ntuplets<TrackerTraits>{},
                            hh,
                            tracks_view,
                            this->device_hitContainer_->data(),
                            this->device_hitToTuple_->data(),
                            k,
                            k + 500,
                            iev);
        alpaka::wait(queue);
      }
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_print_found_ntuplets<TrackerTraits>{},
                          hh,
                          tracks_view,
                          this->device_hitToTuple_->data(),
                          20000,
                          1000000,
                          iev);

      alpaka::wait(queue);
    }
#endif
  }

  /* This will make sense when we will be able to run this once per job in Alpaka

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::printCounters() {
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1,1);
    alpaka::exec<Acc1D>(queue_, workDiv1D, Kernel_printCounters{}, this->counters_->data());
  }
  */

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
