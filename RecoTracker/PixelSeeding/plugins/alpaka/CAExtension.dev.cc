// OT hit attach, the CKF seed-region-rebuild equivalent. The walk runs once per event, in the
// merger: launchMergerAttach is called from PixelTracksSoAMerger::produce over the concatenated,
// HP-selected and twin-merged collection, before the merger's final GBL refit; the CA iterations
// themselves never attach. A pre-gate and compaction stage selects which merged tracks enter the
// search, each accepted extra records its gate chi2 so that shared attached hits are arbitrated
// across tracks by an atomic best-chi2 claim per hit, and the layer geometry comes from the
// producer's run cache.

#include <cassert>
#include <cstdint>
#include <cstdio>  // host-side printf of the compile-gated diagnostic and sizing dumps
#include <cstring>
#include <cmath>
#include <atomic>
#include <optional>  // candidate dump: transient device buffers held only in an instrumented build
#include <sstream>   // candidate dump: chunked host-side LogInfo record formatting
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"  // candidate dump: host readback LogInfo

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"  // per-segment B_bend along the road

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

#include "CAExtensionKernels.h"
#include "CAFitHitSelection.h"
#include "ExtDerivedTables.h"  // the analytic chi2 quantiles of the gate
#include "ExtenderHelixHelpers.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension {

  // Defining EXT_CAND_DUMP compiles in the per-(candidate, visited layer) trace of
  // Kernel_extFindExtras: in-window candidate counts per source round, every scored candidate's id
  // and gate chi2, the committed winner and the layer outcome, read back on the host as fixed-format
  // LogInfo lines whose hit ids join to an all-hit truth table. Diagnostic only: it writes nothing
  // any track reads, and undefined every dump branch folds away with no buffer allocated.
#ifdef EXT_CAND_DUMP
  inline constexpr bool kExtCandDump = true;
#else
  inline constexpr bool kExtCandDump = false;
#endif

  using Quality = ::pixelTrack::Quality;

  // Kernel_extFindExtras runs one block per candidate with this many cooperating lanes (threads on
  // GPU, elements-per-thread on the serial backend). It sets both the launch block size and the
  // in-kernel per-lane shared arrays / hit-partition modulus, so the two must agree -- keep this the
  // single source of truth.
  constexpr uint32_t kExtFindLanes = 128u;
  // The lane partition sets only which lane scores a hit; the (chi2, hitId) argmin is
  // partition-independent, so the count is free to change within two limits, asserted below: it must
  // be a power of two, since every reduce halves the stride and would otherwise drop lanes, and at
  // most 256, since the per-lane shared arrays are sized by it and the serial backend's block shared
  // arena is 47 KiB (roughly a quarter of it in use at 128 lanes).
  static_assert(kExtFindLanes > 0u && (kExtFindLanes & (kExtFindLanes - 1u)) == 0u,
                "kExtFindLanes must be a power of two: the shared-memory reduces halve the stride");
  static_assert(kExtFindLanes <= 256u, "kExtFindLanes > 256 overflows the serial backend's 47 KiB block shared arena");

  // Highland multiple-scattering angle variance per unit X/X0, for a pion of total momentum pTot
  // crossing radLen radiation lengths. The only MS coefficient in the walk: one function, full
  // momentum, pion 1/beta, no momentum cap -- the same form the fast BL builds, so road and fit
  // charge scattering the same way.
  //   c = (13.6 MeV / (p beta))^2 (1 + 0.038 ln W)^2      [rad^2 per unit X/X0]
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float extHighlandC(TAcc const& acc, float pTot, float radLen) {
    constexpr float kMeVToGeV = 13.6e-3f;
    constexpr float kMPi2 = 0.13957f * 0.13957f;
    if (!(radLen > 1e-9f) || !(pTot > 1e-6f))
      return 0.f;
    const float p2 = pTot * pTot;
    const float p2beta2 = p2 * p2 / (p2 + kMPi2);  // (p beta)^2 = p^4/(p^2 + m_pi^2)
    const float lg = 1.f + 0.038f * alpaka::math::log(acc, radLen);
    return kMeVToGeV * kMeVToGeV / p2beta2 * lg * lg;
  }

  // The walk's gate statistic. S is the innovation covariance, packed symmetric n x n with n <= 3 in
  // the order (00, 01, 02, 11, 12, 22); d the residual (pred - meas). Returns the chi2 and, in det,
  // |S| -- which the hole hypothesis prices the candidate's window volume with. Returns -1 when S is
  // not positive definite.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float extChi2FromS(
      TAcc const& acc, int n, const float S[6], const float d[3], float& det) {
    if (n == 1) {
      det = S[0];
      if (!(det > 0.f) || !alpaka::math::isfinite(acc, det))
        return -1.f;
      return d[0] * d[0] / det;
    }
    if (n == 2) {
      det = S[0] * S[3] - S[1] * S[1];
      if (!(det > 0.f) || !alpaka::math::isfinite(acc, det))
        return -1.f;
      return (S[3] * d[0] * d[0] - 2.f * S[1] * d[0] * d[1] + S[0] * d[1] * d[1]) / det;
    }
    const float a0 = S[3] * S[5] - S[4] * S[4];
    const float a1 = S[4] * S[2] - S[1] * S[5];
    const float a2 = S[1] * S[4] - S[3] * S[2];
    det = S[0] * a0 + S[1] * a1 + S[2] * a2;
    if (!(det > 0.f) || !alpaka::math::isfinite(acc, det))
      return -1.f;
    const float b1 = S[0] * S[5] - S[2] * S[2];
    const float b2 = S[1] * S[2] - S[0] * S[4];
    const float c2 = S[0] * S[3] - S[1] * S[1];
    const float q = a0 * d[0] * d[0] + b1 * d[1] * d[1] + c2 * d[2] * d[2] +
                    2.f * (a1 * d[0] * d[1] + a2 * d[0] * d[2] + b2 * d[1] * d[2]);
    return q / det;
  }

  // Tail probability of a chi2 with n dof, 1 - F_n(x): the quantity competing candidates are ranked
  // by, so a 2-dof (position-only) and a 3-dof (position + stub bend) candidate compete fairly
  // instead of the 2-dof one winning the argmin by its missing row. Closed form for n = 1, 2, 3.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float extChi2Tail(TAcc const& acc, int n, float x) {
    if (!(x > 0.f))
      return 1.f;
    const float e = alpaka::math::exp(acc, -0.5f * x);
    if (n == 2)
      return e;
    const float sx = alpaka::math::sqrt(acc, x);
    const float erfc = 1.f - alpaka::math::erf(acc, sx * 0.70710678f);  // erfc(sqrt(x/2))
    if (n == 1)
      return erfc;
    constexpr float kSqrt2OverPi = 0.79788456f;  // sqrt(2/pi)
    return erfc + kSqrt2OverPi * sx * e;
  }

  // How far the crossing may sit from a hit along the secondary row: a strip (and a pixel along its length)
  // reports the centre of a segment the track crossed anywhere inside, so the reading is uniform over the
  // segment and its support is exactly +-sqrt(3 V). On a 2S module the strip is centimetres long, and the
  // Gaussian tail of the same variance would admit hits the sensor cannot have produced.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float extSecSupport(TAcc const& acc, float Rss, float predVar, float qGate1) {
    return alpaka::math::sqrt(acc, 3.f * Rss) + alpaka::math::sqrt(acc, qGate1 * alpaka::math::max(acc, predVar, 0.f));
  }

  // Does the secondary reading still separate candidates, or is it only a window? The reading is uniform over
  // the strip with support +-sqrt(3 Rss); inside that support the likelihood is flat, and a chi2 row with a
  // free residual would let a 2S strip outrank an honest candidate. The row stays a chi2 row while 3 Rss is
  // below the road's own spread qGate1 * M_ss, and becomes the window extSecSupport otherwise (never on a PS
  // module or a pixel).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool extSecIsWindow(float Rss, float predVar, float qGate1) {
    return 3.f * Rss > qGate1 * predVar;
  }

  // Maps a hit's CA module index to its extender layer via binary search over layerStarts.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int hitLayer(uint32_t moduleIdx, ::reco_extender::ExtenderLayersConstView layers) {
    const int nLayers = layers.metadata().size() - 1;
    int lo = 0;
    int hi = nLayers;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (layers.layerStarts()[mid + 1] <= moduleIdx)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }

  // Pack (gate chi2, tuple id) into one word so an atomicMin resolves the best claimant per hit:
  // for non-negative floats the IEEE bit pattern is order-preserving, so the smaller chi2 wins and
  // the tuple id in the low half breaks exact ties deterministically.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint64_t packClaim(float chi2, uint32_t tupleId) {
    const float nonNeg = chi2 < 0.f ? 0.f : chi2;
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(nonNeg));
    memcpy(&bits, &nonNeg, sizeof(bits));  // IEEE bit pattern is order-preserving for non-negative floats
    return (uint64_t(bits) << 32) | uint64_t(tupleId);
  }

  // Inverse of packClaim's chi2 half: recover the gate chi2 from a packed claim's high word (the claimant
  // tuple id lives in the low 32 bits). Only meaningful for a real claim (caller must exclude the
  // kUnclaimed sentinel 0xff..ff, whose high word is a NaN bit pattern). Used by the ambiguity gate
  // to compare the top-2 claimants' gate chi2 on a contested hit.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float unpackClaimChi2(uint64_t claim) {
    const uint32_t bits = uint32_t(claim >> 32);
    float chi2;
    static_assert(sizeof(chi2) == sizeof(bits));
    memcpy(&chi2, &bits, sizeof(chi2));
    return chi2;
  }

  // The walk's pre-gate predicate, as a per-tuple mask for the prediction pass: exactly the quality,
  // finiteness, pT and |eta| tests Kernel_extPreGate applies, so the two sets agree.
  struct Kernel_extHostMask {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const float preGateMinPt,
                                  const float maxAbsEta,
                                  const uint32_t nTracksCap,
                                  int32_t* __restrict__ hostMask,
                                  // Optional smoothed-prediction payload. This sweep visits every slot of
                                  // [0, nTracksCap), so it also clears the payload's `valid` flag there; only
                                  // `valid` is consulted before the payload is written (every reader tests
                                  // pc.valid > 0.5f first), so clearing that one float is sufficient.
                                  ExtPredCoeff* __restrict__ pred = nullptr) const {
      const uint32_t nT = alpaka::math::min(acc, nTracksCap, uint32_t(alpaka::math::max(acc, 0, tracks.nTracks())));
      const float cotMax = alpaka::math::sinh(acc, maxAbsEta);
      for (auto i : cms::alpakatools::uniform_elements(acc, nTracksCap)) {
        hostMask[i] = -1;
        if (pred != nullptr)
          pred[i].valid = 0.f;
        if (i >= nT)
          continue;
        if (tracks[i].quality() == ::pixelTrack::Quality::edup)
          continue;
        bool finite = alpaka::math::isfinite(acc, tracks[i].chi2());
        for (int a = 0; a < 5; ++a)
          finite = finite && alpaka::math::isfinite(acc, tracks[i].state()(a));
        if (!finite)
          continue;
        if (!(tracks[i].pt() >= preGateMinPt))
          continue;
        if (!(alpaka::math::abs(acc, tracks[i].state()(3)) <= cotMax))
          continue;
        hostMask[i] = 0;
      }
    }
  };

  // Pre-gate and compaction: select the tuples that enter the attach search. Never-fitted tuples are
  // identified by their zeroed passBuf row (the fitted-circle radius cannot be exactly zero). The
  // surviving tuple ids are compacted into candList so the extras buffers scale with the candidate
  // cap rather than the tuple capacity. The same predicate and atomic compaction run twice: a count
  // pass (candList == nullptr) computes nCands and the candidate stats on the device, then a fill
  // pass writes candList without re-counting the stats. The compaction is atomic in both passes, so
  // the slot order is run-to-run arbitrary; the per-hit arbitration downstream is order-independent.
  struct Kernel_extPreGate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const double* __restrict__ passBuf,
                                  uint32_t maxNumberOfTuples,
                                  float preGateMinPt,
                                  float maxAbsCotTheta,
                                  // nullptr => no candidate restriction; else a tuple enters
                                  // only if acceptedMask[tuple] >= 0 (a previous pass's extended set).
                                  const int32_t* __restrict__ acceptedMask,
                                  uint32_t cap,
                                  uint32_t* __restrict__ candList,  // nullptr => count-only pass
                                  uint32_t* __restrict__ nCands,
                                  uint32_t* __restrict__ stats) const {
      const bool countOnly = (candList == nullptr);
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      for (auto i : cms::alpakatools::uniform_elements(acc, nT)) {
        if (tracks[i].quality() == Quality::edup)
          continue;
        // never fitted (see header comment). passBuf == nullptr means the caller guarantees every row
        // is fitted, so there is no never-fitted row to skip.
        if (passBuf != nullptr && passBuf[kPassBufStride * i + 2] == 0.)
          continue;
        bool finite = alpaka::math::isfinite(acc, tracks[i].chi2());
        for (int a = 0; a < 5; ++a)
          finite = finite && alpaka::math::isfinite(acc, tracks[i].state()(a));
        if (!finite)
          continue;
        if (tracks[i].pt() < preGateMinPt)
          continue;
        // The |eta| pre-gate (|cotTheta| > sinh(maxAbsEta)). Counted, because nothing else in stats[]
        // sees the population it removes.
        if (alpaka::math::abs(acc, tracks[i].state()(3)) > maxAbsCotTheta) {
          if (countOnly)
            alpaka::atomicAdd(acc, &stats[kStatPreGateEtaSkipped], 1u, alpaka::hierarchy::Grids{});
          continue;
        }
        // Candidate restriction: only the caller-supplied set. A follow-on attach pass passes the
        // previous pass's acceptedByTuple (>=0 iff the tuple got an accepted extension). Null => the
        // predicate is never evaluated and every pre-gate survivor enters.
        if (acceptedMask != nullptr && acceptedMask[i] < 0)
          continue;
        const uint32_t pos = alpaka::atomicAdd(acc, nCands, 1u, alpaka::hierarchy::Grids{});
        if (pos >= cap) {
          // Counted in BOTH passes: the fill pass is the one whose capacity can bind, and a silent
          // drop there costs a run-dependent slice of the extension. The launcher surfaces it.
          alpaka::atomicAdd(acc, &stats[kStatCandOverflow], 1u, alpaka::hierarchy::Grids{});
          continue;
        }
        if (countOnly)
          alpaka::atomicAdd(acc, &stats[kStatCandidates], 1u, alpaka::hierarchy::Grids{});
        else
          candList[pos] = i;
      }
    }
  };

  // Per-candidate layer walk and windowed hit search: one block per candidate (candList slot j ->
  // tuple id), kExtFindLanes lanes. The lanes share two argmin reductions: the walk order (each lane
  // scores the uncovered layers with a geometric nearest-approach proxy; the walk picks the min
  // (proxy distance, L) reachable layer, unreachable layers not counting toward maxWalkLayers) and
  // the per-layer hit scan (in-window hits partitioned round-robin, local best by min chi2 then min
  // hitId, lane-0 reduction). Both are argmins over a strict total order, so the result does not
  // depend on the lane partition. Lane 0 owns the sequential state; on the serial backend the lanes
  // are elements of one thread, so serial validation exercises the same logic.
  struct Kernel_extFindExtras {
    const float* rhoMap_ = nullptr;
    const ExtPhiBinner* phiBinner_ = nullptr;
    // Raw OT-rechit source, held by value (the kernel object is copied to the device; a host
    // pointer could not be dereferenced there, but the source's views/device-pointers copy fine).
    // otSource_.nOTHits == 0 => merged-hits-only walk. When populated, the per-layer hit scan runs a
    // second bin-loop over the OT phi binner after the merged one.
    OTHitsSource otSource_{};
    // Endcap gate-variance decomposition (runtime-gated on params.verbose): accumulate the permil
    // split of sigSec2 (hit/MS/predSec/align) per considered disk hit into the stats buffer. false
    // skips the global atomics, so the hot loop pays nothing.
    bool secFracDiag_ = false;
    // Candidate-level dump (record structs in the header). With candDump_ false the buffers below are
    // null and no accumulator or record write executes. When true, lane 0 writes one ExtCandLayerRec
    // per (candidate, visited layer) into candLayerBuf_, and candDumpOvf_[0] count-and-clamps a
    // visit-index overrun. candHdrBuf_ is allocated and sentinel-filled by the launcher but the walk
    // writes no header record, so the host readback emits no H lines.
    bool candDump_ = false;
    ExtCandLayerRec* candLayerBuf_ = nullptr;  // [maxCandidates * maxWalkLayers]
    ExtCandHdrRec* candHdrBuf_ = nullptr;      // [maxCandidates]
    uint32_t* candDumpOvf_ = nullptr;          // [2] count-and-clamp overflow guard: [0]=visit-index, [1]=member
    // Per-road-candidate member dump. nullptr => off (the candDump_ path is off too). When set, each
    // scored road candidate (both rounds) writes one ExtCandMemberRec into
    // candMemberBuf_[(j*maxWalkLayers + vi)*kExtDumpMaxMembers + m], m from the block shared counter
    // shDumpMemN (reset at layer select); m >= kExtDumpMaxMembers bumps candDumpOvf_[1].
    ExtCandMemberRec* candMemberBuf_ = nullptr;  // [maxCandidates * maxWalkLayers * kExtDumpMaxMembers]

    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const int maxExtraHitsPerTrack,  // per-track extra-slot budget (a compute cap)
        const int maxWalkLayers,         // compile-time sizing of the dump strides
        const int extMaxWalkLayers,      // runtime visit budget K (a compute cap)
        const float bf,                  // Bz(0,0): the scale of the normalised (Bz,Br) map
        const float maxAbsCotTheta,      // sinh(extMaxAbsEta): the |eta| reach of the walk
        const float extGateEps,          // THE efficiency: gate, window, rank and hole prior
        const float qGate1,              // chi2 quantiles of that eps at 1, 2 and 3 dof
        const float qGate2,
        const float qGate3,
        const float* __restrict__ bMap,            // normalised (Bz,Br) r-z field lattice
        const ExtPredCoeff* __restrict__ extPred,  // per-host anchor + exit kink + eloss centre
        const float* __restrict__ extEtaL,         // [kExtOTLayers] measured per-layer stub availability
        const float* __restrict__ extRho,          // [kExtOTLayers] measured per-layer stub areal density [cm^-2]
        const float* __restrict__ extEtaLRaw,      // [kExtOTLayers] raw round conditional availability (null=off)
        const float* __restrict__ extRho3,         // [kExtOTLayers] measured 3-dof stub density [cm^-1 rad^-1]
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView trackHits,
        const caStructures::CAHitsView hits,
        const ::reco::TrackingRecHitsMaskingConstView hitMask,
        const ::reco_extender::ExtenderLayersConstView caLayers,
        const ::reco::CAModulesConstView caModules,
        const uint32_t* __restrict__ candList,
        const uint32_t* __restrict__ nCands,
        const uint32_t maxCandidates,
        uint32_t* __restrict__ extrasIds,
        float* __restrict__ extrasChi2,
        int32_t* __restrict__ nExtras,
        uint32_t* __restrict__ stats) const {  // per-event counter buffer (kStat*/kDiag* indices, CAExtensionKernels.h)
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      const int nLayers = caLayers.metadata().size() - 1;
      // An empty hit mask view (metadata().size() == 0) means "all open": there is no masked hit to
      // skip, and the per-hit mask read is bypassed. A non-empty view is honoured per hit.
      const bool hitMaskArmed = (hitMask.metadata().size() != 0);
      // Walk-state shared arrays are compile-bounded; reads/writes are clamped to the caps so a runtime
      // K/slot budget larger than the compile-time one cannot overrun them.
      constexpr int kChainMaxVisits = 8;  // >= maxWalkLayers (the K visit budget)
      // The runtime walk budget (loop bound) is the requested extMaxWalkLayers clamped to
      // [1, kChainMaxVisits] -- the shared chain/hole arrays are sized at the compile-time
      // kChainMaxVisits, so vi = shWalkSteps-1 must stay < kChainMaxVisits. Buffer sizing/strides keep
      // the compile-time maxWalkLayers below; only the walk loop bounds read extWalkBudget. The
      // static_assert keeps headroom over the shipped default budget.
      static_assert(kChainMaxVisits >= 6, "kChainMaxVisits must cover the shipped maxWalkLayers=6 walk budget");
      const int extWalkBudget =
          (extMaxWalkLayers < 1) ? 1 : ((extMaxWalkLayers > kChainMaxVisits) ? kChainMaxVisits : extMaxWalkLayers);
      constexpr int kMaxOrigHits = 32;
      // Shared bound (reco::kMaxCALayers): the producers of the layers block throw above it, so the
      // walk's shared per-layer arrays and the uint64_t coveredMask can be sized from it here.
      constexpr int kMaxLayersList = ::reco::kMaxCALayers;
      // Per-(candidate,layer) linearization (see the coefficient block below): fall back to the exact
      // per-hit predict when the 2nd-order Taylor term 0.5*|d2phi/dr2|*W^2 (W = the layer half-extent)
      // exceeds this, or when the coefficient solve is near-tangential / has a vanishing denominator.
      constexpr float kTaylor2ndMaxRad = 1.0e-3f;
      constexpr float kLinCondEps = 1.0e-4f;
      const uint32_t nLanes = kExtFindLanes;  // lanes cooperating on one candidate (== launch block size)

      // shared per-block state (reused across grid-stride candidates)
      auto& sh = alpaka::declareSharedVar<RunningHelix, __COUNTER__>(acc);
      auto& shCoveredMask = alpaka::declareSharedVar<uint64_t, __COUNTER__>(acc);
      auto& shNOrigSafe = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shOrigR = alpaka::declareSharedVar<float[kMaxOrigHits], __COUNTER__>(acc);
      auto& shOrigZ = alpaka::declareSharedVar<float[kMaxOrigHits], __COUNTER__>(acc);
      auto& shReachable = alpaka::declareSharedVar<int[kMaxLayersList], __COUNTER__>(acc);
      auto& shReachDist = alpaka::declareSharedVar<float[kMaxLayersList], __COUNTER__>(acc);
      auto& shVisited = alpaka::declareSharedVar<int[kMaxLayersList], __COUNTER__>(acc);
      auto& shNExtra = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // Extras budget in CLUSTERS. The validation matches a track to a particle when at least 75 % of
      // its hits belong to it, so a core of n_core clusters keeps its match only while the extras stay
      // under n_core/4 of the total, i.e. n_extra <= floor((n_core - 1)/3). Not a tuning knob: it is
      // the same matching definition the duplicate removal reads. shNExtraClusters is seeded with the
      // extras an earlier attach pass already appended, so the bound holds over passes.
      auto& shExtraClusterCap = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shNExtraClusters = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shLastArcS = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shLastR = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shLastZ = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      // Per-host payload from the merger-side pre-attach pass, and the per-visit road built from it.
      // shDerQgap is the last FITTED gap's exit-direction kink variance: structurally invisible to the
      // fit (varBeta(n-1) == 0), so the walk carries it until the first accept folds it into P.
      auto& shDerQgap = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // last fitted gap kink var; 0 after
      // Deterministic ionization energy-loss state: the filter above gives the road its width, these
      // four give it its centre. Seeded from the same per-host payload at the same anchor (node n-1),
      // propagated by a deterministic second-order recursion and re-anchored at every
      // accept. The Kalman update does not reset them: the running helix estimates the
      // constant-kappa_0 reference trajectory, so the true track's offset from it keeps accumulating.
      // All four zero leaves every expression below adding an exact float 0.
      auto& shEU = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // u at the anchor [cm]
      auto& shEUp = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // du/ds at the anchor [rad]
      auto& shEDk = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // dkappa at the anchor [1/cm]
      auto& shEK = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // dkappa per column [1/(cm X0)]
      // Per-layer-visit derived road (written by lane 0 with the linearization, read by every lane).
      auto& shDerLayOn = alpaka::declareSharedVar<int, __COUNTER__>(acc);    // 1 = derived gate active on this layer
      auto& shDerWinR = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // r-phi window half-width [cm]
      auto& shDerCeilR = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // r-phi runaway ceiling [cm], envelope
      auto& shDerCeilS = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // secondary runaway ceiling [cm]
      // 1 / (the clusters this layer would put in that patch if it were uniform in this event):
      // the local-occupancy contrast is the patch count times this.
      auto& shOccNorm = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDerHoleK = alpaka::declareSharedVar<float, __COUNTER__>(acc);     // 2 dof, stub round
      auto& shDerHoleKRaw = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // 2 dof, raw-OT round
      auto& shDerHoleK3 = alpaka::declareSharedVar<float, __COUNTER__>(acc);    // 3 dof: rho_3, (2 pi)^{3/2}
      // The bend row's per-layer-visit constants. The track-side prediction and everything
      // in R_bb except the hit's own sigma_b are properties of the layer crossing, so they are formed
      // once here, exactly like the r-phi/secondary road above.
      auto& shDerBendOn = alpaka::declareSharedVar<int, __COUNTER__>(acc);   // 1 = the third row is live
      auto& shDerPredB = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // dphi/dr of the track [1/cm]
      // The prediction block of the innovation covariance, M = H (P+Q) H^T, packed symmetric 3x3 in
      // the order (00, 01, 02, 11, 12, 22) over the rows (r*dphi, secondary, bend). Formed once per
      // visit; the per-hit path only adds the hit's own R to it.
      auto& shM = alpaka::declareSharedVar<float[6], __COUNTER__>(acc);
      // The measurement rows of this layer crossing, from the dual-number crossing code, and the
      // prediction covariance they are evaluated against: P plus the traversed gap's process noise Q.
      // Built once per visit at the reference crossing (the scan is linearised there anyway), so the
      // per-hit path only adds the hit's own R. shHb is the stub-bend row, live when shDerBendOn.
      auto& shHphi = alpaka::declareSharedVar<float[5], __COUNTER__>(acc);    // d(r*phi)/dparams [cm]
      auto& shHsec = alpaka::declareSharedVar<float[5], __COUNTER__>(acc);    // d(sec)/dparams [cm]
      auto& shHb = alpaka::declareSharedVar<float[5], __COUNTER__>(acc);      // d(dPhiDr)/dparams [1/cm]
      auto& shPgate = alpaka::declareSharedVar<float[15], __COUNTER__>(acc);  // P + Q of this gap
      // The gap's Highland coefficients and its material moments about the PERIGEE reference point,
      // kept so the accept site injects exactly the Q the gate was computed with.
      auto& shQcPhi = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // c*(1+cot^2) [rad^2 per X/X0]
      auto& shQcCot = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // c*(1+cot^2)^2
      auto& shGapW = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // W  = sum rho dl
      auto& shGapS1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // S1 = sum rho dl s_k
      auto& shGapS2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // S2 = sum rho dl s_k^2
      // The same gap's moments about the ARRIVAL end (transverse lever), which is the frame the
      // energy-loss recursion is written in.
      auto& shElS1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shElS2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shSegBf = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // this segment's bending field
      // The energy-loss road centre of this layer visit, already projected onto the two gate rows:
      // a phi shift [rad] and a secondary shift [cm], both added to the prediction (never to a
      // width). Zero whenever the payload carries no eloss constants or the material march did not
      // run, which is what keeps the corrections-off path unchanged.
      auto& shDerElossPhi = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDerElossSec = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shCurrentL = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shWalkDone = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shWalkSteps = alpaka::declareSharedVar<int, __COUNTER__>(acc);    // reachable layers walked so far
      auto& shConfirmDone = alpaka::declareSharedVar<int, __COUNTER__>(acc);  // 1 once all candidate layers confirmed
      auto& shMergedHit = alpaka::declareSharedVar<int, __COUNTER__>(acc);    // round 0 (merged) attached here?
      auto& shHasOT = alpaka::declareSharedVar<int, __COUNTER__>(acc);  // >=1 OT extra committed on this candidate
      // Count of accepted OT-layer (CA >= 28) extras so far this walk, seeded with any tagged OT extras
      // already in the hit list. shNOTExtraAcc >= 1 == the track is "anchored" (a prior OT accept exists),
      // which is the class-aware condition for the cluster-cap exemption. Written unconditionally; read
      auto& shNOTExtraAcc = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& laneChi2 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneHit = alpaka::declareSharedVar<int32_t[kExtFindLanes], __COUNTER__>(acc);
      // The winning candidate's measurement, staged per lane for the post-reduce commit: the residual
      // rows d (r*dphi [cm], secondary [cm], bend [1/cm]), the hit's own noise block R (the position
      // 2x2 plus the uncorrelated bend variance), the number of live rows, and |S| -- which the hole
      // hypothesis prices the candidate's window volume with. The measurement rows themselves are
      // per-visit shared state (shHphi/shHsec/shHb), so no lane copy of them is needed.
      auto& laneD0 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneD1 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneD2 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneRpp = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneRps = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneRss = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneRbb = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneDet = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      // Full width of the secondary window when that row is not in the chi2 (0 when it is): the hole
      // hypothesis prices the winner's acceptance volume with it.
      auto& laneSecWin = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      // Clusters this lane saw inside the scan's own (eta, phi) patch around the crossing: the local
      // occupancy the hole hypothesis is priced with. Summed by the same tree reduce as the argmin.
      auto& laneOcc = alpaka::declareSharedVar<uint32_t[kExtFindLanes], __COUNTER__>(acc);
      auto& laneChi2Val = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneNRows = alpaka::declareSharedVar<int[kExtFindLanes], __COUNTER__>(acc);
      auto& laneRh = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneZh = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneArcS = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      // Winning original-lane index, carried through the hit-scan argmin tree reduce so lane 0 can index
      // the winner's KF payload (laneDPhi / laneSigPhi2 / ...) after the reduction.
      auto& laneWin = alpaka::declareSharedVar<int32_t[kExtFindLanes], __COUNTER__>(acc);
      // Per-(candidate,layer) crossing-quantity linearization coefficients (lane 0 -> shared, one set
      // per scanned layer). shUseLin selects the Taylor path (1) or the exact per-hit predict (0).
      auto& shLinRef = alpaka::declareSharedVar<float, __COUNTER__>(acc);    // expansion point (R barrel / Z endcap)
      auto& shLinPhi0 = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // phi at ref
      auto& shLinDphi1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // dphi/du
      auto& shLinDphi2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // d2phi/du2
      auto& shLinSec0 = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // secondary (z barrel / r endcap) at ref
      auto& shLinDsec1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // dsec/du
      auto& shLinDsec2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // d2sec/du2
      auto& shLinArc0 = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // arcS at ref
      auto& shLinDarc1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // darcS/du
      auto& shLinArgLo = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // barrel radial validity band lo
      auto& shLinArgHi = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // barrel radial validity band hi
      auto& shUseLin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // 1 iff the per-layer reference predict p0 was valid, so shLinDphi1 (the track's local
      // dphi/dr) is a real derivative and the 2S stub-bend term has a track curvature expectation to compare
      // against. Written by lane 0 in the linearization block.
      auto& shLinValid = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // Candidate-dump per-(candidate,layer) accumulators, always declared but written only under the
      // runtime candDump_ guard: reset at layer select, accumulated by the scan lanes via block
      // atomics, recorded by lane 0 at the record-write site. shDumpOccM/O = genuine (post-mask)
      // window candidates; shDumpPassM/O = gate passers; shDumpBestM = min gate chi2*1000 over
      // considered hits; shDumpWin*/Round = committed winner; shDumpCapDropped = winner dropped by the
      // MTV cluster cap; shDumpVeto = raw-OT round veto-skipped; shDumpPartner = stack-partner extra.
      auto& shDumpOccM = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpOccO = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpPassM = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpPassO = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpBestM = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpWinHit = alpaka::declareSharedVar<int32_t, __COUNTER__>(acc);
      auto& shDumpWinChi2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDumpWinRound = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shDumpCapDropped = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shDumpVeto = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shDumpPartner = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // Per-(candidate,layer) member write cursor. Reset at layer select (under candDump_), atomicAdd'd
      // once per scored road candidate in both rounds to allocate its slot m in the member buffer.
      // Always declared (a shared allocation only); written and read only under candDump_.
      auto& shDumpMemN = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      // Layer-outcome accumulator for the walk instrument's hole counter: 1 iff the visited layer
      // carried >=1 gate-passing candidate, or-ed across both source rounds' argmin-winner presence.
      auto& shChainHasPass = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // occupancy-gated hole counter (always declared, written only under candDump_). A "hole" is a
      // visited-reachable layer whose outcome was rejGate (occupancy present -- shHoleOcc>0 -- but no
      // gate-passer -- shChainHasPass==0), not rejEmpty (empty road = dead-module / no-crossing carve-out)
      // and not rejCap (a passer existed). shHoleRun = cumulative holes so far; shHoleConsec = consecutive
      // holes since the last accept (reset on any gate-passer). shHoleOcc = per-layer genuine-window
      // occupancy count (reset at layer select).
      auto& shHoleRun = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shHoleConsec = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shHoleOcc = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      // Per-lane best (min-gate-chi2) road candidate over all considered hits of both rounds; it
      // supplies the id and the base-gate-pass flag that accompany bestFailChi2. Same lane-array
      // reduction as the walk argmin: each lane keeps its slot's running min-chi2 (tie by min id) and
      // lane 0 reduces across lanes at the record write. Written only under candDump_, so no atomic
      // is needed. Chi2 sentinel 3.4e38, id sentinel -1; laneBestAnyHit carries the bit30-tagged id.
      // Sized to one element outside an EXT_CAND_DUMP build, so the production kernel does not pay
      // 1.5 kB of block shared memory on a one-block-per-candidate kernel.
      constexpr uint32_t kBestAnySlots = kExtCandDump ? kExtFindLanes : 1u;
      auto& laneBestAnyChi2 = alpaka::declareSharedVar<float[kBestAnySlots], __COUNTER__>(acc);
      auto& laneBestAnyHit = alpaka::declareSharedVar<int32_t[kBestAnySlots], __COUNTER__>(acc);
      auto& laneBestAnyPass = alpaka::declareSharedVar<int[kBestAnySlots], __COUNTER__>(acc);

      // grid-stride over candidates: one group (block) per candidate, all lanes see the same groups
      for (uint32_t j : cms::alpakatools::uniform_groups(acc, nC * nLanes)) {
        const uint32_t i = candList[j];

        // setup (lane 0): running helix, covered-layer mask, cached orig (r,z), walk state
        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          if (element.local != 0u)
            continue;
          // The walk starts in the origin field; every layer visit then re-evaluates the bending
          // field on its own road segment from the (Bz,Br) map and rebuilds the state geometry in it.
          shSegBf = bf;
          sh = makeRunningHelix(acc, tracks, int(i), shSegBf);
          const auto hitBegin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
          const auto hitEnd = tracks[i].hitOffsets();
          const int nOrig = int(hitEnd - hitBegin);
          uint64_t coveredMask = 0;
          int nPixHitsOrig = 0;
          // Tagged OT extras already on the list (from an earlier attach pass): each one is a prior
          // OT accept, so the walk starts anchored.
          int nPriorExtraClusters = 0;
          int nCoreClusters = 0;
          for (auto idx = hitBegin; idx < hitEnd; ++idx) {
            const auto hitId = trackHits[idx].id();
            // A hit list that a previous attach pass extended already contains tagged OT extras (bit30)
            // which index the raw OT source, not the merged SoA -- dispatch like WriteFinal's detOf (both
            // sources share the CA module numbering). On a first pass no id here is tagged.
            const bool ot = isOTId(hitId);
            const uint32_t detIdx =
                ot ? uint32_t(otSource_.otHits[otIdx(hitId)].detectorIndex()) : hits[hitId].detectorIndex();
            const int layer = hitLayer(detIdx, caLayers);
            if (layer >= 0 && layer < 64) {
              coveredMask |= (uint64_t(1) << layer);
              if (layer < 28)
                ++nPixHitsOrig;
            }
            if (ot)
              ++nPriorExtraClusters;  // an appended extra: one cluster, and not part of the core
            else
              nCoreClusters += isStub(hits, int32_t(hitId)) ? 2 : 1;  // a 2-hit stub = 2 clusters
          }
          shCoveredMask = coveredMask;
          const int nOrigSafe = nOrig < kMaxOrigHits ? nOrig : kMaxOrigHits;
          shNOrigSafe = nOrigSafe;
          // The original (r,z) hit array is cached cooperatively across all lanes (below), so the
          // lane-0 setup does not fill it here.
          shNExtra = 0;
          shExtraClusterCap = (nCoreClusters - 1) / 3;
          shNExtraClusters = nPriorExtraClusters;
          shWalkSteps = 0;
          // Far-first window-ambiguity state: unarmed until the eager-confirm round says otherwise.
          // Hole counter: reset the per-candidate cumulative/consecutive occupancy-gated hole run.
          // Written and read only under candDump_, and never read into track output.
          if (candDump_) {
            shHoleRun = 0;
            shHoleConsec = 0;
          }
          shConfirmDone = 0;
          shHasOT = 0;
          // Seed the anchor counter with the tagged OT extras already in the list (each one an OT accept),
          // so a track anchored by an earlier attach pass stays anchored -- consistent with the cap
          // seeding shNExtraClusters from the same extras. On a first pass nPriorExtraClusters == 0.
          shNOTExtraAcc = nPriorExtraClusters;
          // Seed the walk's anchor and its energy-loss centre from the merger-side pre-attach pass.
          // The anchor is the host's LAST FITTED NODE, not the PCA: the material inward of it is
          // already in the fit's covariance, and starting the march at the perigee would count it
          // twice. Without a valid payload the anchor falls back to the PCA.
          shDerQgap = 0.f;
          shEU = 0.f;
          shEUp = 0.f;
          shEDk = 0.f;
          shEK = 0.f;
          shLastArcS = 0.f;
          shLastR = alpaka::math::abs(acc, sh.helix().tip);
          shLastZ = sh.helix().zip;
          int derOn = 0;
          if (extPred != nullptr) {
            const ExtPredCoeff pc = extPred[i];  // indexed by tuple id (the producer's own indexing)
            if (pc.valid > 0.5f) {
              shLastArcS = pc.anchorS;
              shLastR = pc.anchorR;
              shLastZ = pc.anchorZ;
              shDerQgap = pc.qgapCoef > 0.f ? pc.qgapCoef : 0.f;
              // Guarded on a strictly positive growth rate, which is what the fit's own gate produces:
              // with the fit corrections off, or a degenerate column/momentum, the producer leaves all
              // four at 0 and every expression below reduces to the uncorrected form.
              if (pc.elossK > 0.f) {
                shEU = pc.elossU;
                shEUp = pc.elossUp;
                shEDk = pc.elossDkAnchor;
                shEK = pc.elossK;
              }
              derOn = 1;
            }
            alpaka::atomicAdd(acc, &stats[derOn ? kStatDerHostOn : kStatDerHostOff], 1u, alpaka::hierarchy::Grids{});
          }
        }
        // Warp-coop cache of the original (r,z) hit array: each lane owns k = lane, lane+nLanes, ...
        // over the [0, nOrigSafe) set. nOrigSafe / hitBegin are recomputed per lane from the same
        // read-only track SoA (cheap, redundant, no cross-lane dependency), so no extra sync is needed
        // -- the existing syncBlockThreads below publishes the stores to every reader (the candidate-
        // layer proxy loop). Independent per-hit writes of identical values; serial backend (warpSize
        // == 1) fills every k in 0,1,2,... order.
        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          const uint32_t lane = element.local;
          const auto hitBegin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
          const auto hitEnd = tracks[i].hitOffsets();
          const int nOrig = int(hitEnd - hitBegin);
          const int nOrigSafe = nOrig < kMaxOrigHits ? nOrig : kMaxOrigHits;
          for (int k = int(lane); k < nOrigSafe; k += int(nLanes)) {
            const auto hitId = trackHits[hitBegin + k].id();
            if (isOTId(hitId)) {  // tagged extra from an earlier pass -> fetch (r,z) from the OT source
              const uint32_t o = otIdx(hitId);
              const float xg = otSource_.otHits[o].xGlobal();
              const float yg = otSource_.otHits[o].yGlobal();
              shOrigR[k] = alpaka::math::sqrt(acc, xg * xg + yg * yg);
              shOrigZ[k] = otSource_.otHits[o].zGlobal();
            } else {
              shOrigR[k] = hits[hitId].rGlobal();
              shOrigZ[k] = hits[hitId].zGlobal();
            }
          }
        }
        alpaka::syncBlockThreads(acc);

        // Candidate-layer list, parallel over layers: for each uncovered layer a nearest-approach
        // distance from the candidate's cached original hits to the layer's (r,z) envelope, using
        // only the nominal R/Z and halfExtentR/Z rather than predictOnBarrel/Endcap, which the walk
        // runs on demand. shReachable[L] flags a candidate layer, the actual reachability being
        // decided in the walk from the full predict; shReachDist[L] is the ordering proxy, a
        // min-over-original-hits reduction plus the type-priority bias. Each lane owns layers
        // L = lane, lane+nLanes, ...
        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          const uint32_t lane = element.local;
          for (int L = int(lane); L < nLayers; L += int(nLanes)) {
            shVisited[L] = 0;
            int cand = 0;
            float dist = 0.f;
            if (!(shCoveredMask & (uint64_t(1) << L))) {
              const bool isBarrel = caLayers.isBarrel()[L];
              const float R = caLayers.layerR()[L];
              const float Z = caLayers.layerZ()[L];
              const float hR = caLayers.halfExtentR()[L];
              const float hZ = caLayers.halfExtentZ()[L];
              float d = 1e9f;
              for (int k = 0; k < shNOrigSafe; ++k) {
                const float rk = shOrigR[k];
                const float zk = shOrigZ[k];
                float dd;
                if (isBarrel) {
                  // nearest approach to the barrel cylinder R (z spans [Z-hZ, Z+hZ]): radial gap
                  // |R-r_ref| combined with the z-envelope overlap gap (0 inside the envelope).
                  const float dr = R - rk;
                  const float zGap = alpaka::math::max(acc, alpaka::math::abs(acc, zk - Z) - hZ, 0.f);
                  dd = alpaka::math::sqrt(acc, dr * dr + zGap * zGap);
                } else {
                  // nearest approach to the endcap disk Z (r spans [R-hR, R+hR]): z gap |Z-z_ref|
                  // combined with the r-envelope overlap gap (0 inside the envelope).
                  const float dz = Z - zk;
                  const float rGap = alpaka::math::max(acc, alpaka::math::abs(acc, rk - R) - hR, 0.f);
                  dd = alpaka::math::sqrt(acc, dz * dz + rGap * rGap);
                }
                if (dd < d)
                  d = dd;
              }
              cand = 1;
              dist = d;  // provisional; the confirm round replaces it by the information-gain key
            }
            shReachable[L] = cand;
            shReachDist[L] = dist;
          }
        }
        alpaka::syncBlockThreads(acc);

        // Walk the candidate layers in (proxy dist, then L) order. The full predict confirms every candidate layer
        // at once, lane-parallel, the first time the confirmed set is dry (L = lane, lane+nLanes, ...); layers
        // failing the envelope/arc test are consumed without spending the maxWalkLayers budget. shReachable: 0 =
        // covered or confirmed unreachable, 1 = candidate, 2 = confirmed reachable. shWalkDone: 0 = scan
        // shCurrentL, 1 = walk finished, 2 = eager-confirm pending.
        while (true) {
          // warp-parallel select: each lane keeps the local-best confirmed-reachable unvisited layer
          // over its stride L = lane, lane+nLanes, ... (min proxy dist, ties by smaller L via the
          // ascending strided scan + strict <), staged into laneHit/laneChi2 (both free here: the hit
          // scan below rewrites them before its reduction reads them).
          for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
            const uint32_t lane = element.local;
            int bestL = -1;
            float bestD = 0.f;
            for (int L = int(lane); L < nLayers; L += int(nLanes)) {
              if (shReachable[L] == 2 && !shVisited[L]) {
                if (bestL < 0 || shReachDist[L] < bestD) {
                  bestD = shReachDist[L];
                  bestL = L;
                }
              }
            }
            laneHit[lane] = bestL;
            laneChi2[lane] = bestD;
          }
          alpaka::syncBlockThreads(acc);

          // In-place shared-memory tree reduction across all lanes. The comparator is a strict total
          // order (dist, then layer L, and each candidate layer lives in exactly one lane's stride),
          // so min is associative and commutative and any reduction order yields the same winner on
          // every backend. Three reductions share one log2(nLanes)-step tree: primary (laneChi2=dist,
          // laneHit=L), OT-disk (laneChi2Disk, laneHitDisk) and the nDiskUnvis integer sum
          // (laneNDisk). Invalid entries carry laneHit/laneHitDisk == -1.
          for (uint32_t stride = nLanes >> 1; stride > 0u; stride >>= 1) {
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              const uint32_t lane = element.local;
              if (lane < stride) {
                const uint32_t o = lane + stride;
                const bool takeP = (laneHit[lane] < 0) ||
                                   (laneHit[o] >= 0 && (laneChi2[o] < laneChi2[lane] ||
                                                        (laneChi2[o] == laneChi2[lane] && laneHit[o] < laneHit[lane])));
                if (takeP) {
                  laneChi2[lane] = laneChi2[o];
                  laneHit[lane] = laneHit[o];
                }
              }
            }
            alpaka::syncBlockThreads(acc);
          }

          // lane 0: read the reduced global bests (gL / gLD / nDiskUnvis) from slot 0, budget-check,
          // then decide: visit the winner, eager-confirm all candidates once, or finish.
          for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
            if (element.local != 0u)
              continue;
            const int gL = laneHit[0];  // -1 when no confirmed-reachable unvisited layer
            // The visit order is the information-gain key alone: most informative reachable
            // uncovered layer first, within the K budget. The OT-disk seat reserve, the pixel-first
            // reserve and the far-first re-key all existed to correct a nearest-first proxy that no
            // longer decides anything.
            const int selL = gL;
            // The walk stops on its two compute budgets -- extra slots and layer visits -- and on the
            // matching bound, once not even a one-cluster extra would fit under it.
            if (shNExtra >= maxExtraHitsPerTrack || shWalkSteps >= extWalkBudget ||
                shNExtraClusters >= shExtraClusterCap) {
              shWalkDone = 1;
              if (shNExtra >= maxExtraHitsPerTrack) {  // this walk ended on the extras-slot budget
                alpaka::atomicAdd(acc, &stats[kDiagSlotExhaust], 1u, alpaka::hierarchy::Grids{});
                if (shHasOT != 0)
                  alpaka::atomicAdd(acc, &stats[kDiagSlotExhaustOT], 1u, alpaka::hierarchy::Grids{});
              }
            } else if (selL >= 0) {
              shCurrentL = selL;
              shVisited[selL] = 1;
              // far-first window-ambiguity condition: is this visit one of the far crossings the
              // re-key promoted? Endcap pixel layer, armed host, and beyond the host's own outermost
              // |z| -- the interior discs are the fallback and keep the unconditioned commit rule.
              // shFarArmed is 0 for every host the ordering does not touch and for every host when the
              // ordering is off, so both the scan-side count and the commit-side decline below are then
              // unreachable. Lane-0 store; the syncBlockThreads after this block publishes it to the
              // scan lanes.
              if (candDump_) {  // dump: reset this layer's per-(candidate,layer) accumulators
                shDumpOccM = 0u;
                shDumpOccO = 0u;
                shDumpPassM = 0u;
                shDumpPassO = 0u;
                shDumpBestM = 0xFFFFFFFFu;
                shDumpWinHit = -1;
                shDumpWinChi2 = -1.f;
                shDumpWinRound = -1;
                shDumpCapDropped = 0;
                shDumpVeto = 0;
                shDumpPartner = 0;
                shDumpMemN = 0u;  // reset this layer's member write cursor
                // Reset every lane's best-any (min-chi2 over all considered) slot to the sentinel
                // for this layer (both rounds accumulate into it). Lane-0 store; a syncBlockThreads follows
                // the layer-select block before the scan, so all lanes see the reset.
                for (uint32_t l2 = 0; l2 < nLanes; ++l2) {
                  laneBestAnyChi2[l2] = 3.4e38f;
                  laneBestAnyHit[l2] = -1;
                  laneBestAnyPass[l2] = 0;
                }
              }
              // Reset this layer's gate-passer + occupancy accumulators (or-ed / summed across
              // rounds). Under candDump_ so the hole counter can consume them below.
              if (candDump_) {
                shChainHasPass = 0;
                shHoleOcc = 0u;
              }
              shWalkDone = 0;
              ++shWalkSteps;  // only reachable-visited layers count toward the K budget
            } else if (!shConfirmDone) {
              shWalkDone = 2;     // confirmed set dry, candidates still unexamined -> eager-confirm all
              shConfirmDone = 1;  // one round leaves no state-1 layer; the walk never stages again
            } else {
              shWalkDone = 1;  // no confirmed unvisited layer and no candidate left -> walk finished
            }
          }
          alpaka::syncBlockThreads(acc);
          if (shWalkDone == 1)
            break;
          if (shWalkDone == 2) {
            // eager batch confirm (parallel over lanes): confirm every state-1 (candidate) layer in
            // <=2 strided rounds, each lane owning L = lane, lane+nLanes, ... one on-demand full
            // predict per layer, tested against the module-surface envelope widened by the
            // evaluated on the current running helix -- which at this first dry point is still the
            // initial fitted helix, no hit having been attached yet. Sets state 2 (reachable) or 0
            // (consumed, never counted against K).
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              const uint32_t lane = element.local;
              for (int L = int(lane); L < nLayers; L += int(nLanes)) {
                if (shReachable[L] != 1)
                  continue;
                const bool isBarrel = caLayers.isBarrel()[L];
                const float R = caLayers.layerR()[L];
                const float Z = caLayers.layerZ()[L];
                const Prediction pr =
                    isBarrel ? predictOnBarrel(acc, sh.helix(), R) : predictOnEndcap(acc, sh.helix(), Z);
                bool ok = pr.valid;
                if (ok) {
                  // Reachability from geometry: the layer's module-SURFACE envelope (built from the
                  // module plane corners, not from centres) widened by the prediction's own 1-sigma
                  // at this eps. Nothing tuned is left -- the slack constant, the pixel-only relax and
                  // the force-visit bypass that compensated a centre-only envelope are all gone.
                  const float hEnv = isBarrel ? caLayers.halfExtentZ()[L] : caLayers.halfExtentR()[L];
                  const float nom = isBarrel ? Z : R;
                  float Hp[5], Hs[5];
                  float slack = 0.f;
                  if (crossWithGrad5(acc,
                                     sh.phi0,
                                     sh.tip,
                                     sh.invPt,
                                     sh.cotTheta,
                                     sh.zip,
                                     isBarrel,
                                     isBarrel ? R : Z,
                                     shSegBf,
                                     pr.branch,
                                     Hp,
                                     Hs)) {
                    const float vSec = sh.predVar(Hs);
                    if (vSec > 0.f && alpaka::math::isfinite(acc, vSec))
                      slack = alpaka::math::sqrt(acc, qGate1 * vSec);
                    // Ordering key: the expected Gaussian information this crossing carries about the
                    // state, ln det(S) - ln det(R), weighted by the layer's own probability of having
                    // produced a usable hit. R is taken at the geometric bound on any sensor's sigma,
                    // which is enough for a monotone comparison between layers and adds no parameter.
                    // The key is formed once, on the state at the first dry point, like the proxy it
                    // replaces; it is not re-derived after each accept.
                    constexpr float kSigRef2 = 0.1f * 0.1f;  // (1 mm)^2, the sensor-sigma upper bound
                    const float vPhi = alpaka::math::max(acc, sh.predVar(Hp), 0.f);
                    const float etaLw =
                        (L >= 28 && extEtaL != nullptr)
                            ? alpaka::math::min(acc, 1.f, alpaka::math::max(acc, 1e-3f, extEtaL[L - 28]))
                            : 1.f;
                    const float info =
                        etaLw * (alpaka::math::log(acc, 1.f + vPhi / kSigRef2) +
                                 alpaka::math::log(acc, 1.f + alpaka::math::max(acc, vSec, 0.f) / kSigRef2));
                    if (alpaka::math::isfinite(acc, info))
                      shReachDist[L] = -info;  // select-min == most informative first
                  }
                  ok = !(alpaka::math::abs(acc, pr.secondary - nom) > hEnv + slack);
                }
                // A near-tangential crossing is not rejected here: with dc = |circle centre|, a
                // crossing at radius r_x has c = (r_x^2 + dc^2 - rho^2)/(2 dc r_x), and as |c| -> 1 the
                // phi-window width diverges as 1/sqrt(1 - c^2). What bounds that amplification for
                // low-pT curlers and displaced large-tip tangencies is the phi-window clamp and the
                // runaway ceilings in the scan below, not this reachability test.
                shReachable[L] = ok ? 2 : 0;  // unreachable: consumed, never counted toward K
              }
            }
            alpaka::syncBlockThreads(acc);
            alpaka::syncBlockThreads(acc);  // publish the confirmed set + its ordering key
            continue;                       // re-select among the newly confirmed layers
          }

          // Per-(candidate,layer) linearization of the crossing quantities. The per-hit scan crosses the
          // layer at each hit's own rh (barrel) or zh (endcap); the exact predict would cost a dozen
          // transcendentals per hit, so phi and the secondary coordinate are expanded to 2nd order and arcS to
          // 1st order about the layer reference R0/Z0, with analytic derivatives (shUseLin=0 keeps the exact
          // predict for near-tangential or Taylor-stressed layers).
          // Barrel (u = rh - R0), centre (xc,yc), |centre| dc, signed radius rho, c(R) = (R^2+dc^2-rho^2)/(2 dc R),
          //   theta(R) = phi_c + sBranch*acos(c):
          //     c'   = 1/(2dc) - (dc^2-rho^2)/(2 dc R^2),   c'' = (dc^2-rho^2)/(dc R^3)
          //     dphi/dr  = -sBranch * c'/sqrt(1-c^2)
          //     d2phi/dr2= -sBranch * ( c''/sqrt(1-c^2) + c c'^2/(1-c^2)^{3/2} )
          //     ds/dR = rho / (xc sin theta - yc cos theta),  dz/dr = cotTheta * ds/dr,
          //     d2s/dr2 = -rho*(xc cos theta + yc sin theta)*(dphi/dr) / (xc sin theta - yc cos theta)^2
          // Endcap (u = zh - Z0): arcS = (z - zip)/cotTheta and alphaH(z) = alphaOrigin - arcS/rho are linear
          //   (a1 = -1/(rho cotTheta)); with x = xc + |rho| cos alphaH, y = yc + |rho| sin alphaH:
          //     dphi/dz = (x y' - y x')/r^2,  d2phi/dz2 = (x y'' - y x'' - (dphi/dz)*2(x x'+y y'))/r^2
          //     dr/dz   = (x x' + y y')/r,    d2r/dz2   = (x'^2+x x''+y'^2+y y'')/r - (x x'+y y')^2/r^3
          for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
            if (element.local != 0u)
              continue;
            const int L = shCurrentL;
            const bool isBarrel = caLayers.isBarrel()[L];
            const float R0 = isBarrel ? caLayers.layerR()[L] : caLayers.layerZ()[L];
            const float W = isBarrel ? caLayers.halfExtentR()[L] : caLayers.halfExtentZ()[L];
            // The bending field to cross this layer in, from the same (Bz,Br) map the fits use: over a 60 cm forward
            // gap the field falls by a few percent and the road centre moves by O(cm). The walk rebuilds the helix from
            // the PCA in one constant field, and the constant that reproduces the crossing position is the
            // (s_end - s)-weighted average of B_bend along the path (x(s_end) = int_0^s_end kappa(s) (s_end - s) ds):
            // three samples with the weights exact for a quadratic profile, two fixed-point steps. B_bend is normalised
            // to Bz(0,0), so the scale factor is bf.
            {
              constexpr float kBSampU[3] = {0.f, 1.f / 3.f, 2.f / 3.f};
              constexpr float kBSampW[3] = {0.25f, 0.5f, 0.25f};
              float bSeg = shSegBf;
              for (int it = 0; it < 2; ++it) {
                sh.recomputeHelix(acc, bSeg);
                const HelixState hf = sh.helix();
                const Prediction pf = isBarrel ? predictOnBarrel(acc, hf, R0) : predictOnEndcap(acc, hf, R0);
                if (!pf.valid)
                  break;
                const float sEnd = pf.arcS;
                const float absRhoF = alpaka::math::abs(acc, hf.rho);
                float bSum = 0.f;
                for (int k = 0; k < 3; ++k) {
                  const float sK = kBSampU[k] * sEnd;
                  const float aK = hf.alphaOrigin - sK / hf.rho;
                  const float xK = hf.xc + absRhoF * alpaka::math::cos(acc, aK);
                  const float yK = hf.yc + absRhoF * alpaka::math::sin(acc, aK);
                  const float rK = alpaka::math::sqrt(acc, xK * xK + yK * yK);
                  const float zK = hf.zip + sK * hf.cotTheta;
                  // tanLambda cos(alpha), the track-radial cosine of blEffectiveBField, at this
                  // sample: cos(alpha) = -sign(rho) (xc*y - yc*x)/(|rho| r), so the charge cancels
                  // against the signed rho and what multiplies it is tanLambda = cotTheta.
                  const float den = (hf.cotTheta != 0.f) ? (-hf.rho * rK / hf.cotTheta) : 0.f;
                  const float tlca = (den != 0.f) ? -(hf.xc * yK - hf.yc * xK) / den : 0.f;
                  bSum += kBSampW[k] * float(blBFieldMap::bBendAt(bMap, double(rK), double(zK), double(tlca)));
                }
                const float bNew = bf * bSum;
                if (!(alpaka::math::abs(acc, bNew) > 1e-3f) || !alpaka::math::isfinite(acc, bNew))
                  break;
                bSeg = bNew;
              }
              shSegBf = bSeg;
              sh.recomputeHelix(acc, shSegBf);
            }
            const HelixState hh = sh.helix();
            const Prediction p0 = isBarrel ? predictOnBarrel(acc, hh, R0) : predictOnEndcap(acc, hh, R0);
            shLinRef = R0;
            shLinPhi0 = p0.phi;
            shLinSec0 = p0.secondary;
            shLinArc0 = p0.arcS;
            shLinDphi1 = 0.f;
            shLinDphi2 = 0.f;
            shLinDsec1 = 0.f;
            shLinDsec2 = 0.f;
            shLinDarc1 = 0.f;
            shLinArgLo = -1.0e30f;
            shLinArgHi = 1.0e30f;
            int useLin = 0;
            if (p0.valid) {
              const float xc = hh.xc, yc = hh.yc, rho = hh.rho;
              const float absRho = alpaka::math::abs(acc, rho);
              if (isBarrel) {
                const float dc2 = xc * xc + yc * yc;
                const float dc = alpaka::math::sqrt(acc, dc2);
                const float Aa = dc2 - rho * rho;
                float c0 = (R0 * R0 + Aa) / (2.f * dc * R0);
                c0 = alpaka::math::min(acc, 1.f, alpaka::math::max(acc, -1.f, c0));
                const float sq2 = alpaka::math::max(acc, 1.f - c0 * c0, kLinCondEps * kLinCondEps);
                const float sq = alpaka::math::sqrt(acc, sq2);
                const float cp1 = 1.f / (2.f * dc) - Aa / (2.f * dc * R0 * R0);
                const float cpp = Aa / (dc * R0 * R0 * R0);
                const float phic = alpaka::math::atan2(acc, yc, xc);
                const float sBranch = (foldPi(p0.phi - phic) >= 0.f) ? 1.f : -1.f;
                const float dphi1 = -sBranch * cp1 / sq;
                const float dphi2 = -sBranch * (cpp / sq + c0 * cp1 * cp1 / (sq * sq2));
                const float st = alpaka::math::sin(acc, p0.phi);
                const float ct = alpaka::math::cos(acc, p0.phi);
                const float Q0 = xc * st - yc * ct;
                const float dsdr = rho / Q0;
                const float Qp = (xc * ct + yc * st) * dphi1;
                const float d2sdr2 = -rho * Qp / (Q0 * Q0);
                shLinDphi1 = dphi1;
                shLinDphi2 = dphi2;
                shLinDsec1 = hh.cotTheta * dsdr;
                shLinDsec2 = hh.cotTheta * d2sdr2;
                shLinDarc1 = dsdr;
                shLinArgLo = dc - absRho - 1.0e-3f;  // predictOnBarrel radial validity band
                shLinArgHi = dc + absRho + 1.0e-3f;
                const float t2 = 0.5f * alpaka::math::abs(acc, dphi2) * W * W;
                useLin = ((1.f - alpaka::math::abs(acc, c0)) > kLinCondEps &&
                          alpaka::math::abs(acc, Q0) > kLinCondEps && t2 < kTaylor2ndMaxRad)
                             ? 1
                             : 0;
              } else {
                const float cot = hh.cotTheta;
                const float a1 = -1.f / (rho * cot);  // dalphaH/dz (constant in z)
                const float alphaH = hh.alphaOrigin - p0.arcS / rho;
                const float sA = alpaka::math::sin(acc, alphaH);
                const float cA = alpaka::math::cos(acc, alphaH);
                const float x = xc + absRho * cA;
                const float y = yc + absRho * sA;
                const float r2 = alpaka::math::max(acc, x * x + y * y, 1.0e-6f);
                const float rr = alpaka::math::sqrt(acc, r2);
                const float xp = -absRho * sA * a1;
                const float yp = absRho * cA * a1;
                const float xpp = -absRho * cA * a1 * a1;
                const float ypp = -absRho * sA * a1 * a1;
                const float Nn = x * yp - y * xp;
                const float dphi1 = Nn / r2;
                const float Np = x * ypp - y * xpp;
                const float Dp = 2.f * (x * xp + y * yp);
                const float dphi2 = (Np - dphi1 * Dp) / r2;
                const float Mm = x * xp + y * yp;
                const float drdz = Mm / rr;
                const float Mp = xp * xp + x * xpp + yp * yp + y * ypp;
                shLinDphi1 = dphi1;
                shLinDphi2 = dphi2;
                shLinDsec1 = drdz;
                shLinDsec2 = Mp / rr - Mm * Mm / (rr * r2);
                shLinDarc1 = 1.f / cot;  // arcS = (z - zip)/cotTheta is exactly linear
                const float t2 = 0.5f * alpaka::math::abs(acc, dphi2) * W * W;
                useLin = (t2 < kTaylor2ndMaxRad) ? 1 : 0;  // NaN-safe: NaN < thr is false -> fall back
              }
            }
            shUseLin = useLin;
            shLinValid = p0.valid ? 1 : 0;  // shLinDphi1 is a real derivative iff p0 was valid

            // The road of this layer visit, built once on the reference crossing p0.
            // Everything here is a property of the crossing, not of an individual candidate hit: the
            // measurement rows, the prediction covariance including the traversed gap's scattering,
            // the material integral, the energy-loss road centre and the gate threshold. The per-hit
            // path then only adds the hit's own R. The one material march per visit is the walk's
            // dominant cost and is not repeated per candidate.
            shDerLayOn = 0;
            for (int q = 0; q < 6; ++q)
              shM[q] = 0.f;
            for (int q = 0; q < 5; ++q) {
              shHphi[q] = 0.f;
              shHsec[q] = 0.f;
              shHb[q] = 0.f;
            }
            shDerWinR = 0.f;
            shDerCeilR = 0.f;
            shDerCeilS = 0.f;
            shOccNorm = 0.f;
            shDerHoleK = -1e30f;
            shDerHoleKRaw = -1e30f;
            shDerHoleK3 = -1e30f;
            shDerBendOn = 0;
            shDerPredB = 0.f;
            shQcPhi = 0.f;
            shQcCot = 0.f;
            shGapW = 0.f;
            shGapS1 = 0.f;
            shGapS2 = 0.f;
            shElS1 = 0.f;
            shElS2 = 0.f;
            shDerElossPhi = 0.f;
            shDerElossSec = 0.f;
            if (p0.valid) {
              // Runaway ceilings from this layer's module envelope: a geometric bound on a pathological
              // state, never the selector. The secondary prediction and the hit both lie inside the
              // envelope, so |dSec| <= 2 h_sec is a hard bound; in r-phi the envelope bounds nothing
              // azimuthally and the only transverse scale it supplies is the crossing surface's
              // thickness. Both binding rates are counted (kStatDerCapR / kStatDerCapS).
              const float hSecEnv = isBarrel ? caLayers.halfExtentZ()[L] : caLayers.halfExtentR()[L];
              const float hPropEnv = isBarrel ? caLayers.halfExtentR()[L] : caLayers.halfExtentZ()[L];
              shDerCeilS = 2.f * hSecEnv;
              shDerCeilR = 2.f * hPropEnv;
              const float rCross = isBarrel ? R0 : p0.secondary;
              const float zCross = isBarrel ? p0.secondary : R0;
              const float rSafe = alpaka::math::max(acc, rCross, 1.f);
              const float cot = hh.cotTheta;
              const float cot2 = cot * cot;
              const float coslam2 = 1.f / (1.f + cot2);
              const float coslam = alpaka::math::sqrt(acc, coslam2);

              // --- the traversed gap: one material march, anchor -> crossing -------------------------
              // segmentXX0Moments publishes the Kleinwort two-thin-scatterer pair, i.e. the moments of
              // rho*dl about the ARRIVAL end with a 3-D lever: S1_arr = w1 W d1, S2_arr = S1_arr d1.
              // The perigee state needs them about the PERIGEE reference point and on the TRANSVERSE
              // arc the state's levers are measured in, so convert: one cos(lambda) per lever power,
              // then shift the origin from the arrival end (at transverse arc s_a) to the perigee.
              double d1 = 0., w1 = 0.;
              // The map holds X/X0 per cm of 3-D path, and the (r,z) chord the walk marches is shorter
              // than the helix arc between the same two points: hand the march the gap's real 3-D path
              // so the total and the lever come out in path units, the same rescaling the fit's own
              // marchers apply.
              const float sGapT = alpaka::math::abs(acc, p0.arcS - shLastArcS);
              const double path3D = double(sGapT) / double(alpaka::math::max(acc, coslam, 1e-3f));
              const float Wm = float(brokenline::segmentXX0Moments(
                  acc, rhoMap_, double(shLastR), double(shLastZ), double(rSafe), double(zCross), d1, w1, path3D));
              const float S1arr = float(w1) * Wm * float(d1) * coslam;
              const float S2arr = float(w1) * Wm * float(d1) * float(d1) * coslam2;
              const float sA = p0.arcS;  // transverse arc of the crossing from the perigee
              const float S1pca = Wm * sA - S1arr;
              const float S2pca = Wm * sA * sA - 2.f * sA * S1arr + S2arr;
              const float pT = alpaka::math::abs(acc, shSegBf * hh.rho);
              const float pTot = pT * alpaka::math::sqrt(acc, 1.f + cot2);
              const float cH = extHighlandC(acc, pTot, Wm);
              // Azimuthal deflection: variance theta0^2/sin^2(theta) = theta0^2 (1+cot^2). Polar:
              // d(cot)/d(lambda) = -(1+cot^2), so its variance carries that Jacobian squared.
              shQcPhi = cH * (1.f + cot2);
              shQcCot = cH * (1.f + cot2) * (1.f + cot2);
              shGapW = Wm;
              shGapS1 = S1pca;
              shGapS2 = S2pca;
              shElS1 = S1arr;
              shElS2 = S2arr;
              // The last fitted gap's exit-direction kink is structurally invisible to the fit
              // (varBeta(n-1) == 0), so it is injected here as an extra kink at the anchor arc.
              const float qGapExit = shDerQgap;

              // --- P + Q of this gap, and the measurement rows ---------------------------------------
              for (int q = 0; q < 15; ++q)
                shPgate[q] = sh.C[q];
              {
                RunningHelix tmp;  // a covariance-only scratch: addKinkNoise touches C alone
                for (int q = 0; q < 15; ++q)
                  tmp.C[q] = shPgate[q];
                tmp.addKinkNoise(shQcPhi, shQcCot, shGapW, shGapS1, shGapS2);
                if (qGapExit > 0.f) {
                  const float sk = shLastArcS;
                  tmp.addKinkNoise(qGapExit * (1.f + cot2), qGapExit * (1.f + cot2) * (1.f + cot2), 1.f, sk, sk * sk);
                }
                for (int q = 0; q < 15; ++q)
                  shPgate[q] = tmp.C[q];
              }
              float Hphi[5], Hsec[5];
              const bool rowsOk = crossWithGrad5(
                  acc, sh.phi0, sh.tip, sh.invPt, sh.cotTheta, sh.zip, isBarrel, R0, shSegBf, p0.branch, Hphi, Hsec);
              if (rowsOk) {
                for (int q = 0; q < 5; ++q) {
                  shHphi[q] = Hphi[q];
                  shHsec[q] = Hsec[q];
                }
                auto proj = [&](const float* Ha, const float* Hb2) {
                  float v = 0.f;
                  for (int x = 0; x < 5; ++x)
                    for (int y = 0; y < 5; ++y)
                      v += Ha[x] * shPgate[RunningHelix::cIdx(x, y)] * Hb2[y];
                  return v;
                };
                shM[0] = proj(Hphi, Hphi);
                shM[1] = proj(Hphi, Hsec);
                shM[3] = proj(Hsec, Hsec);
                shDerLayOn = (alpaka::math::isfinite(acc, shM[0]) && alpaka::math::isfinite(acc, shM[3]) &&
                              shM[0] > 0.f && shM[3] > 0.f)
                                 ? 1
                                 : 0;

                // --- the stub-bend row -------------------------------------------------------------
                // The track side is the closed-form dphi/dr of the same crossing; its state Jacobian is
                // the analytic gradient of that expression (the only row not produced by the crossing's
                // own dual-number pass: it is a derivative OF a derivative). Q enters through shPgate,
                // so the row shares the state's scattering with the position rows.
                if (shDerLayOn) {
                  const float predB = extBendPredDPhiDr(acc, hh, isBarrel, R0);
                  float Hb1 = 0.f, Hb2c = 0.f, Hb3 = 0.f, Hb4 = 0.f;
                  if (predB != 0.f &&
                      extBendPredDPhiDrGrad(acc, hh, isBarrel, R0, shSegBf, predB, Hb1, Hb2c, Hb3, Hb4)) {
                    const float Hbv[5] = {0.f, Hb1, Hb2c, Hb3, Hb4};  // the phi0 partial is exactly 0
                    const float m22 = proj(Hbv, Hbv);
                    if (alpaka::math::isfinite(acc, m22) && m22 >= 0.f) {
                      for (int q = 0; q < 5; ++q)
                        shHb[q] = Hbv[q];
                      shM[2] = proj(Hphi, Hbv);
                      shM[4] = proj(Hsec, Hbv);
                      shM[5] = m22;
                      shDerPredB = predB;
                      shDerBendOn = 1;
                    }
                  }
                }
              }

              // --- the energy-loss road centre ------------------------------------------------------
              // The walk propagates the curvature published at the vertex while the real curvature grows along the path,
              // so at the layer the track sits inside that circle. In the band's local frame (u = radial offset, outward
              // positive) the offset is uE = shEU + shEUp*ds - 0.5*(shEDk*ds^2 + shEK*S2_arr), S2_arr the second moment
              // about the arrival end; projected on the two rows with the crossing geometry (a bias, never a variance).
              if (shDerLayOn && shEK > 0.f) {
                const float ds = sA - shLastArcS;
                const float xCr = rSafe * alpaka::math::cos(acc, p0.phi);
                const float yCr = rSafe * alpaka::math::sin(acc, p0.phi);
                const float uE = shEU + shEUp * ds - 0.5f * (shEDk * ds * ds + shEK * S2arr);
                const float absRhoS = alpaka::math::max(acc, alpaka::math::abs(acc, hh.rho), 1e-6f);
                const float bN = (hh.xc * yCr - hh.yc * xCr) / (rSafe * absRhoS);          // n.phihat, signed cos(psi)
                const float aN = (rSafe - (hh.xc * xCr + hh.yc * yCr) / rSafe) / absRhoS;  // n.rhat
                const float bSgn = (bN >= 0.f) ? 1.f : -1.f;
                const float bSafe = bSgn * alpaka::math::max(acc, alpaka::math::abs(acc, bN), 1e-3f);
                const float dRPhiE = isBarrel ? (uE / bSafe) : (uE * bN);
                const float dSecE = isBarrel ? (-uE * aN * shLinDsec1) : (uE * aN);
                if (alpaka::math::isfinite(acc, dRPhiE) && alpaka::math::isfinite(acc, dSecE)) {
                  shDerElossPhi = dRPhiE / rSafe;
                  shDerElossSec = dSecE;
                }
              }

              // --- the window, from the gate itself --------------------------------------------------
              // The r-phi half-window is the bounding half-width of the same chi2 ball the gate cuts on:
              // a derived consequence of the one eps, not a second choice. The hit term is taken at a
              // safe sensor upper bound, since the window is sized before the hits are read.
              if (shDerLayOn) {
                constexpr float kMaxSigPhiHitCmW = 0.1f;
                const float qWin = shDerBendOn ? alpaka::math::max(acc, qGate2, qGate3) : qGate2;
                shDerWinR = alpaka::math::sqrt(acc, qWin * (shM[0] + kMaxSigPhiHitCmW * kMaxSigPhiHitCmW));
                alpaka::atomicAdd(acc, &stats[kStatDerLayers], 1u, alpaka::hierarchy::Grids{});
              }

              // --- the hole hypothesis ---------------------------------------------------------------
              // "This layer produced no usable hit" competes in the same currency as a candidate: the PDA no-detection
              // weight chi2_hole = 2 ln[ eta_L / ((1 - eta_L eps) nu) ] with nu = rho_d (2 pi)^{d/2} |S|^{1/2}; the |S|
              // half is per-candidate and is added at the commit site with the local/layer density ratio the scan
              // measures. eta_L and rho are measured detector properties per layer.
              if (shDerLayOn && L >= 28 && extEtaL != nullptr && extRho != nullptr) {
                const float etaLc = alpaka::math::min(acc, 0.9999f, alpaka::math::max(acc, 1e-4f, extEtaL[L - 28]));
                const float rhoL = extRho[L - 28];
                const float ne = etaLc * extGateEps;
                if (ne > 0.f && ne < 1.f) {
                  constexpr float kTwoPi = 6.2831853f;
                  constexpr float kTwoPi32 = 15.7496099f;  // (2 pi)^{3/2}
                  if (rhoL > 0.f)
                    shDerHoleK = 2.f * alpaka::math::log(acc, etaLc / ((1.f - ne) * rhoL * kTwoPi));
                  const float rho3L = (extRho3 != nullptr) ? extRho3[L - 28] : 0.f;
                  if (shDerBendOn && rho3L > 0.f)
                    shDerHoleK3 = 2.f * alpaka::math::log(acc, etaLc / ((1.f - ne) * rho3L * kTwoPi32));
                  // The raw-OT round runs only where the stub round attached nothing, so it arbitrates
                  // the conditional availability of a raw cluster given that no stub formed, against a
                  // background of raw clusters. Its density is the event's own: the OT layer occupancy
                  // over the layer's envelope area, not a frozen multiple of the stub density.
                  const float etaLR =
                      (extEtaLRaw != nullptr)
                          ? alpaka::math::min(acc, 0.9999f, alpaka::math::max(acc, 1e-4f, extEtaLRaw[L - 28]))
                          : etaLc;
                  float rhoR = rhoL;
                  if (otSource_.nOTHits > 0u && otSource_.layerStart != nullptr) {
                    const float nOTL = float(otSource_.layerStart[L + 1] - otSource_.layerStart[L]);
                    const float rEnv = alpaka::math::max(acc, caLayers.layerR()[L], 1.f);
                    const float hEnv = isBarrel ? caLayers.halfExtentZ()[L] : caLayers.halfExtentR()[L];
                    const float area = 2.f * kExtenderPi * rEnv * 2.f * alpaka::math::max(acc, hEnv, 0.1f);
                    if (nOTL > 0.f && area > 0.f)
                      rhoR = nOTL / area;
                  }
                  const float neR = etaLR * extGateEps;
                  if (rhoR > 0.f && neR > 0.f && neR < 1.f)
                    shDerHoleKRaw = 2.f * alpaka::math::log(acc, etaLR / ((1.f - neR) * rhoR * kTwoPi));
                }
              }
              if (!shDerLayOn) {
                shDerElossPhi = 0.f;
                shDerElossSec = 0.f;
              }
            }
          }
          alpaka::syncBlockThreads(acc);

          // Additive attach policy: two rounds per layer visit sharing the same lane arrays and
          // syncBlockThreads pattern. Round 0 scans the merged hit source; round 1 scans the raw OT
          // source only where round 0 attached nothing and the OT source is active, so OT hits add on
          // stub-less layers instead of replacing a merged attachment. The lane arrays are reused for
          // round 1 after round 0's lane-0 reduce has consumed them, separated by a syncBlockThreads.
          // With no OT source round 1 never runs.
          for (int round = 0; round < 2; ++round) {
            // The stack-partner scan lives inside round 1, so it is skipped whenever round 1 is.
            if (round == 1 && (otSource_.nOTHits == 0u || shMergedHit != 0))
              break;  // no OT source, or the merged round already attached here
            // source-scoped raw-OT veto: skip the raw-OT round on the impure layer classes (TOB4-6
            // CA 31-33 / TID CA 34-53). The merged-stub round 0 already ran on this layer and is
            // untouched; only the low-purity raw-OT source is withheld here.

            // parallel hit scan on shCurrentL: each lane scans a round-robin subset of the phi window
            // and keeps its local best (min chi2, ties by min hitId). Round 0 reads the merged binner /
            // SoA; round 1 reads the OT binner / SoA (same window, same gate, same KF update).
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              const uint32_t lane = element.local;
              const int L = shCurrentL;
              const bool isBarrel = caLayers.isBarrel()[L];
              const float R = caLayers.layerR()[L];
              const float Z = caLayers.layerZ()[L];

              // One statistic, one threshold, one ranking key. The argmin runs on the tail probability
              // -ln(1 - F_d(chi2)) so candidates of different dof compete fairly; the seed is +inf and
              // each candidate applies the quantile of its own dof explicitly.
              float bestScore = 3.4e38f;
              float bestChi2 = 3.4e38f;
              float bestD0 = 0.f, bestD1 = 0.f, bestD2 = 0.f;
              float bestRpp = 0.f, bestRps = 0.f, bestRss = 0.f, bestRbb = 0.f;
              float bestDet = 0.f;
              float bestSecWin = 0.f;
              int bestNRows = 0;
              float bestRh = 0.f, bestZh = 0.f, bestArcS = 0.f;
              int32_t bestHit = -1;
              // candDump_ only: per-lane, per-round best over all considered road candidates
              // (passers and failers), min score then min id. Combined into laneBestAny* after the scan.
              float bestAnyChi2 = 3.4e38f;
              int32_t bestAnyId = -1;
              int bestAnyPass = 0;

              uint32_t nOccSeen = 0;  // clusters this lane sees in the local-occupancy patch below
              const Prediction pred =
                  isBarrel ? predictOnBarrel(acc, sh.helix(), R) : predictOnEndcap(acc, sh.helix(), Z);
              if (pred.valid) {
                // Phi window over the CA per-layer phi histogram (identical derivation to the serial
                // version; every lane computes it redundantly from the shared helix + layer, so all
                // lanes agree on the bin range and hit-counter partition). The half-width is the
                // gate's own sigma at eps -- the bounding box of the same chi2 ball -- so there is no
                // window multiplier and no cm cap: one eps sizes window, gate, rank and hole together.
                const float capRPhiEffMax = alpaka::math::min(acc, shDerWinR, shDerCeilR);
                const float rMinLayer = alpaka::math::max(acc, R - caLayers.halfExtentR()[L], 1.f);
                float phiExtA = pred.phi, phiExtB = pred.phi;
                if (isBarrel) {
                  const Prediction pA = predictOnBarrel(acc, sh.helix(), rMinLayer);
                  const Prediction pB = predictOnBarrel(acc, sh.helix(), R + caLayers.halfExtentR()[L]);
                  phiExtA = pA.valid ? pA.phi : pred.phi;
                  phiExtB = pB.valid ? pB.phi : pred.phi;
                } else {
                  const Prediction pA = predictOnEndcap(acc, sh.helix(), Z - caLayers.halfExtentZ()[L]);
                  const Prediction pB = predictOnEndcap(acc, sh.helix(), Z + caLayers.halfExtentZ()[L]);
                  phiExtA = pA.valid ? pA.phi : pred.phi;
                  phiExtB = pB.valid ? pB.phi : pred.phi;
                }
                const float deltaMax = alpaka::math::max(acc,
                                                         alpaka::math::abs(acc, foldPi(phiExtA - pred.phi)),
                                                         alpaka::math::abs(acc, foldPi(phiExtB - pred.phi)));
                const float dPhiWin = capRPhiEffMax / rMinLayer + deltaMax;
                // the total phi half-window, before it becomes the iphi bin window (guard bins added
                // below).
                const float halfWin = dPhiWin;
                const float kToShort = 32768.f / kExtenderPi;  // short units per radian (matches hit iphi scale)
                constexpr int kGuardShorts = 3 * 256;          // ~3 phi bins for iphi quantization + atan approx
                const int iphicut = int(alpaka::math::round(acc, halfWin * kToShort)) + kGuardShorts;
                // The phi-bin window is centred on the energy-loss-corrected prediction: the window is
                // no wider than the road, so a mis-centred scan would drop the very hits the corrected
                // gate accepts. shDerElossPhi is 0 off the corrected path (adding an exact float zero).
                const int16_t mep = int16_t(alpaka::math::round(acc, foldPi(pred.phi + shDerElossPhi) * kToShort));
                const uint32_t hoff = ExtPhiBinner::histOff(uint32_t(L));
                const auto kl = ExtPhiBinner::bin(int16_t(mep - iphicut));
                auto khh = ExtPhiBinner::bin(int16_t(mep + iphicut));
                khh = (khh + 1) % ExtPhiBinner::nbins();
                // The occupancy the scan itself sees: the hole is priced with the number of random clusters expected where
                // this candidate is, and a layer average understates that inside a jet. The patch is the phi bins the scan
                // traverses times the same angular span in pseudorapidity; the count is compared with this layer's own
                // occupancy in this event spread uniformly over the patch (shOccNorm), a dimensionless contrast. The
                // counting is one compare per hit the scan reads anyway; the sum comes out of the argmin's tree reduce.
                const uint32_t nBinsScan =
                    uint32_t((int(khh) - int(kl) + int(ExtPhiBinner::nbins())) % int(ExtPhiBinner::nbins()));
                const float spanPhi = float(nBinsScan) * (2.f * kExtenderPi / float(ExtPhiBinner::nbins()));
                const float cotOcc = sh.helix().cotTheta;
                const float coshEtaOcc = alpaka::math::sqrt(acc, 1.f + cotOcc * cotOcc);
                const float rCrossOcc = alpaka::math::max(acc, isBarrel ? R : pred.secondary, 1.f);
                // z = r sinh(eta) at fixed r, r = z / sinh(eta) at fixed z.
                const float dSecPerEta =
                    isBarrel ? rCrossOcc * coshEtaOcc
                             : rCrossOcc * coshEtaOcc / alpaka::math::max(acc, alpaka::math::abs(acc, cotOcc), 1e-3f);
                const float hSecOcc = isBarrel ? caLayers.halfExtentZ()[L] : caLayers.halfExtentR()[L];
                const float wSecOcc = alpaka::math::min(acc, 0.5f * spanPhi * dSecPerEta, hSecOcc);
                const float secPredOcc = pred.secondary + shDerElossSec;
                if (lane == 0u) {
                  const auto* __restrict__ binL = (round == 0) ? phiBinner_ : otSource_.phiBinner;
                  const float nLayerHits = float(binL->end(hoff + ExtPhiBinner::nbins() - 1u) - binL->begin(hoff));
                  const float patchArea = spanPhi * rCrossOcc * 2.f * wSecOcc;
                  const float layerArea = 2.f * kExtenderPi * rCrossOcc * 2.f * hSecOcc;
                  shOccNorm = (nLayerHits > 0.f && patchArea > 0.f && layerArea > 0.f)
                                  ? layerArea / (nLayerHits * patchArea)
                                  : 0.f;
                }
                uint32_t hitCounter = 0;  // running window-hit index; hit g is owned by lane (g % nLanes)
                // Round 0 only: the merged-source scan. In round 1 the loop condition is false at entry
                // (body never runs) so hitCounter stays 0 for the fresh OT pass below; a different lane
                // ownership partition than the merged pass, but the argmin (min chi2, tie by id) result is
                // partition-independent, so physics is unchanged.
                for (auto kk = kl; round == 0 && kk != khh; kk = (kk + 1) % ExtPhiBinner::nbins()) {
                  auto const* __restrict__ pbeg = phiBinner_->begin(kk + hoff);
                  auto const* __restrict__ pend = phiBinner_->end(kk + hoff);
                  // Per-bin strided ownership: each lane owns the hits g = lane (mod nLanes) and jumps
                  // straight to them, instead of traversing every window hit and masking off the ones it
                  // does not own. A uniform traversal would execute the gate body one owner lane at a
                  // time -- a warp-serial scan at ~1 active thread per warp -- whereas with the stride up
                  // to nLanes gates execute in the same warp instructions. The partition sets only which
                  // lane scores a hit; the (chi2, hitId) argmin over all of them is partition-independent.
                  const uint32_t nBin = uint32_t(pend - pbeg);
                  const uint32_t o0 = (nLanes + lane - (hitCounter % nLanes)) % nLanes;  // first owned in-bin offset
                  hitCounter += nBin;
                  for (uint32_t o = o0; o < nBin; o += nLanes) {
                    const uint32_t hitId = pbeg[o];
                    // Local occupancy, counted BEFORE the mask: the denominator it is compared against
                    // is this layer's whole content, so the numerator has to be the same population.
                    if (alpaka::math::abs(
                            acc, (isBarrel ? hits[hitId].zGlobal() : hits[hitId].rGlobal()) - secPredOcc) <= wSecOcc)
                      ++nOccSeen;
                    if (hitMaskArmed && hitMask[hitId].recHitMask() != 0u)
                      continue;
                    if (candDump_)  // dump: genuine (post-mask) merged candidate in the phi window
                      alpaka::atomicAdd(acc, &shDumpOccM, 1u, alpaka::hierarchy::Blocks{});
                    if (candDump_)  // hole counter: layer occupancy present
                      alpaka::atomicAdd(acc, &shHoleOcc, 1u, alpaka::hierarchy::Blocks{});
                    const float xh = hits[hitId].xGlobal();
                    const float yh = hits[hitId].yGlobal();
                    const float zh = hits[hitId].zGlobal();
                    const float rh = hits[hitId].rGlobal();
                    const float phiH = alpaka::math::atan2(acc, yh, xh);

                    // Per-hit crossing prediction: the linearized Taylor path (few FMAs) when this layer's
                    // coefficients were well-conditioned (shUseLin), else the exact per-hit closed form.
                    // Both feed the identical downstream gate through p2.phi/p2.secondary/p2.arcS.
                    Prediction p2;
                    if (shUseLin) {
                      const float u = isBarrel ? (rh - shLinRef) : (zh - shLinRef);
                      const float arcLin = shLinArc0 + shLinDarc1 * u;
                      bool okLin = (arcLin > 0.f && arcLin <= kExtenderMaxArcLengthCm);
                      if (isBarrel)
                        okLin = okLin && (rh >= shLinArgLo && rh <= shLinArgHi);
                      if (!okLin)
                        continue;
                      p2.phi = foldPi(shLinPhi0 + shLinDphi1 * u + 0.5f * shLinDphi2 * u * u);
                      p2.secondary = shLinSec0 + shLinDsec1 * u + 0.5f * shLinDsec2 * u * u;
                      p2.arcS = arcLin;
                      p2.valid = true;
                    } else {
                      p2 = isBarrel ? predictOnBarrel(acc, sh.helix(), rh) : predictOnEndcap(acc, sh.helix(), zh);
                      if (!p2.valid)
                        continue;
                    }

                    // Cheap pre-filter: a hit beyond the window's own half-width can only fail the gate.
                    const float dPhiPF = foldPi(p2.phi + shDerElossPhi - phiH);
                    if (alpaka::math::abs(acc, dPhiPF) * rh > capRPhiEffMax)
                      continue;

                    const float secH = isBarrel ? zh : rh;
                    // road centre: the deterministic dE/dx offset of this crossing, formed once per
                    // visit in the lane-0 block and added to the prediction (both rows). It is a known
                    // bias, not an uncertainty, so it belongs here and not in any sigma. Exactly 0 with
                    // the fit corrections off.
                    const float dPhi = foldPi(p2.phi + shDerElossPhi - phiH);
                    const float dSec = p2.secondary + shDerElossSec - secH;
                    const float rh_safe = alpaka::math::max(acc, rh, 1.f);

                    // The hit's own noise R: the local errors rotated through the module frame and
                    // projected onto the two position rows, as a full 2x2, the cross term being free.
                    // No variance floor and no alignment floor: a real sensor frame cannot project to
                    // zero, and a floor of the size one would reach for here (1e-6 rad^2 = 1 mm at
                    // r = 1 m, or 1e-4 cm^2) exceeds the intrinsic variance by up to three orders of
                    // magnitude and stops an accepted OT hit from ever sharpening the state.
                    const float xerr = hits[hitId].xerrLocal();
                    const float yerr = hits[hitId].yerrLocal();
                    const auto detIdx = hits[hitId].detectorIndex();
                    const auto frame = caModules[detIdx].innerSensorFrame();
                    float ge[6];
                    frame.toGlobal(xerr, 0.f, yerr, ge);
                    const float rh2 = rh_safe * rh_safe;
                    const float gxx = ge[0], gxy = ge[1], gyy = ge[2], gxz = ge[3], gyz = ge[4], gzz = ge[5];
                    // (r*dphi) direction = (-y, x)/r ; secondary = z (barrel) or (x, y)/r (endcap).
                    const float Rpp = (yh * yh * gxx - 2.f * xh * yh * gxy + xh * xh * gyy) / rh2;
                    const float Rss = isBarrel ? gzz : (xh * xh * gxx + 2.f * xh * yh * gxy + yh * yh * gyy) / rh2;
                    const float Rps = isBarrel ? ((-yh * gxz + xh * gyz) / rh_safe)
                                               : ((-yh * xh * gxx + (xh * xh - yh * yh) * gxy + xh * yh * gyy) / rh2);
                    if (!(Rpp > 0.f) || !(Rss > 0.f))
                      continue;

                    // The stub's local bend is an independent third row of the same statistic, with the
                    // track-side prediction uncertainty already inside S through shM. sigma_b is the
                    // leak-free formation error; non-stub candidates keep the 2-row statistic.
                    float dBend = 0.f, Rbb = 0.f;
                    bool bendRow = false;
                    if (shDerBendOn && isStub(hits, int32_t(hitId))) {
                      auto const stub = hits.stub(int32_t(hitId));
                      const float sPrec = stub.dPhiDrErrorPrec();
                      if (sPrec > 0.f) {
                        dBend = shDerPredB - stub.dPhiDr();
                        Rbb = sPrec * sPrec;
                        bendRow = true;
                      }
                    }
                    const float dR0 = dPhi * rh_safe;  // the r-phi residual in cm, the row's own metric
                    // A secondary reading whose support is wider than the road is a window, not a
                    // measurement: it leaves the chi2 and the candidate loses that degree of freedom.
                    const bool secWin = extSecIsWindow(Rss, shM[3], qGate1);
                    const float secHalf = extSecSupport(acc, Rss, shM[3], qGate1);
                    const int nRows = (secWin ? 1 : 2) + (bendRow ? 1 : 0);
                    float Sm[6] = {shM[0] + Rpp, 0.f, 0.f, 0.f, 0.f, 0.f};
                    float dv[3] = {dR0, 0.f, 0.f};
                    if (secWin) {
                      if (bendRow) {
                        Sm[1] = shM[2];
                        Sm[3] = shM[5] + Rbb;
                        dv[1] = dBend;
                      }
                    } else {
                      Sm[1] = shM[1] + Rps;
                      Sm[2] = shM[2];
                      Sm[3] = shM[3] + Rss;
                      Sm[4] = shM[4];
                      Sm[5] = shM[5] + Rbb;
                      dv[1] = dSec;
                      dv[2] = dBend;
                    }
                    float detS = 0.f;
                    const float chi2 = extChi2FromS(acc, nRows, Sm, dv, detS);
                    if (!(chi2 >= 0.f))
                      continue;  // S not positive definite: no usable statistic for this candidate
                    // One gate, one eps: the chi2 quantile of this candidate's own dof. Ranking is by
                    // the tail probability, so a 2-row and a 3-row candidate compete on equal terms
                    // instead of the row-poorer one winning the argmin by its missing row.
                    const float qGate = (nRows == 1) ? qGate1 : (nRows == 2 ? qGate2 : qGate3);
                    const float score =
                        -alpaka::math::log(acc, alpaka::math::max(acc, extChi2Tail(acc, nRows, chi2), 1e-30f));
                    if (candDump_)  // dump: min gate chi2 over considered merged hits (best fail)
                      alpaka::atomicMin(acc,
                                        &shDumpBestM,
                                        chi2 >= 4.29e6f ? 0xFFFFFFFEu : uint32_t(chi2 * 1000.f + 0.5f),
                                        alpaka::hierarchy::Blocks{});
                    // Runaway ceilings from the layer's module envelope: a guard on a pathological
                    // state, never the selector. Each time one rejects a hit the gate admitted, it is
                    // counted -- if that happens often the delivered efficiency is not the stated eps.
                    const bool gatePass = (chi2 < qGate);
                    const bool okR = alpaka::math::abs(acc, dR0) < shDerCeilR;
                    const bool okS =
                        alpaka::math::abs(acc, dSec) < shDerCeilS && alpaka::math::abs(acc, dSec) < secHalf;
                    if (gatePass && !okR)
                      alpaka::atomicAdd(acc, &stats[kStatDerCapR], 1u, alpaka::hierarchy::Grids{});
                    if (gatePass && !okS)
                      alpaka::atomicAdd(acc, &stats[kStatDerCapS], 1u, alpaka::hierarchy::Grids{});
                    const bool pass = gatePass && okR && okS;

                    if (candDump_ && pass)  // dump: merged gate passer
                      alpaka::atomicAdd(acc, &shDumpPassM, 1u, alpaka::hierarchy::Blocks{});
                    // This lane's best over all considered merged hits (min score, tie min id).
                    if (candDump_ && ((score < bestAnyChi2) || (score == bestAnyChi2 && int32_t(hitId) < bestAnyId))) {
                      bestAnyChi2 = score;
                      bestAnyId = int32_t(hitId);
                      bestAnyPass = pass ? 1 : 0;
                    }
                    // Record this merged road candidate (id + gate chi2 + pass) for the offline join.
                    if (candDump_ && candMemberBuf_ != nullptr) {
                      const int viM = shWalkSteps - 1;
                      if (viM >= 0 && uint32_t(viM) < uint32_t(maxWalkLayers) && j < maxCandidates) {
                        const uint32_t m = alpaka::atomicAdd(acc, &shDumpMemN, 1u, alpaka::hierarchy::Blocks{});
                        if (m < kExtDumpMaxMembers) {
                          ExtCandMemberRec mr;
                          mr.hitId = int32_t(hitId);
                          mr.chi2 = chi2;
                          mr.round = int16_t(0);
                          mr.pass = int16_t(pass ? 1 : 0);
                          candMemberBuf_[(uint32_t(j) * uint32_t(maxWalkLayers) + uint32_t(viM)) * kExtDumpMaxMembers +
                                         m] = mr;
                        } else {
                          if (candDumpOvf_ != nullptr)
                            alpaka::atomicAdd(acc, &candDumpOvf_[1], 1u, alpaka::hierarchy::Blocks{});
                        }
                      }
                    }
                    const bool isBetter = (score < bestScore) || (score == bestScore && int32_t(hitId) < bestHit);
                    if (isBetter && pass) {
                      bestScore = score;
                      bestChi2 = chi2;
                      bestHit = int32_t(hitId);
                      bestD0 = dR0;
                      bestD1 = dSec;
                      bestD2 = dBend;
                      bestRpp = Rpp;
                      bestRps = Rps;
                      bestRss = Rss;
                      bestRbb = Rbb;
                      bestNRows = nRows;
                      bestDet = detS;
                      bestSecWin = secWin ? 2.f * secHalf : 0.f;
                      bestRh = rh;
                      bestZh = zh;
                      bestArcS = p2.arcS;
                    }
                  }
                }

                // Round 1: raw OT rechits, on the same layer visit, phi window and per-layer
                // linearization coefficients as the merged scan, reached only when round 0 attached
                // nothing here and the OT source is active, so an OT hit adds on a stub-less layer
                // instead of competing with a merged attachment. hitCounter restarts at 0, which only
                // re-partitions lane ownership. The gate is the merged path's, with the same
                // toGlobal(variance) convention.
                if (round == 1) {
                  const auto& otHits = otSource_.otHits;
                  for (auto kk = kl; kk != khh; kk = (kk + 1) % ExtPhiBinner::nbins()) {
                    auto const* __restrict__ pbeg = otSource_.phiBinner->begin(kk + hoff);
                    auto const* __restrict__ pend = otSource_.phiBinner->end(kk + hoff);
                    const uint32_t nBin = uint32_t(pend - pbeg);
                    const uint32_t o0 = (nLanes + lane - (hitCounter % nLanes)) % nLanes;  // first owned in-bin offset
                    hitCounter += nBin;
                    for (uint32_t oo = o0; oo < nBin; oo += nLanes) {
                      const uint32_t o = pbeg[oo];
                      // Local occupancy, counted before the skips, for the same reason as round 0.
                      {
                        const float sOcc = isBarrel ? otHits[o].zGlobal()
                                                    : alpaka::math::sqrt(acc,
                                                                         otHits[o].xGlobal() * otHits[o].xGlobal() +
                                                                             otHits[o].yGlobal() * otHits[o].yGlobal());
                        if (alpaka::math::abs(acc, sOcc - secPredOcc) <= wSecOcc)
                          ++nOccSeen;
                      }
                      if (otSource_.usedInStub[o] || (otSource_.ownership != nullptr && otSource_.ownership[o] != 0u)) {
                        continue;  // stub member or already owned
                      }
                      if (candDump_)  // dump: genuine (post-mask) raw-OT candidate in the phi window
                        alpaka::atomicAdd(acc, &shDumpOccO, 1u, alpaka::hierarchy::Blocks{});
                      if (candDump_)  // hole counter: layer occupancy present
                        alpaka::atomicAdd(acc, &shHoleOcc, 1u, alpaka::hierarchy::Blocks{});
                      const float xh = otHits[o].xGlobal();
                      const float yh = otHits[o].yGlobal();
                      const float zh = otHits[o].zGlobal();
                      const float rh = alpaka::math::sqrt(acc, xh * xh + yh * yh);  // rGlobal not stored in OT SoA
                      const float phiH = alpaka::math::atan2(acc, yh, xh);

                      // Per-hit crossing prediction: the same linearized Taylor path / exact fallback as merged.
                      Prediction p2;
                      if (shUseLin) {
                        const float u = isBarrel ? (rh - shLinRef) : (zh - shLinRef);
                        const float arcLin = shLinArc0 + shLinDarc1 * u;
                        bool okLin = (arcLin > 0.f && arcLin <= kExtenderMaxArcLengthCm);
                        if (isBarrel)
                          okLin = okLin && (rh >= shLinArgLo && rh <= shLinArgHi);
                        if (!okLin)
                          continue;
                        p2.phi = foldPi(shLinPhi0 + shLinDphi1 * u + 0.5f * shLinDphi2 * u * u);
                        p2.secondary = shLinSec0 + shLinDsec1 * u + 0.5f * shLinDsec2 * u * u;
                        p2.arcS = arcLin;
                        p2.valid = true;
                      } else {
                        p2 = isBarrel ? predictOnBarrel(acc, sh.helix(), rh) : predictOnEndcap(acc, sh.helix(), zh);
                        if (!p2.valid)
                          continue;
                      }

                      // Cheap r-phi residual pre-filter (same bound as merged; OT xerr/yerr).
                      const float dPhiPF = foldPi(p2.phi + shDerElossPhi - phiH);
                      if (alpaka::math::abs(acc, dPhiPF) * rh > capRPhiEffMax)
                        continue;  // beyond the window's own half-width: it can only fail the gate

                      const float secH = isBarrel ? zh : rh;
                      // Same road centre as the merged round (same layer visit, same crossing).
                      const float dPhi = foldPi(p2.phi + shDerElossPhi - phiH);
                      const float dSec = p2.secondary + shDerElossSec - secH;
                      const float rh_safe = alpaka::math::max(acc, rh, 1.f);

                      // The hit's own noise R through the OT sensor frame (lower/upper per the hit's
                      // position in the stack); same toGlobal(variance) convention as the merged round.
                      // A raw-OT rechit is a single cluster: two rows, no bend.
                      const float xerr = otHits[o].xerrLocal();
                      const float yerr = otHits[o].yerrLocal();
                      const uint32_t geomIdx = uint32_t(otHits[o].detectorIndex()) - ::phase2PixelTopology::nModulesPix;
                      const bool isUpper = (o >= otSource_.otHitModules.upperSensorStart()[geomIdx]);
                      const auto& frame = isUpper ? otSource_.stackedGeometry.upperSensorFrame()[geomIdx]
                                                  : otSource_.stackedGeometry.lowerSensorFrame()[geomIdx];
                      float ge[6];
                      frame.toGlobal(xerr, 0.f, yerr, ge);
                      const float rh2 = rh_safe * rh_safe;
                      const float gxx = ge[0], gxy = ge[1], gyy = ge[2], gxz = ge[3], gyz = ge[4], gzz = ge[5];
                      const float Rpp = (yh * yh * gxx - 2.f * xh * yh * gxy + xh * xh * gyy) / rh2;
                      const float Rss = isBarrel ? gzz : (xh * xh * gxx + 2.f * xh * yh * gxy + yh * yh * gyy) / rh2;
                      const float Rps = isBarrel ? ((-yh * gxz + xh * gyz) / rh_safe)
                                                 : ((-yh * xh * gxx + (xh * xh - yh * yh) * gxy + xh * yh * gyy) / rh2);
                      if (!(Rpp > 0.f) || !(Rss > 0.f))
                        continue;
                      const float dR0 = dPhi * rh_safe;
                      // Same row decision as the merged round: a lone 2S cluster on a disc measures
                      // r-phi and nothing else, so it is judged on that one row inside the window.
                      const bool secWin = extSecIsWindow(Rss, shM[3], qGate1);
                      const float secHalf = extSecSupport(acc, Rss, shM[3], qGate1);
                      const int nRows = secWin ? 1 : 2;
                      float Sm[6] = {shM[0] + Rpp, 0.f, 0.f, 0.f, 0.f, 0.f};
                      float dv[3] = {dR0, 0.f, 0.f};
                      if (!secWin) {
                        Sm[1] = shM[1] + Rps;
                        Sm[3] = shM[3] + Rss;
                        dv[1] = dSec;
                      }
                      float detS = 0.f;
                      const float chi2 = extChi2FromS(acc, nRows, Sm, dv, detS);
                      if (!(chi2 >= 0.f))
                        continue;
                      const float score =
                          -alpaka::math::log(acc, alpaka::math::max(acc, extChi2Tail(acc, nRows, chi2), 1e-30f));
                      if (candDump_)  // dump: min gate chi2 over considered raw-OT hits (best fail)
                        alpaka::atomicMin(acc,
                                          &shDumpBestM,
                                          chi2 >= 4.29e6f ? 0xFFFFFFFEu : uint32_t(chi2 * 1000.f + 0.5f),
                                          alpaka::hierarchy::Blocks{});
                      const bool gatePass = (chi2 < (nRows == 1 ? qGate1 : qGate2));
                      const bool okR = alpaka::math::abs(acc, dR0) < shDerCeilR;
                      const bool okS =
                          alpaka::math::abs(acc, dSec) < shDerCeilS && alpaka::math::abs(acc, dSec) < secHalf;
                      if (gatePass && !okR)
                        alpaka::atomicAdd(acc, &stats[kStatDerCapR], 1u, alpaka::hierarchy::Grids{});
                      if (gatePass && !okS)
                        alpaka::atomicAdd(acc, &stats[kStatDerCapS], 1u, alpaka::hierarchy::Grids{});
                      const bool pass = gatePass && okR && okS;
                      const int32_t candId = int32_t((uint32_t(o) | caOTHitTag::kOTHitTag));
                      if (candDump_ && pass)
                        alpaka::atomicAdd(acc, &shDumpPassO, 1u, alpaka::hierarchy::Blocks{});
                      if (candDump_ && ((score < bestAnyChi2) || (score == bestAnyChi2 && candId < bestAnyId))) {
                        bestAnyChi2 = score;
                        bestAnyId = candId;
                        bestAnyPass = pass ? 1 : 0;
                      }
                      if (candDump_ && candMemberBuf_ != nullptr) {
                        const int viM = shWalkSteps - 1;
                        if (viM >= 0 && uint32_t(viM) < uint32_t(maxWalkLayers) && j < maxCandidates) {
                          const uint32_t m = alpaka::atomicAdd(acc, &shDumpMemN, 1u, alpaka::hierarchy::Blocks{});
                          if (m < kExtDumpMaxMembers) {
                            ExtCandMemberRec mr;
                            mr.hitId = candId;  // bit30-tagged raw-OT id
                            mr.chi2 = chi2;
                            mr.round = int16_t(1);
                            mr.pass = int16_t(pass ? 1 : 0);
                            candMemberBuf_[(uint32_t(j) * uint32_t(maxWalkLayers) + uint32_t(viM)) * kExtDumpMaxMembers +
                                           m] = mr;
                          } else {
                            alpaka::atomicAdd(acc, &candDumpOvf_[1], 1u, alpaka::hierarchy::Grids{});
                          }
                        }
                      }
                      const bool isBetter = (score < bestScore) || (score == bestScore && candId < bestHit);
                      if (isBetter && pass) {
                        bestScore = score;
                        bestChi2 = chi2;
                        bestHit = candId;
                        bestD0 = dR0;
                        bestD1 = dSec;
                        bestD2 = 0.f;
                        bestRpp = Rpp;
                        bestRps = Rps;
                        bestRss = Rss;
                        bestRbb = 0.f;
                        bestNRows = nRows;
                        bestDet = detS;
                        bestSecWin = secWin ? 2.f * secHalf : 0.f;
                        bestRh = rh;
                        bestZh = zh;
                        bestArcS = p2.arcS;
                      }
                    }
                  }
                }
              }
              laneOcc[lane] = nOccSeen;
              laneChi2[lane] = bestScore;  // the reduce ranks on the tail probability, not on chi2
              laneHit[lane] = bestHit;
              // Min-combine this round's best-any into the lane's persistent slot (initialized to
              // the sentinel at layer-select; round 0 then round 1 accumulate). Each lane owns its slot -> no
              // race, no atomic. Tie by min id (an invalid stored slot, id -1, always loses to a valid id).
              if (candDump_ && bestAnyId >= 0 &&
                  ((bestAnyChi2 < laneBestAnyChi2[lane]) ||
                   (bestAnyChi2 == laneBestAnyChi2[lane] &&
                    (laneBestAnyHit[lane] < 0 || bestAnyId < laneBestAnyHit[lane])))) {
                laneBestAnyChi2[lane] = bestAnyChi2;
                laneBestAnyHit[lane] = bestAnyId;
                laneBestAnyPass[lane] = bestAnyPass;
              }
              laneWin[lane] = int(lane);  // seed the argmin tree's carried lane index
              laneChi2Val[lane] = bestChi2;
              laneD0[lane] = bestD0;
              laneD1[lane] = bestD1;
              laneD2[lane] = bestD2;
              laneRpp[lane] = bestRpp;
              laneRps[lane] = bestRps;
              laneRss[lane] = bestRss;
              laneRbb[lane] = bestRbb;
              laneNRows[lane] = bestNRows;
              laneDet[lane] = bestDet;
              laneSecWin[lane] = bestSecWin;
              laneRh[lane] = bestRh;
              laneZh[lane] = bestZh;
              laneArcS[lane] = bestArcS;
            }
            alpaka::syncBlockThreads(acc);

            // In-place shared-memory tree reduction over the per-lane bests. The key (chi2, then
            // hitId) is a strict total order, each lane's best id being distinct since merged hitIds
            // and bit30-tagged OT candIds never collide, so any reduction order yields the same
            // winner on every backend. laneWin carries the winner's original lane so lane 0 can index
            // its KF payload; the reduce touches only laneChi2/laneHit/laneWin. No threshold needs
            // pre-loading: an accepted lane always has laneChi2 < baseChi2Cut, isBetter requiring
            // chi2 < bestChi2 with bestChi2 seeded at the cut.
            for (uint32_t stride = nLanes >> 1; stride > 0u; stride >>= 1) {
              for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
                const uint32_t lane = element.local;
                if (lane < stride) {
                  const uint32_t o = lane + stride;
                  laneOcc[lane] += laneOcc[o];  // the local-occupancy count rides the same tree
                  const bool takeO =
                      (laneHit[lane] < 0) ||
                      (laneHit[o] >= 0 &&
                       (laneChi2[o] < laneChi2[lane] || (laneChi2[o] == laneChi2[lane] && laneHit[o] < laneHit[lane])));
                  if (takeO) {
                    laneChi2[lane] = laneChi2[o];
                    laneHit[lane] = laneHit[o];
                    laneWin[lane] = laneWin[o];
                  }
                }
              }
              alpaka::syncBlockThreads(acc);
            }

            // lane 0 records the extra + applies the Kalman update to the shared running helix, reading
            // the reduced global best (gHit / gChi2 / gLane) from slot 0.
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              if (element.local != 0u)
                continue;
              const int L = shCurrentL;
              const bool isBarrel = caLayers.isBarrel()[L];
              int32_t gHit = laneHit[0];        // -1 when no lane passed the gate
              const float gChi2 = laneChi2[0];  // winner chi2 (used only when gHit >= 0)
              // A gate-passer existed on this layer iff the argmin winner exists (laneHit[0] >= 0, read
              // before any cap mutation below). or-ed across both rounds, so shChainHasPass equals the
              // dump's nPassM + nPassOT > 0. Independent of the cap and the veto, because a continuation
              // is about a passer existing, not about it being committed. Maintained under candDump_ so
              // the hole counter (rejGate == occupancy && !pass) can consume it.
              if (candDump_ && laneHit[0] >= 0)
                shChainHasPass = 1;
              // The hole hypothesis, "attach nothing on this layer", competes in the argmin: the
              // winner is committed only if it beats it. Its price comes from the measured per-layer
              // stub availability eta_L, the measured stub areal density rho and the window volume,
              //     chi2_hole = 2 ln[ P / ((1 - eta_L eps) nu) ],  nu = rho (2 pi) sqrt(|R|),
              // with the numerator P set at the layer site above and |R| = sigma_R^2 sigma_S^2 for
              // this 2-coordinate statistic. It is a competing hypothesis rather than a cut on the
              // hit, and introduces no tuned constant: the same eps and two measured detector tables.
              if (shDerLayOn && gHit >= 0 && (shDerHoleK > -1e29f || shDerHoleKRaw > -1e29f || shDerHoleK3 > -1e29f)) {
                alpaka::atomicAdd(acc, &stats[kStatDerHoleCand], 1u, alpaka::hierarchy::Grids{});
                // raw-channel monitor: the round-1 projection of the same counter. The raw round's own
                // decline rate is what says whether the hole actually arbitrates that channel, and it
                // must be readable without a truth dump.
                if (round == 1)
                  alpaka::atomicAdd(acc, &stats[kStatDerHoleCandRaw], 1u, alpaka::hierarchy::Grids{});
                const int wLane = laneWin[0];
                // |S| is the volume of the statistic the winner was actually judged with, and the
                // density must live in that same space: a stub winner carries the bend row (3 dof,
                // rho_3 [cm^-1 rad^-1]), a raw-OT or non-stub winner does not (2 dof, rho_A [cm^-2]).
                // Mixing the two is a units error, not a rounding one.
                const float detW = (wLane >= 0) ? laneDet[wLane] : 0.f;
                const bool win3 = (wLane >= 0) && (laneRbb[wLane] > 0.f);
                // When the winner's secondary row is a window rather than a chi2 row, its acceptance
                // region is that window times the Gaussian ball of the rows that remain, so the
                // background count nu loses one (2 pi)^{1/2} and gains the window's width. nu is then
                // literally the expected number of random clusters of this layer compatible with the
                // candidate, and the hole wins as soon as it passes one.
                const float secWinW = (wLane >= 0) ? laneSecWin[wLane] : 0.f;
                // A round-1 winner is a raw cluster on a layer whose stub round came up empty, so it
                // is priced with the raw round's own conditional availability and per-event density.
                const float hole2 = (round == 1 && shDerHoleKRaw > -1e29f) ? shDerHoleKRaw : shDerHoleK;
                float holeK = win3 ? shDerHoleK3 : hole2;
                // Price the hole with the occupancy the scan measured here: laneOcc[0] is the patch count summed over the
                // lanes and shOccNorm the count the same patch would hold if this layer's clusters were spread uniformly,
                // so their product is the local contrast and the constant shifts by -2 ln of it. Only the areal part of the
                // 3-dof density moves (the bend dimension is a property of the stub). The floor at one only guards a patch
                // clipped by the layer's envelope.
                if (holeK > -1e29f && shOccNorm > 0.f) {
                  const uint32_t nLoc = (laneOcc[0] > 0u) ? laneOcc[0] : 1u;
                  holeK -= 2.f * alpaka::math::log(acc, float(nLoc) * shOccNorm);
                }
                float lnVol = 0.f;
                if (secWinW > 0.f && holeK > -1e29f) {
                  constexpr float kLnTwoPi = 1.8378771f;
                  holeK += kLnTwoPi;
                  lnVol = 2.f * alpaka::math::log(acc, secWinW);
                }
                if (detW > 0.f && holeK > -1e29f) {
                  const float chi2Hole = holeK - alpaka::math::log(acc, detW) - lnVol;
                  // Both sides are -2 ln(likelihood): the winner's own gate chi2 against the
                  // no-detection weight. gChi2 here is the reduce's ranking key (-ln tail), so the
                  // comparison is made on the chi2 the winner actually scored.
                  const float winChi2 = (wLane >= 0) ? laneChi2Val[wLane] : 0.f;
                  if (winChi2 >= chi2Hole) {
                    gHit = -1;  // the hole wins: this layer contributes no measurement
                    alpaka::atomicAdd(acc, &stats[kStatDerHoleFire], 1u, alpaka::hierarchy::Grids{});
                    if (round == 1)
                      alpaka::atomicAdd(acc, &stats[kStatDerHoleFireRaw], 1u, alpaka::hierarchy::Grids{});
                  }
                }
              }
              // Matching bound: this winner resolves to gExtraClusters output clusters (a merged 2-hit
              // stub -> 2, a raw-OT or pixel winner -> 1). Over the budget it is dropped exactly as if
              // the layer had no hit, so in round 0 round 1 may still try a one-cluster OT hit.
              int gExtraClusters = 0;
              if (gHit >= 0) {
                gExtraClusters = isOTId(uint32_t(gHit)) ? 1 : (isStub(hits, gHit) ? 2 : 1);
                if (shNExtraClusters + gExtraClusters > shExtraClusterCap) {
                  gHit = -1;
                  if (candDump_)
                    shDumpCapDropped = 1;
                }
              }
              const int gLane = (gHit >= 0) ? laneWin[0] : -1;
              if (round == 0)
                shMergedHit = (gHit >= 0) ? 1 : 0;  // gate round 1: OT scans only where the merged round missed
              if (gHit >= 0) {
                extrasIds[j * maxExtraHitsPerTrack + shNExtra] = uint32_t(gHit);
                extrasChi2[j * maxExtraHitsPerTrack + shNExtra] = gChi2;
                ++shNExtra;
                shNExtraClusters += gExtraClusters;
                if (candDump_) {  // dump: capture this layer's committed winner (round 0 or 1)
                  shDumpWinHit = gHit;
                  shDumpWinChi2 = gChi2;
                  shDumpWinRound = round;
                }
                // Tally OT-layer (CA >= 28) accepts -- the anchor count the cluster-cap exemption reads
                // on subsequent layers. Written here, after the per-accept guard, so at that guard it
                if (L >= 28)
                  ++shNOTExtraAcc;
                // Diagnostic: split walk-committed extras by impurity layer class (TOB1-3 / TOB4-6 /
                // TID). Same cheap-atomic pattern as kDiagWalk*; read back only in the verbose summary.
                if (L >= 28 && L <= 30)
                  alpaka::atomicAdd(acc, &stats[kStatExtraTOB13], 1u, alpaka::hierarchy::Grids{});
                else if (L >= 31 && L <= 33)
                  alpaka::atomicAdd(acc, &stats[kStatExtraTOB456], 1u, alpaka::hierarchy::Grids{});
                else if (L >= 34 && L <= 53)
                  alpaka::atomicAdd(acc, &stats[kStatExtraTID], 1u, alpaka::hierarchy::Grids{});
                // raw-channel monitor: the same split restricted to round 1, i.e. the single-cluster
                // channel's own per-class yield -- the class counters above are round-blind and the round
                // counters below are class-blind, so neither alone reads it. Round-1 commits are a small
                // subset (round 1 runs only where round 0 attached nothing), so the merged round's hot
                // path is untouched.
                if (round == 1) {
                  if (L >= 28 && L <= 30)
                    alpaka::atomicAdd(acc, &stats[kStatRawTOB13], 1u, alpaka::hierarchy::Grids{});
                  else if (L >= 31 && L <= 33)
                    alpaka::atomicAdd(acc, &stats[kStatRawTOB456], 1u, alpaka::hierarchy::Grids{});
                  else if (L >= 34 && L <= 53)
                    alpaka::atomicAdd(acc, &stats[kStatRawTID], 1u, alpaka::hierarchy::Grids{});
                }
                // Split walk-committed extras by source (round 0 = merged, round 1 = raw OT).
                if (round == 1) {
                  alpaka::atomicAdd(acc, &stats[kDiagWalkOT], 1u, alpaka::hierarchy::Grids{});
                  shHasOT = 1;
                } else {
                  alpaka::atomicAdd(acc, &stats[kDiagWalkMerged], 1u, alpaka::hierarchy::Grids{});
                }
                // The measurement update. The traversed gap's process noise goes into P first --
                // exactly the Q the gate was computed with, so the two never disagree and the gap is
                // counted once, in the state and never in R -- then the winner's rows update the
                // state. The rows are the per-visit ones (shHphi/shHsec/shHb); R is the hit's own.
                sh.addKinkNoise(shQcPhi, shQcCot, shGapW, shGapS1, shGapS2);
                if (shDerQgap > 0.f) {
                  // The last FITTED gap's exit-direction kink is structurally invisible to the fit
                  // (varBeta(n-1) == 0). It sits at the anchor arc and enters the state once, here.
                  const float cot2a = sh.cotTheta * sh.cotTheta;
                  const float sk = shLastArcS;
                  sh.addKinkNoise(
                      shDerQgap * (1.f + cot2a), shDerQgap * (1.f + cot2a) * (1.f + cot2a), 1.f, sk, sk * sk);
                  shDerQgap = 0.f;
                }
                {
                  // The update sees exactly the rows the gate did: a windowed secondary is not a
                  // Gaussian measurement, so it does not update the state either.
                  const int nR = laneNRows[gLane];
                  const bool hasBend = laneRbb[gLane] > 0.f;
                  const bool secIn = (laneSecWin[gLane] <= 0.f);
                  float Hm[3][5] = {{0.f}};
                  float dm[3] = {laneD0[gLane], 0.f, 0.f};
                  float Rm[3][3] = {{laneRpp[gLane], 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
                  for (int q = 0; q < 5; ++q)
                    Hm[0][q] = shHphi[q];
                  int r = 1;
                  if (secIn) {
                    for (int q = 0; q < 5; ++q)
                      Hm[1][q] = shHsec[q];
                    dm[1] = laneD1[gLane];
                    Rm[1][1] = laneRss[gLane];
                    Rm[0][1] = Rm[1][0] = laneRps[gLane];
                    r = 2;
                  }
                  if (hasBend) {
                    for (int q = 0; q < 5; ++q)
                      Hm[r][q] = shHb[q];
                    dm[r] = laneD2[gLane];
                    Rm[r][r] = laneRbb[gLane];
                  }
                  sh.updateState5(acc, nR, Hm, dm, Rm, true);
                }
                sh.recomputeHelix(acc, shSegBf);
                // The energy-loss centre re-anchors by its own recursion on the same gap moments the process noise used:
                //   u      <- u + u'*ds - (dkappa*ds^2 + K*S2)/2
                //   u'     <- u' - (dkappa*ds + K*S1)
                //   dkappa <- dkappa + K*W
                // It is not reset by the measurement update: the running helix tracks the constant-kappa_0 reference the
                // fit published, and these three carry the true track's growing offset from it.
                if (shEK > 0.f) {
                  const float dsA = laneArcS[gLane] - shLastArcS;
                  const float uN = shEU + shEUp * dsA - 0.5f * (shEDk * dsA * dsA + shEK * shElS2);
                  const float upN = shEUp - (shEDk * dsA + shEK * shElS1);
                  shEDk += shEK * shGapW;
                  shEU = uN;
                  shEUp = upN;
                }
                shLastArcS = laneArcS[gLane];
                shLastR = laneRh[gLane];
                shLastZ = laneZh[gLane];
                // Both-sensors attach on stub-less layers (round 1 only, lane-0-serial): the round-1 winner sits on one
                // sensor of an OT stack and the killed doublet's partner rechit usually lives on the other sensor of the
                // same module, so that sensor range is scanned against the updated helix with the same OT gate and the
                // best passing hit is accepted as a second extra. It needs one more cluster of room under the matching bound.
                if (round == 1 && shNExtra < maxExtraHitsPerTrack && shNExtraClusters < shExtraClusterCap &&
                    shDerLayOn) {
                  const auto& otHits = otSource_.otHits;
                  const uint32_t oWin = otIdx(uint32_t(gHit));
                  const uint32_t geomIdx = uint32_t(otHits[oWin].detectorIndex()) - ::phase2PixelTopology::nModulesPix;
                  const uint32_t upStart = otSource_.otHitModules.upperSensorStart()[geomIdx];
                  const bool winUpper = (oWin >= upStart);
                  // partner = the other sensor range of the same module (ranges from otHitModules)
                  const uint32_t pBeg = winUpper ? otSource_.otHitModules.moduleStart()[geomIdx] : upStart;
                  const uint32_t pEnd = winUpper ? upStart : otSource_.otHitModules.moduleStart()[geomIdx + 1u];
                  // The partner rides the same eps and the same statistic as the primary gate. Its
                  // rows are the visit's, but the state was just updated by the winner, so the
                  // prediction block is rebuilt on the post-update covariance.
                  float Mp[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                  {
                    auto projP = [&](const float* Ha, const float* Hb2) {
                      float v = 0.f;
                      for (int x = 0; x < 5; ++x)
                        for (int y = 0; y < 5; ++y)
                          v += Ha[x] * sh.C[RunningHelix::cIdx(x, y)] * Hb2[y];
                      return v;
                    };
                    Mp[0] = projP(shHphi, shHphi);
                    Mp[1] = projP(shHphi, shHsec);
                    Mp[3] = projP(shHsec, shHsec);
                  }
                  float bestPScore = 3.4e38f;
                  int32_t bestPHit = -1;
                  float bestPD0 = 0.f, bestPD1 = 0.f;
                  float bestPRpp = 0.f, bestPRps = 0.f, bestPRss = 0.f;
                  bool bestPSecIn = true;
                  float bestPRh = 0.f, bestPZh = 0.f, bestPArcS = 0.f;
                  for (uint32_t p = pBeg; p < pEnd; ++p) {
                    if (otSource_.usedInStub[p] || (otSource_.ownership != nullptr && otSource_.ownership[p] != 0u))
                      continue;  // stub member or already owned
                    const float xh = otHits[p].xGlobal();
                    const float yh = otHits[p].yGlobal();
                    const float zh = otHits[p].zGlobal();
                    const float rh = alpaka::math::sqrt(acc, xh * xh + yh * yh);  // rGlobal not stored in OT SoA
                    const float phiH = alpaka::math::atan2(acc, yh, xh);
                    // fresh exact prediction on the updated helix (the per-layer linearization was built
                    // on the pre-winner helix and is now stale, so predict directly at rh/zh).
                    const Prediction p2 =
                        isBarrel ? predictOnBarrel(acc, sh.helix(), rh) : predictOnEndcap(acc, sh.helix(), zh);
                    if (!p2.valid)
                      continue;
                    const float secH = isBarrel ? zh : rh;
                    // The partner sits on the same module as the just-accepted winner, so the road
                    // centre of this layer visit is the offset that applies to it too.
                    const float dPhi = foldPi(p2.phi + shDerElossPhi - phiH);
                    const float dSec = p2.secondary + shDerElossSec - secH;
                    const float rh_safe = alpaka::math::max(acc, rh, 1.f);
                    const float xerr = otHits[p].xerrLocal();
                    const float yerr = otHits[p].yerrLocal();
                    const bool isUpper = (p >= upStart);
                    const auto& frame = isUpper ? otSource_.stackedGeometry.upperSensorFrame()[geomIdx]
                                                : otSource_.stackedGeometry.lowerSensorFrame()[geomIdx];
                    float ge[6];
                    frame.toGlobal(xerr, 0.f, yerr, ge);
                    const float rh2 = rh_safe * rh_safe;
                    const float gxx = ge[0], gxy = ge[1], gyy = ge[2], gxz = ge[3], gyz = ge[4], gzz = ge[5];
                    const float Rpp = (yh * yh * gxx - 2.f * xh * yh * gxy + xh * xh * gyy) / rh2;
                    const float Rss = isBarrel ? gzz : (xh * xh * gxx + 2.f * xh * yh * gxy + yh * yh * gyy) / rh2;
                    const float Rps = isBarrel ? ((-yh * gxz + xh * gyz) / rh_safe)
                                               : ((-yh * xh * gxx + (xh * xh - yh * yh) * gxy + xh * yh * gyy) / rh2);
                    if (!(Rpp > 0.f) || !(Rss > 0.f))
                      continue;
                    const float dR0 = dPhi * rh_safe;
                    // Same row decision as the primary gate, on the post-update prediction block.
                    const bool secWin = extSecIsWindow(Rss, Mp[3], qGate1);
                    const float secHalf = extSecSupport(acc, Rss, Mp[3], qGate1);
                    const int nRows = secWin ? 1 : 2;
                    float Sm[6] = {Mp[0] + Rpp, 0.f, 0.f, 0.f, 0.f, 0.f};
                    float dv[3] = {dR0, 0.f, 0.f};
                    if (!secWin) {
                      Sm[1] = Mp[1] + Rps;
                      Sm[3] = Mp[3] + Rss;
                      dv[1] = dSec;
                    }
                    float detS = 0.f;
                    const float chi2 = extChi2FromS(acc, nRows, Sm, dv, detS);
                    if (!(chi2 >= 0.f) || chi2 >= (nRows == 1 ? qGate1 : qGate2))
                      continue;
                    if (!(alpaka::math::abs(acc, dR0) < shDerCeilR) || !(alpaka::math::abs(acc, dSec) < shDerCeilS) ||
                        !(alpaka::math::abs(acc, dSec) < secHalf))
                      continue;
                    const float score =
                        -alpaka::math::log(acc, alpaka::math::max(acc, extChi2Tail(acc, nRows, chi2), 1e-30f));
                    const int32_t cid = int32_t((uint32_t(p) | caOTHitTag::kOTHitTag));
                    if ((score < bestPScore) || (score == bestPScore && cid < bestPHit)) {
                      bestPScore = score;
                      bestPHit = cid;
                      bestPD0 = dR0;
                      bestPD1 = dSec;
                      bestPRpp = Rpp;
                      bestPRps = Rps;
                      bestPRss = Rss;
                      bestPSecIn = !secWin;
                      bestPRh = rh;
                      bestPZh = zh;
                      bestPArcS = p2.arcS;
                    }
                  }
                  if (bestPHit >= 0) {
                    extrasIds[j * maxExtraHitsPerTrack + shNExtra] = uint32_t(bestPHit);
                    extrasChi2[j * maxExtraHitsPerTrack + shNExtra] = bestPScore;
                    ++shNExtra;
                    ++shNExtraClusters;
                    ++shNOTExtraAcc;
                    // Same-crossing second sensor: no new gap, so no process noise is injected here.
                    float Hm[3][5] = {{0.f}};
                    for (int q = 0; q < 5; ++q) {
                      Hm[0][q] = shHphi[q];
                      Hm[1][q] = bestPSecIn ? shHsec[q] : 0.f;
                    }
                    const float dm[3] = {bestPD0, bestPSecIn ? bestPD1 : 0.f, 0.f};
                    float Rm[3][3] = {{bestPRpp, bestPSecIn ? bestPRps : 0.f, 0.f},
                                      {bestPSecIn ? bestPRps : 0.f, bestPRss, 0.f},
                                      {0.f, 0.f, 0.f}};
                    sh.updateState5(acc, bestPSecIn ? 2 : 1, Hm, dm, Rm, true);
                    sh.recomputeHelix(acc, shSegBf);
                    shLastArcS = bestPArcS;
                    shLastR = bestPRh;
                    shLastZ = bestPZh;
                    alpaka::atomicAdd(acc, &stats[kStatOTPartner], 1u, alpaka::hierarchy::Grids{});
                    if (candDump_)  // dump: a stack-partner second raw-OT extra was attached here
                      shDumpPartner = 1;
                  }
                }
              }
            }
            alpaka::syncBlockThreads(acc);
          }  // for round (0 = merged scan, 1 = OT-only additive scan)
          // Hole counter: update the cumulative/consecutive occupancy-gated hole run from this
          // layer's outcome (lane-0; shHoleOcc summed by scan lanes is visible after the round-loop sync
          // above, shChainHasPass written by lane-0). A hole == rejGate (occupancy present, no gate-passer);
          // rejEmpty (no occupancy) and rejCap (a passer existed) are not holes. A gate-passer breaks the
          // consecutive run. Runs before the dump record write so holeRunToHere reflects this layer,
          // and only under candDump_.
          if (candDump_) {
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              if (element.local != 0u)
                continue;
              const bool hole = (shHoleOcc > 0u) && (shChainHasPass == 0);
              if (hole) {
                ++shHoleRun;
                ++shHoleConsec;
              } else if (shChainHasPass != 0) {
                shHoleConsec = 0;  // a gate-passer was found: reset the consecutive-hole run
              }
            }
          }
          // Candidate dump: record this (candidate, visited layer) trace for all layer families.
          // Lane 0 writes one ExtCandLayerRec at
          // the fixed index j * maxWalkLayers + vi (vi = 0-based visit index). candDump_ is uniform
          // across the block, so the trailing sync is reached by every lane.
          if (candDump_) {
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              if (element.local != 0u)
                continue;
              const int vi = shWalkSteps - 1;  // this scanned layer's 0-based visit index
              if (vi >= 0 && uint32_t(vi) < uint32_t(maxWalkLayers) && j < maxCandidates) {
                ExtCandLayerRec r;
                r.trackId = int32_t(i);
                r.layerId = shCurrentL;
                r.nWinMerged = int32_t(shDumpOccM);
                r.nWinOT = int32_t(shDumpOccO);
                r.nPassMerged = int32_t(shDumpPassM);
                r.nPassOT = int32_t(shDumpPassO);
                r.winnerHitId = shDumpWinHit;
                r.round = shDumpWinRound;
                r.winnerChi2 = shDumpWinChi2;
                r.bestFailChi2 = (shDumpBestM == 0xFFFFFFFFu) ? -1.f : float(shDumpBestM) * 1e-3f;
                // Lane-0 serial reduction over the per-lane best-any slots (min chi2, tie by min id)
                // -> the overall-best (min-chi2) road candidate over all considered hits in both rounds.
                // The winner tree-reduce above does not touch laneBestAny*. bAnyHit carries the
                // bit30-tagged id (-1 = none considered), bAnyPass its base-gate pass; bestFailChi2 (from
                // shDumpBestM) is that same candidate's chi2 by construction.
                float bAnyChi2 = 3.4e38f;
                int32_t bAnyHit = -1;
                int bAnyPass = 0;
                for (uint32_t l = 0; l < nLanes; ++l) {
                  const int32_t h = laneBestAnyHit[l];
                  if (h < 0)
                    continue;
                  const float c = laneBestAnyChi2[l];
                  if (bAnyHit < 0 || c < bAnyChi2 || (c == bAnyChi2 && h < bAnyHit)) {
                    bAnyChi2 = c;
                    bAnyHit = h;
                    bAnyPass = laneBestAnyPass[l];
                  }
                }
                r.bestHitId = bAnyHit;        // already bit30-tagged if raw-OT; -1 = none considered
                r.holeRunToHere = shHoleRun;  // cumulative occupancy-gated holes through this layer
                int32_t flags = 0;
                if (shDumpVeto)
                  flags |= kExtDumpFlagVetoSkip;
                if (shDumpPartner)
                  flags |= kExtDumpFlagPartner;
                if (bAnyHit >= 0 && bAnyPass)
                  flags |= kExtDumpFlagBestPass;  // the min-chi2 road candidate cleared the base gate
                r.flags = flags;
                if (shDumpWinHit >= 0)
                  r.outcome = kExtDumpAccept;
                else if (shDumpCapDropped)
                  r.outcome = kExtDumpRejCap;
                else if (shDumpOccM + shDumpOccO == 0u)
                  r.outcome = kExtDumpRejEmpty;
                else if (shDumpPassM + shDumpPassO == 0u)
                  r.outcome = kExtDumpRejGate;
                else
                  r.outcome = kExtDumpRejOther;
                candLayerBuf_[uint32_t(j) * uint32_t(maxWalkLayers) + uint32_t(vi)] = r;
              } else if (candDumpOvf_ != nullptr) {  // count-and-clamp: visit index out of the sized stride
                alpaka::atomicAdd(acc, &candDumpOvf_[0], 1u, alpaka::hierarchy::Grids{});
              }
            }
            alpaka::syncBlockThreads(acc);
          }
        }  // while (walk)

        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          if (element.local == 0u) {
            nExtras[j] = shNExtra;
          }
        }
        alpaka::syncBlockThreads(acc);  // finish this candidate before shared memory is reused
      }  // for (candidate j)
    }
  };

  // Cross-track arbitration, init phase: set the atomicMin identity on exactly the claim
  // rows the claim/resolve pair will touch. Same loop shape, same (j, k) index space and the same
  // claimIdx expression as Kernel_extClaimExtras below, so the set of rows initialised here is exactly
  // the set of rows read there and in Kernel_extResolveExtras. Two candidates proposing the same hit
  // write the same value to the same row, so the duplicate writes are harmless.
  struct Kernel_extInitClaims {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const uint32_t nHits,  // OT claims live at nHits + otIdx (0 => merged-only)
                                  const uint32_t* __restrict__ extrasIds,
                                  const int32_t* __restrict__ nExtras,
                                  uint64_t* __restrict__ hitClaims) const {
      const uint64_t kUnclaimed = ~uint64_t(0);  // 0xff..ff, the unclaimed identity
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto idx : cms::alpakatools::uniform_elements(acc, nC * uint32_t(maxExtraHitsPerTrack))) {
        const uint32_t j = idx / maxExtraHitsPerTrack;
        const int k = int(idx % maxExtraHitsPerTrack);
        if (k >= nExtras[j])
          continue;
        const uint32_t id = extrasIds[j * maxExtraHitsPerTrack + k];
        const uint32_t claimIdx = isOTId(id) ? nHits + otIdx(id) : id;
        hitClaims[claimIdx] = kUnclaimed;
      }
    }
  };

  // Cross-track arbitration, claim phase: every accepted extra bids for its hit with its gate chi2.
  struct Kernel_extClaimExtras {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const uint32_t* __restrict__ candList,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const uint32_t nHits,  // OT claims live at nHits + otIdx (0 => merged-only)
                                  const uint32_t* __restrict__ extrasIds,
                                  const float* __restrict__ extrasChi2,
                                  const int32_t* __restrict__ nExtras,
                                  uint64_t* __restrict__ hitClaims) const {
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto idx : cms::alpakatools::uniform_elements(acc, nC * uint32_t(maxExtraHitsPerTrack))) {
        const uint32_t j = idx / maxExtraHitsPerTrack;
        const int k = int(idx % maxExtraHitsPerTrack);
        if (k >= nExtras[j])
          continue;
        const uint32_t id = extrasIds[j * maxExtraHitsPerTrack + k];
        const uint32_t claimIdx = isOTId(id) ? nHits + otIdx(id) : id;
        const uint64_t claim = packClaim(extrasChi2[j * maxExtraHitsPerTrack + k], candList[j]);
        // Exclusive ownership: one claim slot per hit, taken by the best claimant. The key is the
        // walk's own ranking score (-ln tail probability), so the hit goes to the track it fits best;
        // a track that loses a hit here simply does not get it, and the duplicate removal downstream
        // decides whether the two tracks are the same particle.
        alpaka::atomicMin(acc, &hitClaims[claimIdx], claim, alpaka::hierarchy::Grids{});
      }
    }
  };

  // Cross-track arbitration, resolve phase: keep only the extras whose claim won; compact each
  // candidate's list in place and accumulate the per-event counters.
  struct Kernel_extResolveExtras {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const uint32_t* __restrict__ candList,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const uint32_t nHits,  // OT claims live at nHits + otIdx (0 => merged-only)
                                  uint32_t* __restrict__ extrasIds,
                                  float* __restrict__ extrasChi2,
                                  int32_t* __restrict__ nExtras,
                                  const uint64_t* __restrict__ hitClaims,
                                  uint32_t* __restrict__ stats) const {
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto j : cms::alpakatools::uniform_elements(acc, nC)) {
        const uint32_t tupleId = candList[j];
        const int n = nExtras[j];
        int kept = 0;
        for (int k = 0; k < n; ++k) {
          const uint32_t hitId = extrasIds[j * maxExtraHitsPerTrack + k];
          const uint32_t claimIdx = isOTId(hitId) ? nHits + otIdx(hitId) : hitId;
          const bool won = (uint32_t(hitClaims[claimIdx] & 0xffffffffu) == tupleId);
          if (won) {
            extrasIds[j * maxExtraHitsPerTrack + kept] = hitId;
            extrasChi2[j * maxExtraHitsPerTrack + kept] = extrasChi2[j * maxExtraHitsPerTrack + k];
            // Keep the chain score aligned with extrasIds through the arbitration compaction.
            ++kept;
          } else {
            // split arbitration losses by source (merged vs raw OT)
            alpaka::atomicAdd(
                acc, &stats[isOTId(hitId) ? kDiagArbLostOT : kDiagArbLostMerged], 1u, alpaka::hierarchy::Grids{});
          }
        }
        if (kept < n)
          alpaka::atomicAdd(acc, &stats[kStatArbLost], uint32_t(n - kept), alpaka::hierarchy::Grids{});
        nExtras[j] = kept;
        if (kept > 0) {
          alpaka::atomicAdd(acc, &stats[kStatExtended], 1u, alpaka::hierarchy::Grids{});
          alpaka::atomicAdd(acc, &stats[kStatTotalExtras], uint32_t(kept), alpaka::hierarchy::Grids{});
        }
        const int bucket = kept < kStatHistBuckets ? kept : kStatHistBuckets - 1;
        alpaka::atomicAdd(acc, &stats[kStatHistFirst + bucket], 1u, alpaka::hierarchy::Grids{});
      }
    }
  };

  // One-line per-event attach summary (single thread), gated by AttachParams::verbose.
  struct Kernel_extPrintSummary {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, const uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      printf("[CAExtension] cand=%u overflow=%u extended=%u extras=%u arbLost=%u hist:",
             stats[kStatCandidates],
             stats[kStatCandOverflow],
             stats[kStatExtended],
             stats[kStatTotalExtras],
             stats[kStatArbLost]);
      for (int b = 0; b < kStatHistBuckets; ++b)
        printf(" %u", stats[kStatHistFirst + b]);
      printf("\n");
      // Pre-gate skips, per-layer-class walk-committed extras, and the candidate-cap overflow.
      printf("[CAExtension] preGateSkipped=%u etaSkipped=%u extras[TOB1-3=%u TOB4-6=%u TID=%u] candOverflow=%u\n",
             stats[kStatPreGateSkipped],
             stats[kStatPreGateEtaSkipped],
             stats[kStatExtraTOB13],
             stats[kStatExtraTOB456],
             stats[kStatExtraTID],
             stats[kStatCandOverflow]);
      // The derived selection's own alarms. capR/capS count the ball-passers the module-envelope
      // runaway ceilings rejected; divided by derLayers (the layer visits the derived gate armed on)
      // that is the ceilings' binding rate -- << 1 % means the ceiling is a guard, anything more means
      // it is silently doing the selection and the delivered efficiency is not the stated eps. The
      // hole* counters stay 0 while the hypothesis is gated to d >= 3 (kExtDerHoleMinDim). All zero
      // unless the derived package is armed, and printed only under verbose.
      printf("[CAExtension] derived: layers=%u capR=%u capS=%u holeCand=%u holeFire=%u hostOn=%u hostOff=%u\n",
             stats[kStatDerLayers],
             stats[kStatDerCapR],
             stats[kStatDerCapS],
             stats[kStatDerHoleCand],
             stats[kStatDerHoleFire],
             stats[kStatDerHostOn],
             stats[kStatDerHostOff]);
      // raw-channel monitor: the round-1 projections of the two lines above -- the single-cluster
      // channel's per-class yield, and the rate at which the per-source-round hole pricing declines it.
      // rawExtras against extras[] above gives the merged round's own split; holeRaw against
      // holeCand/holeFire gives the raw round's own hole-test and decline rates.
      printf("[CAExtension] rawRound: extras[TOB1-3=%u TOB4-6=%u TID=%u] holeCand=%u holeFire=%u\n",
             stats[kStatRawTOB13],
             stats[kStatRawTOB456],
             stats[kStatRawTID],
             stats[kStatDerHoleCandRaw],
             stats[kStatDerHoleFireRaw]);
    }
  };

  AttachBuffers allocateAttachBuffers(Queue& queue,
                                      const AttachParams& params,
                                      uint32_t nHits,
                                      uint32_t nOTHits,
                                      uint32_t maxNumberOfTuples,
                                      ::reco::TrackSoAConstView tracks,
                                      const double* passBuf,
                                      const int32_t* candidateMask,
                                      uint32_t knownCandCapacity) {
    // candCapacity-independent buffers, needed by (or unaffected by) the count pass
    auto nCands = cms::alpakatools::make_device_buffer<uint32_t>(queue);
    auto stats = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kStatSize);
    // Per-hit arbitration claim buffer: merged hits occupy [0, nHits); OT extras (when active) occupy
    // [nHits, nHits + nOTHits). nOTHits == 0 keeps this sized to nHits exactly.
    // N-way sharing (extMaxSharedOwners): each hit keeps N sorted claim slots at
    // [claimIdx*N, claimIdx*N+N); N == 1 reduces to one slot per hit (a single atomicMin winner).
    auto hitClaims = cms::alpakatools::make_device_buffer<uint64_t[]>(queue, std::max<std::size_t>(1, nHits + nOTHits));
    alpaka::memset(queue, nCands, 0);
    alpaka::memset(queue, stats, 0);
    // hitClaims is not memset here: the 0xFF "unclaimed" identity is needed only on the rows the
    // arbitration touches (at most nCands * maxExtraHitsPerTrack, a tiny fraction of the
    // nHits + nOTHits rows this buffer spans). Kernel_extInitClaims writes exactly those rows in
    // launchAttach, immediately before Kernel_extClaimExtras, over the same (candidate, slot) index
    // space the claim and resolve kernels read; rows outside that set are never read.

    // count-only pre-gate pass: nCands (+ candidate/overflow counters), no candList writes
    constexpr auto threadsPerBlock = 256u;
    const auto blocksTuples = cms::alpakatools::divide_up_by(maxNumberOfTuples, threadsPerBlock);
    const auto workDivTuples = cms::alpakatools::make_workdiv<Acc1D>(blocksTuples, threadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDivTuples,
                        Kernel_extPreGate{},
                        tracks,
                        passBuf,
                        maxNumberOfTuples,
                        params.preGateMinPt,
                        std::sinh(params.maxAbsEta),
                        candidateMask,  // restrict the count to the caller's set (null = no restriction)
                        knownCandCapacity,
                        /*candList=*/static_cast<uint32_t*>(nullptr),
                        nCands.data(),
                        stats.data());

    // Size the scratch from the caller's host-known candidate bound. knownCandCapacity > 0 is part
    // of the contract: the count pass above still fills the stats and overflow counters but its
    // result is not read back, so nothing here blocks the host. It need only be an upper bound, the
    // fill pass capping at candCapacity so that no candidate is dropped. extRefitMaxCandidates is an
    // optional ceiling on it, for callers that can only offer the whole track capacity as a bound;
    // the scratch, the refit scaffold and the per-candidate grids all shrink with it.
    assert(knownCandCapacity > 0);
    // The candidate set is a subset of the tuples the pre-gate iterates, so the caller's track
    // capacity IS the structural bound and no ceiling knob is needed: overflow is impossible by
    // construction, and kStatCandOverflow (counted in both passes) reports it if it ever is not.
    const uint32_t candCapacity = std::max(knownCandCapacity, 16u);

    // Reset the counter so the fill pass in launchAttach re-runs the same predicate from scratch.
    alpaka::memset(queue, nCands, 0);

    const auto nSlots = std::size_t(candCapacity) * std::size_t(params.maxExtraHitsPerTrack);
    AttachBuffers bufs{cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, candCapacity)),
                       std::move(nCands),
                       cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, nSlots)),
                       cms::alpakatools::make_device_buffer<float[]>(queue, std::max<std::size_t>(1, nSlots)),
                       cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, candCapacity)),
                       std::move(hitClaims),
                       std::move(stats)};
    bufs.candCapacity = candCapacity;
    // bufs.nExtras needs no memset: Kernel_extFindExtras stores nExtras[j] = shNExtra unconditionally
    // for every candidate group j of its uniform_groups(acc, nC * nLanes) loop, and it is the first
    // kernel of launchAttach to touch the buffer. Every reader clamps to the same
    // nC = min(*nCands, maxCandidates), so the [nC, candCapacity) tail is never read.
    return bufs;
  }

  // Full-hits OT source. These kernels fill the per-event buffers that describe the raw OT rechit
  // collection -- iphi, the used-in-stub mask, the per-CA-layer row offsets and the phi binner -- which
  // the walk's round 1, its stack-partner scan and the final rewrite then read through OTHitsSource.

  // iphi per OT rechit, using the same convention as the merged SoA (OTRecHitsSoA.h documents
  // iphi = unsafe_atan2s<7>(yGlobal, xGlobal)).
  struct Kernel_otFillIphi {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::OTRecHitsConstView otHits,
                                  int16_t* __restrict__ iphi,
                                  uint32_t nOTHits) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, nOTHits)) {
        const float xg = otHits[i].xGlobal();
        const float yg = otHits[i].yGlobal();
        iphi[i] = unsafe_atan2s<7>(yg, xg);
      }
    }
  };

  // Mark every OT rechit that is a member of some stub (used[lowerHitIdx] = used[upperHitIdx] = 1).
  // Guards the UINT32_MAX sentinel (P-hit-only stubs) and the nOTHits bound.
  struct Kernel_otMarkUsedInStub {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::StubsConstView stubs,
                                  uint8_t* __restrict__ used,
                                  uint32_t nStubs,
                                  uint32_t nOTHits) const {
      for (auto s : cms::alpakatools::uniform_elements(acc, nStubs)) {
        const uint32_t lo = stubs[s].lowerHitIdx();
        const uint32_t up = stubs[s].upperHitIdx();
        if (lo != UINT32_MAX && lo < nOTHits)
          used[lo] = uint8_t(1);
        if (up != UINT32_MAX && up < nOTHits)
          used[up] = uint8_t(1);
      }
    }
  };

  // Per-CA-layer OT-row offsets (mirror of SetHitsLayerStart, but over the OT stacked-module start
  // array). For an OT layer i the start is otHitModules.moduleStart[ll.layerStarts()[i] - nPixelModules]
  // (detectorIndex numbering = nPixelModules + geom index in CA order). Pixel layers (layerStarts <
  // nPixelModules) precede all OT layers, so they get an empty range anchored at the first OT row (0).
  // The final entry (i == nLayers, layerStarts == total modules) resolves to the nOTHits sentinel that
  // fillManyFromVector needs as the upper bound.
  struct Kernel_otSetLayerStart {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::OTHitModuleConstView otHitModules,
                                  ::reco::CALayersSoAConstView ll,
                                  uint32_t nPixelModules,
                                  uint32_t nOTHits,
                                  uint32_t* __restrict__ otHitsLayerStart) const {
      // OTHitModules block is sized nOTModules + 1 (cumulative-sum convention).
      const uint32_t nOTModules = uint32_t(otHitModules.metadata().size()) - 1u;
      for (auto i : cms::alpakatools::uniform_elements(acc, uint32_t(ll.metadata().size()))) {
        const int caMod = ll.layerStarts()[i];
        if (caMod < int(nPixelModules)) {
          otHitsLayerStart[i] = 0u;  // pixel layer: empty OT range at the first OT row
        } else {
          const uint32_t geom = uint32_t(caMod) - nPixelModules;
          otHitsLayerStart[i] = (geom < nOTModules) ? otHitModules.moduleStart()[geom] : nOTHits;
        }
      }
    }
  };

  OTHitsBuffers buildOTHitsSource(Queue& queue,
                                  const AttachParams& params,
                                  ::reco::OTRecHitsConstView otHits,
                                  ::reco::OTHitModuleConstView otHitModules,
                                  ::reco::StackedModuleGeometryConstView stackedGeometry,
                                  ::reco::StubsConstView stubs,
                                  ::reco::CALayersSoAConstView caLayers,
                                  uint32_t nOTHits,
                                  uint32_t nStubs) {
    using namespace cms::alpakatools;
    constexpr uint32_t nPixelModules = ::phase2PixelTopology::nModulesPix;  // 4000; OT detIdx = 4000 + geom
    const uint32_t nLayersP1 = uint32_t(caLayers.metadata().size());        // nLayers + 1

    // Per-event buffers (transient, caching-allocator backed).
    auto iphi = make_device_buffer<int16_t[]>(queue, nOTHits);
    auto usedInStub = make_device_buffer<uint8_t[]>(queue, nOTHits);
    auto layerStart = make_device_buffer<uint32_t[]>(queue, nLayersP1);
    auto phiHist = make_device_buffer<ExtPhiBinner>(queue);
    auto phiStorage = make_device_buffer<ExtPhiBinner::value_type[]>(queue, nOTHits);

    // usedInStub is filled sparsely, so it is zeroed every event. The per-hit `ownership` array is not
    // allocated: nothing writes it, so OTHitsSource::ownership stays null and the walk's two vetoes
    // are guarded on the pointer (a null array masks nothing).
    alpaka::memset(queue, usedInStub, 0);

    constexpr uint32_t kThreads = 256u;
    {
      const auto blocks = divide_up_by(nOTHits, kThreads);
      alpaka::exec<Acc1D>(
          queue, make_workdiv<Acc1D>(blocks, kThreads), Kernel_otFillIphi{}, otHits, iphi.data(), nOTHits);
    }
    if (nStubs > 0) {
      const auto blocks = divide_up_by(nStubs, kThreads);
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(blocks, kThreads),
                          Kernel_otMarkUsedInStub{},
                          stubs,
                          usedInStub.data(),
                          nStubs,
                          nOTHits);
    }
    {
      const auto blocks = divide_up_by(nLayersP1, kThreads);
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(blocks, kThreads),
                          Kernel_otSetLayerStart{},
                          otHitModules,
                          caLayers,
                          nPixelModules,
                          nOTHits,
                          layerStart.data());
    }
    // OT phi binner: same 256-bin, per-CA-layer partition as the merged-hit binner (exactly the
    // CAHitNtupletGeneratorKernels.dev.cc prepareHits() pattern), sized to nOTHits.
    ExtPhiBinner::View phiView{phiHist.data(), nullptr, phiStorage.data(), cms::alpakatools::kDynamicSize, nOTHits};
    fillManyFromVector<Acc1D>(phiHist.data(),
                              phiView,
                              ::pixelTopology::Phase2OTStubs::numberOfLayers,
                              iphi.data(),
                              layerStart.data(),
                              nOTHits,
                              (uint32_t)256,
                              queue);

#ifdef EXT_OTFILL_DIAG
    // Compile-gated one-shot OT-fill sanity dump (first event only): nOTHits, used-in-stub fraction,
    // and per-CA-layer OT bin totals (layerStart diffs == hits binned per layer). Not compiled on a
    // plain build; enable with -DEXT_OTFILL_DIAG to check the OT source wiring and sizing.
    static std::atomic<bool> printedOT{false};
    bool expected = false;
    if (printedOT.compare_exchange_strong(expected, true)) {
      std::vector<uint32_t> hLayer(nLayersP1);
      std::vector<uint8_t> hUsed(nOTHits);
      auto hLayerView = make_host_view(hLayer.data(), nLayersP1);
      auto hUsedView = make_host_view(hUsed.data(), nOTHits);
      alpaka::memcpy(queue, hLayerView, make_device_view(queue, layerStart.data(), nLayersP1));
      alpaka::memcpy(queue, hUsedView, make_device_view(queue, usedInStub.data(), nOTHits));
      alpaka::wait(queue);
      std::size_t nUsed = 0;
      for (uint32_t k = 0; k < nOTHits; ++k)
        nUsed += (hUsed[k] != 0);
      const double frac = nOTHits ? double(nUsed) / double(nOTHits) : 0.;
      printf("[CAExtension][OTfill] nOTHits=%u nStubs=%u usedInStub=%zu (frac=%.4f) perLayerOT:",
             nOTHits,
             nStubs,
             nUsed,
             frac);
      for (uint32_t i = 0; i + 1 < nLayersP1; ++i) {
        const uint32_t cnt = hLayer[i + 1] - hLayer[i];
        if (cnt > 0)
          printf(" L%u:%u", i, cnt);
      }
      printf("\n");
    }
#endif

    return OTHitsBuffers{std::move(iphi),
                         std::move(usedInStub),
                         std::move(layerStart),
                         std::move(phiHist),
                         std::move(phiStorage),
                         otHits,
                         stubs,
                         otHitModules,
                         stackedGeometry,
                         nOTHits};
  }

  void launchExtHostMask(Queue& queue,
                         const AttachParams& params,
                         ::reco::TrackSoAConstView tracks,
                         uint32_t nTracksCap,
                         int32_t* hostMask,
                         ExtPredCoeff* pred) {
    if (nTracksCap == 0u || hostMask == nullptr)
      return;
    constexpr uint32_t bs = 128;
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracksCap, bs), bs),
                        Kernel_extHostMask{},
                        tracks,
                        params.preGateMinPt,
                        params.maxAbsEta,
                        nTracksCap,
                        hostMask,
                        pred);
  }

  void launchAttach(Queue& queue,
                    const AttachParams& params,
                    float bf,
                    const float* rhoMap,
                    const float* bMap,
                    const ExtPhiBinner* phiBinner,
                    ::reco::TrackSoAConstView tracks,
                    ::reco::TrackHitSoAConstView trackHits,
                    caStructures::CAHitsView hits,
                    ::reco::TrackingRecHitsMaskingConstView hitMask,
                    ::reco_extender::ExtenderLayersConstView extLayers,
                    ::reco::CAModulesConstView caModules,
                    const double* passBuf,
                    uint32_t maxNumberOfTuples,
                    uint32_t nHits,
                    AttachBuffers& bufs,
                    const OTHitsSource* otSource,
                    const int32_t* candidateMask) {
    // The full-hits OT source is consumed by the walk's second bin-loop (Kernel_extFindExtras round 1)
    // and its stack-partner scan. Held by value so the kernel object copies it to the device. A default
    // source (nOTHits == 0) => the walk scans merged hits only.
    const bool otActive = (otSource != nullptr && otSource->nOTHits > 0u);
    const OTHitsSource otSrcVal = otActive ? *otSource : OTHitsSource{};
    // Runtime candidate capacity the scratch was sized to (allocateAttachBuffers ran the count pass).
    const uint32_t candCapacity = bufs.candCapacity;

    // The walk's candidate-level instrument (see EXT_CAND_DUMP at the top of this file). Compiled in
    // only when the macro is defined; in a production build kExtCandDump is a compile-time false, so
    // nothing below is allocated, no readback or sync happens, and the walk carries no dump code.
    const bool candDumpOn = kExtCandDump && candCapacity > 0u;
    const uint32_t candDumpStride = uint32_t(params.maxWalkLayers);
    std::optional<cms::alpakatools::device_buffer<Device, ExtCandLayerRec[]>> dumpLayer;
    std::optional<cms::alpakatools::device_buffer<Device, ExtCandHdrRec[]>> dumpHdr;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> dumpOvf;
    // Per-road-candidate member buffer, sized candCapacity*maxWalkLayers*kExtDumpMaxMembers.
    std::optional<cms::alpakatools::device_buffer<Device, ExtCandMemberRec[]>> dumpMember;
    ExtCandLayerRec* dumpLayerPtr = nullptr;
    ExtCandHdrRec* dumpHdrPtr = nullptr;
    uint32_t* dumpOvfPtr = nullptr;
    ExtCandMemberRec* dumpMemberPtr = nullptr;
    if (candDumpOn) {
      dumpLayer.emplace(
          cms::alpakatools::make_device_buffer<ExtCandLayerRec[]>(queue, std::size_t(candCapacity) * candDumpStride));
      dumpHdr.emplace(cms::alpakatools::make_device_buffer<ExtCandHdrRec[]>(queue, candCapacity));
      dumpOvf.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2));  // [0]=visit-index, [1]=member
      dumpMember.emplace(cms::alpakatools::make_device_buffer<ExtCandMemberRec[]>(
          queue, std::size_t(candCapacity) * candDumpStride * std::size_t(kExtDumpMaxMembers)));
      alpaka::memset(queue, *dumpLayer, 0xFF);  // int32 trackId = -1 sentinel marks unwritten slots
      alpaka::memset(queue, *dumpHdr, 0xFF);
      alpaka::memset(queue, *dumpOvf, 0);
      alpaka::memset(queue, *dumpMember, 0xFF);  // int32 hitId = -1 sentinel marks unwritten member slots
      dumpLayerPtr = dumpLayer->data();
      dumpHdrPtr = dumpHdr->data();
      dumpOvfPtr = dumpOvf->data();
      dumpMemberPtr = dumpMember->data();
    }

    constexpr auto threadsPerBlock = 256u;
    const auto blocksTuples = cms::alpakatools::divide_up_by(maxNumberOfTuples, threadsPerBlock);
    const auto workDivTuples = cms::alpakatools::make_workdiv<Acc1D>(blocksTuples, threadsPerBlock);
    const auto blocksCands = cms::alpakatools::divide_up_by(candCapacity, threadsPerBlock);
    const auto workDivCands = cms::alpakatools::make_workdiv<Acc1D>(blocksCands, threadsPerBlock);
    const auto blocksSlots =
        cms::alpakatools::divide_up_by(candCapacity * uint32_t(params.maxExtraHitsPerTrack), threadsPerBlock);
    const auto workDivSlots = cms::alpakatools::make_workdiv<Acc1D>(blocksSlots, threadsPerBlock);

    // fill pass: re-run the pre-gate predicate, now writing candList (cap = candCapacity). nCands was
    // reset to 0 by the allocator; stats were already tallied by the count pass and are not touched.
    alpaka::exec<Acc1D>(queue,
                        workDivTuples,
                        Kernel_extPreGate{},
                        tracks,
                        passBuf,
                        maxNumberOfTuples,
                        params.preGateMinPt,
                        std::sinh(params.maxAbsEta),
                        candidateMask,  // same restriction as the count pass (null = no restriction)
                        candCapacity,
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        bufs.stats.data());

    // Kernel_extFindExtras is block-per-candidate: one block (kExtFindLanes lanes) per candidate.
    // Cap the grid and grid-stride the rest (uniform_groups); with runtime sizing candCapacity is
    // already a tight bound on the candidate count, so the grid is essentially exact.
    constexpr uint32_t kExtFindMaxBlocks = 16384u;
    const auto blocksFind = std::min<uint32_t>(candCapacity, kExtFindMaxBlocks);
    const auto workDivFind = cms::alpakatools::make_workdiv<Acc1D>(blocksFind, kExtFindLanes);
    alpaka::exec<Acc1D>(
        queue,
        workDivFind,
        Kernel_extFindExtras{
            rhoMap, phiBinner, otSrcVal, params.verbose, candDumpOn, dumpLayerPtr, dumpHdrPtr, dumpOvfPtr, dumpMemberPtr},
        params.maxExtraHitsPerTrack,
        params.maxWalkLayers,
        params.extMaxWalkLayers,
        bf,
        std::sinh(params.maxAbsEta),
        params.extGateEps,
        float(extDerivedTables::chi2Quantile(1, double(params.extGateEps))),
        float(extDerivedTables::chi2Quantile(2, double(params.extGateEps))),
        float(extDerivedTables::chi2Quantile(3, double(params.extGateEps))),
        bMap,
        params.extPred,
        params.extEtaL,
        params.extRho,
        params.extEtaLRaw,
        params.extRho3,
        tracks,
        trackHits,
        hits,
        hitMask,
        extLayers,
        caModules,
        bufs.candList.data(),
        bufs.nCands.data(),
        candCapacity,
        bufs.extrasIds.data(),
        bufs.extrasChi2.data(),
        bufs.nExtras.data(),
        bufs.stats.data());  // per-event counter buffer (kStat*/kDiag* indices)

    // Seed the atomicMin identity on exactly the rows the next two kernels read.
    alpaka::exec<Acc1D>(queue,
                        workDivSlots,
                        Kernel_extInitClaims{},
                        params.maxExtraHitsPerTrack,
                        bufs.nCands.data(),
                        candCapacity,
                        otActive ? nHits : 0u,
                        bufs.extrasIds.data(),
                        bufs.nExtras.data(),
                        bufs.hitClaims.data());

    alpaka::exec<Acc1D>(queue,
                        workDivSlots,
                        Kernel_extClaimExtras{},
                        params.maxExtraHitsPerTrack,
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        candCapacity,
                        otActive ? nHits : 0u,
                        bufs.extrasIds.data(),
                        bufs.extrasChi2.data(),
                        bufs.nExtras.data(),
                        bufs.hitClaims.data());

    alpaka::exec<Acc1D>(queue,
                        workDivCands,
                        Kernel_extResolveExtras{},
                        params.maxExtraHitsPerTrack,
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        candCapacity,
                        otActive ? nHits : 0u,
                        bufs.extrasIds.data(),
                        bufs.extrasChi2.data(),
                        bufs.nExtras.data(),
                        bufs.hitClaims.data(),
                        bufs.stats.data());

    if (params.verbose) {
      const auto oneThread = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
      alpaka::exec<Acc1D>(queue, oneThread, Kernel_extPrintSummary{}, bufs.stats.data());
    }

    // Candidate dump: host readback (D2H + wait) then the fixed-format edm::LogInfo lines the offline
    // parser consumes. Records with trackId < 0 are unwritten slots.
    if (candDumpOn) {
      const std::size_t nLay = std::size_t(candCapacity) * candDumpStride;
      const std::size_t nMem = nLay * std::size_t(kExtDumpMaxMembers);  // road-candidate members
      std::vector<ExtCandLayerRec> hLay(nLay);
      std::vector<ExtCandHdrRec> hHdr(candCapacity);
      std::vector<ExtCandMemberRec> hMem(nMem);
      uint32_t hOvf[2] = {0u, 0u};  // [0] = visit-index overrun, [1] = member-slot overflow
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hLay.data(), nLay), *dumpLayer);
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hHdr.data(), candCapacity), *dumpHdr);
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hMem.data(), nMem), *dumpMember);
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hOvf, 2u), *dumpOvf);
      // Host wait: hLay/hHdr/hMem/hOvf are plain host storage that the census loop below reads
      // immediately. Only reached in an instrumented build (candDumpOn).
      alpaka::wait(queue);

      // Aggregate census (mirrors the finalDedup single-line summary).
      uint32_t nCand = 0, oAcc = 0, oEmpty = 0, oGate = 0, oCap = 0, oOther = 0, nVeto = 0, nRec = 0;
      for (uint32_t s = 0; s < candCapacity; ++s) {
        if (hHdr[s].trackId < 0)
          continue;
        ++nCand;
        for (uint32_t vi = 0; vi < candDumpStride; ++vi) {
          const ExtCandLayerRec& r = hLay[std::size_t(s) * candDumpStride + vi];
          if (r.trackId < 0)
            continue;
          ++nRec;
          switch (r.outcome) {
            case kExtDumpAccept:
              ++oAcc;
              break;
            case kExtDumpRejEmpty:
              ++oEmpty;
              break;
            case kExtDumpRejGate:
              ++oGate;
              break;
            case kExtDumpRejCap:
              ++oCap;
              break;
            default:
              ++oOther;
              break;
          }
          if (r.flags & kExtDumpFlagVetoSkip)
            ++nVeto;
        }
      }
      edm::LogInfo("CAExtensionCandDump")
          << "[extCandDump] SUMMARY nCand=" << nCand << " nRec=" << nRec << " overflow=" << hOvf[0]
          << " memberOverflow=" << hOvf[1] << " outcome{accept=" << oAcc << " rejEmpty=" << oEmpty
          << " rejGate=" << oGate << " rejCap=" << oCap << " rejOther=" << oOther << "} vetoSkips=" << nVeto
          << " stride=" << candDumpStride << " memberCap=" << kExtDumpMaxMembers;

      // Per-record detail: one H line per candidate + one L line per visited layer, batched into modest
      // LogInfo messages (bounds per-message overhead / category throttling on high-volume runs). Fixed
      // field order; every line independently greppable by the "[extCandDump]" prefix regardless of
      // message boundaries.
      std::ostringstream oss;
      uint32_t inChunk = 0;
      constexpr uint32_t kChunkCands = 32u;  // candidates per LogInfo message (bounds message size)
      auto flush = [&]() {
        const std::string msg = oss.str();
        if (!msg.empty())
          edm::LogInfo("CAExtensionCandDump") << msg;
        oss.str("");
        oss.clear();
        inChunk = 0;
      };
      for (uint32_t s = 0; s < candCapacity; ++s) {
        const ExtCandHdrRec& h = hHdr[s];
        if (h.trackId < 0)
          continue;
        oss << "[extCandDump] H slot=" << s << " trk=" << h.trackId << " nVis=" << h.nVisited
            << " hostFlags=" << h.hostFlags << std::hex << " covered=0x" << h.coveredMask << " reachable=0x"
            << h.reachableMask << " visited=0x" << h.visitedMask << std::dec << "\n";
        for (uint32_t vi = 0; vi < candDumpStride; ++vi) {
          const ExtCandLayerRec& r = hLay[std::size_t(s) * candDumpStride + vi];
          if (r.trackId < 0)
            continue;
          oss << "[extCandDump] L slot=" << s << " trk=" << r.trackId << " vi=" << vi << " layer=" << r.layerId
              << " nWinM=" << r.nWinMerged << " nWinOT=" << r.nWinOT << " nPassM=" << r.nPassMerged
              << " nPassOT=" << r.nPassOT << " win=" << r.winnerHitId << " winChi2=" << r.winnerChi2
              << " bestFail=" << r.bestFailChi2 << " round=" << r.round << " outcome=" << r.outcome
              << " flags=" << r.flags << " bestHit=" << r.bestHitId
              << " bestPass=" << ((r.flags & kExtDumpFlagBestPass) ? 1 : 0) << " holeRun=" << r.holeRunToHere << "\n";
          // One M line per road candidate scored on this (slot, layer). hit is the join key to the
          // HitTruth all-hit table (bit30 = raw-OT). Written in atomic-cursor order, not sorted.
          const std::size_t memBase = (std::size_t(s) * candDumpStride + vi) * std::size_t(kExtDumpMaxMembers);
          for (uint32_t m = 0; m < kExtDumpMaxMembers; ++m) {
            const ExtCandMemberRec& mm = hMem[memBase + m];
            if (mm.hitId < 0)
              continue;  // unwritten member slot (0xFF sentinel)
            oss << "[extCandDump] M slot=" << s << " vi=" << vi << " hit=" << mm.hitId << " chi2=" << mm.chi2
                << " round=" << int(mm.round) << " pass=" << int(mm.pass) << "\n";
          }
        }
        if (++inChunk >= kChunkCands)
          flush();
      }
      flush();
    }
  }

  // The post-walk stage: compact the extended candidates, rebuild their arc-sorted hit lists, and
  // rewrite the accepted extensions into the track SoA in place.

  using SeqContainer = caStructures::SequentialContainer;

  // Upper bound on originals + extras per merged list. Sizes the extension container's content
  // capacity (see `contentCap` below) and the per-candidate `sizes[]` bound.
  constexpr int kMaxMergedHits = 32;
  // Scratch bound for the per-track hit-list rebuild in Kernel_extFillContainer. The originals count
  // is the post-twin-union hit count, which the merger caps at kTwinMaxMergedHits (== kMaxMergedHits),
  // and the extras count is bounded by AttachParams::maxExtraHitsPerTrack, so the two stack arrays
  // must hold kMaxMergedHits + maxExtraHitsPerTrack entries: an overrun would be guarded only by an
  // ALPAKA_ASSERT_ACC, which is a no-op in the production (ndebug) build. This constant sizes only the
  // two stack arrays; `contentCap` and `sizes[]` keep using kMaxMergedHits.
  constexpr int kMaxExtraHitsScratch = 8;  // >= AttachParams::maxExtraHitsPerTrack; raise if a cfi does
  constexpr int kMaxRebuildHits = kMaxMergedHits + kMaxExtraHitsScratch;
  // CA-layer lookup (the CA's own partition, for the nLayers recount of accepted extensions).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int caLayerOf(uint32_t moduleIdx, ::reco::CALayersSoAConstView layers) {
    const int nLayers = layers.metadata().size() - 1;
    int lo = 0;
    int hi = nLayers;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (layers.layerStarts()[mid + 1] <= moduleIdx)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }

  // Compact the arbitration-surviving extended candidates into the dense extIdx space the rest of the
  // post-walk stage indexes by, carrying the tuple id and attach slot along.
  struct Kernel_extCompact {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint32_t* __restrict__ candList,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const int32_t* __restrict__ nExtras,
                                  uint32_t* __restrict__ extTuple,
                                  uint32_t* __restrict__ extCandSlot,
                                  uint32_t* __restrict__ nExt,
                                  int32_t* __restrict__ extNExtras) const {
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto j : cms::alpakatools::uniform_elements(acc, nC)) {
        if (nExtras[j] <= 0)
          continue;
        const uint32_t extIdx = alpaka::atomicAdd(acc, nExt, 1u, alpaka::hierarchy::Grids{});
        const uint32_t tupleId = candList[j];
        extTuple[extIdx] = tupleId;
        extCandSlot[extIdx] = j;
        extNExtras[extIdx] = nExtras[j];
      }
    }
  };

  // Merged (originals + kept extras) list sizes per extIdx; zero padding beyond nExt so the
  // prefix scan runs over the fixed maxCandidates extent.
  struct Kernel_extSizes {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const uint32_t* __restrict__ extTuple,
                                  const uint32_t* __restrict__ nExt,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t maxCandidates,
                                  int32_t* __restrict__ sizes) const {
      for (auto e : cms::alpakatools::uniform_elements(acc, maxCandidates)) {
        if (e >= *nExt) {
          sizes[e] = 0;
          continue;
        }
        const uint32_t i = extTuple[e];
        const uint32_t begin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
        const uint32_t end = tracks[i].hitOffsets();
        sizes[e] = int32_t(end - begin) + extNExtras[e];
        ALPAKA_ASSERT_ACC(sizes[e] <= kMaxMergedHits);
      }
    }
  };

  // Build the merged hit container: per extIdx the originals + extras insertion-sorted by arc length
  // along the host's own fitted helix, plus the CSR offsets with sentinel rows past nExt.
  struct Kernel_extFillContainer {
    // OT source (by value) so a tagged extra's arc-sort key reads the OT global position.
    // Default (nOTHits == 0) => no tagged ids can be present and only merged hits are read.
    OTHitsSource otSource_{};

    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const float bf,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  caStructures::CAHitsView hits,
                                  const uint32_t* __restrict__ extTuple,
                                  const uint32_t* __restrict__ extCandSlot,
                                  const uint32_t* __restrict__ nExt,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t maxCandidates,
                                  const uint32_t* __restrict__ extrasIds,
                                  const int32_t* __restrict__ offsets,  // inclusive prefix of sizes
                                  SeqContainer* extContainer) const {
      const uint32_t nE = alpaka::math::min(acc, *nExt, maxCandidates);
      const int32_t total = (maxCandidates > 0) ? offsets[maxCandidates - 1] : 0;
      for (auto k : cms::alpakatools::uniform_elements(acc, maxCandidates + 1)) {
        if (k == 0u)
          extContainer->off[0] = 0;
        else if (k <= nE)
          extContainer->off[k] = uint32_t(offsets[k - 1]);
        else
          extContainer->off[k] = uint32_t(total);
      }
      for (auto e : cms::alpakatools::uniform_elements(acc, nE)) {
        const uint32_t i = extTuple[e];
        const uint32_t j = extCandSlot[e];
        const uint32_t outStart = (e == 0) ? 0u : uint32_t(offsets[e - 1]);

        const HelixState h = makeHelixState(acc, tracks, int(i), bf);
        auto arcOf = [&](uint32_t hitId) {
          float xh, yh;
          if (isOTId(hitId)) {
            const uint32_t o = otIdx(hitId);
            xh = otSource_.otHits[o].xGlobal();
            yh = otSource_.otHits[o].yGlobal();
          } else {
            xh = hits[hitId].xGlobal();
            yh = hits[hitId].yGlobal();
          }
          const float alphaH = alpaka::math::atan2(acc, yh - h.yc, xh - h.xc);
          return h.rho * foldPi(h.alphaOrigin - alphaH);
        };

        const uint32_t hitBegin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
        const uint32_t hitEnd = tracks[i].hitOffsets();
        const int nOrig = int(hitEnd - hitBegin);
        const int nAdd = extNExtras[e];
        const int nTot = nOrig + nAdd;
        ALPAKA_ASSERT_ACC(nAdd <= kMaxExtraHitsScratch);
        ALPAKA_ASSERT_ACC(nTot <= kMaxRebuildHits);

        uint32_t ids[kMaxRebuildHits];
        float arcs[kMaxRebuildHits];
        for (int a = 0; a < nOrig; ++a) {
          ids[a] = trackHits[hitBegin + a].id();
          arcs[a] = arcOf(ids[a]);
        }
        for (int a = 0; a < nAdd; ++a) {
          ids[nOrig + a] = extrasIds[j * maxExtraHitsPerTrack + a];
          arcs[nOrig + a] = arcOf(ids[nOrig + a]);
        }
        for (int a = 1; a < nTot; ++a) {
          const uint32_t id = ids[a];
          const float s = arcs[a];
          int b = a - 1;
          while (b >= 0 && arcs[b] > s) {
            ids[b + 1] = ids[b];
            arcs[b + 1] = arcs[b];
            --b;
          }
          ids[b + 1] = id;
          arcs[b + 1] = s;
        }
        for (int a = 0; a < nTot; ++a)
          extContainer->content[outStart + a] = ids[a];
      }
    }
  };

  // Admits every walk commit: there is no post-walk re-fit test: a chi2-increase test on the extended hit list would discard a
  // significant share of the correct attachments for a negligible gain in purity, since a background
  // hit that wins an argmin is close by construction and its residual carries almost no information
  // about whether it is the right hit. That population is addressed by the hole hypothesis inside the
  // walk instead. This kernel therefore only fills the scaffold slots the rewrite reads, each
  // extended candidate keeping its walk extras and its original fitted row; the merger's final GBL
  // refit is what re-fits the extended hit list.
  struct Kernel_extAcceptAll {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const uint32_t* __restrict__ extTuple,
                                  const uint32_t* __restrict__ nExt,
                                  const uint32_t candCapacity,
                                  const uint32_t* __restrict__ extCandSlot,
                                  const int32_t* __restrict__ nExtras,
                                  int32_t* __restrict__ extNExtras,
                                  float* __restrict__ extNewState,
                                  float* __restrict__ extNewCov,
                                  float* __restrict__ extNewChi2,
                                  int32_t* __restrict__ extNewNdof,
                                  int32_t* __restrict__ acceptedByTuple,
                                  uint32_t* __restrict__ stats) const {
      const uint32_t nE = alpaka::math::min(acc, *nExt, candCapacity);
      for (auto e : cms::alpakatools::uniform_elements(acc, nE)) {
        const uint32_t i = extTuple[e];
        const uint32_t j = extCandSlot[e];
        const int32_t nA = nExtras[j];
        if (nA <= 0)
          continue;
        extNExtras[e] = nA;
        for (int a = 0; a < 5; ++a)
          extNewState[e * 5u + uint32_t(a)] = tracks[i].state()(a);
        for (int m = 0; m < 15; ++m)
          extNewCov[e * 15u + uint32_t(m)] = tracks[i].covariance()(m);
        extNewChi2[e] = tracks[i].chi2();
        extNewNdof[e] = tracks[i].ndof();
        acceptedByTuple[i] = int32_t(e);
        alpaka::atomicAdd(acc, &stats[kStatAccepted], 1u, alpaka::hierarchy::Grids{});
      }
    }
  };

  // Snapshot the original hit lists (ids + detIds + attached flags + per-tuple ends) before the
  // in-place re-layout. snapAttached is what lets a follow-on attach pass preserve the attach marks an
  // earlier pass set.
  struct Kernel_extSnapshot {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  const uint32_t maxNumberOfTuples,
                                  uint32_t* __restrict__ snapIds,
                                  uint32_t* __restrict__ snapDetIds,
                                  uint8_t* __restrict__ snapAttached,
                                  uint32_t* __restrict__ snapEnds) const {
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      const uint32_t totalHits = nT > 0u ? tracks[nT - 1].hitOffsets() : 0u;
      for (auto i : cms::alpakatools::uniform_elements(acc, nT))
        snapEnds[i] = tracks[i].hitOffsets();
      for (auto k : cms::alpakatools::uniform_elements(acc, totalHits)) {
        snapIds[k] = trackHits[k].id();
        snapDetIds[k] = trackHits[k].detId();
        snapAttached[k] = uint8_t(trackHits[k].attached());
      }
    }
  };

  // Final per-tuple list sizes after acceptance (zero padding beyond nTracks for the prefix scan).
  struct Kernel_extFinalSizes {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t maxNumberOfTuples,
                                  int32_t* __restrict__ finalSizes) const {
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      for (auto i : cms::alpakatools::uniform_elements(acc, maxNumberOfTuples)) {
        if (i >= nT) {
          finalSizes[i] = 0;
          continue;
        }
        const uint32_t begin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
        const int nOrig = int(tracks[i].hitOffsets() - begin);
        const int32_t e = acceptedByTuple[i];
        finalSizes[i] = nOrig + ((e >= 0) ? extNExtras[e] : 0);
      }
    }
  };

  // Capacity guard: if the re-laid-out lists would overflow the hit-list allocation, flag the
  // event for whole-event fallback (nothing is rewritten; counted and reported).
  struct Kernel_extCapacityCheck {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const int32_t* __restrict__ finalOffsets,
                                  const uint32_t maxNumberOfTuples,
                                  const uint32_t hitCapacity,
                                  uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      if (nT > 0u && uint32_t(finalOffsets[nT - 1]) > hitCapacity)
        stats[kStatRewriteOverflow] = 1u;
    }
  };

  // The in-place re-layout: new hitOffsets + TrackHitSoA rows + hit-container offsets/content for
  // every tuple (accepted tuples from the merged sorted lists + the scratch fit row + an nLayers
  // recount; everything else verbatim from the snapshot). Skipped entirely on capacity fallback.
  struct Kernel_extWriteFinal {
    // OT source (by value) so a tagged extra's detId is fetched from the raw OT SoA. Default
    // (nOTHits == 0) => no tagged ids can be present and only merged hits are read.
    OTHitsSource otSource_{};

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAView tracks,
                                  ::reco::TrackHitSoAView trackHits,
                                  const float* __restrict__ extNewState,
                                  const float* __restrict__ extNewCov,
                                  const float* __restrict__ extNewChi2,
                                  const int32_t* __restrict__ extNewNdof,
                                  caStructures::CAHitsView hits,
                                  ::reco::CALayersSoAConstView caLayers,
                                  SeqContainer* hitContainer,
                                  SeqContainer const* __restrict__ extContainer,
                                  const int maxExtraHitsPerTrack,
                                  const uint32_t* __restrict__ extCandSlot,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t* __restrict__ extrasIds,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  const uint32_t* __restrict__ snapIds,
                                  const uint32_t* __restrict__ snapDetIds,
                                  const uint8_t* __restrict__ snapAttached,
                                  const uint32_t* __restrict__ snapEnds,
                                  const int32_t* __restrict__ finalOffsets,
                                  const uint32_t maxNumberOfTuples,
                                  uint32_t* __restrict__ stats) const {
      if (stats[kStatRewriteOverflow] != 0u)
        return;
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      const uint32_t total = nT > 0u ? uint32_t(finalOffsets[nT - 1]) : 0u;
      // CSR offsets of the (rewritten) hit container, sentinels past the live tuples.
      for (auto k : cms::alpakatools::uniform_elements(acc, maxNumberOfTuples + 1)) {
        if (k == 0u)
          hitContainer->off[0] = 0;
        else if (k <= nT)
          hitContainer->off[k] = uint32_t(finalOffsets[k - 1]);
        else
          hitContainer->off[k] = total;
      }
      for (auto i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t outStart = (i == 0) ? 0u : uint32_t(finalOffsets[i - 1]);
        const uint32_t outEnd = uint32_t(finalOffsets[i]);
        tracks[i].hitOffsets() = outEnd;
        const int32_t e = acceptedByTuple[i];
        if (e >= 0) {
          const uint32_t srcBegin = extContainer->off[e];
          const uint32_t extrasBase = uint32_t(extCandSlot[e]) * uint32_t(maxExtraHitsPerTrack);
          const int nAdd = extNExtras[e];
          // The snapshot range of this tuple, to preserve attach marks written by an earlier pass.
          // On a first pass every snapshot flag is 0, so the lookup below is a no-op.
          const uint32_t snapBegin = (i == 0) ? 0u : snapEnds[i - 1];
          const uint32_t snapEnd = snapEnds[i];
          uint16_t nCALayers = 0;
          int prevLayer = -1;
          // detectorIndex for either source; both use the same CA module numbering (OT = nPixel + geom),
          // so caLayerOf works for both. The tagged id itself is kept on trackHits[d].id() (downstream
          // OT-aware consumers key on it).
          auto detOf = [&](uint32_t theId) -> uint32_t {
            return isOTId(theId) ? uint32_t(otSource_.otHits[otIdx(theId)].detectorIndex())
                                 : hits[theId].detectorIndex();
          };
          for (uint32_t d = outStart, s = srcBegin; d < outEnd; ++d, ++s) {
            const uint32_t id = extContainer->content[s];
            const uint32_t det = detOf(id);
            trackHits[d].id() = id;
            trackHits[d].detId() = det;
            // isNewExtra: added by this pass, which is what the stats count. wasAttached: a mark
            // carried in from an earlier pass -- preserved, but never re-counted. On a first pass every
            // snapshot flag is 0, so wasAttached is always false.
            bool isNewExtra = false;
            for (int a = 0; a < nAdd && !isNewExtra; ++a)
              isNewExtra = (extrasIds[extrasBase + a] == id);
            bool wasAttached = false;
            if (!isNewExtra) {
              // id-lookup over the snapshot range: the merged list is arc-sorted, so the container
              // position does not map to the snapshot position. nOrig <= kMaxMergedHits.
              for (uint32_t q = snapBegin; q < snapEnd && !wasAttached; ++q)
                wasAttached = (snapIds[q] == id && snapAttached[q] != 0);
            }
            trackHits[d].attached() = (isNewExtra || wasAttached) ? 1 : 0;
            if (isNewExtra && isOTId(id))  // diagnostic: tagged OT hit written into an accepted tuple
              alpaka::atomicAdd(acc, &stats[kStatOTWritten], 1u, alpaka::hierarchy::Grids{});
            if (isNewExtra && !isOTId(id))  // diagnostic: merged extra written into an accepted tuple
              alpaka::atomicAdd(acc, &stats[kDiagMergedWritten], 1u, alpaka::hierarchy::Grids{});
            hitContainer->content[d] = id;
            // merged lists are arc-sorted, not layer-sorted, so count distinct layers exactly
            const int layer = caLayerOf(det, caLayers);
            if (layer != prevLayer) {
              bool seen = false;
              for (uint32_t q = outStart; q < d && !seen; ++q)
                seen = (caLayerOf(detOf(hitContainer->content[q]), caLayers) == layer);
              if (!seen)
                ++nCALayers;
              prevLayer = layer;
            }
          }
          // state/cov/chi2 from the scaffold row; kinematics recomputed from that state
          // (state(2) holds signed 1/pt, same convention as Kernel_BLFit: pt = 1/|state(2)|).
          for (int a = 0; a < 5; ++a)
            tracks[i].state()(a) = extNewState[e * 5u + uint32_t(a)];
          for (int c = 0; c < 15; ++c)
            tracks[i].covariance()(c) = extNewCov[e * 15u + uint32_t(c)];
          tracks[i].pt() = 1.f / alpaka::math::max(acc, alpaka::math::abs(acc, extNewState[e * 5u + 2u]), 1e-9f);
          tracks[i].eta() = alpaka::math::asinh(acc, extNewState[e * 5u + 3u]);
          tracks[i].chi2() = extNewChi2[e];
          tracks[i].ndof() = extNewNdof[e];
          tracks[i].nLayers() = nCALayers;
        } else {
          const uint32_t srcBegin = (i == 0) ? 0u : snapEnds[i - 1];
          const uint32_t srcEnd = snapEnds[i];
          for (uint32_t d = outStart, s = srcBegin; s < srcEnd; ++d, ++s) {
            trackHits[d].id() = snapIds[s];
            trackHits[d].detId() = snapDetIds[s];
            // Restore the pre-rewrite flag: all-zero on a first pass, and on a follow-on pass this is
            // what preserves the earlier pass's attach marks on unmodified tuples.
            trackHits[d].attached() = snapAttached[s];
            hitContainer->content[d] = snapIds[s];
          }
        }
      }
    }
  };

  // Per-event accept-stage summary appended to the attach line.
  struct Kernel_extPrintAcceptSummary {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, const uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      printf(
          "[CAExtension] accept: accepted=%u capacityFallback=%u\n", stats[kStatAccepted], stats[kStatRewriteOverflow]);
      // The OT-vs-merged attachment ledger: where extras are committed, lost and written. The OT
      // columns read 0 when no OT source is active.
      printf(
          "[CAExtension][DIAG] walk(merged=%u OT=%u) otPartner=%u slotExhaust(tot=%u withOT=%u) "
          "arbLost(merged=%u OT=%u) "
          "mergedWritten=%u OTwritten=%u\n",
          stats[kDiagWalkMerged],
          stats[kDiagWalkOT],
          stats[kStatOTPartner],
          stats[kDiagSlotExhaust],
          stats[kDiagSlotExhaustOT],
          stats[kDiagArbLostMerged],
          stats[kDiagArbLostOT],
          stats[kDiagMergedWritten],
          stats[kStatOTWritten]);
      // Mean permil decomposition of the endcap gate variance sigSec2 over the considered disk hits,
      // per source round. With secFracDiag_ off the counts are zero and the line prints n=0 all-zeros.
      const uint32_t nsm = stats[kDiagSecNM] ? stats[kDiagSecNM] : 1u;
      const uint32_t nso = stats[kDiagSecNOT] ? stats[kDiagSecNOT] : 1u;
      printf(
          "[CAExtension][SIGSEC] endcap sigSec2 permil (merged n=%u: hit=%u ms=%u pred=%u align=%u | "
          "OT n=%u: hit=%u ms=%u pred=%u align=%u)\n",
          stats[kDiagSecNM],
          stats[kDiagSecFracHitM] / nsm,
          stats[kDiagSecFracMsM] / nsm,
          stats[kDiagSecFracPredM] / nsm,
          stats[kDiagSecFracAlignM] / nsm,
          stats[kDiagSecNOT],
          stats[kDiagSecFracHitOT] / nso,
          stats[kDiagSecFracMsOT] / nso,
          stats[kDiagSecFracPredOT] / nso,
          stats[kDiagSecFracAlignOT] / nso);
      // Merged-endcap predSecVar circle/line split plus the exact-5x5 shadow projection
      // (predSecVar_full), all as permil of the same sigSec2, bucketed by prior disk accepts.
      const uint32_t nd0 = stats[kDiagNDisk0M] ? stats[kDiagNDisk0M] : 1u;
      const uint32_t nd1 = stats[kDiagNDisk1M] ? stats[kDiagNDisk1M] : 1u;
      printf(
          "[CAExtension][SIGSEC-PRED] merged-endcap predSec permil-of-sigSec2 (n=%u: circle=%u line=%u "
          "bd=%u full=%u | disk0 n=%u bd=%u full=%u | disk1+ n=%u bd=%u full=%u)\n",
          stats[kDiagSecNM],
          stats[kDiagPredCircleM] / nsm,
          stats[kDiagPredLineM] / nsm,
          stats[kDiagSecFracPredM] / nsm,
          stats[kDiagPredFullM] / nsm,
          stats[kDiagNDisk0M],
          stats[kDiagPredBd0M] / nd0,
          stats[kDiagPredFull0M] / nd0,
          stats[kDiagNDisk1M],
          stats[kDiagPredBd1M] / nd1,
          stats[kDiagPredFull1M] / nd1);
    }
  };

  // One-shot per-event OT summary. Launched only when the OT source is active, so on a merged-only
  // run this line never appears.
  struct Kernel_extPrintOTSummary {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, const uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      printf("[CAExtension][OT] otHitsWritten=%u (tagged OT rechits added to accepted tuples)\n",
             stats[kStatOTWritten]);
    }
  };

  RefitScaffold buildRefitScaffold(Queue& queue,
                                   const AttachParams& params,
                                   float bf,
                                   ::reco::TrackSoAConstView tracks,
                                   ::reco::TrackHitSoAConstView trackHits,
                                   caStructures::CAHitsView hits,
                                   uint32_t maxNumberOfTuples,
                                   uint32_t hitCapacity,
                                   AttachBuffers& bufs,
                                   const OTHitsSource* otSource) {
    const OTHitsSource otSrcVal = (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : OTHitsSource{};
    // Runtime candidate capacity (the count pass in allocateAttachBuffers sized bufs to it): every
    // stage-B scratch array and candidate-loop grid is sized/bounded by this, not params.maxCandidates.
    const uint32_t candCapacity = bufs.candCapacity;
    const auto mc = std::size_t(candCapacity);
    const auto contentCap = mc * std::size_t(kMaxMergedHits);
    RefitScaffold s{
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, mc)),
        // nExt + pfxCounter + pfxCounter2 in one 3-word allocation (see RefitScaffold).
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 3),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<SeqContainer>(queue),
        cms::alpakatools::make_device_buffer<SeqContainer::Counter[]>(queue, mc + 1),
        cms::alpakatools::make_device_buffer<SeqContainer::value_type[]>(queue, contentCap + 1),
        cms::alpakatools::make_device_buffer<float[]>(queue, 5 * std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<float[]>(queue, 15 * std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<float[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, hitCapacity)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, hitCapacity)),
        // snapAttached: the pre-rewrite attached() flags (see the struct comment)
        cms::alpakatools::make_device_buffer<uint8_t[]>(queue, std::max<std::size_t>(1, hitCapacity)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples))};
    alpaka::memset(queue, s.counters, 0);            // one memset for all three words
    alpaka::memset(queue, s.acceptedByTuple, 0xff);  // -1 = no accepted extension

    typename SeqContainer::View extContainerView{s.extContainer.data(),
                                                 s.extContainerOffsets.data(),
                                                 s.extContainerStorage.data(),
                                                 uint32_t(mc + 1),
                                                 uint32_t(contentCap + 1)};
    SeqContainer::template launchZero<Acc1D>(extContainerView, queue);
    constexpr uint32_t blockSize = 256;
    const uint32_t blocksC = cms::alpakatools::divide_up_by(candCapacity, blockSize);
    const auto wdC = cms::alpakatools::make_workdiv<Acc1D>(blocksC, blockSize);
    const uint32_t blocksC1 = cms::alpakatools::divide_up_by(candCapacity + 1, blockSize);
    const auto wdC1 = cms::alpakatools::make_workdiv<Acc1D>(blocksC1, blockSize);

    alpaka::exec<Acc1D>(queue,
                        wdC,
                        Kernel_extCompact{},
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        candCapacity,
                        bufs.nExtras.data(),
                        s.extTuple.data(),
                        s.extCandSlot.data(),
                        s.nExt(),
                        s.extNExtras.data());

    alpaka::exec<Acc1D>(queue,
                        wdC,
                        Kernel_extSizes{},
                        tracks,
                        s.extTuple.data(),
                        s.nExt(),
                        s.extNExtras.data(),
                        candCapacity,
                        s.sizes.data());

    alpaka::exec<Acc1D>(queue,
                        wdC,
                        cms::alpakatools::multiBlockPrefixScan<int32_t>(),
                        s.sizes.data(),
                        s.offsets.data(),
                        int32_t(candCapacity),
                        int(blocksC),
                        s.pfxCounter(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    alpaka::exec<Acc1D>(queue,
                        wdC1,
                        Kernel_extFillContainer{otSrcVal},
                        params.maxExtraHitsPerTrack,
                        bf,
                        tracks,
                        trackHits,
                        hits,
                        s.extTuple.data(),
                        s.extCandSlot.data(),
                        s.nExt(),
                        s.extNExtras.data(),
                        candCapacity,
                        bufs.extrasIds.data(),
                        s.offsets.data(),
                        s.extContainer.data());

    return s;
  }

  void launchVerifyRewrite(Queue& queue,
                           const AttachParams& params,
                           ::reco::TrackSoAView tracks,
                           ::reco::TrackHitSoAView trackHits,
                           caStructures::CAHitsView hits,
                           ::reco::CALayersSoAConstView caLayers,
                           ::reco_extender::ExtenderLayersConstView extLayers,
                           const ::reco::CAModulesConstView modules,
                           const float* rhoMap,
                           float bfield,
                           caStructures::SequentialContainer* hitContainer,
                           uint32_t maxNumberOfTuples,
                           uint32_t hitCapacity,
                           RefitScaffold& scaffold,
                           AttachBuffers& bufs,
                           const OTHitsSource* otSource) {
    const OTHitsSource otSrcVal = (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : OTHitsSource{};
    // Runtime candidate capacity (see allocateAttachBuffers/buildRefitScaffold): bounds the
    // per-candidate grid + extent. Per-tuple grids (wdT/wdT1) stay on maxNumberOfTuples.
    const uint32_t candCapacity = bufs.candCapacity;
    constexpr uint32_t blockSize = 256;
    const uint32_t blocksC = cms::alpakatools::divide_up_by(candCapacity, blockSize);
    const auto wdC = cms::alpakatools::make_workdiv<Acc1D>(blocksC, blockSize);
    const uint32_t blocksT = cms::alpakatools::divide_up_by(maxNumberOfTuples, blockSize);
    const auto wdT = cms::alpakatools::make_workdiv<Acc1D>(blocksT, blockSize);
    const uint32_t blocksT1 = cms::alpakatools::divide_up_by(maxNumberOfTuples + 1, blockSize);
    const auto wdT1 = cms::alpakatools::make_workdiv<Acc1D>(blocksT1, blockSize);

    // Accept every arbitration-surviving extra: the per-hit gate has already priced each one against
    // its own prediction covariance and against the hole hypothesis, so no separate post-walk test is
    // applied.
    alpaka::exec<Acc1D>(queue,
                        wdC,
                        Kernel_extAcceptAll{},
                        tracks,
                        scaffold.extTuple.data(),
                        scaffold.nExt(),
                        candCapacity,
                        scaffold.extCandSlot.data(),
                        bufs.nExtras.data(),
                        scaffold.extNExtras.data(),
                        scaffold.extNewState.data(),
                        scaffold.extNewCov.data(),
                        scaffold.extNewChi2.data(),
                        scaffold.extNewNdof.data(),
                        scaffold.acceptedByTuple.data(),
                        bufs.stats.data());

    alpaka::exec<Acc1D>(queue,
                        wdT,
                        Kernel_extSnapshot{},
                        tracks,
                        trackHits,
                        maxNumberOfTuples,
                        scaffold.snapIds.data(),
                        scaffold.snapDetIds.data(),
                        scaffold.snapAttached.data(),
                        scaffold.snapEnds.data());

    alpaka::exec<Acc1D>(queue,
                        wdT,
                        Kernel_extFinalSizes{},
                        tracks,
                        scaffold.acceptedByTuple.data(),
                        scaffold.extNExtras.data(),
                        maxNumberOfTuples,
                        scaffold.finalSizes.data());

    alpaka::exec<Acc1D>(queue,
                        wdT,
                        cms::alpakatools::multiBlockPrefixScan<int32_t>(),
                        scaffold.finalSizes.data(),
                        scaffold.finalOffsets.data(),
                        int32_t(maxNumberOfTuples),
                        int(blocksT),
                        scaffold.pfxCounter2(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1u, 1u),
                        Kernel_extCapacityCheck{},
                        tracks,
                        scaffold.finalOffsets.data(),
                        maxNumberOfTuples,
                        hitCapacity,
                        bufs.stats.data());

    alpaka::exec<Acc1D>(queue,
                        wdT1,
                        Kernel_extWriteFinal{otSrcVal},
                        tracks,
                        trackHits,
                        scaffold.extNewState.data(),
                        scaffold.extNewCov.data(),
                        scaffold.extNewChi2.data(),
                        scaffold.extNewNdof.data(),
                        hits,
                        caLayers,
                        hitContainer,
                        scaffold.extContainer.data(),
                        params.maxExtraHitsPerTrack,
                        scaffold.extCandSlot.data(),
                        scaffold.extNExtras.data(),
                        bufs.extrasIds.data(),
                        scaffold.acceptedByTuple.data(),
                        scaffold.snapIds.data(),
                        scaffold.snapDetIds.data(),
                        scaffold.snapAttached.data(),
                        scaffold.snapEnds.data(),
                        scaffold.finalOffsets.data(),
                        maxNumberOfTuples,
                        bufs.stats.data());

    if (params.verbose) {
      const auto oneThread = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
      alpaka::exec<Acc1D>(queue, oneThread, Kernel_extPrintAcceptSummary{}, bufs.stats.data());
      if (otSrcVal.nOTHits > 0u)
        alpaka::exec<Acc1D>(queue, oneThread, Kernel_extPrintOTSummary{}, bufs.stats.data());
    }
  }

  // Merger-side attach: the one entry point of the stage. Builds the search structures the merger does
  // not already hold, then drives the attach pipeline over the merged collection.

  // Per-CA-layer pixel-hit-row offsets over the shared rechit SoA (mirror of the CA's SetHitsLayerStart
  // in CAHitNtupletGeneratorKernelsImpl.h): hitsLayerStart[i] = moduleStart[caLayers.layerStarts()[i]].
  // It seeds the merged-hit ExtPhiBinner with the same 256-bin per-CA-layer partition prepareHits
  // builds -- so a merger-attach candidate's per-layer phi window scan is identical to the CA's.
  struct Kernel_mergerHitSetLayerStart {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  caStructures::CAHitsView mm,
                                  ::reco::CALayersSoAConstView ll,
                                  uint32_t* __restrict__ hitsLayerStart) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, uint32_t(ll.metadata().size())))
        hitsLayerStart[i] = caStructures::moduleStartOf(mm, int32_t(ll.layerStarts()[i]));
    }
  };

  // Diagnostic coverage histogram: per CA layer, total hits on the (post-attach) merged tracks and how
  // many of them carry the attached() flag, i.e. were placed by this stage. Iterates per track over the
  // compacted CSR hit span, so the unused hit-capacity tail is never read. Only launched when
  // params.verbose; it is what makes the odd OT disks' (CA layers ~34-53) coverage checkable.
  struct Kernel_mergerCoverage {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  ::reco::CALayersSoAConstView caLayers,
                                  uint32_t nTracksCap,
                                  uint32_t* __restrict__ hitsByLayer,
                                  uint32_t* __restrict__ attachedByLayer) const {
      const uint32_t nT = alpaka::math::min(acc, nTracksCap, uint32_t(std::max(0, tracks.nTracks())));
      for (auto i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t beg = (i == 0u) ? 0u : tracks[i - 1].hitOffsets();
        const uint32_t end = tracks[i].hitOffsets();
        for (uint32_t d = beg; d < end; ++d) {
          const int L = caLayerOf(trackHits[d].detId(), caLayers);
          alpaka::atomicAdd(acc, &hitsByLayer[L], 1u, alpaka::hierarchy::Grids{});
          if (trackHits[d].attached() != 0u)
            alpaka::atomicAdd(acc, &attachedByLayer[L], 1u, alpaka::hierarchy::Grids{});
        }
      }
    }
  };

  void launchMergerAttach(Queue& queue,
                          const AttachParams& params,
                          float bfield,
                          const float* rhoMap,
                          const float* bMap,
                          ::reco::TrackSoAView tracks,
                          ::reco::TrackHitSoAView trackHits,
                          caStructures::CAHitsView hits,
                          ::reco::TrackingRecHitsMaskingConstView hitMask,
                          ::reco::CALayersSoAConstView caLayers,
                          ::reco::CAModulesConstView modules,
                          ::reco_extender::ExtenderLayersConstView extLayers,
                          uint32_t nHits,
                          uint32_t nTracksCap,
                          uint32_t hitCapacity,
                          const OTHitsSource* otSource,
                          int32_t* extendedMaskOut) {
    using namespace cms::alpakatools;
    if (nTracksCap == 0u || nHits == 0u)
      return;
    const uint32_t nLayersP1 = uint32_t(caLayers.metadata().size());  // nLayers + 1
    const uint32_t nOTForClaims = (otSource != nullptr && otSource->nOTHits > 0u) ? otSource->nOTHits : 0u;
    const float* rho = rhoMap;

    // (a) merged-hit ExtPhiBinner over the shared pixelRecHits SoA. Pure fn of the rechit iphi column +
    // the CA-layer partition (layerStart); identical convention to the CA prepareHits() phi binner.
    auto layerStart = make_device_buffer<uint32_t[]>(queue, nLayersP1);
    auto phiHist = make_device_buffer<ExtPhiBinner>(queue);
    auto phiStorage = make_device_buffer<ExtPhiBinner::value_type[]>(queue, nHits);
    {
      constexpr uint32_t kThreads = 256u;
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(nLayersP1, kThreads), kThreads),
                          Kernel_mergerHitSetLayerStart{},
                          hits,
                          caLayers,
                          layerStart.data());
    }
    ExtPhiBinner::View phiView{phiHist.data(), nullptr, phiStorage.data(), cms::alpakatools::kDynamicSize, nHits};
    // Same phi binner the CA's prepareHits() builds, over the same (pixel + stub) global index space:
    // the view-plus-accessor overload, because the facade has no single contiguous iphi column.
    auto accessor_iphi = [] ALPAKA_FN_ACC(auto const& v) { return v.iphi(); };
    fillManyFromVector<Acc1D>(phiHist.data(),
                              phiView,
                              ::pixelTopology::Phase2OTStubs::numberOfLayers,
                              hits,
                              accessor_iphi,
                              layerStart.data(),
                              nHits,
                              (uint32_t)256,
                              queue);

    // (b) mutable per-tuple SequentialContainer WriteFinal re-lays the extended hit lists into (both
    // the offsets and the content are fully overwritten by the rewrite, so it only needs to be sized +
    // initialised here -- no pre-fill from the merged trackHits is required). Sized to the same hit
    // capacity as the merged SoA so a rewrite that fits the SoA fits the container too.
    const uint32_t contentCap = std::max(hitCapacity, 1u);
    auto contHeader = make_device_buffer<SeqContainer>(queue);
    auto offBuf = make_device_buffer<SeqContainer::Counter[]>(queue, std::size_t(nTracksCap) + 1u);
    auto contentBuf = make_device_buffer<SeqContainer::value_type[]>(queue, contentCap);
    typename SeqContainer::View contView{
        contHeader.data(), offBuf.data(), contentBuf.data(), nTracksCap + 1u, contentCap};
    SeqContainer::template launchZero<Acc1D>(contView, queue);

    // (c) all-fitted pre-gate: every merged row is a fitted HP-selected track, so the pre-gate's
    // never-fitted sentinel (passBuf slot 2, the fitted-circle radius) can never fire here.
    // Kernel_extPreGate reads a null passBuf as exactly that statement: no row is never-fitted.
    const double* const passBuf = nullptr;

    // The attach pipeline: pre-gate/compact -> walk + arbitration -> scaffold -> accept + in-place rewrite. The
    // walk seeds from the merged tracks' fitted running-helix state and scans the full attachable layer set.
    // knownCandCapacity = nTracksCap: the candidate set is a subset of the tuples [0, nTracksCap), so no
    // candidate-count readback is needed; the looser bound costs memory and grid width, not correctness.
    auto extBufs =
        allocateAttachBuffers(queue, params, nHits, nOTForClaims, nTracksCap, tracks, passBuf, nullptr, nTracksCap);
    launchAttach(queue,
                 params,
                 bfield,
                 rho,
                 bMap,
                 phiHist.data(),
                 tracks,
                 trackHits,
                 hits,
                 hitMask,
                 extLayers,
                 modules,
                 passBuf,
                 nTracksCap,
                 nHits,
                 extBufs,
                 otSource);
#ifdef CA_SIZING_DUMP
    // Per-event demand dump of the attach stage, same line form as the CA producer's. nCands now holds
    // the fill pass's candidate count, i.e. the number of merged tracks that cleared the pre-gate, and
    // candCapacity is what the candidate scratch, the refit scaffold and the per-candidate grids were
    // sized to. One 4-byte D2H and
    // one host wait, compiled in only under the toggle.
    {
      auto nCandsHost = cms::alpakatools::make_host_buffer<uint32_t>(queue);
      alpaka::memcpy(queue, nCandsHost, extBufs.nCands);
      alpaka::wait(queue);
      printf("[CA Sizing] iter=mergerAttach nCands=%u capCands=%u\n", *nCandsHost.data(), extBufs.candCapacity);
    }
#endif
    auto scaffold =
        buildRefitScaffold(queue, params, bfield, tracks, trackHits, hits, nTracksCap, hitCapacity, extBufs, otSource);
    launchVerifyRewrite(queue,
                        params,
                        tracks,
                        trackHits,
                        hits,
                        caLayers,
                        extLayers,
                        modules,
                        rho,
                        bfield,
                        contHeader.data(),
                        nTracksCap,
                        hitCapacity,
                        scaffold,
                        extBufs,
                        otSource);

    // Expose this pass's per-tuple extended mask (acceptedByTuple: candidate index >= 0 iff the tuple
    // gained an accepted extra, -1 otherwise). It lets the merger scope a sharpening refit to the
    // extended subset only; non-extended tracks keep their state and are still covered by the
    // unconditional final refit. Sized to nTracksCap (== acceptedByTuple's maxNumberOfTuples).
    // Null => not written.
    if (extendedMaskOut != nullptr)
      alpaka::memcpy(queue,
                     make_device_view(queue, extendedMaskOut, nTracksCap),
                     make_device_view(queue, scaffold.acceptedByTuple.data(), nTracksCap));

    // Per-CA-layer attach coverage (verbose only): OT hits available in the scan domain (from the OT
    // source's per-CA-layer offsets) + hits/attached per layer on the post-attach merged tracks. Proves
    // the odd OT disks are covered. One host sync (the merger already syncs freely on this path).
    if (params.verbose) {
      auto hitsByLayer = make_device_buffer<uint32_t[]>(queue, nLayersP1);
      auto attachedByLayer = make_device_buffer<uint32_t[]>(queue, nLayersP1);
      alpaka::memset(queue, hitsByLayer, 0);
      alpaka::memset(queue, attachedByLayer, 0);
      constexpr uint32_t kThreads = 256u;
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(nTracksCap, kThreads), kThreads),
                          Kernel_mergerCoverage{},
                          tracks,
                          trackHits,
                          caLayers,
                          nTracksCap,
                          hitsByLayer.data(),
                          attachedByLayer.data());
      std::vector<uint32_t> hHits(nLayersP1), hAtt(nLayersP1), hOT(nLayersP1, 0u);
      alpaka::memcpy(
          queue, make_host_view(hHits.data(), nLayersP1), make_device_view(queue, hitsByLayer.data(), nLayersP1));
      alpaka::memcpy(
          queue, make_host_view(hAtt.data(), nLayersP1), make_device_view(queue, attachedByLayer.data(), nLayersP1));
      if (nOTForClaims > 0u)
        alpaka::memcpy(queue,
                       make_host_view(hOT.data(), nLayersP1),
                       make_device_view(queue, const_cast<uint32_t*>(otSource->layerStart), nLayersP1));
      // Host wait: hHits/hAtt/hOT are host vectors printed on the next lines (verbose only).
      alpaka::wait(queue);
      const int nLayers = int(nLayersP1) - 1;
      printf("[MergerExtend] per-CA-layer coverage (nLayers=%d, nMergedHits=%u, nOTHits=%u):\n",
             nLayers,
             nHits,
             nOTForClaims);
      printf("[MergerExtend]   layer : otAvail  hitsOnTrk  attached\n");
      for (int L = 0; L < nLayers; ++L) {
        const uint32_t otA = (nOTForClaims > 0u) ? (hOT[L + 1] - hOT[L]) : 0u;
        if (otA != 0u || hHits[L] != 0u || hAtt[L] != 0u)
          printf("[MergerExtend]   L%-4d : %7u  %9u  %8u%s\n",
                 L,
                 otA,
                 hHits[L],
                 hAtt[L],
                 (L >= 34) ? "   <-- OT disk" : "");
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension
