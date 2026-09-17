#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAExtensionKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAExtensionKernels_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"

#include "ExtDerivedTables.h"   // the walk's fixed acceptance
#include "CASizingDumpMacro.h"  // CA_SIZING_DUMP toggle for the attach stage's demand dump
#include "CAStructures.h"
#include "ExtenderGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension {

  // The attach stage is compiled only for Phase2OTStubs; alias its concrete per-layer phi
  // histogram (the CA's hit PhiBinner) so the attach search can window its hit scan by phi.
  using ExtPhiBinner = caStructures::PhiBinnerT<::pixelTopology::Phase2OTStubs>;

  // Configuration of the OT hit-attach stage. It runs once, in the merger, over the concatenated,
  // HP-selected and twin-merged collection, before the merger's final GBL refit; the CA itself only
  // builds, base-fits and HP-selects. The search and gate parameters size the per-layer hit window and
  // its accept test, the pre-gate parameters decide which merged tracks enter the search.
  struct AttachParams {
    bool enable = false;
    bool verbose = false;  // one-line per-event attach summary (counters)
    // Full-hits attach path: when true the producer builds the raw-OT-rechit source (OTHitsSource
    // below) and threads it to launchAttach, so the walk searches the full OT hit population in a
    // second per-layer round instead of only the stub-derived merged hits. When false no OT source is
    // built and the walk sees merged hits only.
    bool useOTRecHits = false;
    // The two compute caps of the walk: how many extra hits one track may gain, and how many layer
    // crossings it may examine. Neither selects physics -- the gate and the hole hypothesis do -- they
    // bound the work and the output buffers.
    int maxExtraHitsPerTrack = 4;
    int maxWalkLayers = 6;     // compile-time sizing of the dump strides and buffer geometry
    int extMaxWalkLayers = 6;  // runtime visit budget K, clamped in-kernel to [1, kChainMaxVisits]
    // Pre-gate: a candidate enters the search only if its fit is finite and its pt is above
    // preGateMinPt (edup and never-fitted tuples are always skipped).
    float preGateMinPt = 0.9f;  // = the downstream selector's pt cut; extending below it is pointless
    // Pre-gate |eta| ceiling: a host with |cotTheta| above sinh(maxAbsEta) is not extended at all. A
    // geometric reach bound, and the same bound the duplicate removal uses for its drop authority.
    float maxAbsEta = 4.5f;
    // THE efficiency of the walk, spent once: the probability mass of the innovation chi2 the gate
    // must contain. It sets the accept threshold (the chi2 quantile of each candidate's own dof), the
    // phi window (the bounding box of that same ball), the reachability slack and the hole prior.
    // Fixed at extDerivedTables::kExtGateEps; it is not configurable.
    float extGateEps = extDerivedTables::kExtGateEps;
    // Per-host payload written by the merger-side pre-attach pass, indexed by attach slot: the
    // material anchor (the last fitted node), the last fitted gap's exit kink and the ionisation
    // road-centre state. nullptr => the walk anchors at the PCA and applies no road-centre shift.
    const struct ExtPredCoeff* extPred = nullptr;
    // Measured detector properties the hole hypothesis is priced from -- not tuning constants.
    // eta_L: P(the host's own particle left a usable stub on this layer). rho: the stub areal density
    // [cm^-2]. rho3: the same density in the 3-dof (position + bend) space [cm^-1 rad^-1], which is the
    // one that makes nu = rho_d (2 pi)^{d/2} |S|^{1/2} dimensionless for a stub candidate. etaLRaw: the
    // raw round's conditional availability, P(raw cluster | no stub). Both eta_L rows are layer-keyed
    // with no |eta| axis, so a layer's row is dominated by whichever |eta| population supplies most of
    // its samples. The raw round's density is NOT tabulated: it is the event's own OT occupancy.
    const float* extEtaL = nullptr;     // [kExtOTLayers]
    const float* extRho = nullptr;      // [kExtOTLayers]
    const float* extEtaLRaw = nullptr;  // [kExtOTLayers]
    const float* extRho3 = nullptr;     // [kExtOTLayers]
  };

  // The smoothed-prediction payload, one record per walk host: the 3x3 local covariance of (u, u', kappa) at
  // the last fitted node (C_loc = T A^-1 T^T, T = [[0,1,0],[-1/h,1/h,-h/2],[0,0,1]]), the anchor the walk
  // extrapolates from and the last fitted gap's exit-direction kink variance. The walk forms
  //     V_u(ds) = g^T C_loc g ,  g = [1, ds, -ds^2/2] ,  ds = s_target - anchorS
  // and converts it to the target surface with conv^2.
  struct ExtPredCoeff {
    float c00, c01, c02, c11, c12, c22;  // symmetric C_loc on (u [cm], u' [rad], kappa [1/cm])
    float anchorS;                       // transverse arc of the last fitted node, in the WALK's convention
    float anchorR, anchorZ;              // its global (r, z) -- the material anchor of every road integral
    float qgapCoef;                      // theta_T^2 * (1 - m1^2/m2) of the last fitted gap [rad^2]
    // The per-host effective bending field the published q/pT is expressed in, i.e. the
    // blEffectiveBField the fast BL fit divided its geometric curvature by (copyFromCircle's
    // 1/bFieldEff). The walk inverts that conversion, rho_geom = 1/(q/pT * B); with the origin scalar
    // instead the reconstructed circle would be off by B_eff/B(0,0), the forward field correction the
    // fit applied. <= 0 (or an invalid payload) => the walk falls back to the scalar.
    float bFieldEff;  // [GeV/cm], the fit's own curvature->pT conversion field
    // The ionization energy-loss road centre. The real curvature grows along the path, kappa(s) = kappa_0
    // (1 + dE(X(s))/p), and the fit publishes kappa_0 at node 0. With u the radial offset from the reference
    // circle (outward positive), u'' = -dkappa(s), the offset at a crossing an arc ds beyond the anchor is
    //     u(ds) = elossU + elossUp*ds - 0.5*( elossDkAnchor*ds^2 + elossK*S2 )
    // with S2 the second moment of the traversed material about the arrival end (segmentXX0Moments, w1*W*d1^2).
    // fitCorrections off, or an invalid payload, leaves all four at 0.
    float elossK;         // dkappa per unit material column = kappa_0*(dE/dX)_eff/p   [1/(cm X0)]
    float elossDkAnchor;  // the curvature excess ALREADY accumulated at the anchor     [1/cm]
    float elossU;         // u at the anchor (the fit's own u_eloss at node n-1)        [cm]
    float elossUp;        // du/ds at the anchor                                        [rad]
    float valid;          // 1 = usable; 0 = the walk falls back to the fixed-cut gate
  };

  constexpr int kExtOTLayers = 26;  // CA layers 28..53

  // The coefficient pass's N-bin ladder. The band solve is templated on the hit count, so the walk hosts
  // are binned by multiplicity: one bin per N in [3, kExtPredMaxN], the top bin absorbing every host
  // above the cap. The bins are launched together in one grid, block b -> bin b / blocksPerBin.
  // kExtPredMaxN must track HelixFit<Phase2OTStubs>::kRefitMaxN; the launcher static_asserts it.
  constexpr int kExtPredMinN = 3;
  constexpr int kExtPredMaxN = 12;
  constexpr uint32_t kExtPredNBins = uint32_t(kExtPredMaxN - kExtPredMinN + 1);  // 10

  // Candidate-level attach dump (diagnostic; compiled in only under -DEXT_CAND_DUMP).
  // Per-(candidate, visited layer) trace of the attach walk, read back to host and emitted as
  // fixed-format edm::LogInfo lines. The kernel writes these buffers only when its candDump_ member is
  // true, which the launcher sets only in an EXT_CAND_DUMP build; in a production build the buffers are
  // never allocated and there is no readback and no sync.
  //
  // Per-visited-layer outcome (the layer-level result recorded once per scanned layer).
  enum ExtCandDumpOutcome : int32_t {
    kExtDumpAccept = 0,    // a hit committed on this layer (winnerHitId >= 0)
    kExtDumpRejEmpty = 1,  // no candidate hit fell in the phi window (nWinMerged + nWinOT == 0)
    kExtDumpRejGate = 2,   // candidates present but none cleared the per-hit chi2 + abs-residual gate
    kExtDumpRejCap = 3,    // a gate-passing winner refused by the extras cluster cap
    kExtDumpRejOther = 4,  // gate-passing winner uncommitted for another reason (defensive; ~unreachable)
  };
  // ExtCandLayerRec::flags bit meanings.
  constexpr int32_t kExtDumpFlagVetoSkip = 1 << 0;  // the raw-OT round was skipped on this layer
  constexpr int32_t kExtDumpFlagPartner = 1 << 1;   // a stack-partner second raw-OT extra was attached here
  // The layer's overall-best (min-gate-chi2) road candidate, over all considered hits of both rounds,
  // itself cleared the per-hit gate (chi2 below its own dof's quantile). Rides alongside
  // bestHitId, so that a geometric continuation that exists but is not committed is still visible.
  constexpr int32_t kExtDumpFlagBestPass = 1 << 2;

  // Per-road-candidate member cap. Each (candidate slot j, visit index vi) owns a block of up to
  // kExtDumpMaxMembers ExtCandMemberRec, one per scored road candidate of either round, i.e. the hits
  // that reached the gate chi2. The cap bounds the buffer and log volume on dense layers, whose window
  // occupancies reach several hundred; a layer exceeding it is clipped and candMemberOvf counts the clip.
  constexpr uint32_t kExtDumpMaxMembers = 64u;

  // Per-road-candidate member record. The hit id is the offline join key to the all-hit truth table, so
  // this dump carries only the ids, the per-candidate gate chi2 and the round/pass context. Buffer index
  // = (j * maxWalkLayers + vi) * kExtDumpMaxMembers + m; hitId < 0 marks an unwritten slot.
  struct ExtCandMemberRec {
    int32_t hitId;  // scored road-candidate id (bit30-tagged if raw-OT); join key to HitTruth tpKey/features
    float chi2;     // per-candidate gate chi2 (dPhi^2/sigPhi2 + dSec^2/sigSec2), the SAME value the argmin ranks
    int16_t round;  // 0 = merged (stub/pixel) round, 1 = raw-OT round
    int16_t pass;   // 1 iff this candidate cleared the base per-hit gate (chi2 < cut && abs-residual ok)
  };

  // Per-(candidate slot j, visit index vi in [0, maxWalkLayers)) record; buffer index = j *
  // maxWalkLayers + vi. trackId < 0 marks an unwritten slot (skip offline). Counts are post-detector-
  // masking genuine candidates (recHitMask / usedInStub / ownership rejects are NOT counted in nWin*).
  struct ExtCandLayerRec {
    int32_t trackId;      // CA tuple id (candList[j]); -1 = unwritten slot
    int32_t layerId;      // CA layer visited (0..nLayers-1; OT barrel >= 28)
    int32_t nWinMerged;   // genuine candidate hits in the phi window, merged round (round 0)
    int32_t nWinOT;       // genuine candidate hits in the phi window, raw-OT round (round 1); 0 if not run
    int32_t nPassMerged;  // merged-round hits clearing the per-hit gate (chi2 < cut && abs-residual ok)
    int32_t nPassOT;      // raw-OT-round hits clearing the per-hit gate
    int32_t winnerHitId;  // committed winner id (bit30-tagged if raw-OT); -1 = none committed
    int32_t round;        // committed round (0 merged, 1 raw-OT); -1 = none
    int32_t outcome;      // ExtCandDumpOutcome
    int32_t flags;        // OR of kExtDumpFlag*
    float winnerChi2;     // committed winner gate chi2; -1 = none committed
    float bestFailChi2;   // min gate chi2 over all considered hits (both rounds); -1 = none considered
    // Id of the overall-best (min-gate-chi2) road candidate on this layer, over all considered hits
    // regardless of gate outcome (bit30-tagged if raw-OT); -1 = none considered. bestFailChi2 is its
    // chi2 and kExtDumpFlagBestPass its base-gate pass.
    int32_t bestHitId;
    // Cumulative occupancy-gated hole count (rejGate
    // layers -- occupancy present, no gate-passer; NOT rejEmpty dead-module layers) from walk start through
    // this visited layer. For an accept record, == holes before the accept (an accept adds no hole).
    int32_t holeRunToHere;
  };
  // Per-candidate header (one per slot j). The covered/reachable/visited layer bitmasks give the
  // reachable-but-unvisited set exactly (K-budget / ordering losses = reachable & ~visited & ~covered),
  // so the offline budget/captured/lost decomposition needs no reachability proxy.
  struct ExtCandHdrRec {
    int32_t trackId;         // CA tuple id (candList[j]); -1 = unwritten slot
    int32_t nVisited;        // reachable layers scanned this walk (== walk steps)
    int32_t hostFlags;       // bit0 = displacement-gate host, bit1 = forward-pocket host
    int32_t pad;             // alignment padding (unused)
    uint64_t coveredMask;    // original (pre-walk) hit layers (bit L set)
    uint64_t reachableMask;  // confirmed-reachable layers (walk state 2)
    uint64_t visitedMask;    // scanned layers
  };

  // OT-rechit id tag (bit30): a walk-extra id with this bit set indexes the raw OT-rechit source
  // (OTHitsSource) instead of the merged TrackingRecHit SoA. The single definition lives in
  // interface/OTHitTag.h, so the plugin and CATrackFeatures.h share one literal.
  using caOTHitTag::isOTId;
  using caOTHitTag::kOTHitTag;
  using caOTHitTag::otIdx;

  // Raw device handles of the full-hits OT source consumed by the attach stage. Holds the two
  // source SoA views plus device pointers into the per-event buffers owned by OTHitsBuffers below.
  // All pointers null / nOTHits == 0 when the full-hits path is off. The walk indexes phiBinner /
  // layerStart uniformly, using the SAME 256-bin, per-CA-layer partition convention as the merged-hit
  // binner (so a candidate's per-layer phi window scan is identical whether it reads merged or raw
  // OT hits).
  struct OTHitsSource {
    ::reco::OTRecHitsConstView otHits;          // raw OT rechits (local/global position, error, detectorIndex)
    ::reco::StubsConstView stubs;               // OT stubs (lower/upper rechit indices, dPhiDr, flags)
    ::reco::OTHitModuleConstView otHitModules;  // per-OT-module moduleStart / upperSensorStart
    ::reco::StackedModuleGeometryConstView stackedGeometry;  // per-OT-module lower/upper sensor frames
    const ExtPhiBinner* phiBinner = nullptr;                 // per-CA-layer phi histogram over OT iphi, sized nOTHits
    const uint32_t* layerStart = nullptr;                    // [nLayers+1] first OT row of each CA layer (fill offsets)
    const uint8_t* usedInStub = nullptr;                     // [nOTHits] 1 iff the rechit is a member of some stub
    uint32_t* ownership = nullptr;                           // [nOTHits] per-hit ownership tag; the walk's two
    // `ownership[o] != 0` tests are guarded on the pointer, so a null
    // array rejects nothing
    uint32_t nOTHits = 0;
  };

  // Owns the per-event OT-source device buffers (caching-allocator backed, transient) plus the two
  // source views. Returned by buildOTHitsSource and kept alive by the producer for the duration of the
  // attach launch; makeSource() packages the raw handles the attach stage consumes.
  struct OTHitsBuffers {
    cms::alpakatools::device_buffer<Device, int16_t[]> iphi;         // [nOTHits] discretized phi
    cms::alpakatools::device_buffer<Device, uint8_t[]> usedInStub;   // [nOTHits] stub-membership mask
    cms::alpakatools::device_buffer<Device, uint32_t[]> layerStart;  // [nLayers+1] per-layer OT-row offsets
    cms::alpakatools::device_buffer<Device, ExtPhiBinner> phiHist;   // phi histogram header
    cms::alpakatools::device_buffer<Device, ExtPhiBinner::value_type[]> phiStorage;  // [nOTHits] index payload
    ::reco::OTRecHitsConstView otHits;
    ::reco::StubsConstView stubs;
    ::reco::OTHitModuleConstView otHitModules;               // per-OT-module upperSensorStart (walk module map)
    ::reco::StackedModuleGeometryConstView stackedGeometry;  // per-OT-module lower/upper sensor frames (gate)
    uint32_t nOTHits = 0;
    // The ownership slot is left null: the veto is unarmed (see OTHitsSource::ownership above).
    OTHitsSource makeSource() {
      return OTHitsSource{otHits,
                          stubs,
                          otHitModules,
                          stackedGeometry,
                          phiHist.data(),
                          layerStart.data(),
                          usedInStub.data(),
                          /*ownership=*/nullptr,
                          nOTHits};
    }
  };

  // Build the per-event full-hits OT source: allocate the buffers, fill iphi (same convention as the
  // merged SoA), the used-in-stub mask, the per-CA-layer OT-row offsets, and the OT phi binner (sized
  // nOTHits). Called by the producer only when the attach stage is enabled and the event has OT hits.
  // otHitModules / caLayers feed the per-layer offset kernel; nStubs sizes the stub-mask pass.
  OTHitsBuffers buildOTHitsSource(Queue& queue,
                                  const AttachParams& params,
                                  ::reco::OTRecHitsConstView otHits,
                                  ::reco::OTHitModuleConstView otHitModules,
                                  ::reco::StackedModuleGeometryConstView stackedGeometry,
                                  ::reco::StubsConstView stubs,
                                  ::reco::CALayersSoAConstView caLayers,
                                  uint32_t nOTHits,
                                  uint32_t nStubs);

  // Device scratch of the attach stage, allocated per event by allocateAttachBuffers. candList/extras are
  // indexed by the compacted candidate slot (0..nCands); candList maps the slot back to the tuple id.
  // candCapacity is the per-event candidate cap the scratch is allocated to,
  // min(max(min(knownCandCapacity, extRefitMaxCandidates), 16), params.maxCandidates); every downstream
  // kernel and launch loop reads it from here.
  struct AttachBuffers {
    cms::alpakatools::device_buffer<Device, uint32_t[]> candList;   // [candCapacity] tuple ids
    cms::alpakatools::device_buffer<Device, uint32_t> nCands;       // compacted candidate count
    cms::alpakatools::device_buffer<Device, uint32_t[]> extrasIds;  // [candCapacity * maxExtra]
    cms::alpakatools::device_buffer<Device, float[]> extrasChi2;    // [candCapacity * maxExtra]
    cms::alpakatools::device_buffer<Device, int32_t[]> nExtras;     // [candCapacity]
    cms::alpakatools::device_buffer<Device, uint64_t[]> hitClaims;  // [nHits] packed (chi2, tuple)
    cms::alpakatools::device_buffer<Device, uint32_t[]> stats;      // counters, see kStats* below
    // Runtime candidate capacity the scratch above was allocated to; the extent bound for every candidate
    // loop, grid size and RefitScaffold allocation downstream.
    uint32_t candCapacity = 0;
  };

  // Row stride of the extension's per-track fit buffer (nextRef[0..3], chi2). The pre-gate reads slot 2
  // (fitted-circle radius) as the never-fitted sentinel.
  constexpr uint32_t kPassBufStride = 5;

  // stats slots
  constexpr int kStatCandidates = 0;       // tuples passing the pre-gate
  constexpr int kStatCandOverflow = 1;     // tuples lost to the maxCandidates cap
  constexpr int kStatExtended = 2;         // candidates with >= 1 extra after arbitration
  constexpr int kStatTotalExtras = 3;      // extras after arbitration
  constexpr int kStatArbLost = 4;          // extras dropped by cross-track arbitration
  constexpr int kStatAccepted = 5;         // extended candidates rewritten into the track SoA
  constexpr int kStatUnfittable = 6;       // unwritten slot; kept so the indices below do not renumber
  constexpr int kStatRewriteOverflow = 7;  // events falling back whole (hit-list capacity)
  constexpr int kStatHistFirst = 8;        // extras-per-track histogram (last bucket = overflow)
  constexpr int kStatHistBuckets = 8;
  // Diagnostic: tagged OT-rechit hits written into accepted tuples. Stays 0 on the merged-only path;
  // reported by a one-shot print only when the OT source is active.
  constexpr int kStatOTWritten = kStatHistFirst + kStatHistBuckets;  // 16
  constexpr int kStatOTFirstFind = kStatOTWritten + 1;               // 17: unwritten slot (index placeholder)
  // Attachment ledger: where the walk's extras come from and where they are lost. All grid-wide
  // atomics, printed by the one-shot summaries below.
  constexpr int kDiagWalkMerged = kStatOTFirstFind + 1;     // 18: walk-committed merged extras (round 0)
  constexpr int kDiagWalkOT = kStatOTFirstFind + 2;         // 19: walk-committed OT extras (round 1)
  constexpr int kDiagSlotExhaust = kStatOTFirstFind + 3;    // 20: candidates whose walk ended by slot budget
  constexpr int kDiagSlotExhaustOT = kStatOTFirstFind + 4;  // 21: ...with >=1 OT extra committed
  constexpr int kDiagArbLostMerged = kStatOTFirstFind + 5;  // 22: arbitration-dropped merged extras
  constexpr int kDiagArbLostOT = kStatOTFirstFind + 6;      // 23: arbitration-dropped OT extras
  // Unwritten slots, kept so the indices below do not renumber.
  constexpr int kDiagVerRejOT = kStatOTFirstFind + 7;        // 24
  constexpr int kDiagVerRejMerged = kStatOTFirstFind + 8;    // 25
  constexpr int kDiagVerAccOT = kStatOTFirstFind + 9;        // 26
  constexpr int kDiagMergedWritten = kStatOTFirstFind + 14;  // 31: merged attached hits written to tuples
  // Raw-OT stack-partner extras accepted on stub-less layers (second OT hit on the same
  // module as the round-1 winner, gated against the updated helix). Grid-wide atomic, printed in the
  // [DIAG] summary. Stays 0 on the merged-only path (round 1 never runs).
  constexpr int kStatOTPartner = kDiagMergedWritten + 1;  // 32
  // Endcap secondary-variance decomposition: sigSec2 = hit + MS + predSec + align, accumulated over disk
  // candidate-hit evaluations, merged and OT rounds separately. Each considered endcap hit adds the
  // permil fraction of each term plus a count. Gated on the kernel functor's secFracDiag_, so production
  // runs skip the global atomics entirely.
  constexpr int kDiagSecFracHitM = kStatOTPartner + 1;    // 33: merged-round sum permil(hit)
  constexpr int kDiagSecFracMsM = kStatOTPartner + 2;     // 34: merged-round sum permil(MS)
  constexpr int kDiagSecFracPredM = kStatOTPartner + 3;   // 35: merged-round sum permil(predSec)
  constexpr int kDiagSecFracAlignM = kStatOTPartner + 4;  // 36: merged-round sum permil(align)
  constexpr int kDiagSecNM = kStatOTPartner + 5;          // 37: merged-round considered endcap hits
  constexpr int kDiagSecFracHitOT = kStatOTPartner + 6;   // 38-42: same, raw-OT round
  constexpr int kDiagSecFracMsOT = kStatOTPartner + 7;
  constexpr int kDiagSecFracPredOT = kStatOTPartner + 8;
  constexpr int kDiagSecFracAlignOT = kStatOTPartner + 9;
  constexpr int kDiagSecNOT = kStatOTPartner + 10;
  // Merged-endcap decomposition of predSecVar (verbose-gated too): circle-block vs line-block split of
  // the block-diagonal predSecVar, and the exact full-5x5 shadow-cov projection predSecVar_full, split
  // by whether a prior DISK hit was already folded in.
  constexpr int kDiagPredCircleM = kDiagSecNOT + 1;  // 43: sum permil(circle-part of predSecVar_bd)
  constexpr int kDiagPredLineM = kDiagSecNOT + 2;    // 44: sum permil(line-part of predSecVar_bd)
  constexpr int kDiagPredFullM = kDiagSecNOT + 3;    // 45: sum permil(predSecVar_full, exact 5x5 shadow)
  constexpr int kDiagNDisk0M = kDiagSecNOT + 4;      // 46: considered w/ 0 prior disk accepts
  constexpr int kDiagPredBd0M = kDiagSecNOT + 5;     // 47: sum permil(predSecVar_bd)   | 0 prior disk
  constexpr int kDiagPredFull0M = kDiagSecNOT + 6;   // 48: sum permil(predSecVar_full) | 0 prior disk
  constexpr int kDiagNDisk1M = kDiagSecNOT + 7;      // 49: considered w/ >=1 prior disk accept
  constexpr int kDiagPredBd1M = kDiagSecNOT + 8;     // 50: sum permil(predSecVar_bd)   | >=1 prior disk
  constexpr int kDiagPredFull1M = kDiagSecNOT + 9;   // 51: sum permil(predSecVar_full) | >=1 prior disk
  // Host-quality pre-gate and per-layer-class attach diagnostics, read back only when verbose.
  // kStatPreGateSkipped counts hosts skipped by the host-quality predicate alone, not by the always-on
  // base pre-gate. The per-class counters tally walk-committed extras of both rounds, pre-arbitration.
  constexpr int kStatPreGateSkipped = kDiagPredFull1M + 1;  // 52: hosts skipped by the host-quality pre-gate
  constexpr int kStatExtraTOB13 = kDiagPredFull1M + 2;      // 53: walk-committed extras on TOB1-3 (CA 28-30)
  constexpr int kStatExtraTOB456 = kDiagPredFull1M + 3;     // 54: ... TOB4-6 (CA 31-33)
  constexpr int kStatExtraTID = kDiagPredFull1M + 4;        // 55: ... TID (CA 34-53)
  // Ambiguity gate: extras vetoed from the worse claimant of a contested near-tie hit (verbose readout).
  constexpr int kStatAmbigVetoed = kDiagPredFull1M + 5;  // 56: extras dropped by the ambiguity gate
  // Derived-gate monitors. On the derived path the three cm residual caps are runaway ceilings, not
  // selectors, so their binding rate is the alarm: a ceiling that binds often means the delivered
  // efficiency is not the stated eps. kStatDerLayers is the denominator, the layer visits scored by the
  // derived gate. All are diagnostic-only; stats[] never reaches track content.
  constexpr int kStatDerLayers = kStatAmbigVetoed + 1;  // derived-gate layer visits
  constexpr int kStatDerCapR = kStatDerLayers + 1;      // 65: r-phi ceiling bound a ball-passer
  constexpr int kStatDerCapS = kStatDerLayers + 2;      // 66: secondary ceiling likewise
  constexpr int kStatDerHoleFire = kStatDerLayers + 3;  // 67: the hole hypothesis vetoed the argmin winner
  constexpr int kStatDerHoleCand = kStatDerLayers + 4;  // 68: winners the hole was tested on
  constexpr int kStatDerHostOff = kStatDerLayers + 5;   // 69: hosts without a usable payload
  constexpr int kStatDerHostOn = kStatDerLayers + 6;    // 70: hosts running the derived gate
  // Raw-channel monitors: the round-1 projections of the counters above, which otherwise split commits
  // by layer class or by source round but never by both. Incremented only on the round-1 path, so the
  // merged round pays nothing.
  constexpr int kStatRawTOB13 = kStatDerLayers + 7;         // 71: round-1 commits on TOB1-3 (CA 28-30)
  constexpr int kStatRawTOB456 = kStatDerLayers + 8;        // 72: ... TOB4-6 (CA 31-33)
  constexpr int kStatRawTID = kStatDerLayers + 9;           // 73: ... TID/TEDD (CA 34-53)
  constexpr int kStatDerHoleCandRaw = kStatDerLayers + 10;  // 74: hole tested on a round-1 winner
  constexpr int kStatDerHoleFireRaw = kStatDerLayers + 11;  // 75: ... and it declined the attach
  // The |eta| pre-gate's own rejection count (kStatPreGateSkipped covers only the three host-quality
  // predicates). Counted in the count pass only, matching the kStatCandidates single-count convention.
  constexpr int kStatPreGateEtaSkipped = kStatDerLayers + 12;  // 76: host skipped on |cotTheta| > sinh(maxAbsEta)
  // The far-first disc ordering arms per host on three conditions, only one of which is visible in any
  // other counter.
  constexpr int kStatAttachFarArmed = kStatDerLayers + 13;  // 77: host whose disc ordering was re-keyed far-first
  // A far commit declined by the window-ambiguity condition leaves no other trace: the layer's outcome
  // reads as "no gate-passer" in every other counter.
  constexpr int kStatAttachFarDecline = kStatDerLayers + 14;  // 78: far commit declined on window multiplicity
  // Length of the stats[] counter buffer: one slot past the last counter above.
  constexpr int kStatSize = kStatDerLayers + 15;  // 79

  // The smoothed-prediction pass. Defined in its own translation unit (ExtPredCoeff.dev.cc) for the
  // reason documented at the top of that file: compiling its band instantiations alongside Kernel_BLFit
  // changes the FMA contraction the compiler applies to the fit itself. Runs the fast-BL circle band
  // over every merged track selected by hostMask and publishes the walk's prediction covariance.
  // Read-only on the tracks.
  void launchExtPredCoeff(Queue& queue,
                          caStructures::CAHitsView hv,
                          ::reco::CAModulesConstView cm,
                          ::reco::TrackSoAView mergedTracks,
                          ::reco::TrackHitSoAView mergedHits,
                          const int32_t* hostMask,
                          const OTHitsSource* otSource,
                          float bfield,
                          const float* rhoMap,
                          // The normalized (Bz,Br) r-z field map (BLBFieldMap), the same EventSetup
                          // condition the GBL refits read. Consumed only under fitCorrections; null =>
                          // the origin scalar everywhere.
                          const float* bMap,
                          bool fitCorrections,
                          ExtPredCoeff* out,
                          uint32_t nTracksCap);

  // Replicate the walk's own pre-gate predicate as a per-tuple mask, so the smoothed-prediction pass
  // runs on exactly the hosts the walk will use and on nothing else. Writes 0 for a host and -1
  // otherwise (the refit-ladder compaction's own convention); when pred is given, also clears its
  // `valid` field for every slot.
  void launchExtHostMask(Queue& queue,
                         const AttachParams& params,
                         ::reco::TrackSoAConstView tracks,
                         uint32_t nTracksCap,
                         int32_t* hostMask,
                         struct ExtPredCoeff* pred = nullptr);

  // Allocate the attach-stage scratch. Runs the count-only pre-gate (Kernel_extPreGate with a null candList),
  // whose counts are not read back; launchAttach re-runs the predicate to fill candList.
  // nOTHits: with the OT full-hits path active, the per-hit claim buffer must cover the OT domain too
  // (indices nHits + otIdx); 0 keeps hitClaims sized to nHits.
  // candidateMask: when non-null, only tuples with candidateMask[tuple] >= 0 enter the pre-gate.
  // knownCandCapacity (> 0): host-known upper bound on the candidate set; the scratch is sized to it (capped
  // at extRefitMaxCandidates, clamped to [16, maxCandidates]) without a device->host readback.
  AttachBuffers allocateAttachBuffers(Queue& queue,
                                      const AttachParams& params,
                                      uint32_t nHits,
                                      uint32_t nOTHits,
                                      uint32_t maxNumberOfTuples,
                                      ::reco::TrackSoAConstView tracks,
                                      const double* passBuf,
                                      const int32_t* candidateMask,
                                      uint32_t knownCandCapacity);

  // Post-walk scratch: the compacted extended candidates and the containers of their merged
  // (arc-sorted) hit lists, consumed by the accept stage + the final in-place rewrite.
  // extIdx (the dense candidate index) maps back to the CA tuple id via extTuple.
  struct RefitScaffold {
    cms::alpakatools::device_buffer<Device, uint32_t[]> extTuple;     // [candCapacity] extIdx -> tuple
    cms::alpakatools::device_buffer<Device, uint32_t[]> extCandSlot;  // [candCapacity] extIdx -> attach slot
    // nExt, pfxCounter and pfxCounter2 share one uint32_t[3] allocation (one allocation and one memset)
    // with three accessors. Each word is reached only through its own accessor, so no word is ever
    // touched as two different types; the prefix-scan counters are handed out as int32_t* because
    // multiBlockPrefixScan takes int32_t*, and an atomicAdd of 1 on a 4-byte aligned word is
    // bit-identical either way.
    cms::alpakatools::device_buffer<Device, uint32_t[]> counters;   // [3] = {nExt, pfxCounter, pfxCounter2}
    cms::alpakatools::device_buffer<Device, int32_t[]> extNExtras;  // [candCapacity] kept extras
    cms::alpakatools::device_buffer<Device, caStructures::SequentialContainer> extContainer;
    cms::alpakatools::device_buffer<Device, caStructures::SequentialContainer::Counter[]> extContainerOffsets;
    cms::alpakatools::device_buffer<Device, caStructures::SequentialContainer::value_type[]> extContainerStorage;
    // The fit row the rewrite writes back for each accepted candidate (indexed by dense extIdx):
    // 5-param state, 15-element covariance, chi2/ndof-normalized chi2, and ndof.
    cms::alpakatools::device_buffer<Device, float[]> extNewState;        // [5*candCapacity]
    cms::alpakatools::device_buffer<Device, float[]> extNewCov;          // [15*candCapacity]
    cms::alpakatools::device_buffer<Device, float[]> extNewChi2;         // [candCapacity]
    cms::alpakatools::device_buffer<Device, int32_t[]> extNewNdof;       // [candCapacity]
    cms::alpakatools::device_buffer<Device, int32_t[]> sizes;            // [candCapacity] merged sizes
    cms::alpakatools::device_buffer<Device, int32_t[]> offsets;          // [candCapacity] inclusive prefix
    cms::alpakatools::device_buffer<Device, int32_t[]> acceptedByTuple;  // [maxNumberOfTuples] extIdx or -1
    // Snapshot of the original per-tuple hit lists + list ends, taken before the in-place
    // re-layout (the new offsets shift every list after the first accepted extension).
    cms::alpakatools::device_buffer<Device, uint32_t[]> snapIds;     // [hitCapacity]
    cms::alpakatools::device_buffer<Device, uint32_t[]> snapDetIds;  // [hitCapacity]
    // Pre-rewrite attached() flags. WriteFinal recomputes attached() from the CURRENT pass's extras
    // only, so a follow-on attach pass would otherwise clear the marks an earlier pass set; restoring
    // from this snapshot preserves them. All-zero on a first pass (the CA initialises attached = 0 and
    // no extras are in the lists yet), where restoring it is a no-op.
    cms::alpakatools::device_buffer<Device, uint8_t[]> snapAttached;  // [hitCapacity]
    cms::alpakatools::device_buffer<Device, uint32_t[]> snapEnds;     // [maxNumberOfTuples]
    cms::alpakatools::device_buffer<Device, int32_t[]> finalSizes;    // [maxNumberOfTuples]
    cms::alpakatools::device_buffer<Device, int32_t[]> finalOffsets;  // [maxNumberOfTuples] inclusive

    // Accessors for the three counter words.
    uint32_t* nExt() { return counters.data(); }
    int32_t* pfxCounter() { return reinterpret_cast<int32_t*>(counters.data()) + 1; }
    int32_t* pfxCounter2() { return reinterpret_cast<int32_t*>(counters.data()) + 2; }
  };

  // Post-walk phase 1 (after launchAttach): compact the extended candidates and build the merged
  // (arc-sorted) hit container the accept stage and the final rewrite consume.
  RefitScaffold buildRefitScaffold(Queue& queue,
                                   const AttachParams& params,
                                   float bf,
                                   ::reco::TrackSoAConstView tracks,
                                   ::reco::TrackHitSoAConstView trackHits,
                                   caStructures::CAHitsView hits,
                                   uint32_t maxNumberOfTuples,
                                   uint32_t hitCapacity,
                                   AttachBuffers& bufs,
                                   // The full-hits OT source, so the merged (arc-sorted) container build can fetch
                                   // an OT extra's global position for its arc key. nullptr => merged-hits-only.
                                   const OTHitsSource* otSource = nullptr);

  // Post-walk phase 2: accept every extended candidate (the gate and the hole hypothesis already priced each
  // extra inside the walk), then rewrite the track SoA row from the scratch fit, the hit lists (TrackHitSoA +
  // hit container + hitOffsets, via a snapshot copy + full re-layout) and nLayers. Unextended tuples are
  // untouched; if the re-layout would overflow the hit-list capacity the whole event falls back (counted).
  // extLayers, modules, rhoMap and bfield are the geometry and material handles of this stage.
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
                           // The full-hits OT source, so the final rewrite can fetch a tagged extra's OT
                           // position / frame / detId. nullptr => merged-hits-only.
                           const OTHitsSource* otSource = nullptr);

  // Run the attach search: pre-gate + compaction, per-candidate layer walk with the chi2 gates, and
  // cross-track arbitration of shared attached hits (atomic best-chi2 claim per hit). On return the
  // buffers hold, per candidate slot, the arbitration-surviving extra hit ids. passBuf is the
  // two-pass fit buffer: its zeroed rows identify never-fitted tuples (the caller must memset it
  // before use).
  void launchAttach(Queue& queue,
                    const AttachParams& params,
                    float bf,  // Bz(0,0): the scale of the normalised (Bz,Br) map
                    const float* rhoMap,
                    const float* bMap,  // normalised (Bz,Br) r-z lattice, read per road segment
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
                    // The full-hits OT source, threaded to this entry point so the walk's round 1, its
                    // stack-partner scan and the rewrite can read raw OT rechits. nullptr => the walk
                    // scans merged hits only and no tagged extra can be produced.
                    const OTHitsSource* otSource = nullptr,
                    // Candidate mask: nullptr => no restriction; when set, only tuples with
                    // candidateMask[tuple] >= 0 enter the walk (a follow-on pass passes the previous
                    // pass's extended set here).
                    const int32_t* candidateMask = nullptr);

  // Merger-side attach, the entry point of the attach stage (allocateAttachBuffers -> launchAttach ->
  // buildRefitScaffold -> launchVerifyRewrite), run once over the merged HP-selected collection with the
  // EventSetup full geometry. It builds the merged-hit ExtPhiBinner (same 256-bin per-layer partition
  // as the CA's prepareHits), the mutable per-tuple SequentialContainer the extended hit lists are
  // re-laid into, and passes a null passBuf (every merged row is fitted). The caller supplies an empty
  // hitMask ("all open"), the full OTHitsSource and the extender layer surfaces, and allocates hit-list
  // headroom for the post-attach total. Modifies the merged track SoA in place (state/cov/chi2/ndof +
  // hit lists). extendedMaskOut (optional, size >= nTracksCap): candidate index >= 0 iff the tuple
  // gained an accepted extra, else -1; usable as a refit mask.
  void launchMergerAttach(Queue& queue,
                          const AttachParams& params,
                          float bfield,
                          const float* rhoMap,
                          const float* bMap,  // normalised (Bz,Br) r-z lattice, read per road segment
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
                          int32_t* extendedMaskOut = nullptr);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAExtensionKernels_h
