import FWCore.ParameterSet.Config as cms

# Per-topology alpaka ESProducer (Phase2OT) for the geometry-only blocks of the CA geometry SoA
# (per-module surface frames, per-CA-layer geometry classification, the layer-pair graph, and the
# per-layer fishbone cut). Registered by CAGeometryESProducer<pixelTopology::Phase2OT>; used by the
# non-stubs prompt (hltPhase2PixelTracksSoA) and low-pt (hltPhase2PixelTracksSoADisplaced) arms that are
# active when the phase2CAStubs modifier is off. This topology collects OT-barrel P-in-PS modules
# from the tracker geometry and does NOT read the StackedModuleGeometry record. The framework selects
# the backend (CPU serial, CUDA, ROCm) at runtime.
#
# `geometry` MUST match the CA producer that shares this geometry (same layer-pair graph +
# fishbone thresholds); copy its pairGraph/startingPairs/skipsLayers/fishboneCuts when instantiating.
caGeometryESProducerPhase2OT = cms.ESProducer('CAGeometryESProducerPhase2OT@alpaka',
    # The geometry-only members of upstream's CA `geometry` PSet (pairGraph = flat [i0,o0,i1,o1,...],
    # startingPairs = list of pair IDs, skipsLayers per pair, fishboneCuts per layer), copied from the
    # CA producer that shares this geometry.
    geometry = cms.PSet(
        pairGraph     = cms.vuint32(),
        startingPairs = cms.vuint32(),
        skipsLayers   = cms.vuint32(),
        fishboneCuts  = cms.vdouble(),
    ),
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')  # Empty string = use default backend
    )
)
