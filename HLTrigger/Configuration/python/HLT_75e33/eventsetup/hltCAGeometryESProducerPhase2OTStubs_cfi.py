import FWCore.ParameterSet.Config as cms

# EventSetup producer of the CA-ordered module geometry SoA (per-module surface frames + per-CA-layer
# geometry classification + the layer-pair graph) for the Phase2OTStubs OT-stubs chain. The
# PixelTracksSoAMerger twin full-refit reads this geometry from the EventSetup (CAGeometryRecord)
# rather than from a per-event product: the geometry-only blocks are pure per-run conditions and
# describe the same geometry the CA arms build.
#
# The geometry members MUST match the CA producer that shares this geometry (same layer-pair graph +
# fishbone thresholds): the module/layer fill is sized by fishboneCuts (n_layers) and asserts on the
# layer-pair graph. They are copied from the displaced OT-stubs arm
# (hltPhase2PixelTracksSoADisplacedWithStubs, the phase2CAStubs swap target whose geometry the merger
# currently consumes) so the produced SoA is identical to that arm's.
from RecoTracker.PixelSeeding.caGeometryESProducerPhase2OTStubs_cfi import (
    caGeometryESProducerPhase2OTStubs as _caGeometryESProducerPhase2OTStubs,
)
from ..modules.hltPhase2PixelTracksSoADisplacedWithStubs_cfi import (
    hltPhase2PixelTracksSoADisplacedWithStubs as _hltPhase2PixelTracksSoADisplacedWithStubs,
)

hltCAGeometryESProducerPhase2OTStubs = _caGeometryESProducerPhase2OTStubs.clone(
    geometry = dict(
        pairGraph     = list(_hltPhase2PixelTracksSoADisplacedWithStubs.geometry.pairGraph),
        startingPairs = list(_hltPhase2PixelTracksSoADisplacedWithStubs.geometry.startingPairs),
        skipsLayers   = list(_hltPhase2PixelTracksSoADisplacedWithStubs.geometry.skipsLayers),
        fishboneCuts  = list(_hltPhase2PixelTracksSoADisplacedWithStubs.geometry.fishboneCuts),
    ),
)
