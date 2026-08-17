import FWCore.ParameterSet.Config as cms
from .hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA as _hltPhase2PixelTracksSoA

# Displaced arm of the default (non-stub) chain: the prompt CA producer reading the hit mask instead
# of all hits. With phase2CAStubs it is replaced by the stub version.
hltPhase2PixelTracksSoADisplaced = _hltPhase2PixelTracksSoA.clone(
    iterationName = cms.string('displaced'),
    hitMask       = cms.InputTag('hltPhase2PixelTrackHighPtMasking'),
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from .hltPhase2PixelTracksSoADisplacedWithStubs_cfi import (
    hltPhase2PixelTracksSoADisplacedWithStubs as _hltPhase2PixelTracksSoADisplacedWithStubs,
)
phase2CAStubs.toReplaceWith(hltPhase2PixelTracksSoADisplaced,
                            _hltPhase2PixelTracksSoADisplacedWithStubs)
