import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackHighPtMasking = cms.EDProducer('PixelTracksMaskingSoA@alpaka',
    iterationIndex = cms.uint32(1),
    minQuality = cms.string('tight'),
    # Masking reads the pre-extension high-purity selector on purpose: masking with the
    # stub-extended tracks removes too many hits and starves the displaced iteration.
    tracksSoASrc = cms.InputTag('hltPhase2PixelTrackTorchHighPuritySelector'),
    # First masking stage of the chain: no mask to inherit, so it seeds an all-open one over the
    # global hit index space the CA indexes, [0, nPixelHits + nStubs). Both collections are read for
    # their row count only.
    recHitsMaskSoASrc = cms.InputTag(''),
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    stubsSrc = cms.InputTag('hltOTStubProducer'),
)
