import FWCore.ParameterSet.Config as cms

# Merges the high-purity tracks of both iterations, then runs the OT-hit attach walk, the GBL refit
# and the duplicate removal in one pass. Everything the walk and the duplicate removal need beyond
# the one efficiency epsilon is geometry, the material and field maps, and the hit errors.
hltPhase2PixelTracksSoAMerger = cms.EDProducer('PixelTracksSoAMerger@alpaka',
    # Both iterations' high-purity selector outputs, still un-extended: the OT-hit extension runs here.
    inputTkSoAs = cms.VInputTag("hltPhase2PixelTrackTorchHighPuritySelector", "hltPhase2PixelTrackHighPuritySelectorDisplaced"),
    # Arm of each input collection: 0 = prompt-side, 1 = displaced-side.
    inputArms = cms.vuint32(0, 1),
    minQuality = cms.string('tight'),
)
