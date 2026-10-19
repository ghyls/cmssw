import FWCore.ParameterSet.Config as cms

# Input for testOffloadedModulesDoNotRunMoreThanNeeded.sh. Defines the following
# modules:
#
# - producerA, which is consumed by moduleX
# - moduleX, which is consumed by moduleY.
# - evenEventIDFilter, who sits between moduleX and moduleY, and passes only if
#   eventID is even.
#
# moduleY should run only when evenEventIDFilter passes. moduleX runs always.
#
# moduleY consumes moduleX's products, and thus the two should still be
# offloaded together. The two should be gated by moduleY's own condition, not
# moduleX's. Otherwise moduleY would run even when it does not need to.


process = cms.Process("CorrectRemoteGrupingTest")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

process.source = cms.Source("EmptySource")
process.maxEvents.input = 20

# producerA: an ordinary local producer. Needed (indirectly) by moduleY. Runs on
# its own path.
process.producerA = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

# moduleX: consumes producerA. Scheduled unconditionally, on its own Path.
process.moduleX = cms.EDProducer("AddIntsProducer",
    labels=cms.VInputTag(cms.InputTag("producerA")))

# evenEventIDFilter: passes only fi eventID is even.
process.evenEventIDFilter = cms.EDFilter("ModuloEventIDFilter",
    modulo=cms.uint32(2), offset=cms.uint32(0))

# moduleY: Should only run when evenEventIDFilter passes. Consumes moduleX's
# product, and thus both moduleX and moduleY should be put together in the same
# remote group.
process.moduleY = cms.EDProducer("AddIntsProducer",
    labels=cms.VInputTag(cms.InputTag("moduleX")))

process.pathA = cms.Path(process.producerA)
process.pathX = cms.Path(process.moduleX)
process.pathY = cms.Path(process.evenEventIDFilter + process.moduleY)

process.schedule = cms.Schedule(process.pathA, process.pathX, process.pathY)
