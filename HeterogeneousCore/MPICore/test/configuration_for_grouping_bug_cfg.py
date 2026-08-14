import FWCore.ParameterSet.Config as cms

# Input for testOffloadGroupOwnGating.sh.
#
#   producerA --(consumed by)--> moduleX --(consumed by)--> moduleY
#                                                             ^
#                                                     gated by gatingFilter,
#                                                     on moduleY's own Path
#
# moduleX and moduleY really do consume each other's products, so offloading them
# together correctly puts them in one group -- but moduleY was only ever meant to run on
# the half of the events where gatingFilter passes, a condition that has nothing to do
# with moduleX.

process = cms.Process("GroupingBugTest")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

process.source = cms.Source("EmptySource")
process.maxEvents.input = 20

# producerA: an ordinary local producer, unconditional, needed (indirectly) by moduleY.
process.producerA = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

# moduleX: consumes producerA. Scheduled, unconditionally, on its own Path.
process.moduleX = cms.EDProducer("AddIntsProducer",
    labels=cms.VInputTag(cms.InputTag("producerA")))

# gatingFilter: passes only on even-numbered events (10 of 20).
process.gatingFilter = cms.EDFilter("ModuloEventIDFilter",
    modulo=cms.uint32(2), offset=cms.uint32(0))

# moduleY: only meant to run when gatingFilter passes. It also directly consumes
# moduleX's product, which is what pulls moduleX and moduleY into the same offload group.
process.moduleY = cms.EDProducer("AddIntsProducer",
    labels=cms.VInputTag(cms.InputTag("moduleX")))

process.pathA = cms.Path(process.producerA)
process.pathX = cms.Path(process.moduleX)
process.pathY = cms.Path(process.gatingFilter + process.moduleY)

process.schedule = cms.Schedule(process.pathA, process.pathX, process.pathY)
