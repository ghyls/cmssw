import FWCore.ParameterSet.Config as cms

# Input for testOffloadGatingDeadlock.sh.
#
# moduleX and moduleY below will be offloaded to the remote process. moduleY
# consumes moduleX, but the two are reached under different conditions:
#
# - moduleX is unconditional, runs always
# - moduleY sits behind evenNumberFilter, which passes only on even eventIDs.
#
# Gating the two together, with the capture taken where moduleY sits (behind
# evenNumberFilter) while moduleX's results are awaited where moduleX sits, would
# hang both processes on the very first event:
#
# - local runs producerA and sends its product to the remote process;
# - local reaches the activity filter that waits for the offloaded results,
#   inserted in front of moduleX, and blocks there;
# - the remote process has producerA's product, but no activation token, so its
#   own activity filter stops the offloaded path: moduleX never runs, and
#   nothing is ever sent back;
# - the capture that would produce that token sits further down the same local
#   path, behind evenNumberFilter, which local cannot reach while it is blocked;
# - each process is now waiting for the other, for ever.
#
# What avoids it is that moduleX and moduleY are reached under different
# conditions, so they end up in separate offload groups, each captured at its own
# position -- and every capture is inserted in front of every position where a
# filter waiting for it ends up.

process = cms.Process("GatingDeadlockTest")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

process.source = cms.Source("EmptySource")
process.maxEvents.input = 20

# producerA: An ordinary local producer, runs always.
process.producerA = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

# moduleX: will be offloaded. Takes producerA's inputs.
process.moduleX = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerA")))

# localReadX: Reads (and therefore requires) moduleX's outputs in the local process.
process.localReadX = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("moduleX")))

# evenNumberFilter: passes only on even eventIDs.
process.evenNumberFilter = cms.EDFilter("ModuloEventIDFilter", modulo=cms.uint32(2), offset=cms.uint32(0))

# moduleY: offloaded as well, behind evenNumberFilter, and also read back on local.
process.moduleY = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("moduleX")))
process.localReadY = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("moduleY")))

process.onePath = cms.Path(
    process.producerA
    + process.moduleX
    + process.localReadX
    + process.evenNumberFilter
    + process.moduleY
    + process.localReadY
)

process.schedule = cms.Schedule(process.onePath)
