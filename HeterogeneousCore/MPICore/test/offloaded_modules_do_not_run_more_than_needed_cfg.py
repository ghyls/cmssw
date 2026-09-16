import FWCore.ParameterSet.Config as cms

# Used by testOffloadedModulesDoNotRunMoreThanNeeded.sh.
#
# Remote modules should be grouped based on when they should run, not on their
# dependencies. Consider producers A, B, C, D, E and F, and a filter. This
# configuration distributes them in the following way
#
#    - PATH 1: A -- B (consumes A) -- C (consumes B) -- D (consumes C)
#    - PATH 2: filter -- E (consumes D) -- F (consumes E)
#
# Producers B, C, and D will run always. Producer E consumes D, but sits behind
# a filter on path 2. Thus it should not be grouped together with the modules in
# path 1, otherwise it would run on every event.

process = cms.Process("testOffloadedModulesDoNotRunMoreThanNeeded")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

process.source = cms.Source("EmptySource")
process.maxEvents.input = 20

# producerA: an ordinary local producer, the offloaded modules' only external input.
process.producerA = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

# producerB, producerC, producerD
process.producerB = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerA")))
process.producerC = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerB")))
process.producerD = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerC")))

# eventIDFilter: passes only on events with even evenIDs (10 out of 20).
process.eventIDFilter = cms.EDFilter("ModuloEventIDFilter",
    modulo=cms.uint32(2), offset=cms.uint32(0))

# producerE: consumes producerD, but is reached under a different condition than
# producerD
process.producerE = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerD")))

# producerF: reads producerE in the local process (requiring its results to be
# sent back)
process.producerF = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerE")))

process.path1 = cms.Path(process.producerA + process.producerB + process.producerC + process.producerD)
process.path2 = cms.Path(process.eventIDFilter + process.producerE + process.producerF)

process.schedule = cms.Schedule(process.path1, process.path2)
