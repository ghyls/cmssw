import FWCore.ParameterSet.Config as cms

# Input for testOffloadGatingDeduplication.sh.
#
#   chainA --(consumed by)--> chainB --> chainC --> gatedD
#
# chainA, chainB and chainC sit back-to-back on one Path with no filter between them, so
# they are reached under the same condition and must share one activation capture when
# offloaded. gatedD consumes chainC but sits behind gateFilter on another Path, so it
# must keep its own.

process = cms.Process("GatingCollapseTest")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

process.source = cms.Source("EmptySource")
process.maxEvents.input = 20

process.chainA = cms.EDProducer("IntProducer", ivalue=cms.int32(1))
process.chainB = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("chainA")))
process.chainC = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("chainB")))

# gateFilter: passes only on even-numbered events (10 of 20).
process.gateFilter = cms.EDFilter("ModuloEventIDFilter",
    modulo=cms.uint32(2), offset=cms.uint32(0))

process.gatedD = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("chainC")))

process.chainPath = cms.Path(process.chainA + process.chainB + process.chainC)
process.gatedPath = cms.Path(process.gateFilter + process.gatedD)

process.schedule = cms.Schedule(process.chainPath, process.gatedPath)
