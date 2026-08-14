import FWCore.ParameterSet.Config as cms

# Exercises everything test_dumpDependencyGraph.sh checks in the JSON that
# DumpDependencyGraph writes: a consumes() dependency, one reached through an EDAlias,
# schedule order on both a Path and an EndPath, and a module on no path at all.

process = cms.Process("DEPGRAPH")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 0
process.options.numberOfThreads = 1

process.DumpDependencyGraph = cms.Service(
    "DumpDependencyGraph",
    fileName=cms.untracked.string("test_dumpDependencyGraph.json"),
)

process.first = cms.EDProducer("IntProducer", ivalue=cms.int32(1))
process.second = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("first")))

# consumed under its alias name, but the dump must report the real producer
process.aliasOfSecond = cms.EDAlias(second=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct"))))
process.viaAlias = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfSecond")))

process.gate = cms.EDFilter("ModuloEventIDFilter", modulo=cms.uint32(2), offset=cms.uint32(0))
process.behindGate = cms.EDProducer("AddIntsProducer",
    labels=cms.VInputTag(cms.InputTag("viaAlias"), cms.InputTag("unscheduled")))

# only ever run on demand for behindGate, so it must be listed but be on no path
process.unscheduled = cms.EDProducer("IntProducer", ivalue=cms.int32(2))
process.onDemand = cms.Task(process.unscheduled)

process.watcher = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel=cms.untracked.InputTag("first"), valueMustMatch=cms.untracked.int32(1))

process.p = cms.Path(process.first + process.second + process.viaAlias + process.gate + process.behindGate, process.onDemand)
process.e = cms.EndPath(process.watcher)

process.schedule = cms.Schedule(process.p, process.e)
