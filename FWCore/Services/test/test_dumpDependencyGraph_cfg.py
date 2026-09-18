import FWCore.ParameterSet.Config as cms


# The configuration in this file will be dumped by the DumpDependencyGraph
# service into a .json file. The json will then be read by
# test_dumpDependencyGraph_check.py, which will check that its contents match
# what this file declares. Changes to this file should be manually reflected
# into test_dumpDependencyGraph_check.py .


# It will be checked that this is the process name
process = cms.Process("DEPGRAPH")


process.DumpDependencyGraph = cms.Service(
    "DumpDependencyGraph",
    fileName=cms.untracked.string("test_dumpDependencyGraph.json"),
)

# It will be checked that process.source's type is "Source"
process.source = cms.Source("IntSource")
process.maxEvents.input = 0
process.options.numberOfThreads = 1

# It will be checked that process.first has no dependencies
process.first = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

# It will be checked that process.second's class is "AddIntsProducer"
# It will be checked that process.second consumes process.first
# It will be checked that process.second has no non-event dependency
process.second = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("first"))
)

process.aliasOfSecond = cms.EDAlias(
    second=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct")))
)

# It will be checked that process.viaAlias's dependency resolves (through
# aliasOfSecond) to process.second
process.viaAlias = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfSecond"))
)


process.moduleD = cms.EDProducer("ThingProducer")
process.moduleE = cms.EDProducer("IntProducer", ivalue=cms.int32(8))
process.moduleC = cms.EDAlias(
    moduleD=cms.VPSet(cms.PSet(type=cms.string("edmtestThings"))),
    moduleE=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct"))),
)
# It will be checked that moduleA consumes moduleD's products
process.moduleA = cms.EDProducer("OtherThingProducer", thingTag=cms.InputTag("moduleC"))
# It will be checked that moduleB consumes moduleE's products
process.moduleB = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("moduleC"))
)


# It will be checked that process.fromSource consumes process.source
process.fromSource = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("source"))
)
process.aliasOfSource = cms.EDAlias(
    source=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct")))
)

# It will be checked that process.viaSourceAlias, through aliasOfSource, also
# consumes to process.source
process.viaSourceAlias = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfSource"))
)

# An EDAlias can stand for the Source for one data product, and for a module for
# another one. It will be checked that process.viaMixedAlias, which consumes
# only the edmtestThings of process.mixedAlias, depends on process.thingMaker
# alone, and not on process.source.
process.thingMaker = cms.EDProducer("ThingProducer")
process.mixedAlias = cms.EDAlias(
    source=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct"))),
    thingMaker=cms.VPSet(cms.PSet(type=cms.string("edmtestThings"))),
)
process.viaMixedAlias = cms.EDProducer(
    "OtherThingProducer", thingTag=cms.InputTag("mixedAlias")
)

# The Source produces edmtestIntProduct, but no edmtestThings. It will be
# checked that process.notFromSource, which asks for an edmtestThings carrying
# the source's label, is reported with no dependency on the source.
process.notFromSource = cms.EDProducer(
    "OtherThingProducer", thingTag=cms.InputTag("source")
)

# It will be checked that process.runConsumer's dependency on
# process.runProducer is a non-event one
process.runProducer = cms.EDProducer("NonEventIntProducer", ivalue=cms.int32(1))
process.runConsumer = cms.EDProducer(
    "NonEventIntProducer",
    ivalue=cms.int32(2),
    consumesBeginRun=cms.InputTag("runProducer", "beginRun"),
)

# It will be checked that process.unknownLabel's dependency (on a module that
# does not exist) is reported as unresolved, and not as a normal edge. It will
# also be checked that no other module in this configuration has an unresolved
# dependency.
process.unknownLabel = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("noSuchModule"))
)

# It will be checked that dependencies on data products from an earlier process
# are marked as "unresolved"
process.fromEarlierProcess = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("first", "", "EARLIER"))
)

# It will be checked that process.gate's type is "EDFilter"
process.gate = cms.EDFilter(
    "ModuloEventIDFilter", modulo=cms.uint32(2), offset=cms.uint32(0)
)
process.behindGate = cms.EDProducer(
    "AddIntsProducer",
    labels=cms.VInputTag(cms.InputTag("viaAlias"), cms.InputTag("unscheduled")),
)

# It will be checked that process.unscheduled is listed, but on no path
process.unscheduled = cms.EDProducer("IntProducer", ivalue=cms.int32(2))
process.onDemand = cms.Task(process.unscheduled)

process.watcher = cms.EDAnalyzer(
    "IntTestAnalyzer",
    moduleLabel=cms.untracked.InputTag("first"),
    valueMustMatch=cms.untracked.int32(1),
)

# It will be checked which Paths and EndPaths exist, and that each of them
# contains these modules, in this order. The order of the Paths and the EndPaths
# themselves is unspecified, and DumpDependencyGraph sorts them alphabetically.

process.p = cms.Path(
    process.first
    + process.second
    + process.viaAlias
    + process.gate
    + process.behindGate,
    process.onDemand,
)
process.q = cms.Path(
    process.fromSource
    + process.viaSourceAlias
    + process.runProducer
    + process.runConsumer
    + process.unknownLabel
    + process.fromEarlierProcess
)
process.r = cms.Path(
    process.moduleD
    + process.moduleE
    + process.moduleA
    + process.moduleB
)
process.s = cms.Path(process.thingMaker + process.viaMixedAlias + process.notFromSource)
process.e = cms.EndPath(process.watcher)
