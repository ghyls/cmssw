import FWCore.ParameterSet.Config as cms

# Used by testSplitterSendsOnlyConsumedProducts.sh.
#
# The splitter should forward, in each direction, only the data products that are
# consumed on the other side, matched by type and product instance name, and put
# them under the labels (module labels or EDAliases) their consumers ask for.

process = cms.Process("SPLITPRODUCTS")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

# the Source produces an edmtest::IntProduct with value 4
process.source = cms.Source("IntSource")
process.maxEvents.input = 10

# local producers
process.localProducer = cms.EDProducer("IntProducer", ivalue=cms.int32(3))
# localSum produces two edmtest::IntProduct, with instances "" and "other", both with value 3
process.localSum = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("localProducer")))
# things produces an Event edmtest::ThingCollection, and some for runs and lumis
process.things = cms.EDProducer("ThingProducer")

# EDAliases of local data products
process.aliasOfSource = cms.EDAlias(source=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct"))))
process.aliasOfLocalSum = cms.EDAlias(
    localSum=cms.VPSet(
        cms.PSet(
            type=cms.string("edmtestIntProduct"),
            fromProductInstance=cms.string("other"),
            toProductInstance=cms.string("fromOther"),
        )
    )
)
# an EDAlias of data products of both the Source and a module
process.mixed = cms.EDAlias(
    source=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct"))),
    things=cms.VPSet(cms.PSet(type=cms.string("edmtestThings"), fromProductInstance=cms.string(""))),
)

# the modules to offload
process.fromSource = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("source")))
process.viaSourceAlias = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfSource")))
process.fromLocalOther = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("localSum", "other")))
process.viaLocalAlias = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfLocalSum", "fromOther"))
)
# reads only the edmtest::ThingCollection of the mixed EDAlias, so nothing of the Source
process.thingsViaMixed = cms.EDProducer("OtherThingProducer", thingTag=cms.InputTag("mixed"))
# 4 + 4 + 3 + 3
process.remoteSum = cms.EDProducer(
    "AddIntsProducer",
    labels=cms.VInputTag("fromSource", "viaSourceAlias", "fromLocalOther", "viaLocalAlias"),
)
process.aliasOfRemoteSum = cms.EDAlias(
    remoteSum=cms.VPSet(
        cms.PSet(
            type=cms.string("edmtestIntProduct"),
            fromProductInstance=cms.string("other"),
            toProductInstance=cms.string(""),
        )
    )
)
# reads an offloaded module through an EDAlias
process.remoteViaAlias = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfRemoteSum")))

# the local modules reading the results of the offloaded ones
process.checkRemoteSum = cms.EDAnalyzer(
    "IntTestAnalyzer", moduleLabel=cms.untracked.InputTag("remoteSum", "other"), valueMustMatch=cms.untracked.int32(14)
)
process.checkAliasOfRemoteSum = cms.EDAnalyzer(
    "IntTestAnalyzer", moduleLabel=cms.untracked.InputTag("aliasOfRemoteSum"), valueMustMatch=cms.untracked.int32(14)
)
process.checkRemoteViaAlias = cms.EDAnalyzer(
    "IntTestAnalyzer", moduleLabel=cms.untracked.InputTag("remoteViaAlias"), valueMustMatch=cms.untracked.int32(14)
)

process.path = cms.Path(
    process.localProducer
    + process.localSum
    + process.things
    + process.fromSource
    + process.viaSourceAlias
    + process.fromLocalOther
    + process.viaLocalAlias
    + process.thingsViaMixed
    + process.remoteSum
    + process.remoteViaAlias
)
process.checks = cms.EndPath(process.checkRemoteSum + process.checkAliasOfRemoteSum + process.checkRemoteViaAlias)
