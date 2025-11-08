import FWCore.ParameterSet.Config as cms

from hlt import process as _process



process = cms.Process("REMOTE")

process.load("Configuration.StandardSequences.Accelerators_cff")

# load the event setup
for module in _process.psets.keys():
    setattr(process, module, getattr(_process, module).clone())
for module in _process.es_sources.keys():
    setattr(process, module, getattr(_process, module).clone())
for module in _process.es_producers.keys():
    setattr(process, module, getattr(_process, module).clone())


process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1

# do not print a final summary
process.options.wantSummary = False
process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)

# enable MPI messages
process.MessageLogger.MPISender = cms.untracked.PSet()
process.MessageLogger.MPIReceiver = cms.untracked.PSet()

# print debug messages
process.MessageLogger.cerr.threshold = "DEBUG"

# set up the MPI communication channel
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = "file:server.uri"

process.source = cms.Source("MPISource",
  # FIXME this should not be necessary
  firstRun = cms.untracked.uint32(383631)
)

process.maxEvents.input = -1

# receive the raw data over MPI
process.rawDataCollector = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("source"),
    instance = cms.int32(1),
    products = cms.VPSet(cms.PSet(
        type = cms.string("FEDRawDataCollection"),
        label = cms.string("")
    ))
)

process.hltGetRaw = _process.hltGetRaw.clone()

# # HBHE local reconstruction from the HLT menu
process.hltHcalDigis = _process.hltHcalDigis.clone()
process.hltHcalDigisSoA = _process.hltHcalDigisSoA.clone()
process.hltHbheRecoSoA = _process.hltHbheRecoSoA.clone()
process.hltParticleFlowRecHitHBHESoA = _process.hltParticleFlowRecHitHBHESoA.clone()
process.hltParticleFlowClusterHBHESoA = _process.hltParticleFlowClusterHBHESoA.clone()


# The problematic module
process.hltHbhereco = _process.hltHbhereco.clone()


# send the HBHE rechits SoA over MPI
process.mpiSenderHbheRecoSoA = cms.EDProducer("MPISenderPortable@alpaka",
    upstream = cms.InputTag("rawDataCollector"),
    # upstream = cms.InputTag("hltHbheRecoSoA"),
    instance = cms.int32(11),
    products = cms.vstring(
        # "128falsehcalHcalRecHitSoALayoutPortableHostCollection_hltHbheRecoSoA__*",
        # "128falsehcalHcalRecHitSoALayoutPortableDeviceCollection_hltHbheRecoSoA__*",
        "128falsehcalHcalRecHitSoALayoutalpakaDevCudaRtvoidPortableDeviceCollectionedmDeviceProduct_hltHbheRecoSoA__*",
        # "ushort_hltHbheRecoSoA_backend_*",
        "*_HBHEActivity__*"
    ) 
)
# process.mpiSenderHbheRecoSoA = cms.EDProducer("MPISenderPortable@alpaka",
#     upstream = cms.InputTag("rawDataCollector"),
#     instance = cms.int32(11),
#     products = cms.VPSet(
#     cms.PSet(
#         type = cms.string("PortableHostCollection<hcal::HcalRecHitSoALayout<128,false> >"),
#         # type = cms.string("PortableHostCollection<hcal::HcalRecHitSoA>"),
#         instance = cms.string(""),
#         label = cms.string("hltHbheRecoSoA")
#     )
#     )
#     # products = cms.vstring(
#     #     "128falsehcalHcalRecHitSoALayoutPortableHostCollection_hltHbheRecoSoA__*",
#     #     # "ushort_hltHbheRecoSoA_backend_*",
#     # )
#     )


# # send the HBHE PF rechits SoA over MPI
# process.mpiSenderParticleFlowRecHitHBHESoA = cms.EDProducer("MPISenderPortable@alpaka",
#     upstream = cms.InputTag("mpiSenderHbheRecoSoA"),
#     instance = cms.int32(12),
#     products = cms.vstring(
#        "128falserecoPFRecHitSoALayoutPortableHostCollection_hltParticleFlowRecHitHBHESoA__*",
#        "uint_hltParticleFlowRecHitHBHESoA__*",
#     #    "ushort_hltParticleFlowRecHitHBHESoA_backend_*",
#     )
# )
# process.mpiSenderParticleFlowRecHitHBHESoA = cms.EDProducer("MPISenderPortable@alpaka",
#     upstream = cms.InputTag("mpiSenderHbheRecoSoA"),
#     instance = cms.int32(12),
#     products = cms.VPSet(
#     cms.PSet
#     (        
#         instance = cms.string(""),
#         label = cms.string("hltParticleFlowRecHitHBHESoA")
#     )
# ))


# # send the HBHE PF clusters SoA over MPI
# process.mpiSenderParticleFlowClusterHBHESoA = cms.EDProducer("MPISenderPortable@alpaka",
#     upstream = cms.InputTag("mpiSenderParticleFlowRecHitHBHESoA"),
#     instance = cms.int32(13),
#     products = cms.vstring(
#         "128falserecoPFClusterSoALayoutPortableHostCollection_hltParticleFlowClusterHBHESoA__*",
#         "128falserecoPFRecHitFractionSoALayoutPortableHostCollection_hltParticleFlowClusterHBHESoA__*",
#         # "ushort_hltParticleFlowClusterHBHESoA_backend_*",
#     )
# )
# process.mpiSenderParticleFlowClusterHBHESoA = cms.EDProducer("MPISenderPortable@alpaka",
#     upstream = cms.InputTag("mpiSenderParticleFlowRecHitHBHESoA"),
#     instance = cms.int32(13),
#     products = cms.VPSet(
#     cms.PSet
#     (
#         instance = cms.string(""),
#         label = cms.string("hltParticleFlowClusterHBHESoA")
#     ),
#     cms.PSet
#     (
#         instance = cms.string(""),
#         label = cms.string("hltParticleFlowClusterHBHESoA")
#     )
#     ))






# run the HBHE local reconstruction
process.HLTLocalHBHE = cms.Path(
    process.rawDataCollector +
    process.hltGetRaw +
    process.hltHcalDigis +
    process.hltHcalDigisSoA +
    process.hltHbheRecoSoA +
    process.hltHbhereco +
    process.mpiSenderHbheRecoSoA #+
    # process.hltParticleFlowRecHitHBHESoA +
    # process.mpiSenderParticleFlowRecHitHBHESoA +
    # process.hltParticleFlowClusterHBHESoA +
    # process.mpiSenderParticleFlowClusterHBHESoA
)

# # # ECAL local reconstruction from the HLT menu
# process.hltEcalDigisSoA = _process.hltEcalDigisSoA.clone()
# process.hltEcalUncalibRecHitSoA = _process.hltEcalUncalibRecHitSoA.clone()

# # send the ECAL digis SoA over MPI
# process.mpiSenderEcalDigisSoA = cms.EDProducer("MPISenderPortableEcalDigiSoA@alpaka",
#     upstream = cms.InputTag("rawDataCollector"),
#     # upstream = cms.InputTag("hltEcalDigisSoA"),
#     instance = cms.int32(20),
#     products = cms.vstring(
#         "128falseEcalDigiSoALayoutPortableHostCollection_hltEcalDigisSoA_ebDigis_*",
#         "128falseEcalDigiSoALayoutPortableHostCollection_hltEcalDigisSoA_eeDigis_*",
#         # "ushort_hltEcalDigisSoA_backend_*",
#     ) 
# )
# process.mpiSenderEcalDigisSoA = cms.EDProducer("MPISenderPortableEcalDigiSoA@alpaka",
#     upstream = cms.InputTag("rawDataCollector"),
#     instance = cms.int32(20),
#     products = cms.VPSet(
#     cms.PSet
#     (        
#         instance = cms.string("ebDigis"),
#         label = cms.string("hltEcalDigisSoA")
#     ),
#     cms.PSet
#     (
#         instance = cms.string("eeDigis"),
#         label = cms.string("hltEcalDigisSoA")
#     )
#     ))

# # send the ECAL uncalibrated rechits SoA over MPI
# process.mpiSenderEcalUncalibRecHitSoA = cms.EDProducer("MPISenderPortableEcalUncalibratedRecHitSoA@alpaka",
#     upstream = cms.InputTag("mpiSenderEcalDigisSoA"),
#     instance = cms.int32(21),
#     products = cms.vstring(
#         # "128falseEcalUncalibratedRecHitSoALayoutPortableHostCollection_hltEcalUncalibRecHitSoA_EcalUncalibRecHitsEB_*",
#         "128falseEcalUncalibratedRecHitSoALayoutPortableHostCollection_hltEcalUncalibRecHitSoA_EcalUncalibRecHitsEE_*",
#         # "ushort_hltEcalUncalibRecHitSoA_backend_*",
#     ) 
# )
# process.mpiSenderEcalUncalibRecHitSoA = cms.EDProducer("MPISenderPortable@alpaka",
#     upstream = cms.InputTag("mpiSenderEcalDigisSoA"),
#     instance = cms.int32(21),
#     products = cms.VPSet(
#     cms.PSet
#     (        
#         instance = cms.string("EcalUncalibRecHitsEB"),
#         label = cms.string("hltEcalUncalibRecHitSoA")
#     ),
#     cms.PSet
#     (
#         instance = cms.string("EcalUncalibRecHitsEE"),
#         label = cms.string("hltEcalUncalibRecHitSoA")
#     )
#     ))




# # run the ECAL local reconstruction
# process.HLTLocalECAL = cms.Path(
#     process.rawDataCollector +
#     process.hltGetRaw +
#     process.hltEcalDigisSoA +
#     process.mpiSenderEcalDigisSoA +
#     process.hltEcalUncalibRecHitSoA +
#     process.mpiSenderEcalUncalibRecHitSoA
# )



# process.ProcessAcceleratorAlpaka.setBackend("serial_sync")

# schedule the reconstruction
process.schedule = cms.Schedule(
    process.HLTLocalHBHE#,
    # process.HLTLocalECAL
)


# process.Tracer = cms.Service("Tracer")



# process.ThroughputService = cms.Service('ThroughputService',
#     enableDQM = cms.untracked.bool(False),
#     printEventSummary = cms.untracked.bool(True),
#     eventResolution = cms.untracked.uint32(100),
#     eventRange = cms.untracked.uint32(10300),
# )

# process.MessageLogger.cerr.ThroughputService = cms.untracked.PSet(
#     limit = cms.untracked.int32(10000000),
#     reportEvery = cms.untracked.int32(1)
# )
