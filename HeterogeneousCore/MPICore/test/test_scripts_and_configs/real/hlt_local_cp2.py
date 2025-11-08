import FWCore.ParameterSet.Config as cms

# run over HLTPhysics data from run 383363
from hlt import process

process.load("Configuration.StandardSequences.Accelerators_cff")


# run with 32 threads, 24 concurrent events, 1 concurrent lumisection, over 10k events
process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1  # MPIController does not support concurrent lumisections
process.maxEvents.input = 600 #10300

# do not print a final summary
process.options.wantSummary = False
process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)

# enable MPI messages
process.MessageLogger.MPISender = cms.untracked.PSet()
process.MessageLogger.MPIReceiver = cms.untracked.PSet()

# set up the MPI communication channel
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = "file:server.uri"

from HeterogeneousCore.MPICore.mpiController_cfi import mpiController as mpiController_
process.mpiController = mpiController_.clone()

# send the raw data over MPI
process.mpiSenderRawData = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(1),
    products = cms.vstring("rawDataCollector") 
)

# schedule the communication before the ECAL local reconstruction
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence.insert(0, process.mpiController)
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence.insert(1, process.mpiSenderRawData)

# # receive the ECAL digis SoA over MPI
# process.hltEcalDigisSoA = cms.EDProducer("MPIReceiverPortableE@alpaka",
#     upstream = cms.InputTag("mpiSenderRawData"),
#     instance = cms.int32(20),
#     products = cms.VPSet(
#     cms.PSet
#     (
#         type = cms.string("PortableHostCollection<EcalDigiSoALayout<128,false> >"),
#         label = cms.string("ebDigis"),
#         instance = cms.string("ebDigis")
#     )
#     , 
#     cms.PSet(
#         type = cms.string("PortableHostCollection<EcalDigiSoALayout<128,false> >"),
#         label = cms.string("eeDigis"),
#         instance = cms.string("eeDigis")
#     )
#     # , cms.PSet(
#     #     type = cms.string("unsigned short"),
#     #     label = cms.string("backend")
#     # )
#     )
# )

# # receive the ECAL uncalibrated rechits SoA over MPI
# process.hltEcalUncalibRecHitSoA = cms.EDProducer("MPIReceiverPortable@alpaka",
#     upstream = cms.InputTag("hltEcalDigisSoA"),
#     instance = cms.int32(21),
#     products = cms.VPSet(
#     cms.PSet(
#         type = cms.string("PortableHostCollection<EcalUncalibratedRecHitSoALayout<128,false> >"),
#         label = cms.string("EcalUncalibRecHitsEB"),
#         instance = cms.string("EcalUncalibRecHitsEB")
#     ), 
#     cms.PSet(
#         type = cms.string("PortableHostCollection<EcalUncalibratedRecHitSoALayout<128,false> >"),
#         label = cms.string("EcalUncalibRecHitsEE"),
#         instance = cms.string("EcalUncalibRecHitsEE")
#     )
#     # , cms.PSet(
#     #     type = cms.string("unsigned short"),
#     #     label = cms.string("backend")
#     # )
#     )
# )

# delete the modules runnig remotely
del process.hltHcalDigisSoA
# del process.hltHcalDigis # TODO: There are modules in the local that depends on this one.

# receive the HBHE rechits SoA over MPI
process.hltHbheRecoSoA = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("mpiSenderRawData"),
    instance = cms.int32(11),
    products = cms.VPSet(
    cms.PSet(
        type = cms.string("PortableHostCollection<hcal::HcalRecHitSoALayout<128,false> >"),
        # type = cms.string("edm::Wrapper<edm::DeviceProduct<PortableHostCollection<hcal::HcalRecHitSoALayout<128ul, false> > > >"),
        # type = cms.string("PortableDeviceCollection<hcal::HcalRecHitSoALayout<128,false>, Device>"),
        # type = cms.string("PortableCollection<hcal::HcalRecHitSoALayout<128, false>, Device>"),
        # type = cms.string("edm::DeviceProduct<PortableCollection<hcal::HcalRecHitSoALayout<128, false>, Device>>"),
        # type = cms.string("PortableHostCollection<hcal::HcalRecHitSoA>"),
        instance = cms.string(""),
        label = cms.string("")
    ),
    # , cms.PSet(
    #    type = cms.string("unsigned short"),
    #    label = cms.string("backend")
    # )
    cms.PSet(
        type = cms.string("edm::PathStateToken"),
        label = cms.string("")
    ))
)



# # receive the HBHE PF rechits SoA over MPI
# process.hltParticleFlowRecHitHBHESoA = cms.EDProducer("MPIReceiverPortable@alpaka",
#     upstream = cms.InputTag("hltHbheRecoSoA"),
#     instance = cms.int32(12),
#     products = cms.vstring(
#         "128falserecoPFRecHitSoALayoutPortableHostCollection_hltParticleFlowRecHitHBHESoA__*",
#         "uint_hltParticleFlowRecHitHBHESoA__*",
#         # "ushort_hltParticleFlowRecHitHBHESoA_backend_*",
#     )
# )
# process.hltParticleFlowRecHitHBHESoA = cms.EDProducer("MPIReceiverPortable@alpaka",
#     upstream = cms.InputTag("hltHbheRecoSoA"),
#     instance = cms.int32(12),
#     products = cms.VPSet(cms.PSet(
#         type = cms.string("PortableHostCollection<reco::PFRecHitSoALayout<128,false> >"),
#         label = cms.string(""),
#         instance = cms.string("")
#     )
#     , cms.PSet(
#         type = cms.string("unsigned int"),
#         label = cms.string("")
#     )#, cms.PSet(
#     #     type = cms.string("unsigned short"),
#     #     label = cms.string("backend")
#     # )
#     )
# )




# # receive the HBHE PF clusters SoA over MPI
# process.hltParticleFlowClusterHBHESoA = cms.EDProducer("MPIReceiverPortable@alpaka",
#     upstream = cms.InputTag("hltParticleFlowRecHitHBHESoA"),
#     instance = cms.int32(13),
#     products = cms.vstring(
#         "128falserecoPFClusterSoALayoutPortableHostCollection_hltParticleFlowClusterHBHESoA__*",
#         "128falserecoPFRecHitFractionSoALayoutPortableHostCollection_hltParticleFlowClusterHBHESoA__*",
#         # "ushort_hltParticleFlowClusterHBHESoA_backend_*",
#     )
# )
# process.hltParticleFlowClusterHBHESoA = cms.EDProducer("MPIReceiverPortable@alpaka",
#     upstream = cms.InputTag("hltParticleFlowRecHitHBHESoA"),
#     instance = cms.int32(13),
#     products = cms.VPSet(cms.PSet(
#         type = cms.string("PortableHostCollection<reco::PFClusterSoALayout<128,false> >"),
#         label = cms.string(""),
#         instance = cms.string("")
#     )
#     , cms.PSet(
#         type = cms.string("PortableHostCollection<reco::PFRecHitFractionSoALayout<128,false> >"),
#         label = cms.string(""),
#         instance = cms.string("")
#     )
#     # , cms.PSet(
#     #     type = cms.string("unsigned short"),
#     #     label = cms.string("backend")
#     # )
#     )
# )





# schedule the communication before the HBHE local reconstruction
process.HLTDoLocalHcalSequence.insert(0, process.mpiController)
process.HLTDoLocalHcalSequence.insert(1, process.mpiSenderRawData)

process.HLTStoppedHSCPLocalHcalReco.insert(0, process.mpiController)
process.HLTStoppedHSCPLocalHcalReco.insert(1, process.mpiSenderRawData)

# schedule the communication before the HBHE PF reconstruction
process.HLTPFHcalClustering.insert(0, process.mpiController)
process.HLTPFHcalClustering.insert(1, process.mpiSenderRawData)


process.ProcessAcceleratorAlpaka.setBackend("serial_sync")


# schedule the communication for every event
process.Offload = cms.Path(
    process.mpiController +
    process.mpiSenderRawData +
    process.hltHbheRecoSoA #+
    # process.hltParticleFlowRecHitHBHESoA +
    # process.hltParticleFlowClusterHBHESoA #+
    # process.hltEcalDigisSoA +
    # process.hltEcalUncalibRecHitSoA
)



process.schedule.append(process.Offload)

# process.Tracer = cms.Service("Tracer")


