// C++ include files
#include <cassert>
#include <string>
#include <vector>

// CMSSW include files
#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/api.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

// Data type headers for template specializations
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/HcalRecHit/interface/alpaka/HcalRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The template parameters document which device product types this module handles.
  // The implementation is runtime-generic, identical to MPIReceiverPortable.
  template <typename... PortableTypes>
  class MPIReceiverPortableTemplated : public stream::SynchronizingEDProducer<> {
  public:
    MPIReceiverPortableTemplated(edm::ParameterSet const& config)
        : stream::SynchronizingEDProducer<>(config),
          upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
          token_(this->producesCollector().template produces<MPIToken>()),
          instance_(config.getParameter<int32_t>("instance")) {
      if (instance_ < 1 or instance_ > 255) {
        throw cms::Exception("InvalidValue")
            << "Invalid MPIReceiverPortableTemplated instance value, please use a value between 1 and 255";
      }

      auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
      products_.reserve(products.size());
      for (auto const& product : products) {
        auto const& type = product.getParameter<std::string>("type");
        auto const& label = product.getParameter<std::string>("label");

        Entry entry;
        entry.typeName = type;

        // detect PathStateToken entries
        if (type == "edm::PathStateToken") {
          entry.isPathState = true;
          entry.token = this->producesCollector().template produces<edm::PathStateToken>();
          products_.emplace_back(std::move(entry));
          continue;
        }

        // look up the device serialiser by human-readable name
        std::unique_ptr<ngt::SerialiserBase> deviceSerialiser{
            ngt::SerialiserFactoryDevice::get()->tryToCreate(type)};
        if (!deviceSerialiser) {
          throw cms::Exception("MPIReceiverPortableTemplated")
              << "No device serialiser found for type '" << type
              << "'. Only device products are supported by MPIReceiverPortableTemplated. "
              << "Use MPIReceiver for host products.";
        }

        // get the event-store type from the serialiser
        edm::TypeID eventStoreType{deviceSerialiser->eventStoreTypeInfo()};

        entry.token = this->producesCollector().produces(eventStoreType, label);

        LogTrace("MPIReceiverPortableTemplated") << "receive type \"" << eventStoreType << "\" (" << type
                                                 << ") for label \"" << label << "\" over MPI channel instance "
                                                 << instance_;

        products_.emplace_back(std::move(entry));
      }
    }

    void acquire(device::Event const& event, device::EventSetup const&) override {
      MPIToken const& token = event.get(upstream_);
      received_meta_ = std::make_shared<ProductMetadataBuilder>();
      token.channel()->receiveMetadata(instance_, received_meta_);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      MPIToken token = event.get(upstream_);

      // if filter was false before the sender, receive nothing
      if (received_meta_->productCount() == -1) {
        event.emplace(token_, token);
        return;
      }

      for (auto const& entry : products_) {
        // PathStateToken entries are not sent as products; produce them if the event was not filtered
        if (entry.isPathState) {
          event.put(entry.token, std::make_unique<edm::PathStateToken>());
          continue;
        }

        auto product_meta = received_meta_->getNext();

        if (product_meta.kind == ProductMetadata::Kind::Missing) {
          edm::LogWarning("MPIReceiverPortableTemplated") << "Product " << entry.typeName << " was not received.";
          continue;
        }

        if (product_meta.kind == ProductMetadata::Kind::TrivialCopy) {
          std::unique_ptr<ngt::SerialiserBase> serialiser{
              ngt::SerialiserFactoryDevice::get()->tryToCreate(entry.typeName)};
          if (!serialiser) {
            throw cms::Exception("MPIReceiverPortableTemplated")
                << "No device serialiser found for type '" << entry.typeName << "'";
          }
          auto writer = serialiser->writer();
          ::ngt::AnyBuffer buffer = writer->uninitialized_parameters();
          assert(buffer.size_bytes() == product_meta.sizeMeta);
          std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);
          writer->initialize(event.queue(), buffer);
          token.channel()->receiveInitializedTrivialCopy(instance_, *writer);
          writer->finalize();
          event.put(entry.token, writer->get(event.metadata()));
        } else {
          throw cms::Exception("MPIReceiverPortableTemplated")
              << "Unexpected product metadata kind for device product '" << entry.typeName << "'. "
              << "Only TrivialCopy is supported for device products.";
        }
      }

      event.emplace(token_, token);
    }

  private:
    struct Entry {
      std::string typeName;  // human-readable type name from config
      edm::EDPutToken token;
      bool isPathState = false;
    };

    edm::EDGetTokenT<MPIToken> const upstream_;
    edm::EDPutTokenT<MPIToken> const token_;
    std::vector<Entry> products_;
    int32_t const instance_;

    std::shared_ptr<ProductMetadataBuilder> received_meta_;
  };

  // Type aliases for the device collections used in the HLT configuration
  using EcalDigiDevice = EcalDigiDeviceCollection;
  using EcalUncalibratedRecHitDevice = EcalUncalibratedRecHitDeviceCollection;
  using HcalRecHitDevice = hcal::RecHitDeviceCollection;
  using PFRecHitDevice = reco::PFRecHitDeviceCollection;
  using PFClusterDevice = reco::PFClusterDeviceCollection;
  using PFRecHitFractionDevice = reco::PFRecHitFractionDeviceCollection;
  using TracksDevice = reco::TracksSoACollection;
  using ZVertexDevice = reco::ZVertexSoACollection;

  // Specializations for each type combination
  using MPIReceiverPortableEcalDigiSoA = MPIReceiverPortableTemplated<EcalDigiDevice, EcalDigiDevice>;
  using MPIReceiverPortableEcalUncalibratedRecHitSoA =
      MPIReceiverPortableTemplated<EcalUncalibratedRecHitDevice, EcalUncalibratedRecHitDevice>;
  using MPIReceiverPortableHbheRecoSoA = MPIReceiverPortableTemplated<HcalRecHitDevice>;
  using MPIReceiverPortablePFRecHitSoA = MPIReceiverPortableTemplated<PFRecHitDevice>;
  using MPIReceiverPortablePFClusterSoA = MPIReceiverPortableTemplated<PFClusterDevice, PFRecHitFractionDevice>;
  using MPIReceiverPortablePixelTracksSoA = MPIReceiverPortableTemplated<TracksDevice>;
  using MPIReceiverPortablePixelVerticesSoA = MPIReceiverPortableTemplated<ZVertexDevice>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableEcalDigiSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableEcalUncalibratedRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableHbheRecoSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePFRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePFClusterSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePixelTracksSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePixelVerticesSoA);
