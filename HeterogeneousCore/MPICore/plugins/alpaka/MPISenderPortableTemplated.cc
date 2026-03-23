// C++ include files
#include <string>
#include <vector>

// CMSSW include files
#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/api.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

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
  // The implementation is runtime-generic, identical to MPISenderPortable.
  template <typename... PortableTypes>
  class MPISenderPortableTemplated : public stream::SynchronizingEDProducer<> {
  public:
    MPISenderPortableTemplated(edm::ParameterSet const& config)
        : stream::SynchronizingEDProducer<>(config),
          upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
          token_(this->producesCollector().template produces<MPIToken>()),
          instance_(config.getParameter<int32_t>("instance")) {
      if (instance_ < 1 or instance_ > 255) {
        throw cms::Exception("InvalidValue")
            << "Invalid MPISenderPortableTemplated instance value, please use a value between 1 and 255";
      }

      auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
      products_.reserve(products.size());
      for (auto const& product : products) {
        auto const& type = product.getParameter<std::string>("type");
        auto const& label = product.getParameter<std::string>("label");
        auto const& instance = product.getParameter<std::string>("instance");

        Entry entry;
        entry.typeName = type;

        // detect PathStateToken entries
        if (type == "edm::PathStateToken") {
          entry.isFilter = true;
          entry.eventStoreType = edm::TypeID(typeid(edm::PathStateToken));
          entry.token =
              this->consumes(edm::TypeToGet{entry.eventStoreType, edm::PRODUCT_TYPE}, edm::InputTag{label, instance});
          products_.emplace_back(std::move(entry));
          continue;
        }

        // look up the device serialiser by human-readable name
        std::unique_ptr<ngt::SerialiserBase> deviceSerialiser{ngt::SerialiserFactoryDevice::get()->tryToCreate(type)};
        if (!deviceSerialiser) {
          throw cms::Exception("MPISenderPortableTemplated")
              << "No device serialiser found for type '" << type
              << "'. Only device products are supported by MPISenderPortableTemplated. "
              << "Use MPISender for host products.";
        }

        // get the event-store type from the serialiser
        edm::TypeID eventStoreType{deviceSerialiser->eventStoreTypeInfo()};
        entry.eventStoreType = eventStoreType;

        entry.token =
            this->consumes(edm::TypeToGet{eventStoreType, edm::PRODUCT_TYPE}, edm::InputTag{label, instance});

        LogTrace("MPISenderPortableTemplated")
            << "send type \"" << eventStoreType << "\" (" << type << ") from label \"" << label << "\" instance \""
            << instance << "\" over MPI channel instance " << instance_;

        products_.emplace_back(std::move(entry));
      }
    }

    void acquire(device::Event const& event, device::EventSetup const&) override {
      edm::Event const& iEvent = event;

      MPIToken const& token = event.get(upstream_);
      // count only non-filter products for the metadata
      size_t productCount = 0;
      for (auto const& entry : products_) {
        if (!entry.isFilter)
          ++productCount;
      }
      auto meta = std::make_shared<ProductMetadataBuilder>(productCount);
      is_active_ = true;
      readers_.clear();
      readers_.resize(productCount);

      size_t index = 0;
      for (auto const& entry : products_) {
        // check the filter entry: if the path didn't run, set productCount(-1) and stop
        if (entry.isFilter) {
          edm::Handle<edm::WrapperBase> handle(entry.eventStoreType.typeInfo());
          iEvent.getByToken(entry.token, handle);
          if (not handle.isValid()) {
            meta->setProductCount(-1);
            is_active_ = false;
            break;
          }
          // path ran — filter entry is valid, skip it (not sent as a product)
          continue;
        }

        edm::Handle<edm::WrapperBase> handle(entry.eventStoreType.typeInfo());
        iEvent.getByToken(entry.token, handle);

        if (handle.isValid()) {
          edm::WrapperBase const* wrapper = handle.product();
          std::unique_ptr<ngt::SerialiserBase> serialiser{
              ngt::SerialiserFactoryDevice::get()->tryToCreate(entry.typeName)};
          auto reader = serialiser->reader(*wrapper, *event.metadata(), not event.wasQueueUsed());
          ::ngt::AnyBuffer buffer = reader->parameters();
          meta->addTrivialCopy(buffer.data(), buffer.size_bytes());
          readers_[index] = std::move(reader);
        } else {
          meta->addMissing();
        }
        index++;
      }

      token.channel()->sendMetadata(instance_, meta);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      MPIToken token = event.get(upstream_);

      if (!is_active_) {
        event.emplace(token_, token);
        return;
      }

      for (auto const& reader : readers_) {
        if (reader) {
          token.channel()->sendTrivialCopyProduct(instance_, *reader);
        }
      }

      event.emplace(token_, token);
    }

  private:
    struct Entry {
      std::string typeName;  // human-readable type name from config
      edm::TypeID eventStoreType;
      edm::EDGetToken token;
      bool isFilter = false;
    };

    edm::EDGetTokenT<MPIToken> const upstream_;
    edm::EDPutTokenT<MPIToken> const token_;
    std::vector<Entry> products_;
    int32_t const instance_;

    std::vector<std::unique_ptr<const ::ngt::ReaderBase>> readers_;
    bool is_active_ = true;
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
  using MPISenderPortableEcalDigiSoA = MPISenderPortableTemplated<EcalDigiDevice, EcalDigiDevice>;
  using MPISenderPortableEcalUncalibratedRecHitSoA =
      MPISenderPortableTemplated<EcalUncalibratedRecHitDevice, EcalUncalibratedRecHitDevice>;
  using MPISenderPortableHbheRecoSoA = MPISenderPortableTemplated<HcalRecHitDevice>;
  using MPISenderPortablePFRecHitSoA = MPISenderPortableTemplated<PFRecHitDevice>;
  using MPISenderPortablePFClusterSoA = MPISenderPortableTemplated<PFClusterDevice, PFRecHitFractionDevice>;
  using MPISenderPortablePixelTracksSoA = MPISenderPortableTemplated<TracksDevice>;
  using MPISenderPortablePixelVerticesSoA = MPISenderPortableTemplated<ZVertexDevice>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableEcalDigiSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableEcalUncalibratedRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableHbheRecoSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePFRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePFClusterSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePixelTracksSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePixelVerticesSoA);
