// C++ include files
#include <utility>

// CMSSW include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/alpaka/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/alpaka/api.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"

#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/chain_first.h"
// #include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TrivialSerialisation/Common/interface/alpaka/SerialiserBase.h"
#include "TrivialSerialisation/Common/interface/alpaka/SerialiserFactory.h"

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"

#include <condition_variable>
#include <mutex>
#include <cassert>

// local include files
#include "HeterogeneousCore/MPICore/interface/messages.h"
#include <TBufferFile.h>
#include <TClass.h>


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class MPIReceiverPortable : public stream::SynchronizingEDProducer<> {
public:
  MPIReceiverPortable(edm::ParameterSet const& config)
      : SynchronizingEDProducer<>(config),
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    upstream_ = consumes(config.getParameter<edm::InputTag>("upstream"));
    token_ = produces();

    eventProducts_.reserve(eventPatterns_.size());

    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");
      static const std::string_view kBackend("backend");

      switch (product.branchType()) {
        case edm::InEvent:
          if (product.className() == kPathStatus or product.className() == kEndPathStatus) {
            return;
          }
          if (product.productInstanceName() == kBackend) {
            return;
          }
          for (auto& pattern : eventPatterns_) {
            if (pattern.match(product)) {
              // check that the product is not transient
              if (product.transient()) {
                // edm::LogWarning("GenericClonerPortable")
                // << "Event product " << product.branchName() << " of type " << product.unwrappedType()
                // << " is transient, will not be cloned.";
                // break;
              }
              if (verbose_) {
                edm::LogInfo("GenericClonerPortable")
                    << "will clone Event product " << product.branchName() << " of type " << product.unwrappedType();
              }
              Entry entry;
              entry.objectType_ = product.unwrappedType();
              entry.wrappedType_ = product.wrappedType();

              // TODO move this to EDConsumerBase::consumes() ?
              entry.getToken_ = this->consumes(
                  edm::TypeToGet{product.unwrappedTypeID(), edm::PRODUCT_TYPE},
                  edm::InputTag{product.moduleLabel(), product.productInstanceName(), product.processName()});
              printf("Type name: %s\n", product.unwrappedTypeID().typeInfo().name());
              printf("Event product: %s\n", product.branchName().c_str());
              printf("Product Instance Name: %s\n", product.productInstanceName().c_str());
              printf("process Name: %s\n", product.processName().c_str());

              entry.putToken_ =
                  this->producesCollector().produces(product.unwrappedTypeID(), product.productInstanceName());
              eventProducts_.emplace_back(std::move(entry));
              break;
            }
          }
          break;

        case edm::InLumi:
        case edm::InRun:
        case edm::InProcess:
          // lumi, run and process products are not supported
          break;

        default:
          throw edm::Exception(edm::errors::LogicError)
              << "Unexpected product type " << product.branchType() << "\nPlease contact a Framework developer.";
      }
    });


    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
          << "Invalid MPIReceiverPortable instance value, please use a value between 1 and 255";
    }

    printf("MPIReceiverPortable constructed with %zu patterns\n", eventProducts_.size());
    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
    printf("MPIReceiverPortable configured to receive %zu products\n", products.size());
  }

  void acquire(device::Event const& event, device::EventSetup const&) final {
    printf("Entering MPIReceiverPortable::acquire()\n");
    MPIToken token = event.get(upstream_);

    //also try unique or optional
    received_meta_ = std::make_shared<ProductMetadataBuilder>();

    token.channel()->receiveMetadata(instance_, received_meta_);

    // edm::Service<edm::Async> as;
    // as->runAsync(
    //     std::move(holder),
    //     [this, token]() { token.channel()->receiveMetadata(instance_, received_meta_); },
    //     []() { return "Calling MPIReceiverPortable::acquire()"; });
  }

  void produce(device::Event& event, device::EventSetup const&) override{
    printf("Entering MPIReceiverPortable::produce()\n");
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);
    // see the summary of metadata for dubug purposes
    // received_meta_->debugPrintMetadataSummary();

    // if filter was false before the sender, receive nothing
    if (received_meta_->productCount() == -1) {
      event.emplace(token_, token);
      return;
    }

    char* buf_ptr = nullptr;
    size_t full_buffer_size = 0;
    size_t buffer_offset_ = 0;

    std::unique_ptr<TBufferFile> serialized_buffer;

    if (received_meta_->hasSerialized()) {
      serialized_buffer = token.channel()->receiveSerializedBuffer(instance_, received_meta_->serializedBufferSize());
      buf_ptr = serialized_buffer->Buffer();
      full_buffer_size = serialized_buffer->BufferSize();
    }

    for (auto const& product : eventProducts_) {

      // std::unique_ptr<edm::WrapperBase> wrapper(
      //     reinterpret_cast<edm::WrapperBase*>(product.wrappedType_.getClass()->New()));

      std::unique_ptr<edm::WrapperBase> wrapper(static_cast<edm::WrapperBase*>(product.wrappedType_.getClass()->New()));
      auto product_meta = received_meta_->getNext();

      if (product_meta.kind == ProductMetadata::Kind::Missing) {
        edm::LogWarning("MPIReceiverPortable") << "Product " << product.type_.name() << " was not received.";
        continue;  // Skip products that weren't sent
      }

      else if (product_meta.kind == ProductMetadata::Kind::Serialized) {
        auto productBuffer = TBufferFile(TBuffer::kRead, product_meta.sizeMeta);
        // assert(!wrapper->hasTrivialCopyTraits() && "mismatch between expected and factual metadata type");
        assert(buffer_offset_ < full_buffer_size && "serialized data buffer is shorter than expected");
        productBuffer.SetBuffer(buf_ptr + buffer_offset_, product_meta.sizeMeta, kFALSE /* adopt = false */);
        buffer_offset_ += product_meta.sizeMeta;
        product.wrappedType_.getClass()->Streamer(wrapper.get(), productBuffer);
      }

      else if (product_meta.kind == ProductMetadata::Kind::TrivialCopy) {
        // assert(wrapper->hasTrivialCopyTraits() && "mismatch between expected and factual metadata type");
        // wrapper->markAsPresent();
        // std::unique_ptr<ngt::SerialiserBase> serialiser{
        //   ngt::SerialiserFactory::get()->tryToCreate(entry.objectType_.typeInfo().name())};
      std::unique_ptr<ngt::SerialiserBase> serialiser{
          ngt::SerialiserFactoryPortable::get()->tryToCreate(product.objectType_.typeInfo().name())};

        if (!serialiser) {
          throw cms::Exception("SerializerError")
          << "Receiver could not retrieve its serializer when it was expected";
        }

        auto writer = serialiser->initialize(*wrapper);
        edm::AnyBuffer buffer = writer->parameters();  // constructs buffer with typeid
        assert(buffer.size_bytes() == product_meta.sizeMeta);
        // TDL: can we add func to AnyBuffer to replace pointer to the data?
        std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);
        // why both of these methods are called initialize? I find this rather confusing
        writer->initialize(buffer, event.queue());
        token.channel()->receiveInitializedTrivialCopy(event.queue(), instance_, *writer);

        
        writer->trivialCopyFinalize();
      }
      // put the data into the Event
      event.put(product.putToken_, std::move(wrapper));
    }

    if (received_meta_->hasSerialized()) {
      assert(static_cast<int>(buffer_offset_) == received_meta_->serializedBufferSize() &&
             "serialized data buffer is not equal to the expected length");
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  struct Entry {
    edm::TypeWithDict type_;
    edm::TypeWithDict wrappedType_;
    edm::TypeWithDict objectType_;
    edm::EDPutToken putToken_;
    edm::EDGetToken getToken_;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<Entry> eventProducts_;             // data to be read over the channel and put into the Event
  int32_t const instance_;                  // instance used to identify the source-destination pair
  std::vector<edm::ProductNamePattern> eventPatterns_;
  bool verbose_;
  std::shared_ptr<ProductMetadataBuilder> received_meta_;
};


}

// #include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortable);