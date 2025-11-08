// C++ include files
#include <cstdio>
#include <utility>

// CMSSW include files
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/api.h"
#include "TrivialSerialisation/Common/interface/alpaka/SerialiserFactory.h"


#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"


#include <condition_variable>
#include <mutex>
#include <cassert>

// local include files
#include <TBufferFile.h>
#include <TClass.h>


#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class MPIReceiverPortable : public stream::SynchronizingEDProducer<> {
public:
  MPIReceiverPortable(edm::ParameterSet const& config)
      : SynchronizingEDProducer<>(config),
        upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(this->producesCollector().produces<MPIToken>()),
        patterns_(edm::productPatterns(config.getParameter<std::vector<std::string>>("products"))),
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    printf("MPIReceiverPortable: Entering constructor\n");
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
          << "Invalid MPIReceiverPortable instance value, please use a value between 1 and 255";
    }

    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
    products_.reserve(products.size());
    printf("MPIReceiverPortable: number of products to receive: %zu\n", products.size());
    for (auto const& product : products) {
      printf("MPIReceiverPortable: getting parameters\n");
      auto const& type = product.getParameter<std::string>("type");
      auto const& label = product.getParameter<std::string>("label");
      printf("MPIReceiverPortable: getting parameters done\n");
      Entry entry;
      printf("MPIReceiverPortable: AAAAAAA\n");
      entry.type = edm::TypeWithDict::byName(type);
      printf("MPIReceiverPortable: BBBBBBB (type: \"%s\")\n", type.c_str());
      entry.wrappedType = edm::TypeWithDict::byName("edm::Wrapper<" + type + ">");
      // entry.wrappedType = edm::TypeWithDict::byName("edm::DeviceProduct<" + type + ">");
      // entry.token = produces(edm::TypeID{entry.type.typeInfo()}, label);
      printf("MPIReceiverPortable: producing token for type \"%s\" and label \"%s\"\n", entry.type.name().c_str(), label.c_str());
      entry.token = this->producesCollector().produces(edm::TypeID{entry.type.typeInfo()}, label);

      edm::LogVerbatim("MPIReceiverPortable") << "receive type \"" << entry.type.name() << "\" for label \"" << label
                                      << "\" over MPI channel instance " << this->instance_;
      printf("MPIReceiverPortable: produced token for type \"%s\" and label \"%s\"\n", entry.type.name().c_str(), label.c_str());
      products_.emplace_back(std::move(entry));
      printf("MPIReceiverPortable: emplaced token for type \"%s\" and label \"%s\"\n", entry.type.name().c_str(), label.c_str());
    }
    printf("MPIReceiverPortable constructed with %zu products\n", products_.size());
  }

  // void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
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
    printf("MPIReceiverPortable::acquire() done\n");
  }

  void produce(device::Event& event, device::EventSetup const& /*unused*/) override {
    printf("Entering MPIReceiverPortable::produce()\n");
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);
    printf("done event.get()\n");
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

    printf("MPIReceiverPortable: Starting to process %zu products\n", products_.size());
    for (auto const& entry : products_) {
      printf("MPIReceiverPortable: Processing entry:\n");
      printf("  type name: %s\n", entry.type.name().c_str());
      printf("  wrapped type name: %s\n", entry.wrappedType.name().c_str());

      printf("MPIReceiverPortable: making wrapper of product of type \"%s\"\n", entry.wrappedType.name().c_str());
      std::unique_ptr<edm::WrapperBase> wrapper(static_cast<edm::WrapperBase*>(entry.wrappedType.getClass()->New()));
      printf("MPIReceiverPortable: wrapper constructed\n");
      auto product_meta = received_meta_->getNext();

      if (product_meta.kind == ProductMetadata::Kind::Missing) {
        edm::LogWarning("MPIReceiverPortable") << "Product " << entry.type.name() << " was not received.";
        continue;  // Skip products that weren't sent
      }

      else if (product_meta.kind == ProductMetadata::Kind::Serialized) {
        printf("YYYYYYYYYY\n");
        auto productBuffer = TBufferFile(TBuffer::kRead, product_meta.sizeMeta);
        // assert(!wrapper->hasTrivialCopyTraits() && "mismatch between expected and factual metadata type");
        assert(buffer_offset_ < full_buffer_size && "serialized data buffer is shorter than expected");
        productBuffer.SetBuffer(buf_ptr + buffer_offset_, product_meta.sizeMeta, kFALSE /* adopt = false */);
        buffer_offset_ += product_meta.sizeMeta;
        entry.wrappedType.getClass()->Streamer(wrapper.get(), productBuffer);
      }

      else if (product_meta.kind == ProductMetadata::Kind::TrivialCopy) {
        // assert(wrapper->hasTrivialCopyTraits() && "mismatch between expected and factual metadata type");
        // wrapper->markAsPresent();
        // std::unique_ptr<ngt::SerialiserBase> serialiser{
        //   ngt::SerialiserFactory::get()->tryToCreate(entry.objectType_.typeInfo().name())};
        printf("MPIReceiverPortable: Trying to create a serializer for type \"%s\"\n", entry.type.name().c_str());
        std::unique_ptr<ngt::SerialiserBase> serialiser{
          ngt::SerialiserFactoryPortable::get()->tryToCreate(entry.type.typeInfo().name())}; // is this ame correct?
        if (!serialiser) {
          throw cms::Exception("SerializerError")
          << "Receiver could not retrieve its serializer when it was expected";
        }
        printf("MPIReceiverPortable: Successfully created a serializer for type \"%s\"\n", entry.type.name().c_str());
        auto writer = serialiser->initialize(*wrapper);
        edm::AnyBuffer buffer = writer->parameters();  // constructs buffer with typeid
        assert(buffer.size_bytes() == product_meta.sizeMeta);
        // TDL: can we add func to AnyBuffer to replace pointer to the data?
        std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);
        // why both of these methods are called initialize? I find this rather confusing
        writer->initialize(buffer, event.queue());
        token.channel()->receiveInitializedTrivialCopyWithQueue(instance_, *writer, event.queue());
        printf("MPIReceiverPortable: receiveInitializedTrivialCopy done\n");
        writer->trivialCopyFinalize();
        printf("MPIReceiverPortable: trivialCopyFinalize done\n");
      }
      // put the data into the Event
      event.put(entry.token, std::move(wrapper));
      printf("MPIReceiverPortable: event.put() done for type \"%s\"\n", entry.type.name().c_str());
    }

    // Verify that we consumed all serialized data from the buffer
    if (received_meta_->hasSerialized() && serialized_buffer) {
      assert(static_cast<int>(buffer_offset_) == received_meta_->serializedBufferSize() &&
             "serialized data buffer is not equal to the expected length");
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
    printf("MPIReceiverPortable::produce() done\n");
  }

private:
  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDPutToken token;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<edm::ProductNamePattern> patterns_;  // patterns to match products to receive over the channel
  std::vector<Entry> products_;             // data to be read over the channel and put into the Event
  int32_t const instance_;                  // instance used to identify the source-destination pair

  std::shared_ptr<ProductMetadataBuilder> received_meta_;
};


}

// #include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortable);