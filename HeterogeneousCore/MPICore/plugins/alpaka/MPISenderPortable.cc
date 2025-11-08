#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

// ROOT headers
#include <TBufferFile.h>
#include <TClass.h>

// CMSSW include files
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
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

#include <iostream>


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class MPISenderPortable : public stream::SynchronizingEDProducer<> {
// class MPISenderPortable : public stream::EDProducer<> {
public:
  MPISenderPortable(edm::ParameterSet const& config)
      : SynchronizingEDProducer<>(config),
      // : EDProducer<>(config),
        upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(this->producesCollector().produces<MPIToken>()),
        patterns_(edm::productPatterns(config.getParameter<std::vector<std::string>>("products"))),
        instance_(config.getParameter<int32_t>("instance")),
        buffer_(std::make_unique<TBufferFile>(TBuffer::kWrite)),
        buffer_offset_(0),
        metadata_size_(0) 
        {
          printf("MPISenderPortable constructor called\n");




    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue") << "Invalid MPISenderPortable instance value, please use a value between 1 and 255";
    }

    products_.resize(patterns_.size());



    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");


      switch (product.branchType()) {
        case edm::InEvent:
          if (product.className() == kPathStatus or product.className() == kEndPathStatus)
            return;
          for (size_t pattern_index = 0; pattern_index < patterns_.size(); pattern_index++) {
            printf("checking if pattern %zu matches product %s\n", pattern_index, product.branchName().c_str());
            if (patterns_[pattern_index].match(product)) {
              Entry entry;
              entry.type = product.unwrappedType();
              entry.wrappedType = product.wrappedType();
              // TODO move this to EDConsumerBase::consumes() ?
              entry.getToken_ = this->consumes(
                  edm::TypeToGet{product.unwrappedTypeID(), edm::PRODUCT_TYPE},
                  edm::InputTag{product.moduleLabel(), product.productInstanceName(), product.processName()});
              printf("Type name: %s\n", product.unwrappedTypeID().typeInfo().name());
              printf("Event product: %s\n", product.branchName().c_str());
              printf("Product Instance Name: %s\n", product.productInstanceName().c_str());
              printf("process Name: %s\n", product.processName().c_str());
              edm::LogVerbatim("MPISenderPortable")
                  << "send product \"" << product.friendlyClassName() << '_' << product.moduleLabel() << '_'
                  << product.productInstanceName() << '_' << product.processName() << "\" of type \""
                  << entry.type.name() << "\" over MPI channel instance " << instance_;
              printf("MPISenderPortable registered product: %s\n", product.branchName().c_str());

              products_[pattern_index] = std::move(entry);
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
              << "Unexpected branch type " << product.branchType() << "\nPlease contact a Framework developer\n";
      }
    });
    printf("MPISenderPortable constructed with %zu patterns\n", patterns_.size());
    // TODO add an error if a pattern does not match any branches? how?
  }

  // void acquire(device::Event const& event, device::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) {
  void acquire(device::Event const& event, device::EventSetup const&) final {
    printf("Entering MPISenderPortable::acquire()\n");
    MPIToken token = event.get(upstream_);
    // we need 1 byte for type, 8 bytes for size and at least 8 bytes for trivial copy parameters buffer
    auto meta = std::make_shared<ProductMetadataBuilder>(products_.size() * 24);
    size_t index = 0;
    // this seems to work fine, but does this vector indeed persist between acquire() and produce()?
    // serializedBuffers_.clear();
    buffer_->Reset();
    buffer_offset_ = 0;
    printf("MPISenderPortable::acquire(): AAAAAA\n");
    meta->setProductCount(products_.size());
    printf("MPISenderPortable::acquire(): BBBBBB product count: %zu\n", products_.size());
    has_serialized_ = false;
    is_active_ = true;

    // estimate buffer size in the constructor

    for (auto const& entry : products_) {
      // Get the product
      printf("MPISenderPortable::acquire(): DDDDDD\n");
      edm::Handle<edm::WrapperBase> handle(entry.type.typeInfo());
      printf("MPISenderPortable::acquire(): EEEEEEE\n");
      static_cast<edm::Event const&>(event).getByToken(entry.getToken_, handle);
      printf("MPISenderPortable::acquire(): FFFFFF\n");

      // product count -1 indicates that the event was filtered out on given path
      printf("MPISenderPortable::acquire(): GGGGGG\n");
      if (!handle.isValid() && entry.type.name() == "edm::PathStateToken") {
        printf("MPISenderPortable::acquire(): FILTER HHHHHH\n");
        meta->setProductCount(-1);
        is_active_ = false;
        break;
      }
      printf("MPISenderPortable::acquire(): CCCCCCC\n");

      if (handle.isValid()) {
        edm::WrapperBase const* wrapper = handle.product();
        std::unique_ptr<ngt::SerialiserBase> serialiser{
          ngt::SerialiserFactoryPortable::get()->tryToCreate(entry.type.typeInfo().name())};

        if (serialiser) {
          auto reader = serialiser->initialize(*wrapper);
          edm::AnyBuffer buffer = reader->parameters();
          printf("MPISenderPortable::acquire(): DDD3DDD\n");
          meta->addTrivialCopy(buffer.data(), buffer.size_bytes());
        } else {
          printf("MPISenderPortable::acquire(): EEEE3EEE\n");
          TClass* cls = entry.wrappedType.getClass();
          if (!cls) {
            throw cms::Exception("MPISenderPortable") << "Failed to get TClass for type: " << entry.type.name();
          }
          size_t bufLen = serializeAndStoreBuffer_(index, cls, wrapper);
          meta->addSerialized(bufLen);
          has_serialized_ = true;
        }

      } else {
        printf("MPISenderPortable::acquire(): Adding missing product for entry %zu\n", index);
        // handle missing product
        meta->addMissing();
      }
      index++;
    }

    // meta_ = std::move(meta);

    printf("MPISenderPortable::acquire(): FFFFeFF\n");
    token.channel()->sendMetadata(instance_, meta);
    // std::this_thread::sleep_for(std::chrono::seconds(1));

    // // Submit sending of all products to run in the additional asynchronous threadpool
    // edm::Service<edm::Async> as;
    // as->runAsync(
    //     std::move(holder),
    //     [this, token, meta = std::move(meta)]() { token.channel()->sendMetadata(instance_, meta); },
    //     []() { return "Calling MPISenderPortable::acquire()"; });
  }

  void produce(device::Event& event, device::EventSetup const&) override {

    
    printf("Entering MPISenderPortable::produce()\n");

    MPIToken token = event.get(upstream_);
    // token.channel()->sendMetadata(instance_, meta_);
    // std::this_thread::sleep_for(std::chrono::seconds(1));

    if (!is_active_) {
      event.emplace(token_, token);
      return;
    }

    if (has_serialized_) {
      token.channel()->sendBuffer(buffer_->Buffer(), buffer_->Length(), instance_, EDM_MPI_SendSerializedProduct);
    }

    for (auto const& entry : products_) {
      edm::Handle<edm::WrapperBase> handle(entry.type.typeInfo());
      // event.getByToken(entry.token, handle);
      static_cast<edm::Event const&>(event).getByToken(entry.getToken_, handle);


      edm::WrapperBase const* wrapper = handle.product();
      // we don't send missing products
      if (handle.isValid()) {
        printf("MPISenderPortable: Trying to create a serializer for type \"%s\"\n", entry.type.name().c_str());
      std::unique_ptr<ngt::SerialiserBase> serialiser{
          ngt::SerialiserFactoryPortable::get()->tryToCreate(entry.type.typeInfo().name())};
        if (serialiser) {
          printf("MPISenderPortable: Successfully created a serializer for type \"%s\"\n", entry.type.name().c_str());
          auto reader = serialiser->initialize(*wrapper);
          printf("MPISenderPortable: initialized reader for type \"%s\"\n", entry.type.name().c_str());
          token.channel()->sendTrivialCopyProductTemplated(instance_, *reader);
          printf("MPISenderPortable: sendTrivialCopyProduct done for type \"%s\"\n", entry.type.name().c_str());
        }
      }
    }
    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  size_t serializeAndStoreBuffer_(size_t index, TClass* type, void const* product) {
    buffer_->ResetMap();
    type->Streamer(const_cast<void*>(product), *buffer_);
    size_t prod_size = buffer_->Length() - buffer_offset_;
    buffer_offset_ = buffer_->Length();
    return prod_size;
  }

  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDGetToken getToken_;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<edm::ProductNamePattern> patterns_;  // branches to read from the Event and send over the MPI channel
  std::vector<Entry> products_;                    // types and tokens corresponding to the branches
  int32_t const instance_;                         // instance used to identify the source-destination pair
  std::unique_ptr<TBufferFile> buffer_;
  size_t buffer_offset_;
  size_t metadata_size_;
  bool has_serialized_ = false;
  bool is_active_ = true;

  // metadata
  std::shared_ptr<ProductMetadataBuilder> meta_;

};

}

// #include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortable);
