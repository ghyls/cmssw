#include <string>
#include <string_view>
#include <vector>

// CMSSW include files
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include <alpaka/alpaka.hpp>

#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/EcalDigi/interface/EcalDigiSoA.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"

#include "HeterogeneousCore/MPICore/interface/api.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"




namespace ALPAKA_ACCELERATOR_NAMESPACE {

// TODO: move this to a common place so it can be used by both the sender and receiver
template<typename... Types, typename T, std::size_t N, typename Func>
void initTupleElementsFromArray(std::tuple<Types...>& tup, 
                          const std::array<T, N>& arr, Func&& func) {
    std::size_t i = 0;
    std::apply([&arr, &func, &i](auto&... elements) {
        ((elements = func(arr[i++])), ...);
    }, tup);
}


template<typename TokenType>
void sendSingleProduct(device::Event& event, MPIToken& mpiToken, 
                       TokenType& token, uint32_t instance) {
    auto const& handle = event.get(token);
    alpaka::wait(event.queue());
    auto const bufferSize = alpaka::getExtentProduct(handle.buffer());
    mpiToken.channel()->sendSurelyTrivialCopyProduct(instance, handle, bufferSize);
}

// TODO: This has too much in common with receiveProducts too.
template<typename... TokenTypes>
void sendProducts(device::Event& event, MPIToken& mpiToken, 
                 std::tuple<TokenTypes...>& tokens, uint32_t instance) {
    std::apply([&event, &mpiToken, instance](auto&... tokens) {
        ((sendSingleProduct(event, mpiToken, tokens, instance)), ...);
    }, tokens);
}


// TODO: Move these to a common place so they can be used by both the sender and receiver
template <template <typename...> class Template, typename T>
inline constexpr bool is_instance_of_v = false;

template <template <typename...> class Template, typename... Args>
inline constexpr bool is_instance_of_v<Template, Template<Args...>> = true;

template <typename T>
inline constexpr bool is_portable_collection_v =
  is_instance_of_v<PortableHostCollection, T> or is_instance_of_v<PortableDeviceCollection, T>;

template <typename... Ts>
inline constexpr bool are_all_portable_collections_v = (is_portable_collection_v<Ts> && ...);

template <typename... Ts>
  requires(are_all_portable_collections_v<Ts...>)
class MPISenderPortable : public stream::EDProducer<> {
public:
    MPISenderPortable(edm::ParameterSet const& config)
      : stream::EDProducer<>(config),
        upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces()),
        instance_(config.getParameter<int32_t>("instance")) {


    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
        << "Invalid MPISenderPortable instance value, please use a value between 1 and 255";
    }

      auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
      assert(products.size() == sizeof...(Ts) && "No template class MPIReceiverPortable<Ts...> found matching the number of products provided in the configuration");

      for (size_t i = 0; i < products.size(); ++i) {
        tags_[i] = edm::InputTag(products[i].getParameter<std::string>("label"),
                                  products[i].getParameter<std::string>("instance"));
      }

      initTupleElementsFromArray(tokens_, tags_, [this](const auto& tag) {
          return this->consumes(tag);
      });

      // TODO: We probably don't need this at all
      callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");

      switch (product.branchType()) {
        case edm::InEvent:
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

    // TODO add an error if a pattern does not match any branches? how?
  }

  void produce(device::Event& event, device::EventSetup const&) override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);

    sendProducts(event, token, tokens_, instance_);

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  
  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::tuple<device::EDGetToken<Ts>... > tokens_;
  std::array<edm::InputTag, std::tuple_size_v<decltype(tokens_)>> tags_;  // tokens for the products
  int32_t const instance_;                         // instance used to identify the source-destination pair
};

template class MPISenderPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>,
                                  PortableCollection<reco::PFRecHitFractionSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<EcalDigiSoALayout<128, false>>, 
                                 PortableCollection<EcalDigiSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE


#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using MPISenderPortableHbheRecoSoA = MPISenderPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
  using MPISenderPortablePFRecHitSoA = MPISenderPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
  using MPISenderPortablePFClusterSoA = MPISenderPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>,
                                                           PortableCollection<reco::PFRecHitFractionSoALayout<128, false>>>;
  using MPISenderPortableEcalDigiSoA = MPISenderPortable<PortableCollection<EcalDigiSoALayout<128, false>>, 
                                                         PortableCollection<EcalDigiSoALayout<128, false>>>;
  using MPISenderPortableEcalUncalibratedRecHitSoA = MPISenderPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>,
                                                                       PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;
}

DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableHbheRecoSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableEcalDigiSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePFRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePFClusterSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableEcalUncalibratedRecHitSoA);
