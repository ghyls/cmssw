// C++ include files
#include <memory>
#include <utility>

// CMSSW include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
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
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"



#include "HeterogeneousCore/MPICore/interface/api.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

template<typename... Types, typename T, std::size_t N, typename Func>
void initTupleElementsFromArray(std::tuple<Types...>& tup,
                        const std::array<T, N>& arr, Func&& func) {
  std::size_t i = 0;
  std::apply([&arr, &func, &i](auto&... elements) {
    ((elements = func(arr[i++])), ...);
  }, tup);
}

template<typename ProductType>
void receiveSingleProduct(device::Event& event, MPIToken& mpiToken,
                         device::EDPutToken<ProductType>& token, uint32_t instance) {
  
  auto product = std::make_unique<ProductType>(edm::Uninitialized{});
  mpiToken.channel()->receiveSurelyTrivialCopyProduct(event.queue(), instance, *product);
  
  event.put(token, std::move(product));
}

template<typename... TokenTypes>
void receiveProducts(device::Event& event, MPIToken& mpiToken,
                     std::tuple<TokenTypes...>& tokens, uint32_t instance) {
    std::apply([&event, &mpiToken, instance](auto&... tokens) {
        ((receiveSingleProduct(event, mpiToken, tokens, instance)), ...);
    }, tokens);
}



template <template <typename...> class Template, typename T>
inline constexpr bool is_instance_of_v = false;

template <template <typename...> class Template, typename... Args>
inline constexpr bool is_instance_of_v<Template, Template<Args...>> = true;

template <typename T>
inline constexpr bool is_portable_collection_v =
  is_instance_of_v<PortableHostCollection, T> or is_instance_of_v<PortableDeviceCollection, T>;

template <typename... Ts>
inline constexpr bool all_are_portable_collections_v = (is_portable_collection_v<Ts> && ...);

template <typename... Ts>
  requires(all_are_portable_collections_v<Ts...>)
class MPIReceiverPortable : public stream::EDProducer<> {
public:
  MPIReceiverPortable(edm::ParameterSet const& config)
      : EDProducer<>(config),                                // Note: EDProducer<>, not stream::EDProducer<>
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
          << "Invalid MPIReceiverPortable instance value, please use a value between 1 and 255";
    }

    upstream_ = consumes(config.getParameter<edm::InputTag>("upstream"));
    token_ = produces();

    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
    assert(products.size() == sizeof...(Ts) && "No template class MPIReceiverPortable<Ts...> found matching the number of products provided in the configuration");

    for (size_t i = 0; i < products.size(); ++i) {
      instanceNames_[i] = products[i].getParameter<std::string>("instance");
    }

    initTupleElementsFromArray(tokens_, instanceNames_, [this](const auto& instanceName) {
      return this->produces(instanceName);
    });
  }

  void produce(device::Event& event, device::EventSetup const&) override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);

    receiveProducts(event, token, tokens_, instance_);

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> upstream_;     // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> token_;        // copy of the MPIToken that may be used to implement an ordering relation
  std::tuple<device::EDPutToken<Ts>...> tokens_;  // tokens for the products
  std::array<std::string, std::tuple_size_v<decltype(tokens_)>> instanceNames_;  // names for the products
  int32_t const instance_;                  // instance used to identify the source-destination pair
};

template class MPIReceiverPortable<PortableCollection<EcalDigiSoALayout<128, false>>, 
                                   PortableCollection<EcalDigiSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>, 
                                   PortableCollection<reco::PFRecHitFractionSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// Define the plugin for each instantiation
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using MPIReceiverPortableHbheRecoSoA = MPIReceiverPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
  using MPIReceiverPortablePFRecHitSoA = MPIReceiverPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
  using MPIReceiverPortablePFClusterSoA = MPIReceiverPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>,
                                                              PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
  using MPIReceiverPortableEcalDigiSoA = MPIReceiverPortable<PortableCollection<EcalDigiSoALayout<128, false>>,
                                                             PortableCollection<EcalDigiSoALayout<128, false>>>;
  using MPIReceiverPortableEcalUncalibratedRecHitSoA = MPIReceiverPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>,
                                                                           PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;

  //using MPIReceiverPortableUShort = MPIReceiverPortable<unsigned short>;
}
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableHbheRecoSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePFRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePFClusterSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableEcalDigiSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableEcalUncalibratedRecHitSoA);