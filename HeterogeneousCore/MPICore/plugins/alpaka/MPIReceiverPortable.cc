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



#include "HeterogeneousCore/MPICore/interface/api.h"

#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

//  template <template <typename...> class Template, typename T>
//  struct is_instance_of : std::false_type {};
//
//  template <template <typename...> class Template, typename... Args>
//  struct is_instance_of<Template, Template<Args...>> : std::true_type {};
//
//  template <template <typename...> class Template, typename T>
//  concept IsInstanceOf = is_instance_of<Template, T>::value;


template <template <typename...> class Template, typename T>
inline constexpr bool is_instance_of_v = false;

template <template <typename...> class Template, typename... Args>
inline constexpr bool is_instance_of_v<Template, Template<Args...>> = true;

template <typename T>
  requires(is_instance_of_v<PortableHostCollection, T> or is_instance_of_v<PortableDeviceCollection, T>)
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
    products_.reserve(products.size());
    for (auto const& product : products) {
      auto const& label = product.getParameter<std::string>("label");
      Entry entry;
      entry.token = produces(label);

      // edm::LogVerbatim("MPIReceiver") << "receive type \"" << entry.type.name() << "\" for label \"" << label
      //                                 << "\" over MPI channel instance " << this->instance_;

      products_.emplace_back(std::move(entry));
    }
  }

  void produce(device::Event& event, device::EventSetup const&) override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);
    auto& queue = event.queue();

    for (auto const& entry : products_) {
      
      auto product = std::make_unique<T>(edm::Uninitialized{});
      
      printf("++++++  MPIReceiverPortable::produce: receiving product into collection of type %s\n",
              typeid(T).name());
      token.channel()->receiveSurelyTrivialCopyProduct(queue, instance_, *product);

      // Put the data into the Event
      event.put(entry.token, std::move(product));
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  struct Entry {
    device::EDPutToken<T> token;  // token for the product
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> upstream_;     // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> token_;        // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<Entry> products_;             // data to be read over the channel and put into the Event
  int32_t const instance_;                  // instance used to identify the source-destination pair
};

template class MPIReceiverPortable<PortableCollection<EcalDigiSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>>;
template class MPIReceiverPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// Define the plugin for each instantiation
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using MPIReceiverPortableHbheRecoSoA = MPIReceiverPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
  using MPIReceiverPortablePFRecHitSoA = MPIReceiverPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
  using MPIReceiverPortablePFClusterSoA = MPIReceiverPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>>;
  using MPIReceiverPortableEcalDigiSoA = MPIReceiverPortable<PortableCollection<EcalDigiSoALayout<128, false>>>;
  using MPIReceiverPortableEcalUncalibratedRecHitSoA = MPIReceiverPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;

  //using MPIReceiverPortableUShort = MPIReceiverPortable<unsigned short>;
}
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableHbheRecoSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePFRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortablePFClusterSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableEcalDigiSoA);
DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortableEcalUncalibratedRecHitSoA);