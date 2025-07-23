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
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"

#include "HeterogeneousCore/MPICore/interface/api.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

template <template <typename...> class Template, typename T>
inline constexpr bool is_instance_of_v = false;

template <template <typename...> class Template, typename... Args>
inline constexpr bool is_instance_of_v<Template, Template<Args...>> = true;

template <typename T>
  requires(is_instance_of_v<PortableHostCollection, T> or is_instance_of_v<PortableDeviceCollection, T>)
class MPISenderPortable : public stream::EDProducer<> {
public:
    MPISenderPortable(edm::ParameterSet const& config)
      : stream::EDProducer<>(config),
        upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces()),
        patterns_(edm::productPatterns(config.getParameter<std::vector<std::string>>("products"))),
        instance_(config.getParameter<int32_t>("instance")) {
    printf("++++++  Calling the constructor of MPISenderPortable with instance %d\n", instance_);
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
        << "Invalid MPISenderPortable instance value, please use a value between 1 and 255";
    }

      products_.reserve(patterns_.size());

    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");

      switch (product.branchType()) {
        case edm::InEvent:
          if (product.className() == kPathStatus or product.className() == kEndPathStatus)
            return;
          for (auto const& pattern : patterns_) {
            if (pattern.match(product)) {
              Entry entry;
              entry.type = product.unwrappedType();
              entry.wrappedType = product.wrappedType();
              // TODO move this to EDConsumerBase::consumes() ?
              entry.token = consumes(
                  edm::InputTag(product.moduleLabel(), product.productInstanceName(), product.processName()));
              edm::LogVerbatim("MPISender")
                  << "send product \"" << product.friendlyClassName() << '_' << product.moduleLabel() << '_'
                  << product.productInstanceName() << '_' << product.processName() << "\" of type \""
                  << entry.type.name() << "\" over MPI channel instance " << instance_;

              printf("++++++  MPISenderPortable: registering product of type %s with label %s\n",
                     entry.type.name().c_str(),
                     product.friendlyClassName().c_str());
              products_.emplace_back(std::move(entry));
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

    // TODO add an error if a pattern does not match any branches? how?
  }

  void produce(device::Event& event, device::EventSetup const&) override {
    printf("+++++++ Calling MPISenderPortable::produce with instance %d\n", instance_);
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);

    printf("++++++  MPISenderPortable::produce: got token\n");

    for (auto const& entry : products_) {
      //edm::Handle<edm::WrapperBase> handle(entry.typeID);
      printf("++++++  MPISenderPortable::produce: entered loop\n");
      auto const& handle = event.get(entry.token);
      printf("++++++  MPISenderPortable::produce: extracting product from wrapper of type %s\n",
             entry.wrappedType.name().c_str());
      printf("++++++  MPISenderPortable::produce: sending unwrapped product of type %s\n", typeid(T).name());
      auto const bufferSize = alpaka::getExtentProduct(handle.buffer());
      token.channel()->sendSurelyTrivialCopyProduct(instance_, handle, bufferSize);
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    device::EDGetToken<T> token;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<edm::ProductNamePattern> patterns_;  // branches to read from the Event and send over the MPI channel
  std::vector<Entry> products_;                    // types and tokens corresponding to the branches
  int32_t const instance_;                         // instance used to identify the source-destination pair
};

template class MPISenderPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<EcalDigiSoALayout<128, false>>>;
template class MPISenderPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE


#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using MPISenderPortableHbheRecoSoA = MPISenderPortable<PortableCollection<hcal::HcalRecHitSoALayout<128, false>>>;
  using MPISenderPortablePFRecHitSoA = MPISenderPortable<PortableCollection<reco::PFRecHitSoALayout<128, false>>>;
  using MPISenderPortablePFClusterSoA = MPISenderPortable<PortableCollection<reco::PFClusterSoALayout<128, false>>>;
  using MPISenderPortableEcalDigiSoA = MPISenderPortable<PortableCollection<EcalDigiSoALayout<128, false>>>;
  using MPISenderPortableEcalUncalibratedRecHitSoA = MPISenderPortable<PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>>>;

  //using MPISenderPortableUShort = MPISenderPortable<unsigned short>;
}

DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableHbheRecoSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableEcalDigiSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePFRecHitSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortablePFClusterSoA);
DEFINE_FWK_ALPAKA_MODULE(MPISenderPortableEcalUncalibratedRecHitSoA);