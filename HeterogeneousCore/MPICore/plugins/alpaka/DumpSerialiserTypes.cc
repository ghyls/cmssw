// C++ include files
#include <fstream>
#include <map>
#include <string>

// JSON headers
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// CMSSW
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/modulePrevalidate.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Dump this backend's device serialiser registry as JSON, so that edmMpiSplitConfig can
  // name a device product in a configuration without pinning it to one backend. One entry
  // per serialiser:
  //
  //   {"product_type": "edm::DeviceProduct<SiPixelClustersDevice<alpaka::DevUniformCudaHipRt<...> > >",
  //    "keys":         ["19SiPixelClustersHost", "alpaka_serial_sync::SiPixelClustersSoACollection"],
  //    "alias":        "SiPixelClustersSoACollection"}
  //
  // "product_type" is the type a product dump reports for such a product on this backend.
  // "keys" are the names the serialiser is registered under, of which the mangled typeid
  // of its host flavour is registered on every backend, and so identifies the same
  // serialiser across two dumps.
  //
  // "alias" is the name a configuration has to use, and only the CPU backends know it.
  // There the macro also registers the serialiser under the literal
  // "alpaka_serial_sync::" followed by TYPE_DEVICE spelled exactly as it received it (a
  // workaround for CMSSW issue #51427), which is what a configuration writes once the
  // namespace is replaced by the ALPAKA_ACCELERATOR_NAMESPACE placeholder. The GPU
  // backends register the mangled typeid of the resolved device type instead, which names
  // no backend but their own; the field is then absent.
  class DumpSerialiserTypes : public edm::global::EDAnalyzer<> {
  public:
    explicit DumpSerialiserTypes(edm::ParameterSet const& config)
        : outputFile_(config.getParameter<std::string>("outputFile")) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override {}

    void endJob() override {
      static std::string const kNamespacePrefix = std::string(EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)) + "::";

      auto const& factory = *ngt::SerialiserFactoryDevice::get();
      auto const& categories = edmplugin::PluginManager::get()->categoryToInfos();
      auto category = categories.find(factory.category());

      // Keyed by product type, which is what collects the several names one serialiser
      // is registered under back into a single entry. Grouping by the host type would be
      // more direct, but hostProductTypeID() is only available for a type that has a
      // CopyToHost specialisation.
      std::map<std::string, json> serialisers;
      if (category != categories.end()) {
        for (edmplugin::PluginInfo const& info : category->second) {
          // Creating the serialiser is the only way to learn which types it moves: the
          // registry maps names to makers, not to types.
          auto serialiser = factory.tryToCreate(info.name_);
          if (not serialiser) {
            continue;
          }
          std::string productType = edm::TypeWithDict(serialiser->productTypeID()).name();
          json& entry = serialisers[productType];
          entry["product_type"] = productType;
          entry["keys"].push_back(info.name_);
          if (info.name_.starts_with(kNamespacePrefix)) {
            entry["alias"] = info.name_.substr(kNamespacePrefix.size());
          }
        }
      }

      json entries = json::array();
      for (auto const& [productType, entry] : serialisers) {
        entries.push_back(entry);
      }

      std::ofstream out(outputFile_);
      out << entries.dump(2);  // pretty-print with indent 2
    }

    static void prevalidate(edm::ConfigurationDescriptions& descriptions) {
      edm::global::EDAnalyzer<>::prevalidate(descriptions);
      cms::alpakatools::modulePrevalidate(descriptions);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.setComment(
          "This module dumps the contents of this Alpaka backend's device serialiser registry."
          " It produces a json file, a list with one entry per registered serialiser."
          " Each entry contains the following fields:"
          "\n - product_type - the C++ type of the product the serialiser moves, as this backend resolves it"
          "\n - keys - the names the serialiser is registered under"
          "\n - alias - the backend-independent name of that type, on the backends that register one");
      desc.add<std::string>("outputFile", "serialiser_types.json");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    std::string const outputFile_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(DumpSerialiserTypes);
