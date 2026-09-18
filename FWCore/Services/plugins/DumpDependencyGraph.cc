/*
 * Service to dump the module dependency graph of a process as a .json
 */

#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/BranchType.h"

using json = nlohmann::json;

namespace {
  // the label the framework gives to the unnamed input source
  std::string const kSourceLabel = "source";

  std::string const kEmptyLabel = "@EmptyLabel@";

  // Get the value of the "@module_edm_type" parameter (e.g. "Source",
  // "EDProducer", etc.) every module carries in its ParameterSet.
  std::string moduleType(edm::ModuleDescription const& module) {
    auto const* pset = edm::pset::Registry::instance()->getMapped(module.parameterSetID());
    if (pset and pset->existsAs<std::string>("@module_edm_type")) {
      return pset->getParameter<std::string>("@module_edm_type");
    }
    return "Unknown";
  }

  // The labels of all the EDAliases of a process. An EDAlias is not a module,
  // so its label has to be known here, to tell it apart from a label that
  // nothing in this process provides.
  std::unordered_set<std::string> aliasLabels(edm::ParameterSetID const& processPSetID) {
    auto const* processPSet = edm::pset::Registry::instance()->getMapped(processPSetID);
    if (not processPSet or not processPSet->existsAs<std::vector<std::string>>("@all_aliases")) {
      // There are no aliases
      return {};
    }
    auto const& aliases = processPSet->getParameter<std::vector<std::string>>("@all_aliases");
    return {aliases.begin(), aliases.end()};
  }

  // the labels of a list of modules, in order, as a JSON array
  json moduleLabels(std::vector<edm::ModuleDescription const*> const& modules) {
    json labels = json::array();
    for (edm::ModuleDescription const* module : modules) {
      labels.push_back(module->moduleLabel());
    }
    return labels;
  }
}  // namespace

class DumpDependencyGraph {
public:
  DumpDependencyGraph(edm::ParameterSet const& pset, edm::ActivityRegistry& registry)
      : fileName_(pset.getUntrackedParameter<std::string>("fileName")) {
    registry.watchPreSourceConstruction([this](edm::ModuleDescription const& module) {
      modules_[module.moduleLabel()] = {{"class", module.moduleName()}, {"type", "Source"}};
    });
    registry.watchLookupInitializationComplete(this, &DumpDependencyGraph::lookupInitializationComplete);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment(
        "Dumps the module dependency graph of a process as JSON."
        "\nThe document has the following fields:"
        "\n - process: the process name"
        "\n - modules: {label: {class, type, consumes, consumesNonEvent, consumesUnresolved}}"
        "\n - paths / endpaths: {name: [labels in schedule order]}"
        "\nEach module's three 'consume' lists are omitted when empty, and hold:"
        "\n - consumes: event-level dependencies on the Source or on modules of this process"
        "\n - consumesNonEvent: the same, for the non-event transitions (run,"
        " lumi and process block), excluding what is already listed in 'consumes'"
        "\n - consumesUnresolved: declared labels, on any transition, that name no module and"
        " no EDAlias of this process, or that are explicitly asked from an earlier process;"
        " those data products have to be read from the input"
        "\nThe same label can show up in both lists, when a module consumes it on both counts.");
    desc.addUntracked<std::string>("fileName", "dependency_graph.json");
    descriptions.add("DumpDependencyGraph", desc);
  }

  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                    edm::ProcessContext const& context) {
    // record the class and the type of all the modules in the process
    for (edm::ModuleDescription const* module : pathsAndConsumes.allModules()) {
      auto& entry = modules_[module->moduleLabel()];
      entry["class"] = module->moduleName();
      entry["type"] = moduleType(*module);
    }

    std::string const& processName = context.processName();
    auto const aliases = aliasLabels(context.parameterSetID());

    for (edm::ModuleDescription const* consumer : pathsAndConsumes.allModules()) {
      std::set<std::string> consumed;
      std::set<std::string> consumesNonEvent;
      std::set<std::string> unresolved;

      // the dependencies within this process, as resolved by the framework
      for (edm::ModuleDescription const* produced :
           pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id(), edm::InEvent)) {
        // Event dependencies
        consumed.insert(produced->moduleLabel());
      }
      if (pathsAndConsumes.consumesSourceProduct(consumer->id(), edm::InEvent)) {
        consumed.insert(kSourceLabel);
      }
      for (edm::BranchType branchType : {edm::InLumi, edm::InRun, edm::InProcess}) {
        // Non event-only dependencies
        for (edm::ModuleDescription const* produced :
             pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id(), branchType)) {
          consumesNonEvent.insert(produced->moduleLabel());
        }
        if (pathsAndConsumes.consumesSourceProduct(consumer->id(), branchType)) {
          consumesNonEvent.insert(kSourceLabel);
        }
      }

      // The data products nothing in this process provides are not reported
      // above: look for those among the declared consumes().
      for (edm::ModuleConsumesInfo const& info : pathsAndConsumes.moduleConsumesInfos(consumer->id())) {
        std::string label{info.label()};
        if (label.empty() or label == kEmptyLabel) {
          continue;
        }

        if (info.skipCurrentProcess() or (not info.process().empty() and info.process() != processName)) {
          // the data product is explicitly asked from an earlier process
          unresolved.insert(label);
        } else if (not modules_.contains(label) and not aliases.contains(label)) {
          // no module and no EDAlias of this process carries this label
          unresolved.insert(label);
        }
      }

      // Remove self-dependencies (modules consuming their own products)
      consumed.erase(consumer->moduleLabel());
      consumesNonEvent.erase(consumer->moduleLabel());

      // Don't repeat in consumesNonEvent a label already reported in consumed
      // This way consumesNonEvent declares the edges that exist *only* because
      // of a non-event transition
      for (std::string const& label : consumed) {
        consumesNonEvent.erase(label);
      }

      // write to the json object
      json& entry = modules_[consumer->moduleLabel()];
      if (not consumed.empty()) {
        entry["consumes"] = consumed;
      }
      if (not consumesNonEvent.empty()) {
        entry["consumesNonEvent"] = consumesNonEvent;
      }
      if (not unresolved.empty()) {
        entry["consumesUnresolved"] = unresolved;
      }
    }

    // save the modules scheduled on each Path and EndPath, in schedule order.
    json paths = json::object();
    for (unsigned int i = 0; i < pathsAndConsumes.paths().size(); ++i) {
      paths[pathsAndConsumes.paths()[i]] = moduleLabels(pathsAndConsumes.modulesOnPath(i));
    }
    json endpaths = json::object();
    for (unsigned int i = 0; i < pathsAndConsumes.endPaths().size(); ++i) {
      endpaths[pathsAndConsumes.endPaths()[i]] = moduleLabels(pathsAndConsumes.modulesOnEndPath(i));
    }

    // write out the dependency graph
    json out;
    out["process"] = context.processName();
    out["modules"] = std::move(modules_);
    out["paths"] = std::move(paths);
    out["endpaths"] = std::move(endpaths);

    std::ofstream file(fileName_);
    file << out.dump();
  }

private:
  std::string const fileName_;

  json modules_ = json::object();
};

// define as a framework service
DEFINE_FWK_SERVICE(DumpDependencyGraph);
