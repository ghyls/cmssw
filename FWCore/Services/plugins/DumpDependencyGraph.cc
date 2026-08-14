/*
 * DumpDependencyGraph: JSON-dumping sibling of DependencyGraph.cc.
 *
 * Where DependencyGraph writes Graphviz .dot for humans to look at, this Service writes
 * a single JSON document for tooling to read back (e.g. the edmMpiSplitConfig /
 * configuration_splitter tools in HeterogeneousCore/MPICore).
 *
 * The document is keyed by module label throughout -- labels are unique within a process
 * and are what tooling works in, so there is no numeric-id indirection to resolve:
 *
 *   {
 *     "process": "HLT",
 *     "modules": {"<label>": {"class": "<C++ class>",
 *                             "type": "EDProducer" | "EDFilter" | ... ,
 *                             "consumes": ["<label>", ...]}, ...},
 *     "paths":    {"<path name>": ["<label>", ...]},   // in schedule order
 *     "endpaths": {"<endpath name>": ["<label>", ...]}
 *   }
 *
 * "consumes" is omitted when empty, and a module's "scheduled" flag is not written at
 * all: a module is scheduled iff it appears in some path or endpath.
 *
 * Consumed labels come from two sources, unioned:
 *   - modulesWhoseProductsAreConsumedBy(), the framework's own resolved dependencies.
 *     These are already EDAlias-resolved, but by contract never include the Source.
 *   - moduleConsumesInfos(), the raw labels each module declared. These do include
 *     Source products and EDAlias labels, so they are resolved here against
 *     "@all_aliases" from the process ParameterSet and then kept only if they name a
 *     module that really exists in this process.
 *
 * Usage mirrors DumpProductNames: attach the Service to a copy of the process, set
 * maxEvents.input = 0 and run cmsRun. No event loop is needed -- the graph is complete
 * once lookupInitializationComplete fires.
 */

#include <fstream>
#include <map>
#include <set>
#include <string>
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

using json = nlohmann::json;

namespace {
  // The value of the "@module_edm_type" parameter every module carries in its ParameterSet.
  std::string moduleType(edm::ModuleDescription const& module) {
    auto const* pset = edm::pset::Registry::instance()->getMapped(module.parameterSetID());
    if (pset and pset->existsAs<std::string>("@module_edm_type")) {
      return pset->getParameter<std::string>("@module_edm_type");
    }
    return "Unknown";
  }

  // EDAlias label -> the real module labels it aliases, read from the process ParameterSet
  // the same way Schedule::finishSetup() does.
  std::map<std::string, std::vector<std::string>> aliasTargets(edm::ParameterSetID const& processPSetID) {
    std::map<std::string, std::vector<std::string>> targets;
    auto const* processPSet = edm::pset::Registry::instance()->getMapped(processPSetID);
    if (not processPSet or not processPSet->existsAs<std::vector<std::string>>("@all_aliases")) {
      return targets;
    }
    for (std::string const& alias : processPSet->getParameter<std::vector<std::string>>("@all_aliases")) {
      targets[alias] = processPSet->getParameterSet(alias).getParameterNamesForType<edm::VParameterSet>();
    }
    return targets;
  }
}  // namespace

class DumpDependencyGraph {
public:
  DumpDependencyGraph(edm::ParameterSet const& pset, edm::ActivityRegistry& registry)
      : fileName_(pset.getUntrackedParameter<std::string>("fileName")) {
    registry.watchPreSourceConstruction(this, &DumpDependencyGraph::preSourceConstruction);
    registry.watchLookupInitializationComplete(this, &DumpDependencyGraph::lookupInitializationComplete);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment(
        "Dumps the module dependency graph of a process as JSON (JSON sibling of the "
        "DependencyGraph Graphviz service)."
        "\nThe document is keyed by module label and has the top-level fields:"
        "\n - process: the process name"
        "\n - modules: {label: {class, type, consumes: [labels]}}"
        "\n - paths / endpaths: {name: [labels in schedule order]}");
    desc.addUntracked<std::string>("fileName", "dependency.json");
    descriptions.add("DumpDependencyGraph", desc);
  }

  // The Source is constructed before the schedule exists, and is never reported by
  // PathsAndConsumesOfModulesBase::allModules(), so record it here.
  void preSourceConstruction(edm::ModuleDescription const& module) {
    modules_[module.moduleLabel()] = {{"class", module.moduleName()}, {"type", "Source"}};
  }

  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                    edm::ProcessContext const& context) {
    processName_ = context.processName();

    for (edm::ModuleDescription const* module : pathsAndConsumes.allModules()) {
      auto& entry = modules_[module->moduleLabel()];
      entry["class"] = module->moduleName();
      entry["type"] = moduleType(*module);
    }

    auto const aliases = aliasTargets(context.parameterSetID());

    for (edm::ModuleDescription const* consumer : pathsAndConsumes.allModules()) {
      std::set<std::string> consumed;

      for (edm::ModuleDescription const* produced :
           pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id())) {
        consumed.insert(produced->moduleLabel());
      }

      // Declared labels, which -- unlike the resolved dependencies above -- also cover
      // products made by the Source, reached either directly or through an EDAlias.
      for (edm::ModuleConsumesInfo const& info : pathsAndConsumes.moduleConsumesInfos(consumer->id())) {
        std::string label{info.label()};
        auto alias = aliases.find(label);
        if (alias != aliases.end()) {
          for (std::string const& target : alias->second) {
            if (modules_.contains(target)) {
              consumed.insert(target);
            }
          }
        } else if (modules_.contains(label)) {
          consumed.insert(label);
        }
      }

      consumed.erase(consumer->moduleLabel());  // a module consuming its own product is not an edge
      if (not consumed.empty()) {
        modules_[consumer->moduleLabel()]["consumes"] = consumed;
      }
    }

    auto fill = [](json& target, std::vector<std::string> const& names, auto modulesOn) {
      for (unsigned int i = 0; i < names.size(); ++i) {
        json labels = json::array();
        for (edm::ModuleDescription const* module : modulesOn(i)) {
          labels.push_back(module->moduleLabel());
        }
        target[names[i]] = labels;
      }
    };
    fill(paths_, pathsAndConsumes.paths(), [&](unsigned int i) -> auto const& {
      return pathsAndConsumes.modulesOnPath(i);
    });
    fill(endpaths_, pathsAndConsumes.endPaths(), [&](unsigned int i) -> auto const& {
      return pathsAndConsumes.modulesOnEndPath(i);
    });

    write();
  }

private:
  void write() const {
    json out;
    out["process"] = processName_;
    out["modules"] = modules_;
    out["paths"] = paths_;
    out["endpaths"] = endpaths_;

    std::ofstream file(fileName_);
    file << out.dump();
  }

  std::string fileName_;
  std::string processName_;

  json modules_ = json::object();
  json paths_ = json::object();
  json endpaths_ = json::object();
};

DEFINE_FWK_SERVICE(DumpDependencyGraph);
