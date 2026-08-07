/*
 * DumpDependencyGraph: JSON-dumping sibling of FWCore/Services/plugins/DependencyGraph.cc
 *
 * Like DependencyGraph, this Service watches the framework's module lookup/scheduling
 * machinery and records:
 *   - every module in the process (id, label, C++ class, EDM type, whether it is scheduled)
 *   - every Path / EndPath and the modules scheduled on it, in order
 *   - "hard" edges: real edm::consumes() data dependencies between modules
 *   - "soft" edges: the scheduled order of modules along a Path/EndPath, plus the
 *     implicit TriggerResults -> PathStatusInserter -> last-module-on-path edges
 *
 * Instead of writing Graphviz .dot/.gv (meant for visualization with `dot`), it writes a
 * single JSON document that downstream Python tooling (e.g. the edmMpiConfigSplitter /
 * configuration_splitter tools) can load directly, the same way DumpProductNames.cc feeds
 * cpp_name_getter.py.
 *
 * Usage pattern mirrors cpp_name_getter.py's CPPNameGetter: attach this Service to a
 * deep-copied process, set maxEvents.input = 0, run cmsRun, then read back the JSON --
 * no event loop is needed since the graph is fully known after lookupInitializationComplete.
 *
 * Note on EDAlias: per PathsAndConsumesOfModulesBase::modulesWhoseProductsAreConsumedBy()'s
 * own doc comment, if a module consumes a label that is an EDAlias, the returned vector
 * already contains the real target module -- the framework resolves the alias for us here,
 * so no separate ConstProductRegistry-based alias-resolution pass is needed.
 */

#include <fstream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

// JSON headers
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// CMSSW
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm;
using namespace edm::service;

namespace {
  // module.id() -> compact 0-based index into our modules_ vector, same trick used by
  // DependencyGraph::m_moduleIDToGraphIndex
  constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();
}  // namespace

class DumpDependencyGraph {
public:
  DumpDependencyGraph(ParameterSet const& pset, ActivityRegistry& registry)
      : fileName_(pset.getUntrackedParameter<std::string>("fileName")),
        showPathDependencies_(pset.getUntrackedParameter<bool>("showPathDependencies")) {
    registry.watchPreSourceConstruction(this, &DumpDependencyGraph::preSourceConstruction);
    registry.watchLookupInitializationComplete(this, &DumpDependencyGraph::lookupInitializationComplete);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment(
        "Dumps the module dependency graph of a process as JSON (JSON sibling of the "
        "DependencyGraph Graphviz service)."
        "\nThe JSON document has the following top-level fields:"
        "\n - process: the process name"
        "\n - modules: list of {id, label, class, type, scheduled}"
        "\n - paths / endpaths: list of {name, modules: [module ids in schedule order]}"
        "\n - edges: list of {from, to, kind}, where 'kind' is either"
        "\n     'consumes' (a real edm::consumes<>() data dependency: from consumes to)"
        "\n     'schedule' (a soft/implicit ordering dependency, e.g. path order,"
        "\n                 TriggerResults -> PathStatusInserter -> last module on path)");
    desc.addUntracked<std::string>("fileName", "dependency.json");
    desc.addUntracked<bool>("showPathDependencies", true);
    descriptions.add("DumpDependencyGraph", desc);
  }

  void preSourceConstruction(ModuleDescription const& module) {
    addModule(module, /*scheduled=*/true, EDMModuleType::Source);
  }

  void lookupInitializationComplete(PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                     ProcessContext const& context) {
    processName_ = context.processName();

    if (pathsAndConsumes.largestModuleID() >= idToIndex_.size()) {
      idToIndex_.resize(pathsAndConsumes.largestModuleID() + 1, kInvalidIndex);
    }

    // add every module in the process (skipping the Source(s) already added)
    for (edm::ModuleDescription const* module : pathsAndConsumes.allModules()) {
      if (idToIndex_[module->id()] != kInvalidIndex)
        continue;  // already added (e.g. Source)
      addModule(*module, /*scheduled=*/false, edmModuleTypeEnum(*module));
    }

    // hard (consumes) edges
    for (edm::ModuleDescription const* consumer : pathsAndConsumes.allModules()) {
      for (edm::ModuleDescription const* produced :
           pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id())) {
        edm::LogInfo("DumpDependencyGraph")
            << "module " << consumer->moduleLabel() << " depends on module " << produced->moduleLabel();
        edges_.push_back({idToIndex_[consumer->id()], idToIndex_[produced->id()], "consumes"});
      }
    }

    auto const& paths = pathsAndConsumes.paths();
    auto const& endps = pathsAndConsumes.endPaths();

    // record Path membership (schedule order) and mark modules scheduled
    for (unsigned int i = 0; i < paths.size(); ++i) {
      json pathModules = json::array();
      edm::ModuleDescription const* previous = nullptr;
      for (edm::ModuleDescription const* module : pathsAndConsumes.modulesOnPath(i)) {
        auto index = idToIndex_[module->id()];
        modules_[index]["scheduled"] = true;
        pathModules.push_back(module->id());

        if (previous and showPathDependencies_) {
          edges_.push_back({index, idToIndex_[previous->id()], "schedule"});
        }
        previous = module;
      }
      paths_.push_back({{"name", paths[i]}, {"modules", pathModules}});

      // implicit PathStatusInserter -> last module, and TriggerResults -> PathStatusInserter
      if (previous and showPathDependencies_) {
        for (std::size_t j = 0; j < modules_.size(); ++j) {
          if (modules_[j]["label"] == paths[i]) {
            edges_.push_back({j, idToIndex_[previous->id()], "schedule"});
            if (triggerResultsIndex_ != kInvalidIndex) {
              edges_.push_back({triggerResultsIndex_, j, "schedule"});
            }
            break;
          }
        }
      }
    }

    for (unsigned int i = 0; i < endps.size(); ++i) {
      json pathModules = json::array();
      edm::ModuleDescription const* previous = nullptr;
      for (edm::ModuleDescription const* module : pathsAndConsumes.modulesOnEndPath(i)) {
        auto index = idToIndex_[module->id()];
        modules_[index]["scheduled"] = true;
        pathModules.push_back(module->id());

        if (previous and showPathDependencies_) {
          edges_.push_back({index, idToIndex_[previous->id()], "schedule"});
        }
        previous = module;
      }
      endpaths_.push_back({{"name", endps[i]}, {"modules", pathModules}});
    }

    write();
  }

private:
  enum class EDMModuleType { Unknown, Source, ESSource, ESProducer, EDAnalyzer, EDProducer, EDFilter, OutputModule };

  static constexpr const char* kModuleTypeNames[]{
      "Unknown", "Source", "ESSource", "ESProducer", "EDAnalyzer", "EDProducer", "EDFilter", "OutputModule"};

  static EDMModuleType edmModuleTypeEnum(edm::ModuleDescription const& module) {
    auto const& registry = *edm::pset::Registry::instance();
    auto const& pset = *registry.getMapped(module.parameterSetID());
    if (not pset.existsAs<std::string>("@module_edm_type"))
      return EDMModuleType::Unknown;
    std::string const& t = pset.getParameter<std::string>("@module_edm_type");
    for (EDMModuleType v : {EDMModuleType::Source,
                            EDMModuleType::ESSource,
                            EDMModuleType::ESProducer,
                            EDMModuleType::EDAnalyzer,
                            EDMModuleType::EDProducer,
                            EDMModuleType::EDFilter,
                            EDMModuleType::OutputModule}) {
      if (t == kModuleTypeNames[static_cast<int>(v)])
        return v;
    }
    return EDMModuleType::Unknown;
  }

  void addModule(ModuleDescription const& module, bool scheduled, EDMModuleType type) {
    if (module.id() >= idToIndex_.size()) {
      idToIndex_.resize(module.id() + 1, kInvalidIndex);
    }
    idToIndex_[module.id()] = modules_.size();

    if (module.moduleLabel() == "TriggerResults") {
      triggerResultsIndex_ = modules_.size();
    }

    modules_.push_back({{"id", module.id()},
                        {"label", module.moduleLabel()},
                        {"class", module.moduleName()},
                        {"type", kModuleTypeNames[static_cast<int>(type)]},
                        {"scheduled", scheduled}});
  }

  void write() const {
    json out;
    out["process"] = processName_;
    out["modules"] = modules_;
    out["paths"] = paths_;
    out["endpaths"] = endpaths_;

    json edgesJson = json::array();
    for (auto const& e : edges_) {
      edgesJson.push_back({{"from", modules_[e.from]["id"]}, {"to", modules_[e.to]["id"]}, {"kind", e.kind}});
    }
    out["edges"] = edgesJson;

    std::ofstream file(fileName_);
    file << out.dump(2);
  }

  struct Edge {
    std::size_t from;
    std::size_t to;
    std::string kind;
  };

  std::string fileName_;
  bool showPathDependencies_;
  std::string processName_;

  std::vector<std::size_t> idToIndex_;
  std::size_t triggerResultsIndex_ = kInvalidIndex;

  json modules_ = json::array();
  json paths_ = json::array();
  json endpaths_ = json::array();
  std::vector<Edge> edges_;
};

constexpr const char* DumpDependencyGraph::kModuleTypeNames[];

DEFINE_FWK_SERVICE(DumpDependencyGraph);
