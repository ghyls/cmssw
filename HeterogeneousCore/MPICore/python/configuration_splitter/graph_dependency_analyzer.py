"""
GraphDependencyAnalyzer: JSON-graph-backed drop-in replacement for
ModuleDependencyAnalyzer's dependency extraction.

ModuleDependencyAnalyzer._build_dependency_maps() currently infers module
dependencies by statically walking each module's Python parameters, pulling out
cms.InputTag / cms.VInputTag objects, and matching tag.getModuleLabel() against
process attribute names. That misses anything the framework resolves at runtime
rather than storing as a literal InputTag parameter (mayConsumes, EventSetup
dependencies, EDAlias/SwitchProducer indirection, etc.).

DumpDependencyGraph.cc (a JSON-dumping sibling of FWCore/Services/DependencyGraph.cc)
gets this information directly from PathsAndConsumesOfModulesBase -- the framework's
own, authoritative record of registered consumes() dependencies -- and writes it as
JSON with edges like:

    {"from": <consumer id>, "to": <producer id>, "kind": "consumes"}

Every other method on ModuleDependencyAnalyzer (direct_dependencies, dependency_groups,
grouped_external_dependencies, producer_to_groups, modules_to_send_back_by_group, ...)
operates purely on the two label-keyed dicts `module_inputs` and `producer_to_consumers`
built here -- they don't touch `self.process` directly. So this subclass only needs to
override the map-building step; everything downstream (including split_remote.py) is
unaffected apart from swapping which class gets instantiated.
"""

import json
from collections import defaultdict
from itertools import chain
from typing import Dict, Set

from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import (
    ModuleDependencyAnalyzer,
)


class GraphDependencyAnalyzer(ModuleDependencyAnalyzer):
    def __init__(self, process, graph_json_path):
        """
        process: the (deep-copied) cms.Process -- kept only because the base class's
            constructor signature expects it, and other tooling in the splitter may
            still want process access for non-dependency purposes (e.g. flatten_all_to_module_list
            reads process.<name>.moduleNames() for cms.Sequence/cms.Task inputs, which
            still needs the live process object, not the graph).
        graph_json_path: path to the JSON file written by DumpDependencyGraph
            (e.g. via a DependencyGraphGetter mirroring cpp_name_getter.py's
            CPPNameGetter -- run a throwaway maxEvents.input=0 cmsRun job with the
            DumpDependencyGraph service attached, then read back its output).
        """
        self.process = process
        self.module_inputs: Dict[str, Set[str]] = defaultdict(set)
        self.producer_to_consumers: Dict[str, Set[str]] = defaultdict(set)
        self._build_dependency_maps_from_graph(graph_json_path)

    def _build_dependency_maps_from_graph(self, graph_json_path):
        with open(graph_json_path) as f:
            graph = json.load(f)

        id_to_label = {m["id"]: m["label"] for m in graph["modules"]}

        # Primary source: real, framework-verified consumes() edges from
        # PathsAndConsumesOfModulesBase (via DumpDependencyGraph.cc). Authoritative.
        for edge in graph["edges"]:
            if edge["kind"] != "consumes":
                continue  # skip "schedule" (soft/order) edges -- only real data deps matter here

            consumer = id_to_label.get(edge["from"])
            producer = id_to_label.get(edge["to"])
            if not consumer or not producer:
                continue

            self.module_inputs[consumer].add(producer)
            self.producer_to_consumers[producer].add(consumer)

        # Supplementary fallback: the framework's own consumes-dependency tracking does NOT
        # resolve a consumer's dependency through an EDAlias -- confirmed empirically against
        # a real HLT menu (rawDataCollector, an EDAlias for the process Source's
        # FEDRawDataCollection product): all real consumers of "rawDataCollector" were
        # scheduled graph nodes, yet none of them had a resolved "consumes" edge to the
        # alias's real target. So PathsAndConsumesOfModulesBase::modulesWhoseProductsAreConsumedBy()'s
        # own doc comment about alias resolution does not hold here, at least not for an
        # EDAlias that targets the process Source itself.
        #
        # edm::ConstProductRegistry (the usual way to resolve EDAlias -> real target from a
        # C++ Service) does not exist in this CMSSW release, so resolve aliases directly from
        # the Python config we already have instead: an EDAlias's keyword parameter names ARE
        # the real target module labels (e.g. `rawDataCollector = cms.EDAlias(source=...)`
        # means "source" is the real producer) -- no C++/DumpDependencyGraph.cc changes needed.
        #
        # Then fall back to the same InputTag-scanning the OLD ModuleDependencyAnalyzer used,
        # but only ever accept a resolved label if it's a REAL, graph-confirmed node
        # (valid_labels). This can never reintroduce the original phantom-dependency bug (a
        # label that doesn't correspond to anything real) -- it can only add edges for labels
        # the graph has already confirmed are real.
        alias_to_targets = {
            alias_label: list(getattr(self.process, alias_label).parameterNames_())
            for alias_label in self.process.aliases_().keys()
        }

        valid_labels = set(id_to_label.values())
        for name in chain(
            self.process.producers_().keys(),
            self.process.analyzers_().keys(),
            self.process.outputModules_().keys(),
            self.process.filters_().keys(),
        ):
            mod = getattr(self.process, name)
            for param in mod.parameters_().values():
                for tag in self._extract_inputtags(param):
                    label = tag.getModuleLabel()
                    if not label:
                        continue
                    for producer in alias_to_targets.get(label, [label]):
                        if not producer or producer not in valid_labels:
                            continue  # NOT accepted on faith -- must be a real, graph-confirmed node
                        if producer in self.module_inputs.get(name, set()):
                            continue  # already have this from the primary "consumes" edges
                        self.module_inputs[name].add(producer)
                        self.producer_to_consumers[producer].add(name)

    # Intentionally do NOT override _build_dependency_maps() itself (base class's
    # signature that reads from self.process) -- overriding __init__ to skip calling
    # it means it's simply never invoked. Kept here as a documented no-op for clarity
    # in case someone greps for it:
    def _build_dependency_maps(self):
        raise NotImplementedError(
            "GraphDependencyAnalyzer builds its maps from the DumpDependencyGraph JSON "
            "(see _build_dependency_maps_from_graph); the InputTag-parsing base class "
            "method is intentionally unused here."
        )