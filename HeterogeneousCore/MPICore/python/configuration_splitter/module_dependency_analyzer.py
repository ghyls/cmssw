"""
Answers the two questions the splitter asks about a configuration:

  - which offloaded modules depend on which, and on what outside the offloaded set
    (`dependency_groups`, `external_dependencies_by_group`, `modules_to_send_back_by_group`)
  - under which condition each module was reached in the original schedule
    (`reachability_signature` and friends)

Everything comes from the DumpDependencyGraph JSON, i.e. from the framework's own record
of registered consumes() dependencies and of the schedule. This used to be inferred
instead by walking each module's Python parameters for cms.InputTag values and matching
them against process attribute names, which both invented dependencies (any string that
happened to look like a module label) and missed real ones (mayConsumes, EDAlias and
SwitchProducer indirection, products made by the Source).
"""

from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


def flatten_all_to_module_list(process, user_args):
    """
    Flatten input arguments into an ordered list of module names.
    Preserves user-provided order and avoids duplicates.
    """
    module_list = []
    seen = set()

    for name in user_args:
        if not hasattr(process, name):
            print(f"[WARN] process has no attribute named '{name}'")
            continue

        obj_ = getattr(process, name)

        if hasattr(obj_, "moduleNames"):
            for mod in obj_.moduleNames():
                if mod not in seen:
                    module_list.append(mod)
                    seen.add(mod)
        else:
            if name not in seen:
                module_list.append(name)
                seen.add(name)

    return module_list


class ModuleDependencyAnalyzer:
    def __init__(self, graph):
        """graph: the JSON document produced by the DumpDependencyGraph service."""
        modules = graph["modules"]

        self.module_inputs: Dict[str, Set[str]] = {
            label: set(module.get("consumes", ())) for label, module in modules.items()
        }
        self.producer_to_consumers: Dict[str, Set[str]] = defaultdict(set)
        for consumer, producers in self.module_inputs.items():
            for producer in producers:
                self.producer_to_consumers[producer].add(consumer)

        self._reachability = self._build_reachability(graph)

    @staticmethod
    def _build_reachability(graph) -> Dict[str, FrozenSet[Tuple[str, int]]]:
        """
        Maps each module to its set of (path key, run index) pairs -- see
        reachability_signature(). A "run" is a maximal EDFilter-free stretch of a
        Path or EndPath: it starts at the beginning of the path, or right after an
        EDFilter on it, and every module inside one is reached under the exact same
        condition (all the EDFilters before it, on that path, passed). EndPaths always
        run, which is just "run 0" of their own path key -- no special case needed.
        """
        types = {label: module["type"] for label, module in graph["modules"].items()}

        reachability = defaultdict(set)
        for kind in ("paths", "endpaths"):
            for name, labels in graph[kind].items():
                run_index = 0
                for label in labels:
                    reachability[label].add((f"{kind}:{name}", run_index))
                    if types.get(label) == "EDFilter":
                        run_index += 1

        return {label: frozenset(runs) for label, runs in reachability.items()}

    def reachability_signature(self, module_name: str) -> Optional[FrozenSet[Tuple[str, int]]]:
        """
        Identifies the condition under which a module is reached, as the set of
        filter-free path stretches it sits in. A module on several paths gets one entry
        per occurrence: its real condition is the OR over them, since the framework runs
        a module as soon as any path reaches its position.

        Two modules with EQUAL signatures are reached under a provably identical
        condition (the same set of alternatives, hence the same OR-of-ANDs) and may
        share one activation capture. Unequal signatures do NOT prove the conditions
        differ -- they only mean the modules must be kept apart, which is always safe.

        None means the module is on no Path or EndPath at all, and only ever runs
        on-demand for a consumer; there is no path-level condition to reproduce.
        """
        return self._reachability.get(module_name)

    def reachability_classes(self, modules: List[str]) -> List[List[str]]:
        """
        Partitions `modules` into sublists sharing an identical reachability_signature(),
        keeping the order they were given in. The modules with no signature at all end up
        together in one class, which callers have to recognise: being equally unscheduled
        is not being equally reached. Pass one offload group at a time, since merging
        across groups is out of scope (see split_remote.py's per-module activation gating).
        """
        classes: Dict[FrozenSet[Tuple[str, int]], List[str]] = {}
        for module_name in modules:
            classes.setdefault(self.reachability_signature(module_name), []).append(module_name)
        return list(classes.values())

    def dependency_groups(self, modules: List[str]) -> List[List[str]]:
        """
        Splits `modules` into the connected components of their dependencies on each
        other, each ordered so that a module comes after everything it consumes from.
        """
        selected = set(modules)
        inputs = {m: self.module_inputs.get(m, set()) & selected for m in modules}

        # union-find over the components
        parent = {m: m for m in modules}

        def root(module_name):
            while parent[module_name] != module_name:
                parent[module_name] = parent[parent[module_name]]
                module_name = parent[module_name]
            return module_name

        for consumer, producers in inputs.items():
            for producer in producers:
                parent[root(producer)] = root(consumer)

        components = defaultdict(list)
        for m in modules:  # iterating `modules` keeps both the groups and their contents deterministic
            components[root(m)].append(m)

        return [self._in_dependency_order(component, inputs) for component in components.values()]

    @staticmethod
    def _in_dependency_order(component: List[str], inputs: Dict[str, Set[str]]) -> List[str]:
        ordered: List[str] = []
        placed: Set[str] = set()
        remaining = component

        while remaining:
            ready = [m for m in remaining if inputs[m] <= placed]
            if not ready:
                return ordered + remaining  # dependency cycle: leave the rest as it came
            ordered += ready
            placed.update(ready)
            remaining = [m for m in remaining if m not in placed]

        return ordered

    def external_dependencies_by_group(self, groups: List[List[str]]) -> Dict[str, Set[int]]:
        """
        Maps each module outside the offloaded set to the indices of the groups that
        consume its products, i.e. exactly the products that have to be sent over.
        """
        mapping = defaultdict(set)
        for group_index, group in enumerate(groups):
            members = set(group)
            for m in group:
                for producer in self.module_inputs.get(m, ()):
                    if producer not in members:
                        mapping[producer].add(group_index)
        return mapping

    def modules_to_send_back_by_group(self, groups: List[List[str]], modules_to_run_on_both: List[str]):
        """
        Splits the offloaded modules into those whose products are still needed outside
        their own group -- which therefore have to be sent back, per group -- and those
        that are not needed anywhere else, which the local process can simply drop.
        """
        group_of = {m: i for i, group in enumerate(groups) for m in group}

        to_send = [[] for _ in groups]
        unused = []

        for group_index, group in enumerate(groups):
            for m in group:
                if m in modules_to_run_on_both:
                    continue
                consumers = self.producer_to_consumers.get(m, set())
                if any(group_of.get(consumer) != group_index for consumer in consumers):
                    to_send[group_index].append(m)
                else:
                    unused.append(m)

        return to_send, unused
