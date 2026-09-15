"""
Answers the two questions the splitter asks about a configuration:

  - which offloaded modules have to be offloaded and gated as one unit, and what each
    unit needs from outside (`dependency_groups`, `external_dependencies_by_group`,
    `modules_to_send_back_by_group`)
  - under which condition each module was reached in the original schedule
    (`reachability_signature`)

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


# Synthetic module label DaqProvenanceHelper files raw FED data under (Issue 45137);
# DumpDependencyGraph reports it as unresolved since no such module exists.
_RAW_DATA_COLLECTOR = "rawDataCollector"


class ModuleDependencyAnalyzer:
    def __init__(self, graph):
        """graph: the JSON document produced by the DumpDependencyGraph service."""
        modules = graph["modules"]
        source = next((label for label, module in modules.items() if module.get("type") == "Source"), None)

        self.module_inputs: Dict[str, Set[str]] = {}
        for label, module in modules.items():
            inputs = set(module.get("consumes", ()))
            if source is not None and _RAW_DATA_COLLECTOR in module.get("consumesUnresolved", ()):
                inputs.add(source)
            self.module_inputs[label] = inputs

        self.producer_to_consumers: Dict[str, Set[str]] = defaultdict(set)
        for consumer, producers in self.module_inputs.items():
            for producer in producers:
                self.producer_to_consumers[producer].add(consumer)

        # Run/lumi/process-block dependencies, which MPISender/MPIReceiver cannot carry
        # (they move products event by event); kept apart from module_inputs so offloading
        # such a module is reported instead of silently dropping the product.
        self._non_event_inputs: Dict[str, Set[str]] = {
            label: set(module.get("consumesNonEvent", ())) for label, module in modules.items()
        }

        self._reachability = self._build_reachability(graph)

    def non_event_dependencies(self, modules: List[str]) -> Dict[str, Set[str]]:
        """
        Maps each of `modules` that has one to the modules outside `modules` it depends on
        through a run, lumi or process block product. Those are exactly the dependencies
        offloading would break, since nothing forwards them: a dependency on another
        offloaded module is fine, both sides running in the same remote process.
        """
        offloaded = set(modules)

        outside = {}
        for module in modules:
            producers = self._non_event_inputs.get(module, set()) - offloaded
            if producers:
                outside[module] = producers

        return outside

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
        The condition under which a module is reached, as the set of filter-free path
        stretches it sits in. A module on several paths gets one entry per occurrence:
        its real condition is the OR over them.

        EQUAL signatures mean a provably identical condition, so the modules may share
        one activation capture. Unequal ones do NOT prove the conditions differ -- they
        only mean the modules must be kept apart, which is always safe.

        None means the module is on no Path or EndPath at all and only ever runs on
        demand for a consumer; there is no path-level condition to reproduce.
        """
        return self._reachability.get(module_name)

    def dependency_groups(self, modules: List[str]) -> List[List[str]]:
        """
        Splits `modules` into the units the splitter offloads and gates together: the
        connected components of their dependencies on each other, each split further by
        reachability_signature() so that one activation gate per unit reproduces its
        members' original condition exactly.

        Two members of one component reached under different conditions therefore end up
        in different groups, chained by the data dependency between them. Groups come out
        ordered so that a group follows the ones it consumes from, and each group's
        members in dependency order; groups that depend on each other in both directions
        are merged back into one, which is always safe, just not always tight.

        See doc/edmMpiSplitConfig.md for why the split has to be this fine.
        """
        selected = set(modules)
        inputs = {m: self.module_inputs.get(m, set()) & selected for m in modules}

        keys = self._grouping_keys(modules, inputs)

        members: Dict[Tuple, List[str]] = {}
        for m in modules:  # iterating `modules` keeps both the groups and their contents deterministic
            members.setdefault(keys[m], []).append(m)

        # how the groups depend on each other, as "key -> the keys it consumes from"
        depends_on = {key: set() for key in members}
        for consumer, producers in inputs.items():
            for producer in producers:
                if keys[producer] != keys[consumer]:
                    depends_on[keys[consumer]].add(keys[producer])

        groups = []
        for component in self._strongly_connected_components(list(members), depends_on):
            merged = set(component)
            groups.append(self._in_dependency_order([m for m in modules if keys[m] in merged], inputs))

        return groups

    def _grouping_keys(self, modules: List[str], inputs: Dict[str, Set[str]]) -> Dict[str, Tuple]:
        """
        What decides which group a module belongs to: its connected component, paired
        with its reachability_signature().

        A module on no Path has no condition of its own, so there is nothing to gate it
        by and no position to capture one at. It takes the key of a neighbour inside the
        offloaded set instead, consumer for preference; a chain of such modules resolves
        from its scheduled end inwards.
        """
        component = self._connected_components(modules, inputs)
        keys = {m: (component[m], self.reachability_signature(m)) for m in modules}

        consumers: Dict[str, List[str]] = defaultdict(list)
        for consumer, producers in inputs.items():
            for producer in producers:
                consumers[producer].append(consumer)

        unscheduled = [m for m in modules if keys[m][1] is None]
        while unscheduled:
            resolved = []
            for m in unscheduled:
                for neighbour in consumers.get(m, []) + sorted(inputs[m]):
                    if keys[neighbour][1] is not None:
                        keys[m] = keys[neighbour]
                        resolved.append(m)
                        break
            if not resolved:
                # nothing left is attached to anything scheduled: each keeps its own key
                break
            unscheduled = [m for m in unscheduled if m not in resolved]

        return keys

    @staticmethod
    def _connected_components(modules: List[str], inputs: Dict[str, Set[str]]) -> Dict[str, str]:
        """Maps each module to a representative of its connected component (union-find)."""
        parent = {m: m for m in modules}

        def root(module_name):
            while parent[module_name] != module_name:
                parent[module_name] = parent[parent[module_name]]
                module_name = parent[module_name]
            return module_name

        for consumer, producers in inputs.items():
            for producer in producers:
                parent[root(producer)] = root(consumer)

        return {m: root(m) for m in modules}

    @staticmethod
    def _strongly_connected_components(nodes: List, edges: Dict) -> List[List]:
        """
        Tarjan's algorithm, iterative so that a deep dependency chain cannot overflow the
        stack. `edges` maps a node to the nodes it depends on, so a component comes out
        after every component it depends on -- the order the groups have to be built in.
        """
        index: Dict = {}
        lowlink: Dict = {}
        on_stack: Set = set()
        stack: List = []
        components: List[List] = []

        for start in nodes:
            if start in index:
                continue

            index[start] = lowlink[start] = len(index)
            stack.append(start)
            on_stack.add(start)
            work = [(start, iter(edges.get(start, ())))]

            while work:
                node, successors = work[-1]
                for successor in successors:
                    if successor not in index:
                        index[successor] = lowlink[successor] = len(index)
                        stack.append(successor)
                        on_stack.add(successor)
                        work.append((successor, iter(edges.get(successor, ()))))
                        break
                    if successor in on_stack:
                        lowlink[node] = min(lowlink[node], index[successor])
                else:
                    if lowlink[node] == index[node]:
                        component = []
                        while True:
                            member = stack.pop()
                            on_stack.discard(member)
                            component.append(member)
                            if member == node:
                                break
                        components.append(component)
                    work.pop()
                    if work:
                        lowlink[work[-1][0]] = min(lowlink[work[-1][0]], lowlink[node])

        return components

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
        consume its products, i.e. exactly the products that have to be sent over. A
        producer that is itself offloaded is never one of them, even when it belongs to
        a different group than its consumer: both run in the remote process, so the
        product never has to travel.
        """
        offloaded = {m for group in groups for m in group}

        mapping = defaultdict(set)
        for group_index, group in enumerate(groups):
            for m in group:
                for producer in self.module_inputs.get(m, ()):
                    if producer not in offloaded:
                        mapping[producer].add(group_index)
        return mapping

    def modules_to_send_back_by_group(self, groups: List[List[str]], modules_to_run_on_both: List[str]):
        """
        Splits the offloaded modules into those whose products are still needed by a
        module running in the local process -- which therefore have to be sent back, per
        group -- and those that are not needed there, which the local process can simply
        drop. A consumer that is itself offloaded does not count, unless it is one of the
        modules running on both sides: that one has a local copy needing the product
        locally too.
        """
        remote_only = {m for group in groups for m in group} - set(modules_to_run_on_both)

        to_send = [[] for _ in groups]
        unused = []

        for group_index, group in enumerate(groups):
            for m in group:
                if m in modules_to_run_on_both:
                    continue
                consumers = self.producer_to_consumers.get(m, set())
                if any(consumer not in remote_only for consumer in consumers):
                    to_send[group_index].append(m)
                else:
                    unused.append(m)

        return to_send, unused
