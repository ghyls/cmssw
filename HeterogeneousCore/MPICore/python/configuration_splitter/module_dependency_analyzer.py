from collections import defaultdict
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

# the origin of the data products of the Source, and of those read from the input
SOURCE = "source"


class ConsumedProduct(NamedTuple):
    """
    One data product a module consumes in the Event.
    """

    # the (label, instance, process) the module asks for; label may be an EDAlias
    tag: Tuple[str, str, str]
    # the module of this process that produces it, or SOURCE for the Source and for the input
    origin: str
    # its product_names.json entry; for an EDAlias, the entry of the original data product
    product: dict
    # the product_instance under which the module finds it (the EDAlias one, for an EDAlias)
    instance: str


class ModuleDependencyAnalyzer:
    def __init__(self, graph, products):
        # graph is the JSON produced by the DumpDependencyGraph service, products the entries
        # produced by DumpProductNames, grouped by module label
        modules = graph["modules"]
        self.process_name = graph["process"]

        # the module dependencies, resolved by the framework
        self.module_inputs: Dict[str, Set[str]] = {
            label: set(module.get("consumes", ())) for label, module in modules.items()
        }

        # the Event data products each module declares it consumes
        self._consumes_products = {label: module.get("consumesProducts", ()) for label, module in modules.items()}

        # the Event data products known to the process, by (module label, product instance)
        self._products: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for entries in products.values():
            for product in entries:
                if product["branch"] == 0:
                    self._products[(product["module"], product["product_instance"])].append(product)

        # Run/lumi/process-block dependencies
        self._non_event_inputs: Dict[str, Set[str]] = {
            label: set(module.get("consumesNonEvent", ())) for label, module in modules.items()
        }

        # the condition under which each module runs
        self._reachability = self._build_reachability(graph)

    def consumed_products(self, module: str) -> List[ConsumedProduct]:
        """
        The Event data products a module consumes, matched against the data products of the
        process by label, product instance, type and process, as the framework does.
        """
        result = []
        for entry in self._consumes_products.get(module, ()):
            tag = (entry["label"], entry["instance"], entry["process"])
            # an edm::View consumer declares the element type, which matches the containers of that
            # type or of a type derived from it
            matches = [
                p
                for p in self._products.get(tag[:2], ())
                if (entry["type"] in p.get("view_types", ()) if entry["elementType"] else p["friendly_type_name"] == entry["type"])
            ]
            if tag[2] == "@skipCurrentProcess":
                matches = [p for p in matches if p["process"] != self.process_name]
            elif tag[2]:
                matches = [p for p in matches if p["process"] == tag[2]]
            else:
                # without a process name, a data product of this process hides those of the input
                matches = [p for p in matches if p["process"] == self.process_name] or matches

            for p in matches:
                if p["process"] != self.process_name:
                    # read from the input, which only the Source provides
                    result.append(ConsumedProduct(tag, SOURCE, p, p["product_instance"]))
                elif "alias_for" in p:
                    original = self._original(p)
                    result.append(ConsumedProduct(tag, original["module"], original, p["product_instance"]))
                else:
                    result.append(ConsumedProduct(tag, p["module"], p, p["product_instance"]))
        return result

    def _original(self, alias: dict) -> dict:
        """
        The product_names.json entry of the data product an EDAlias stands for.
        """
        target = alias["alias_for"]
        return next(
            p
            for p in self._products[(target["module"], target["product_instance"])]
            if p["friendly_type_name"] == alias["friendly_type_name"] and p["process"] == self.process_name
        )

    def non_event_dependencies(self, modules: List[str]) -> Dict[str, Set[str]]:
        offloaded = set(modules)

        # the run, lumi and process block producers the offloaded modules need but which stay local
        outside = {}
        for module in modules:
            # a producer that is offloaded too follows the module to the remote process
            producers = self._non_event_inputs.get(module, set()) - offloaded
            if producers:
                outside[module] = producers

        return outside

    @staticmethod
    def _build_reachability(graph) -> Dict[str, FrozenSet[Tuple[str, int]]]:
        # the edm type of every module, e.g. "Source", "EDProducer", "EDFilter"
        types = {label: module["type"] for label, module in graph["modules"].items()}

        # for each module, the set of positions in the schedule from which it is reached
        reachability = defaultdict(set)
        for kind in ("paths", "endpaths"):
            for name, labels in graph[kind].items():
                run_index = 0
                for label in labels:
                    # the position: which path, and which filter-free section of it
                    reachability[label].add((f"{kind}:{name}", run_index))
                    # an EDFilter can stop the path here, so what follows runs under a new condition
                    if types.get(label) == "EDFilter":
                        run_index += 1

        return {label: frozenset(runs) for label, runs in reachability.items()}

    def reachability_signature(self, module_name: str) -> Optional[FrozenSet[Tuple[str, int]]]:
        """
        The set of (path key, run index) positions that reach this module; None
        for a module that is on no Path or EndPath at all.
        """
        return self._reachability.get(module_name)

    def dependency_groups(self, modules: List[str]) -> List[List[str]]:
        """
        Split the offloaded modules into the units that are moved and gated together: modules that
        depend on each other and that are reached under the same condition end up in one group.
        """
        # the offloaded modules, as a set to test membership against
        selected = set(modules)

        # each module's inputs, restricted to the offloaded set
        inputs = {m: self.module_inputs.get(m, set()) & selected for m in modules}

        # the (component, condition) key that decides which group each module belongs to
        keys = self._grouping_keys(modules, inputs)

        # gather the modules that ended up sharing a key
        members: Dict[Tuple, List[str]] = {}
        for m in modules:
            members.setdefault(keys[m], []).append(m)

        # how the groups depend on each other, as "key -> the keys it consumes from"
        depends_on = {key: set() for key in members}
        for consumer, producers in inputs.items():
            for producer in producers:
                if keys[producer] != keys[consumer]:
                    depends_on[keys[consumer]].add(keys[producer])

        # keys that feed each other in both directions cannot be scheduled apart, so merge them back
        groups = []
        for component in self._strongly_connected_components(list(members), depends_on):
            # every key in the cycle contributes its modules to a single group
            merged = set(component)
            # list that group's modules with each producer ahead of the consumers that read it
            groups.append(self._in_dependency_order([m for m in modules if keys[m] in merged], inputs))

        return groups

    def _grouping_keys(self, modules: List[str], inputs: Dict[str, Set[str]]) -> Dict[str, Tuple]:
        """
        What decides which group a module belongs to: its connected component, paired
        with its reachability_signature().

        A module on no Path has no condition of its own, so there is nothing to filter it
        by. It borrows the key of a neighbour inside the offloaded set instead, a consumer
        for preference, so that a chain of such modules resolves from the scheduled end.
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
                # consumers first, producers second: take the first key that already carries a condition
                for neighbour in consumers.get(m, []) + sorted(inputs[m]):
                    if keys[neighbour][1] is not None:
                        keys[m] = keys[neighbour]
                        resolved.append(m)
                        break
            if not resolved:
                break
            unscheduled = [m for m in unscheduled if m not in resolved]

        return keys

    @staticmethod
    def _connected_components(modules: List[str], inputs: Dict[str, Set[str]]) -> Dict[str, str]:
        # union-find: every consumer-producer edge merges the two components it connects
        parent = {m: m for m in modules}

        def root(module_name):
            # climb to the top of the chain, flattening it on the way
            while parent[module_name] != module_name:
                parent[module_name] = parent[parent[module_name]]
                module_name = parent[module_name]
            return module_name

        for consumer, producers in inputs.items():
            for producer in producers:
                parent[root(producer)] = root(consumer)

        # name each module's component after the top-level one it ends up under
        return {m: root(m) for m in modules}

    @staticmethod
    def _strongly_connected_components(nodes: List, edges: Dict) -> List[List]:
        """
        Tarjan's algorithm: return the sets of nodes that all reach each other
        """
        # the order in which each node was first visited
        index: Dict = {}
        # the lowest visit number reachable from each node, which identifies its component
        lowlink: Dict = {}
        # the nodes visited but not yet assigned to a component, and the membership test for them
        stack: List = []
        on_stack: Set = set()
        # the finished components, in the order they were closed
        components: List[List] = []

        for start in nodes:
            if start in index:
                continue

            index[start] = lowlink[start] = len(index)
            stack.append(start)
            on_stack.add(start)
            # the search is kept explicit rather than recursive: one entry per unfinished node
            work = [(start, iter(edges.get(start, ())))]

            while work:
                # resume the deepest unfinished node, where its iterator of successors left off
                node, successors = work[-1]
                for successor in successors:
                    if successor not in index:
                        # descend into it, and come back to this node's iterator afterwards
                        index[successor] = lowlink[successor] = len(index)
                        stack.append(successor)
                        on_stack.add(successor)
                        work.append((successor, iter(edges.get(successor, ()))))
                        break
                    if successor in on_stack:
                        # a successor still on the stack loops back here, so it shares this component
                        lowlink[node] = min(lowlink[node], index[successor])
                else:
                    # a node whose lowlink never left its own visit number closes a component
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
                    # whatever this node can reach, the parent waiting for it can reach through it
                    if work:
                        lowlink[work[-1][0]] = min(lowlink[work[-1][0]], lowlink[node])

        return components

    @staticmethod
    def _in_dependency_order(component: List[str], inputs: Dict[str, Set[str]]) -> List[str]:
        """
        Sort a group's modules so that each one comes after the modules it consumes from.
        """
        # build the order in passes, tracking what has been placed and what is left
        ordered: List[str] = []
        placed: Set[str] = set()
        remaining = component

        while remaining:
            ready = [m for m in remaining if inputs[m] <= placed]
            if not ready:
                return ordered + remaining
            ordered += ready
            placed.update(ready)
            remaining = [m for m in remaining if m not in placed]

        return ordered

    def external_dependencies_by_group(
        self, groups: List[List[str]]
    ) -> Tuple[Dict[str, Set[int]], Dict[str, List[ConsumedProduct]]]:
        """
        For each producer that stays in the local process (SOURCE for the Source and the input),
        the indices of the groups reading from it, and the data products they read: exactly what
        has to be sent from the local process to the remote one.
        """
        # every module that moves to the remote process, in any group
        offloaded = {m for group in groups for m in group}

        # a data product coming from outside that set is produced locally, so record who needs it
        groups_by_producer = defaultdict(set)
        consumed_by_producer = defaultdict(list)
        for group_index, group in enumerate(groups):
            for m in group:
                for consumed in self.consumed_products(m):
                    if consumed.origin not in offloaded:
                        groups_by_producer[consumed.origin].add(group_index)
                        consumed_by_producer[consumed.origin].append(consumed)
        return groups_by_producer, consumed_by_producer

    def modules_to_send_back_by_group(self, groups: List[List[str]], modules_to_run_on_both: List[str]):
        """
        Split each group's members into the ones something local still reads, whose products have
        to be sent back, and the ones nothing local reads, which can be dropped from the local side.
        Also return, per group, the data products the local modules read from its members.
        """
        # the modules that end up existing only remotely; a duplicated one keeps running locally too
        remote_only = {m for group in groups for m in group} - set(modules_to_run_on_both)

        # what the modules that still run locally read from the remote-only modules
        read_locally = defaultdict(list)
        for consumer in self._consumes_products:
            if consumer not in remote_only:
                for consumed in self.consumed_products(consumer):
                    if consumed.origin in remote_only:
                        read_locally[consumed.origin].append(consumed)

        # per group, the members whose products have to travel back over MPI, and those products
        to_send = [[] for _ in groups]
        consumed_by_group = [[] for _ in groups]

        # the members no local module reads, collected across all the groups
        unused = []

        # a member is worth sending back only if at least one of its data products is read locally
        for group_index, group in enumerate(groups):
            for m in group:
                if m in modules_to_run_on_both:
                    continue
                if read_locally[m]:
                    to_send[group_index].append(m)
                    consumed_by_group[group_index] += read_locally[m]
                else:
                    unused.append(m)

        return to_send, unused, consumed_by_group
