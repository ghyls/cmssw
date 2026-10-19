"""
Unit tests for ModuleDependencyAnalyzer against small synthetic DumpDependencyGraph JSON
documents, exercising the grouping and reachability logic without a cmsRun/MPI round
trip. See testOffloadedModulesDoNotRunMoreThanNeeded.sh,
testOffloadGatingDeduplication.sh and testOffloadGatingDeadlock.sh for the corresponding
end-to-end regression tests.
"""

from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import ModuleDependencyAnalyzer


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def producer(*consumes):
    return {"class": "AddIntsProducer", "type": "EDProducer", "consumes": list(consumes)}


# - producerA is consumed by chainB, then chainC.
# - chainC is consumed by gatedD, behind gateFilter on a different path, and also by
#   localConsumer, which stays local (not offloaded).
# - isolatedE has no relation to anything.
graph = {
    "process": "Test",
    "modules": {
        "producerA": producer(),
        "chainB": producer("producerA"),
        "chainC": producer("chainB"),
        "gateFilter": {"class": "ModuloEventIDFilter", "type": "EDFilter"},
        "gatedD": producer("chainC"),
        "localConsumer": producer("chainC"),
        "isolatedE": producer(),
    },
    "paths": {
        "mainPath": ["producerA", "chainB", "chainC", "localConsumer"],
        "gatedPath": ["gateFilter", "gatedD"],
        "isolatedPath": ["isolatedE"],
    },
    "endpaths": {},
}

analyzer = ModuleDependencyAnalyzer(graph)

# --- reachability: a filter-free chain shares a signature; a filter behind it changes it ---
check(analyzer.reachability_signature("chainB") == analyzer.reachability_signature("chainC"),
      "chainB and chainC sit on the same filter-free stretch of mainPath")
check(analyzer.reachability_signature("gatedD") != analyzer.reachability_signature("chainC"),
      "gatedD is gated by gateFilter on a different path, so its signature must differ from chainC's")
check(analyzer.reachability_signature("producerA") is not None, "producerA is on mainPath")

# --- dependency_groups(): one group per reachability class, ordered by their dependencies ---
groups = analyzer.dependency_groups(["chainB", "chainC", "gatedD", "isolatedE"])
groups_as_sets = [set(g) for g in groups]

check({"chainB", "chainC"} in groups_as_sets,
      f"chainB and chainC are reached under the same condition and must share a group, got {groups}")
check(["gatedD"] in groups, f"gatedD is reached under a condition of its own and must be alone, got {groups}")
check({"isolatedE"} in groups_as_sets, "isolatedE has no dependency edges, so it must be its own group")
check(len(groups) == 3, f"expected 3 groups, got {groups}")

chain_group_idx = next(i for i, g in enumerate(groups) if "chainB" in g)
gated_group_idx = groups.index(["gatedD"])
check(chain_group_idx < gated_group_idx,
      "a group must come after the groups it consumes from, so that its gate can be captured "
      "behind the point where theirs come back")
check(groups[chain_group_idx] == ["chainB", "chainC"],
      f"a group's members must be ordered so one comes after what it consumes, got {groups[chain_group_idx]}")

# --- external_dependencies_by_group(): only producers outside the offloaded set are sent ---
deps = analyzer.external_dependencies_by_group(groups)
check(deps["producerA"] == {chain_group_idx}, "producerA stays local and must be sent to the group consuming it")
check("chainC" not in deps,
      "chainC is offloaded, so gatedD reads it in the remote process: it must not be sent over "
      "just because it now sits in a different group")

# --- modules_to_send_back_by_group(): only products a locally running module reads come back ---
to_send, unused = analyzer.modules_to_send_back_by_group(groups, modules_to_run_on_both=[])
check("chainC" in to_send[chain_group_idx],
      "chainC feeds localConsumer, which stays local, so it must be sent back")
check("chainB" in unused, "chainB is only read by chainC, in the remote process, so it is unused on local")
check("gatedD" in unused, "nothing reads gatedD at all")

to_send_dup, unused_dup = analyzer.modules_to_send_back_by_group(groups, modules_to_run_on_both=["chainC"])
check("chainB" in to_send_dup[chain_group_idx],
      "with chainC running on both sides, its local copy reads chainB locally, so chainB must be sent back")
check("chainC" not in to_send_dup[chain_group_idx] and "chainC" not in unused_dup,
      "a module running on both sides produces its own local copy and is never sent back")

# --- an on-demand module is gated along with a neighbour, having no condition of its own ---
on_demand_graph = {
    "process": "Test",
    "modules": {"onDemandF": producer(), "scheduledG": producer("onDemandF")},
    "paths": {"mainPath": ["scheduledG"]},
    "endpaths": {},
}
on_demand = ModuleDependencyAnalyzer(on_demand_graph)
check(on_demand.reachability_signature("onDemandF") is None,
      "onDemandF is on no path or endpath, so it has no reachability signature")
check(on_demand.dependency_groups(["onDemandF", "scheduledG"]) == [["onDemandF", "scheduledG"]],
      "onDemandF has no condition and no position of its own, so it must be gated with its consumer")

# --- classes that depend on each other in both directions are merged back into one group ---
#   m1 and m3 share a filter-free stretch of pathP; m2, behind a filter on pathQ, sits
#   between them in the dependency chain, so grouping by class alone has no valid order.
cyclic_graph = {
    "process": "Test",
    "modules": {
        "m1": producer(),
        "gateFilter": {"class": "ModuloEventIDFilter", "type": "EDFilter"},
        "m2": producer("m1"),
        "m3": producer("m2"),
    },
    "paths": {"pathP": ["m1", "m3"], "pathQ": ["gateFilter", "m2"]},
    "endpaths": {},
}
cyclic = ModuleDependencyAnalyzer(cyclic_graph)
check(cyclic.reachability_signature("m1") == cyclic.reachability_signature("m3") != cyclic.reachability_signature("m2"),
      "the synthetic graph must really put m1 and m3 in one class and m2 in another")
check(cyclic.dependency_groups(["m1", "m2", "m3"]) == [["m1", "m2", "m3"]],
      "classes depending on each other in both directions have no order to be chained in, "
      "so they must be merged back into a single group")

# --- the two fields DumpDependencyGraph reports but does not interpret ---
#   "rawDataCollector" is the synthetic label raw FED data is filed under (Issue 45137);
#   a run/lumi dependency is reported apart from event-level ones because nothing forwards it.
provenance_graph = {
    "process": "Test",
    "modules": {
        "theSource": {"class": "FedRawDataInputSource", "type": "Source"},
        "readsRawData": {"class": "AddIntsProducer", "type": "EDProducer",
                         "consumesUnresolved": ["rawDataCollector"]},
        "readsNothingKnown": {"class": "AddIntsProducer", "type": "EDProducer",
                              "consumesUnresolved": ["gone"]},
        "runProducer": producer(),
        "runConsumer": {"class": "AddIntsProducer", "type": "EDProducer",
                        "consumesNonEvent": ["runProducer"]},
    },
    "paths": {"pathP": ["readsRawData", "readsNothingKnown", "runProducer", "runConsumer"]},
    "endpaths": {},
}
provenance = ModuleDependencyAnalyzer(provenance_graph)

check(provenance.module_inputs["readsRawData"] == {"theSource"},
      "a consumer of 'rawDataCollector' depends on the Source, whatever the Source is labelled")
check(provenance.producer_to_consumers["theSource"] == {"readsRawData"},
      "and the Source must know it as a consumer, so its products get forwarded")
check(provenance.module_inputs["readsNothingKnown"] == set(),
      "any other unresolved label names nothing this package knows, and is not an edge")

check(provenance.module_inputs["runConsumer"] == set(),
      "a run-level dependency is not an event-level one")
check(provenance.non_event_dependencies(["runConsumer"]) == {"runConsumer": {"runProducer"}},
      "offloading runConsumer alone leaves it depending on a run product produced locally")
check(provenance.non_event_dependencies(["runConsumer", "runProducer"]) == {},
      "offloading both puts producer and consumer in the same process, so there is nothing to report")

print("OK")
