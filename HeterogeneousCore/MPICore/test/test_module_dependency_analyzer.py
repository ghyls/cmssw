"""
Unit tests for ModuleDependencyAnalyzer against a small synthetic DumpDependencyGraph
JSON document, exercising the grouping and reachability logic without a cmsRun/MPI round
trip. See testOffloadGroupOwnGating.sh and testOffloadGatingDeduplication.sh for the
corresponding end-to-end regression tests.
"""

from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import ModuleDependencyAnalyzer


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAILED: {message}")


#   producerA -> chainB -> chainC -> gatedD (behind gateFilter on a different path)
#                                  \-> localConsumer (stays local, not offloaded)
#   isolatedE has no relation to anything. onDemandF is on no path or endpath at all.
graph = {
    "process": "Test",
    "modules": {
        "producerA": {"class": "IntProducer", "type": "EDProducer"},
        "chainB": {"class": "AddIntsProducer", "type": "EDProducer", "consumes": ["producerA"]},
        "chainC": {"class": "AddIntsProducer", "type": "EDProducer", "consumes": ["chainB"]},
        "gateFilter": {"class": "ModuloEventIDFilter", "type": "EDFilter"},
        "gatedD": {"class": "AddIntsProducer", "type": "EDProducer", "consumes": ["chainC"]},
        "localConsumer": {"class": "AddIntsProducer", "type": "EDProducer", "consumes": ["chainC"]},
        "isolatedE": {"class": "IntProducer", "type": "EDProducer"},
        "onDemandF": {"class": "IntProducer", "type": "EDProducer"},
    },
    "paths": {
        "mainPath": ["producerA", "chainB", "chainC", "localConsumer"],
        "gatedPath": ["gateFilter", "gatedD"],
        "isolatedPath": ["isolatedE"],
    },
    "endpaths": {},
}

analyzer = ModuleDependencyAnalyzer(graph)

# --- dependency_groups(): connected components, ordered, no accidental merging ---
groups = analyzer.dependency_groups(["chainB", "chainC", "gatedD", "isolatedE"])
groups_as_sets = [set(g) for g in groups]

check({"chainB", "chainC", "gatedD"} in groups_as_sets,
      "chainB/chainC/gatedD are connected by real consumes() edges and must end up in one group")
check({"isolatedE"} in groups_as_sets, "isolatedE has no dependency edges, so it must be its own group")
check(len(groups) == 2, f"expected 2 groups, got {groups}")

chain_group_idx, chain_group = next((i, g) for i, g in enumerate(groups) if "chainB" in g)
check(chain_group.index("chainB") < chain_group.index("chainC") < chain_group.index("gatedD"),
      f"group must be ordered so a module comes after what it consumes, got {chain_group}")

# --- external_dependencies_by_group(): only genuinely external producers are reported ---
deps = analyzer.external_dependencies_by_group(groups)
check(deps["producerA"] == {chain_group_idx}, "producerA is external to the group and must be reported")
check("chainB" not in deps and "chainC" not in deps,
      "a group member consumed by another member of the same group is not an external dependency")

# --- reachability: a filter-free chain shares a signature; a filter behind it changes it ---
check(analyzer.reachability_signature("chainB") == analyzer.reachability_signature("chainC"),
      "chainB and chainC sit on the same filter-free stretch of mainPath")
check(analyzer.reachability_signature("gatedD") != analyzer.reachability_signature("chainC"),
      "gatedD is gated by gateFilter on a different path, so its signature must differ from chainC's")
check(analyzer.reachability_signature("onDemandF") is None,
      "onDemandF is on no path or endpath, so it has no reachability signature")

classes = analyzer.reachability_classes(chain_group)
check(sorted(len(c) for c in classes) == [1, 2],
      f"expected chainB/chainC to share one class and gatedD to be alone, got {classes}")

# --- modules_to_send_back_by_group(): only products consumed outside their own group are sent ---
to_send, unused = analyzer.modules_to_send_back_by_group(groups, modules_to_run_on_both=[])
check("chainB" in unused, "chainB is only consumed by chainC, inside its own group, so it is unused")
check("chainC" in to_send[chain_group_idx],
      "chainC also feeds localConsumer, which stays local, so it must be sent back")

print("OK")
