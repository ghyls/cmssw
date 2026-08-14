#!/bin/bash

# Checks the JSON that the DumpDependencyGraph service writes for
# test_dumpDependencyGraph_cfg.py.

function die { echo $1: status $2; exit $2; }

LOCAL_TEST_DIR=${LOCAL_TEST_DIR:-$(dirname $0)}

cmsRun ${LOCAL_TEST_DIR}/test_dumpDependencyGraph_cfg.py || die "cmsRun test_dumpDependencyGraph_cfg.py failed" $?

python3 - test_dumpDependencyGraph.json <<'EOF' || die "unexpected dependency graph" $?
import json, sys

graph = json.load(open(sys.argv[1]))
modules = graph["modules"]

def check(condition, message):
    if not condition:
        sys.exit(f"FAILED: {message}")

check(graph["process"] == "DEPGRAPH", "wrong process name: " + graph["process"])

check(modules["source"]["type"] == "Source", "the source is missing or mistyped")
check(modules["gate"]["type"] == "EDFilter", "gate should be an EDFilter")
check(modules["second"]["class"] == "AddIntsProducer", "wrong class for second")

check(modules["second"]["consumes"] == ["first"], "second should consume first")
check(modules["viaAlias"]["consumes"] == ["second"],
      "viaAlias consumes aliasOfSecond, which should be reported as its target 'second'")
check("consumes" not in modules["first"], "first consumes nothing and should say nothing")

check(graph["paths"] == {"p": ["first", "second", "viaAlias", "gate", "behindGate"]},
      "wrong paths: " + str(graph["paths"]))
check(graph["endpaths"] == {"e": ["watcher"]}, "wrong endpaths: " + str(graph["endpaths"]))

check("unscheduled" in modules, "an unscheduled module should still be listed")
check(not any("unscheduled" in labels for labels in graph["paths"].values()),
      "an unscheduled module should be on no path")
EOF

echo "OK"
