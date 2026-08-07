"""
DependencyGraphGetter: mirrors CPPNameGetter's pattern (deep-copy process, run a throwaway
maxEvents=0 cmsRun, read back JSON) but drives the DumpDependencyGraph Service instead of
the DumpProductNames EDAnalyzer.

Unlike DumpProductNames, DumpDependencyGraph is a Service, so it needs no EndPath / schedule
wiring -- it hooks ActivityRegistry callbacks directly and writes its JSON once
lookupInitializationComplete fires, before any event is processed.
"""

import os
import copy
import subprocess
import json

import FWCore.ParameterSet.Config as cms


class DependencyGraphGetter:
    def __init__(
        self,
        original_process,
        helper_file_dir=".depgraphdir",
        cfg_name="print_depgraph_cfg.py",
        json_name="print_depgraph.json",
        log_name="print_depgraph_debug.log",
        reuse=False,
    ):
        os.makedirs(helper_file_dir, exist_ok=True)

        self.cfg_path = os.path.join(helper_file_dir, cfg_name)
        self.json_path = os.path.join(helper_file_dir, json_name)
        self.log_name = os.path.join(helper_file_dir, log_name)

        self.reuse = reuse

        if reuse:
            if not os.path.exists(self.json_path):
                raise RuntimeError(
                    f"--dep-graph-exists was specified but JSON file not found: {self.json_path}"
                )
            self.process = None
        else:
            self.process = copy.deepcopy(original_process)

    def get_dependency_graph(self):
        """
        Returns the parsed DumpDependencyGraph JSON document, e.g.:
        {
          "process": "A",
          "modules": [{"id": ..., "label": ..., "class": ..., "type": ..., "scheduled": ...}, ...],
          "paths": [{"name": ..., "modules": [id, id, ...]}, ...],
          "endpaths": [...],
          "edges": [{"from": id, "to": id, "kind": "consumes" | "schedule"}, ...]
        }
        """
        if self.reuse:
            return self._read_json()

        self._write_dependency_graph_config()
        self._run_cmsrun()

        return self._read_json()

    def _write_dependency_graph_config(self):
        self.process.DumpDependencyGraph = cms.Service(
            "DumpDependencyGraph",
            fileName=cms.untracked.string(self.json_path),
            showPathDependencies=cms.untracked.bool(True),
        )

        self.process.maxEvents.input = 0
        self.process.options.numberOfThreads = 1
        self.process.options.numberOfStreams = 1
        self.process.options.numberOfConcurrentLuminosityBlocks = 1

        with open(self.cfg_path, "w") as f:
            f.write(self.process.dumpPython())

    def _run_cmsrun(self):
        with open(self.log_name, "w") as log:
            subprocess.run(
                ["cmsRun", self.cfg_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,  # fail later with a nicer message
            )

    def _read_json(self):
        if not os.path.exists(self.json_path):
            raise RuntimeError(
                f"JSON file not produced: {self.json_path}\n"
                f"Check that the DumpDependencyGraph service executed correctly at {self.log_name}"
            )

        with open(self.json_path) as f:
            data = json.load(f)

        if not data:
            raise RuntimeError(f"JSON file '{self.json_path}' is empty.\nCheck the log at {self.log_name}")

        return data
