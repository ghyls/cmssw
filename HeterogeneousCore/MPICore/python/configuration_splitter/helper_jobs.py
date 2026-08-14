"""
Throwaway `cmsRun` jobs the splitter uses to interrogate a configuration.

Some of what the splitter needs to know about a process is only available from the
framework itself: the C++ type of every product, and the real consumes() dependency
graph. Both are obtained the same way -- deep-copy the process, attach the module or
service that dumps the information as JSON, run it over zero events, and read the JSON
back -- so that plumbing lives in HelperJob and the two subclasses only say what to
attach and how to interpret the result.
"""

import copy
import json
import os
import subprocess
from collections import defaultdict

import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.MPICore.modules import DumpProductNames


class HelperJob:
    """Subclasses set `directory` and `name`, and implement `configure()`."""

    directory = None
    name = None

    def __init__(self, process, reuse=False):
        os.makedirs(self.directory, exist_ok=True)
        self.cfg_path = os.path.join(self.directory, f"{self.name}_cfg.py")
        self.json_path = os.path.join(self.directory, f"{self.name}.json")
        self.log_path = os.path.join(self.directory, f"{self.name}.log")

        self.reuse = reuse
        if reuse and not os.path.exists(self.json_path):
            raise RuntimeError(f"asked to reuse '{self.json_path}', but that file does not exist")
        self.process = None if reuse else copy.deepcopy(process)

    def run(self):
        if not self.reuse:
            self.configure(self.process)
            self.process.maxEvents.input = 0
            self.process.options.numberOfThreads = 1
            self.process.options.numberOfStreams = 1
            self.process.options.numberOfConcurrentLuminosityBlocks = 1

            with open(self.cfg_path, "w") as cfg:
                cfg.write(self.process.dumpPython())
            with open(self.log_path, "w") as log:
                # Not check=True: a missing or empty JSON gives a much clearer message.
                subprocess.run(["cmsRun", self.cfg_path], stdout=log, stderr=subprocess.STDOUT, check=False)

        if not os.path.exists(self.json_path):
            raise RuntimeError(f"'{self.json_path}' was not produced; check the log at {self.log_path}")

        with open(self.json_path) as f:
            data = json.load(f)
        if not data:
            raise RuntimeError(f"'{self.json_path}' is empty; check the log at {self.log_path}")

        return self.parse(data)

    def parse(self, data):
        return data


class DependencyGraphGetter(HelperJob):
    """
    Returns the DumpDependencyGraph document, keyed by module label:

        {"process": "HLT",
         "modules": {label: {"class": ..., "type": ..., "consumes": [labels]}},
         "paths": {name: [labels in schedule order]},
         "endpaths": {name: [labels]}}

    DumpDependencyGraph is a Service, so it needs no EndPath or schedule wiring: it
    writes its JSON as soon as the schedule is known, before the first event.
    """

    directory = ".depgraphdir"
    name = "dependency_graph"

    def configure(self, process):
        process.DumpDependencyGraph = cms.Service(
            "DumpDependencyGraph",
            fileName=cms.untracked.string(self.json_path),
        )


class CPPNameGetter(HelperJob):
    """Returns the C++ type of every product, as {module label: [product descriptions]}."""

    directory = ".cppnamedir"
    name = "product_names"

    def configure(self, process):
        process.PrintNames = DumpProductNames(outputFile=self.json_path)
        process.PrintNamesPath = cms.EndPath(process.PrintNames)

        if getattr(process, "schedule", None) is None:
            process.schedule = cms.Schedule()
        process.schedule.append(process.PrintNamesPath)

    def parse(self, data):
        products = defaultdict(list)
        for entry in data:
            products[entry["module"]].append(entry)
        return products
