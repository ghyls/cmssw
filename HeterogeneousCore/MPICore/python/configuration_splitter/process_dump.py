import copy
import json
import os
import subprocess
from collections import defaultdict

import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.MPICore.modules import DumpProductNames


class DumpJob:
    """
    A short cmsRun job the splitter runs to ask the framework something the configuration
    alone does not say. The answer comes back as JSON.

    Every job leaves the configuration it ran, its log and its JSON in `directory`, so
    that a failed run can be looked at and a later run can reuse the answer. Subclasses
    set `name` (which names the job's own files) and `json_names` (the documents its
    cmsRun writes), and implement configure() and parse().
    """

    directory = ".edmMpiSplitConfig"
    name = None
    json_names = ()

    def __init__(self, process, reuse=False):
        os.makedirs(self.directory, exist_ok=True)
        self.cfg_path = os.path.join(self.directory, f"{self.name}_cfg.py")
        self.log_path = os.path.join(self.directory, f"{self.name}.log")

        self.reuse = reuse
        self.process = None if reuse else copy.deepcopy(process)

        if reuse:
            for json_name in self.json_names:
                if not os.path.exists(self.json_path(json_name)):
                    raise RuntimeError(f"asked to reuse '{self.json_path(json_name)}', but that file does not exist")

    def json_path(self, json_name):
        return os.path.join(self.directory, f"{json_name}.json")

    def run(self):
        """What this job has to say, from a fresh cmsRun or from what an earlier one left."""
        if not self.reuse:
            self.configure(self.process)
            self.process.maxEvents.input = 0
            self.process.options.numberOfThreads = 1
            self.process.options.numberOfStreams = 1
            self.process.options.numberOfConcurrentLuminosityBlocks = 1

            with open(self.cfg_path, "w") as cfg:
                cfg.write(self.process.dumpPython())

            # Delete stale JSON before running: on failure, reading it back would silently
            # describe a different configuration or machine than this run.
            for json_name in self.json_names:
                if os.path.exists(self.json_path(json_name)):
                    os.remove(self.json_path(json_name))

            with open(self.log_path, "w") as log:
                # Not check=True: a missing or empty JSON gives a much clearer message.
                subprocess.run(["cmsRun", self.cfg_path], stdout=log, stderr=subprocess.STDOUT, check=False)

        return self.parse()

    def load(self, json_name):
        """One of this job's JSON documents, read back."""
        path = self.json_path(json_name)
        if not os.path.exists(path):
            raise RuntimeError(f"'{path}' was not produced; check the log at {self.log_path}")

        with open(path) as f:
            data = json.load(f)
        if not data:
            raise RuntimeError(f"'{path}' is empty; check the log at {self.log_path}")

        return data


class ProcessDump(DumpJob):
    """
    What the framework knows about the configuration being split, which is everything the
    splitter works from: the C++ type of every product, and the module dependency graph
    together with the schedule.

    Returns the pair (products, graph):

        products: {module label: [product descriptions]}

        graph:    {"process": "HLT",
                   "modules": {label: {"class": ..., "type": ..., "consumes": [labels],
                                       "consumesProducts": [{label, instance, process, type, elementType}]}},
                   "paths": {name: [labels in schedule order]},
                   "endpaths": {name: [labels]}}
    """

    name = "process_dump"
    json_names = ("product_names", "dependency_graph")

    def configure(self, process):
        process.DumpDependencyGraph = cms.Service(
            "DumpDependencyGraph",
            fileName=cms.untracked.string(self.json_path("dependency_graph")),
        )

        process.PrintNames = DumpProductNames(outputFile=self.json_path("product_names"))
        process.PrintNamesPath = cms.EndPath(process.PrintNames)
        # without an explicit Schedule every Path and EndPath runs, this one included
        if getattr(process, "schedule", None) is not None:
            process.schedule.append(process.PrintNamesPath)

    def parse(self):
        products = defaultdict(list)
        for entry in self.load("product_names"):
            products[entry["module"]].append(entry)

        return products, self.load("dependency_graph")
