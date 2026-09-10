import copy
import json
import os
import subprocess
from collections import defaultdict

import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.MPICore.modules import DumpProductNames


class HelperJob:
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


class ProcessDump(HelperJob):
    """
    What the framework knows about the configuration being split, which is everything the
    splitter works from: the C++ type of every product, and the module dependency graph
    together with the schedule.

    Returns the pair (products, graph):

        products: {module label: [product descriptions]}

        graph:    {"process": "HLT",
                   "modules": {label: {"class": ..., "type": ..., "consumes": [labels]}},
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
        if getattr(process, "schedule", None) is None:
            process.schedule = cms.Schedule()
        process.schedule.append(process.PrintNamesPath)

    def parse(self):
        products = defaultdict(list)
        for entry in self.load("product_names"):
            products[entry["module"]].append(entry)

        return products, self.load("dependency_graph")


class SerialiserTypeGetter(HelperJob):
    """
    The name every Alpaka backend agrees on for each device product type that can be
    moved over MPI. Only the serialiser registry knows those names, so they have to be
    read out of a running job, and only annotate_portable_types() knows what to do with
    them.

    Returns {product type as a backend spells it: the same type written with the
    "ALPAKA_ACCELERATOR_NAMESPACE::" placeholder}, covering both this machine's backend
    and the host one.
    """

    name = "serialiser_types"
    json_names = ("serialiser_types", "serialiser_types_serial_sync")

    placeholder_namespace = "ALPAKA_ACCELERATOR_NAMESPACE::"

    def __init__(self, reuse=False):
        process = cms.Process("DUMPSERIALISERTYPES")
        process.source = cms.Source("EmptySource")
        process.load("Configuration.StandardSequences.Accelerators_cff")
        process.options.accelerators = ["*"]
        super().__init__(process, reuse)

    def configure(self, process):
        process.dumpSerialiserTypes = cms.EDAnalyzer(
            "DumpSerialiserTypes@alpaka",
            outputFile=cms.string(self.json_path("serialiser_types")),
        )
        process.dumpSerialiserTypesSerialSync = cms.EDAnalyzer(
            "DumpSerialiserTypes@alpaka",
            alpaka=cms.untracked.PSet(backend=cms.untracked.string("serial_sync")),
            outputFile=cms.string(self.json_path("serialiser_types_serial_sync")),
        )
        process.dumpSerialiserTypesPath = cms.EndPath(
            process.dumpSerialiserTypes + process.dumpSerialiserTypesSerialSync
        )

    def parse(self):
        serial_data = self.load("serialiser_types_serial_sync")

        # the name a configuration can use, under every registry key naming the
        # serialiser it belongs to
        aliases = {key: entry["alias"]
                   for entry in serial_data if "alias" in entry
                   for key in entry["keys"]}

        # every serialisable type under its host spelling, which is the same on every
        # backend
        portable_types = {entry["product_type"]: self.placeholder_namespace + entry["alias"]
                          for entry in serial_data if "alias" in entry}

        for entry in self.load("serialiser_types"):
            alias = next((aliases[key] for key in entry["keys"] if key in aliases), None)
            if alias is not None:
                portable_types[entry["product_type"]] = self.placeholder_namespace + alias

        return portable_types
