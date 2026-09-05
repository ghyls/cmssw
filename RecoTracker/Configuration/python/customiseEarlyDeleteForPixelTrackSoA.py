import collections

import FWCore.ParameterSet.Config as cms

# Early deletion of the device products of the stub-seeded pixel-track chain (phase2CAStubs) that
# nothing reads after the next step of the chain: about 27 MiB per event and stream on QCD PU200.
#
# Releasing a device product early is safe only if no reader can still be reading it when the caching
# allocator hands its block to a later allocation on the producer's queue. The readers named below were
# checked for that: they enqueue every read in acquire(), which has drained before their produce() runs,
# and the delete follows produce().
#
# A product is listed only when every alpaka module the schedule runs that reads it is one of the
# readers named for it; any other reader keeps it alive. Readers that are not alpaka modules read the
# host copy, which is declared as holding a reference to the device product so that it is copied before
# the delete. customiseAlpakaServiceMemoryFilling makes a reopened race fail loudly: run it after
# changing a reader.

_devices = ("alpakaDevCudaRt", "alpakaDevHipRt")

# (device product type without the device prefix, host copy type)
_tracks = ("128falserecoTrackBlocksLayoutvoidPortableDeviceCollectionedmDeviceProduct",
           "128falserecoTrackBlocksLayoutPortableHostCollection")
_mask = ("128falserecoTrackingRecHitsMaskingLayoutvoidPortableDeviceCollectionedmDeviceProduct",
         "128falserecoTrackingRecHitsMaskingLayoutPortableHostCollection")

_stubCA = "CAHitNtupletAlpakaPhase2OTStubs@alpaka"
_selector = "PixelTrackForestHighPuritySelector@alpaka"
_masking = "PixelTracksMaskingSoA@alpaka"
_trackMerger = "PixelTracksSoAMerger@alpaka"

# Producer type -> (the products released, the reader types checked for them). The checks, per row:
_released = {
    # the CA track SoAs: read by the high-purity selector in acquire() only
    _stubCA: ((_tracks,), (_selector,)),
    # the merged track SoA: read by the merged high-purity selector in acquire() only
    _trackMerger: ((_tracks,), (_selector,)),
    # the hit mask: read by the displaced CA in acquire() only
    _masking: ((_mask,), (_stubCA,)),
    # a high-purity selection: no device reader (the legacy converter reads the host copy); with two
    # iterations the prompt and displaced selections are read by the masking step and the merger in
    # produce() and stay
    _selector: ((_tracks,), ()),
}


def _inputLabels(parameters):
    # every module label an InputTag anywhere in the parameter set points at
    labels = set()
    for name in parameters.parameterNames_():
        p = getattr(parameters, name)
        if isinstance(p, cms.InputTag):
            labels.add(p.getModuleLabel())
        elif isinstance(p, cms.VInputTag):
            labels.update(cms.InputTag(t).getModuleLabel() if isinstance(t, str) else t.getModuleLabel() for t in p)
        elif isinstance(p, cms.PSet):
            labels.update(_inputLabels(p))
        elif isinstance(p, cms.VPSet):
            for pset in p:
                labels.update(_inputLabels(pset))
    return labels


def _scheduledModules(process):
    # the modules the job runs: those on the scheduled paths and the ones they consume (unscheduled)
    modules = dict(process.producers_())
    modules.update(process.filters_())
    modules.update(process.analyzers_())
    schedule = process.schedule_()
    paths = list(schedule) if schedule is not None else list(process.paths_().values()) + list(process.endpaths_().values())
    reachable = set()
    for path in paths:
        reachable.update(label for label in path.moduleNames() if label in modules)
    frontier = list(reachable)
    while frontier:
        label = frontier.pop()
        for inputLabel in _inputLabels(modules[label]):
            if inputLabel in modules and inputLabel not in reachable:
                reachable.add(inputLabel)
                frontier.append(inputLabel)
    return {label: modules[label] for label in reachable}


def customiseEarlyDeleteForPixelTrackSoA(process, products):
    references = collections.defaultdict(list)

    def branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    modules = _scheduledModules(process)
    if not any(module.type_() == _stubCA for module in modules.values()):
        return (products, references)

    # the alpaka modules reading each producer's output, by type
    readers = collections.defaultdict(list)
    for label, module in modules.items():
        if module.type_().endswith("@alpaka"):
            for producer in _inputLabels(module):
                readers[producer].append(module)

    for label, module in sorted(modules.items()):
        if module.type_() not in _released:
            continue
        released, checked = _released[module.type_()]
        if any(reader.type_() not in checked for reader in readers[label]):
            continue
        for deviceType, hostType in released:
            deviceBranches = [branchName(device + deviceType, label) for device in _devices]
            products[label].extend(deviceBranches)
            if hostType is not None:
                references[branchName(hostType, label)].extend(deviceBranches)

    return (products, references)
