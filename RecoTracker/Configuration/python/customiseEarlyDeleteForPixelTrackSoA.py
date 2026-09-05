import collections

import FWCore.ParameterSet.Config as cms

# Early deletion of the device products of the stub-seeded pixel-track chain (phase2CAStubs) that
# nothing reads after the next step of the chain (about 80 MiB per event and stream on QCD PU200).
#
# A product is released only after readers that enqueue all their reads in acquire(), or whose get()
# is their first access to the event (so they run on the product's queue), or whose reads are ordered
# through device products before such a reader. A product is listed only when every alpaka module
# that reads it is named; non-alpaka readers use the host copy, which is copied before the delete.
# customiseAlpakaServiceMemoryFilling makes a reopened race fail loudly: run it after changing a reader.

_devices = ("alpakaDevCudaRt", "alpakaDevHipRt")

# (device product type without the device prefix, host copy type)
_tracks = ("128falserecoTrackBlocksLayoutvoidPortableDeviceCollectionedmDeviceProduct",
           "128falserecoTrackBlocksLayoutPortableHostCollection")
_mask = ("128falserecoTrackingRecHitsMaskingLayoutvoidPortableDeviceCollectionedmDeviceProduct",
         "128falserecoTrackingRecHitsMaskingLayoutPortableHostCollection")
_hits = ("recoTrackingRecHitDeviceedmDeviceProduct", "recoTrackingRecHitHost")
_stubs = ("recoStubsDeviceedmDeviceProduct", "recoStubsHost")
_otHits = ("recoOTRecHitsDeviceedmDeviceProduct", None)  # the host collection is the original, not a copy

_stubCA = "CAHitNtupletAlpakaPhase2OTStubs@alpaka"
_selector = "PixelTrackForestHighPuritySelector@alpaka"
_masking = "PixelTracksMaskingSoA@alpaka"
_trackMerger = "PixelTracksSoAMerger@alpaka"
_stubProducer = "OTStubProducerVectorHitStyle@alpaka"

# Producer type -> (the products released, the reader types checked for them, and, where the release
# rests on a reader that takes the product in acquire() after every other reader, that reader: its type,
# the input-tag parameter with the fillDescriptions default through which it reads the product, and the
# parameter that turns the read on). The checks, per row:
_released = {
    # the CA track SoAs: read by the high-purity selector in acquire() only
    _stubCA: ((_tracks,), (_selector,), None),
    # the merged track SoA: read by the merged high-purity selector in acquire() only
    _trackMerger: ((_tracks,), (_selector,), None),
    # the hit mask: read by the displaced CA in acquire() only
    _masking: ((_mask,), (_stubCA,), None),
    # a high-purity selection: no device reader (the legacy converter reads the host copy); with two
    # iterations the prompt and displaced selections are read by the masking step and the merger in
    # produce() and stay
    _selector: ((_tracks,), (), None),
    # the pixel rechits: the CA iterations read them in acquire(), the masking step and the track
    # merger in produce(), and all of them feed the last high-purity selector of the chain, which
    # reads them in acquire() (useHitFeatures)
    "SiPixelRecHitAlpakaPhase2OTStubs@alpaka": ((_hits,), (_stubCA, _masking, _trackMerger, _selector),
                                                (_selector, "pixelRecHitSrc", "hltPhase2SiPixelRecHitsSoA",
                                                 "useHitFeatures")),
    # the stubs: the stub half of the same global hit index space, with exactly the same readers and
    # the same anchor
    _stubProducer: ((_stubs,), (_stubCA, _masking, _trackMerger, _selector),
                    (_selector, "stubsSrc", "hltOTStubProducer", "useHitFeatures")),
    # the outer-tracker rechits: the stub producer and the track merger read them in produce() and
    # both feed the high-purity selector, which reads them in acquire() (useHitFeatures)
    "PixelSeedingOTRecHitsSoAConverter@alpaka": ((_otHits,), (_stubProducer, _trackMerger, _selector),
                                                 (_selector, "otRecHitsSoASrc", "hltPixelSeedingOTRecHitsSoA", "useHitFeatures")),
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


def _value(module, parameter, default):
    # a parameter as written in the configuration, or its fillDescriptions default
    return getattr(module, parameter).value() if hasattr(module, parameter) else default


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
        released, checked, anchor = _released[module.type_()]
        if any(reader.type_() not in checked for reader in readers[label]):
            continue
        if anchor is not None:
            anchorType, parameter, default, gate = anchor
            if not any(module.type_() == anchorType and _value(module, gate, True)
                       and cms.InputTag(_value(module, parameter, default)).getModuleLabel() == label
                       for module in modules.values()):
                continue
        for deviceType, hostType in released:
            deviceBranches = [branchName(device + deviceType, label) for device in _devices]
            products[label].extend(deviceBranches)
            if hostType is not None:
                references[branchName(hostType, label)].extend(deviceBranches)

    return (products, references)
