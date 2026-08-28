import collections

import FWCore.ParameterSet.Config as cms

# Early deletion of the pixel digi and cluster SoAs on the device: their only consumer is the alpaka pixel
# rechit producer (about 16 MiB per event and stream on QCD PU200). Only the device branches are listed, for
# every asynchronous backend; the host copies are declared as holding references to the device branches,
# because the device-to-host transform is not a consumes() of any module. Every consumer must enqueue its
# reads on the product's queue before taking a queue of its own; check that before adding a product here.

_clusterizers = ("SiPixelRawToClusterPhase1@alpaka", "SiPixelPhase2DigiToCluster@alpaka")
_devices = ("alpakaDevCudaRt", "alpakaDevHipRt")
# (device product type without the device prefix, host copy type)
_products = (("SiPixelDigisDeviceedmDeviceProduct", "SiPixelDigisHost"),
             ("SiPixelClustersDeviceedmDeviceProduct", "SiPixelClustersHost"))


def customiseEarlyDeleteForPixelClusterSoA(process, products):
    references = collections.defaultdict(list)

    def branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for label, module in process.producers_().items():
        if module.type_() not in _clusterizers:
            continue
        for deviceType, hostType in _products:
            deviceBranches = [branchName(device + deviceType, label) for device in _devices]
            products[label].extend(deviceBranches)
            references[branchName(hostType, label)].extend(deviceBranches)

    return (products, references)
