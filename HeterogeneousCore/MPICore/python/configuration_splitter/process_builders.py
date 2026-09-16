"""
Builds everything the split needs to exist in one of the two processes: the remote
cms.Process itself, the MPIController/MPISender/MPIReceiver modules connecting it to the
local one, the EDAliases putting the offloaded products back under their original labels,
the PathStateCapture/PathStateRelease pair the activity filters are made of, and the
paths holding all of it.

Nothing here decides *what* to build -- that is split_remote.py, the only caller.
"""

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import insert_modules_before

from HeterogeneousCore.MPICore.modules import MPISource


# -- the two processes --

def add_controller_to_local(process, remote_name):
    process.load("Configuration.StandardSequences.Accelerators_cff")
    process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
    process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")
    controller_name = f"mpiController{remote_name.title()}"
    controller = cms.EDProducer("MPIController",
                               followerProcessName = cms.string(remote_name))
    setattr(process, controller_name, controller)
    # Multiple luminocity blocks are currently unsupported
    process.options.numberOfConcurrentLuminosityBlocks = 1
    return controller_name


def create_remote_process(local_process, modules_to_run, remote_process_name, local_process_name):
    remote_process = cms.Process(remote_process_name)
    remote_process.load("Configuration.StandardSequences.Accelerators_cff")

    # load the global psets and event setup modules
    for module in local_process.psets.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())
    for module in local_process.es_sources.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())
    for module in local_process.es_producers.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())

    if hasattr(local_process, "MessageLogger"):
        remote_process.MessageLogger = local_process.MessageLogger.clone()

    remote_process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
    remote_process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")

    # where do i get this firstRun parameter from?
    remote_process.source = MPISource(
    #     firstRun = cms.untracked.uint32(process.Source)
        mode = 'CommWorld',
        controllerProcessName = local_process_name
    )
    remote_process.maxEvents.input = -1

    for module in modules_to_run:
        if not hasattr(local_process, module):
            raise RuntimeError(f"Module {module} not found in source process")
        setattr(remote_process, module, getattr(local_process, module).clone())

    return remote_process


# -- functions to add senders and receivers --

def _is_device_product(prod):
    return prod["type"].startswith("edm::DeviceProduct")


def _branch_label(module, product_instance):
    """
    The product instance name under which a receiver carrying several modules' products
    registers this one, so that modules producing the same instance name do not collide.
    """
    return f"{module}@{product_instance}" if product_instance else module


def _forwarded_products(products):
    """
    Every product an MPISender/MPIReceiver pair carries over MPI.
    """
    return [p for p in products if not _is_device_product(p)]


def products_of(products_by_module, modules):
    return [product for module in modules for product in products_by_module[module]]


def _sender_psets(products):
    return [
        cms.PSet(
            type=cms.string(p["type"]),
            name=cms.InputTag(p["module"], p["product_instance"]),
        )
        for p in _forwarded_products(products)
    ]


def _receiver_psets(products, grouped):
    return [
        cms.PSet(
            type=cms.string(p["type"]),
            label=cms.string(
                _branch_label(p["module"], p["product_instance"]) if grouped else p["product_instance"]
            ),
        )
        for p in _forwarded_products(products)
    ]


def create_sender(products, instance, upstream, activity=None):
    """
    An MPISender carrying `products`, the products of one offloaded module or of several
    (see products_of()). `activity` names the PathStateCapture whose token decides whether
    the send happens at all, when the send is filtered.
    """
    parameters = dict(
        upstream=cms.InputTag(upstream),
        instance=cms.int32(instance),
        products=cms.VPSet(*_sender_psets(products)),
    )
    if activity is not None:
        parameters["activity"] = cms.InputTag(activity)

    return cms.EDProducer("MPISender", **parameters)


def create_receiver(products, instance, upstream, activity=False, grouped=False):
    """
    The MPIReceiver end of a create_sender() pair. `activity` says whether the sender is
    filtered, in which case the receiver puts the token it carries into the event, for an
    activity filter to release the path on.
    """
    return cms.EDProducer(
        "MPIReceiver",
        upstream=cms.InputTag(upstream),
        instance=cms.int32(instance),
        products=cms.VPSet(*_receiver_psets(products, grouped)),
        activity=cms.bool(activity),
    )


def create_receiver_alias(receiver_name, products, module_name):
    """
    The EDAlias that puts a receiver's products back under the offloaded module's own
    label, so that everything downstream finds them where it did before the split.
    """
    psets = []

    for p in products:
        if _is_device_product(p):
            continue

        product_instance = p["product_instance"]
        fromProductInstance_string = _branch_label(module_name, product_instance)

        psets.append(
            cms.PSet(
                type=cms.string(p["friendly_type_name"]),
                fromProductInstance=cms.string(fromProductInstance_string),
                toProductInstance = cms.string(product_instance)
            )
        )

    alias = cms.EDAlias(
            **{
                receiver_name: cms.VPSet(*psets)
            }
        )

    return alias


# -- the activity filters --

def add_activity_filter(process, module_name, filter_name):
    """
    The PathStateRelease that stops its path unless the token carried by `module_name`,
    an MPIReceiver, arrived with the event.
    """
    filter_object =  cms.EDFilter("PathStateRelease",
            state = cms.InputTag(module_name)
        )
    setattr(process, filter_name, filter_object)


def insert_path_state_capture_before(
    process,
    first_modules_in_a_group,
    capture_name,
):
    """
      - create one PathStateCapture EDProducer
      - insert it in front of every one of `first_modules_in_a_group`, wherever on the
        paths of `process` each of them sits
    """

    # create the EDProducer
    setattr(
        process,
        capture_name,
        cms.EDProducer("PathStateCapture"),
    )

    capture = getattr(process, capture_name)

    # insert into sequences
    for module_name in first_modules_in_a_group:
        if not hasattr(process, module_name):
            print(f"[WARN] process has no module '{module_name}'")
            continue

        module = getattr(process, module_name)

        # insert at the beginning
        insert_modules_before(process, module, capture)


# -- paths --

def make_new_path(
    process,
    path_name: str,
    module_names: list[str],
):
    """
    Create a cms.Path from an ordered list of module names
    and optionally append it to the process schedule.
    """

    if not module_names:
        raise ValueError("Offload path must contain at least one module")

    modules = []
    for name in module_names:
        if not hasattr(process, name):
            raise AttributeError(f"Process has no module named '{name}'")
        modules.append(getattr(process, name))

    # Chain modules with +
    sequence = modules[0]
    for mod in modules[1:]:
        sequence = sequence + mod

    path = cms.Path(sequence)
    setattr(process, path_name, path)

    if hasattr(process, "schedule") and process.schedule is not None:
        process.schedule.append(path)

    return path
