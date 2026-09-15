# This module contains functions to edit local and remote processes
import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.MPICore.modules import MPISource


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


def clone_module_from_process(dst, src, name):
    """
    Clone module 'name' from src Process into dst Process.
    """
    if not hasattr(src, name):
        raise RuntimeError(f"Module {name} not found in source process")

    setattr(dst, name, getattr(src, name).clone())


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
        clone_module_from_process(remote_process, local_process, module)
    
    return remote_process


# -- functions to add senders and receivers --

def is_device_product(prod):
    return prod["type"].startswith("edm::DeviceProduct")


def _branch_label(module, product_instance):
    """
    The product instance name under which a receiver carrying several modules' products
    registers this one, so that modules producing the same instance name do not collide.
    """
    return f"{module}@{product_instance}" if product_instance else module

def _forwarded_products(products):
    """
    Every product an MPISender/MPIReceiver pair carries over MPI, as (product, the C++
    type name to write for it in the configuration).
    """
    for p in products:
        if not is_device_product(p):
            yield p, p["type"]


def products_of(products_by_module, modules):
    return [product for module in modules for product in products_by_module[module]]

def _sender_psets(products):
    return [
        cms.PSet(
            type=cms.string(product_type),
            name=cms.InputTag(p["module"], p["product_instance"]),
        )
        for p, product_type in _forwarded_products(products)
    ]


def _receiver_psets(products, grouped):
    return [
        cms.PSet(
            type=cms.string(product_type),
            label=cms.string(
                _branch_label(p["module"], p["product_instance"]) if grouped else p["product_instance"]
            ),
        )
        for p, product_type in _forwarded_products(products)
    ]


def create_sender(products, instance, upstream, activity=None):
    """
    An MPISender carrying `products`, the products of one offloaded module or of several
    (see products_of()). `activity` names the PathStateCapture whose token decides whether
    the send happens at all, if the send is gated.
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
    gated, in which case the receiver puts the token it carries into the event, for an
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
        if is_device_product(p):
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
