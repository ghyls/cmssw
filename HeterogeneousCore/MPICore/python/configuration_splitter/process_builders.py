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

_SENDER_CLASS = {False: "MPISender", True: "MPISenderPortable@alpaka"}
_RECEIVER_CLASS = {False: "MPIReceiver", True: "MPIReceiverPortable@alpaka"}

_HOST_BACKEND = "serial_sync"

# the wrapper the framework puts around a product living on an Alpaka device
_DEVICE_PRODUCT = "edm::DeviceProduct<"


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
    # the ESPrefer statements resolve ambiguities between the event setup modules above,
    # so the remote process needs them as much as the local one does. ESPrefer.copy()
    # preserves the label of the preferred module, while clone() would drop it.
    for module in local_process.es_prefers.keys():
        setattr(remote_process, module, getattr(local_process, module).copy())

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

def is_device_product(prod):
    return prod["type"].startswith("edm::DeviceProduct")


def _is_portable_product(product):
    """
    Whether MPISenderPortable/MPIReceiverPortable have to move this product as a device
    product rather than as a plain host one.
    """
    return is_device_product(product) or "portable_type" in product


def annotate_portable_types(products_by_module, portable_types, process):
    """
    Give every device product, and its host mirror, a name that means the same thing on
    every Alpaka backend, recorded under the "portable_type" key.

    The product names come from a dump taken on one machine, and spell device types the
    way that machine's backend does. The configurations written out of them have to run
    anywhere, so this is what lets a CPU-only machine split a configuration for a GPU one,
    and the other way round.

    `portable_types` is the serialiser registry read by SerialiserTypeGetter, and
    `process` the configuration being split. A device product the registry does not know
    is left alone: most products are never forwarded, and _portable_type_of() raises for
    the ones that are. See doc/portable-product-types.md.
    """
    portable_types = {_without_whitespace(type_name): portable_type
                      for type_name, portable_type in portable_types.items()}

    for module, products in products_by_module.items():
        device_products = [product for product in products if is_device_product(product)]

        # A host-only dump has no device products; fall back to the configuration to find
        # which modules produce them.
        to_annotate = device_products or (products if _produces_device_products(process, module) else [])

        for product in to_annotate:
            mirror = _host_mirror_of(product, products) if is_device_product(product) else None
            portable_type = portable_types.get(_without_whitespace(product["type"]))
            if portable_type is None and mirror is not None:
                # a dump from a machine with a different GPU: nothing here can resolve its
                # device types, but the host mirror is spelled the same everywhere
                portable_type = portable_types.get(_without_whitespace(mirror["type"]))
            if portable_type is not None:
                product["portable_type"] = portable_type
                if mirror is not None:
                    # A module reading only the host mirror has it forwarded under the
                    # portable name too, as a host-only dump does, so that every machine
                    # writes the same configuration.
                    mirror["portable_type"] = portable_type


def _produces_device_products(process, module):
    """
    Whether `module` produces device products on a backend that has them: it is an Alpaka
    module, and the configuration does not pin it to a host backend.
    """
    definition = getattr(process, module, None)
    module_class = definition.type_() if hasattr(definition, "type_") else None
    if module_class is None or not module_class.endswith("@alpaka"):
        return False

    alpaka = getattr(definition, "alpaka", None)
    backend = getattr(alpaka, "backend", None)

    return backend is None or backend.value() != _HOST_BACKEND


def _without_whitespace(cpp_type):
    """A C++ type name with its whitespace removed, so "A<B<C> >" and "A<B<C>>" match."""
    return "".join(cpp_type.split())


def _portable_type_of(product):
    """
    The backend-independent name annotate_portable_types() gave this product, which is
    what a configuration has to write. Raises if the product has none.
    """
    portable_type = product.get("portable_type")
    if portable_type is None:
        raise RuntimeError(
            "no backend-independent type name is known for the device product "
            f"'{product['module']}:{product['product_instance']}' of type "
            f"'{product['type']}'. MPISenderPortable/MPIReceiverPortable can only move a "
            "device product whose type is registered with "
            "DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN; register a serialiser for it, or "
            "keep the module producing it out of --remote-modules."
        )

    return portable_type


def _unwrap_device_product(cpp_type):
    """
    The collection a device product holds, or `cpp_type` itself if it is not one:
    "SiPixelClustersDevice<...>" from "edm::DeviceProduct<SiPixelClustersDevice<...> >".
    """
    cpp_type = cpp_type.strip()
    if not cpp_type.startswith(_DEVICE_PRODUCT) or not cpp_type.endswith(">"):
        return cpp_type

    return cpp_type[len(_DEVICE_PRODUCT):-1].strip()


def _outer_cpp_class_name(cpp_type):
    """
    The class a product's "type" names, without its template arguments:
    "SiPixelClustersDevice" from "edm::DeviceProduct<SiPixelClustersDevice<...> >".
    """
    return _unwrap_device_product(cpp_type).split("<", 1)[0].strip()


def _template_arguments(cpp_type):
    """
    The top-level template arguments of a C++ type name, whitespace removed, as a tuple.
    "A<B<C,D>,E>" gives ("B<C,D>", "E"); a type with no arguments gives ().
    """
    start = cpp_type.find("<")
    if start == -1:
        return ()

    arguments = []
    current = ""
    depth = 0
    for character in cpp_type[start:]:
        if character == "<":
            depth += 1
            if depth == 1:
                # the outermost bracket delimits the arguments, it is not part of one
                continue
        elif character == ">":
            depth -= 1
            if depth == 0:
                break
        elif character == "," and depth == 1:
            arguments.append(current)
            current = ""
            continue
        current += character
    arguments.append(current)

    return tuple("".join(argument.split()) for argument in arguments)


def _portable_payload(cpp_type):
    """
    What a portable collection holds, spelled the same whichever flavour it is asked of.
    A device product and its host mirror agree on it; two different collections of one
    module do not.
    """
    return tuple(
        argument
        for argument in _template_arguments(_unwrap_device_product(cpp_type))
        if argument != "void" and not argument.startswith("alpaka::")
    )


def _host_mirror_of(device_product, products):
    """
    The host copy of `device_product` that the framework offers alongside it, or None if
    it cannot be told apart from the module's other products.

    Both flavours of a collection carry the same module and product instance name, so
    only their types tell them apart: the host one names the same class with "Device"
    replaced by "Host", and holds the same payload. Returning None is always safe -- the
    device flavour alone still carries the data.
    """
    key = (device_product["module"], device_product["product_instance"])
    expected_host_name = _outer_cpp_class_name(device_product["type"]).replace("Device", "Host")
    matches = [
        p
        for p in products
        if not is_device_product(p)
        and (p["module"], p["product_instance"]) == key
        and _outer_cpp_class_name(p["type"]) == expected_host_name
    ]
    if len(matches) > 1:
        payload = _portable_payload(device_product["type"])
        matches = [p for p in matches if _portable_payload(p["type"]) == payload]

    return matches[0] if len(matches) == 1 else None


def _drop_redundant_host_mirrors(products):
    """
    `products` without the host mirrors. A receiver offers those again by itself, so
    forwarding them too would move the same data twice.
    """
    mirrors = set()
    for product in products:
        if is_device_product(product):
            mirror = _host_mirror_of(product, products)
            if mirror is not None:
                mirrors.add(id(mirror))

    return [product for product in products if id(product) not in mirrors]


def _branch_label(module, product_instance):
    """
    The product instance name under which a receiver carrying several modules' products
    registers this one, so that modules producing the same instance name do not collide.
    """
    return f"{module}@{product_instance}" if product_instance else module


def _forwarded_products(products, portable=False):
    """
    Every product an MPISender/MPIReceiver pair carries over MPI, as (product, the C++
    type name to write for it in the configuration).
    """
    if portable:
        products = _drop_redundant_host_mirrors(products)

    for p in products:
        if portable and _is_portable_product(p):
            yield p, _portable_type_of(p)
        elif not is_device_product(p):
            yield p, p["type"]


def _branch_label_of(product):
    return _branch_label(product["module"], product["product_instance"])


def _sender_psets(products, portable):
    return [
        cms.PSet(
            type=cms.string(product_type),
            # a data product read from the input is asked for as the consumer asks for it
            name=cms.InputTag(p["module"], p["product_instance"], p.get("sender_process", "")),
        )
        for p, product_type in _forwarded_products(products, portable)
    ]


def _receiver_psets(products, portable, grouped):
    return [
        cms.PSet(
            type=cms.string(product_type),
            label=cms.string(
                _branch_label(p["module"], p["product_instance"]) if grouped else p["product_instance"]
            ),
        )
        for p, product_type in _forwarded_products(products, portable)
    ]


def create_sender(products, instance, upstream, activity=None, portable=False):
    """
    An MPISender (or MPISenderPortable) carrying `products`, product_names.json entries of
    one module or of several. `activity` names the PathStateCapture whose token decides
    whether the send happens at all, when the send is filtered.
    """
    parameters = dict(
        upstream=cms.InputTag(upstream),
        instance=cms.int32(instance),
        products=cms.VPSet(*_sender_psets(products, portable)),
    )
    if activity is not None:
        parameters["activity"] = cms.InputTag(activity)

    return cms.EDProducer(_SENDER_CLASS[portable], **parameters)


def create_receiver(products, instance, upstream, activity=False, portable=False, grouped=False):
    """
    The MPIReceiver (or MPIReceiverPortable) end of a create_sender() pair. `activity`
    says whether the sender is filtered, in which case the receiver puts the token it
    carries into the event, for an activity filter to release the path on.
    """
    return cms.EDProducer(
        _RECEIVER_CLASS[portable],
        upstream=cms.InputTag(upstream),
        instance=cms.int32(instance),
        products=cms.VPSet(*_receiver_psets(products, portable, grouped)),
        activity=cms.bool(activity),
    )


def alias_entry(product, to_instance, grouped, portable=False):
    """
    What an EDAlias needs to put `product` under the product instance `to_instance`: its
    friendly type, and the product instance it has in the module the EDAlias points to (a
    receiver registers it as modulelabel@instance when `grouped`).
    """
    from_instance = _branch_label_of(product) if grouped else product["product_instance"]
    # With the portable MPI modules, a device product is aliased with a "*" wildcard: naming its
    # friendly class name would pin the EDAlias to one backend.
    type_name = "*" if portable and _is_portable_product(product) else product["friendly_type_name"]
    return (type_name, from_instance, to_instance)


def _without_shadowed_entries(module_entries):
    """
    The alias_entry() tuples of one module, without the named entries that a "*" wildcard entry
    for the same product instances already covers: mixing both is an EDAlias conflict.
    """
    wildcards = {(from_instance, to_instance) for type_name, from_instance, to_instance in module_entries if type_name == "*"}
    return sorted(entry for entry in module_entries if entry[0] == "*" or entry[1:] not in wildcards)


def create_alias(entries):
    """
    An EDAlias from {module label: {alias_entry()}}, so that the modules reading from it find
    the data products under the label and product instance they ask for.
    """
    return cms.EDAlias(
        **{
            module: cms.VPSet(
                *(
                    cms.PSet(
                        type=cms.string(type_name),
                        fromProductInstance=cms.string(from_instance),
                        toProductInstance=cms.string(to_instance),
                    )
                    for type_name, from_instance, to_instance in _without_shadowed_entries(module_entries)
                )
            )
            for module, module_entries in sorted(entries.items())
        }
    )


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
