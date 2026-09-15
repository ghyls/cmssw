"""
This module implements the logic to separate one remote process from the local one
"""

from collections import defaultdict

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import insert_modules_before

from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import (
    SOURCE,
    ModuleDependencyAnalyzer,
)
from HeterogeneousCore.MPICore.configuration_splitter.process_builders import (
    add_activity_filter,
    add_controller_to_local,
    alias_entry,
    create_alias,
    create_receiver,
    create_remote_process,
    create_sender,
    insert_path_state_capture_before,
    is_device_product,
    make_new_path,
)


def _flatten_all_to_module_list(process, user_args):
    """
    Flatten input arguments into an ordered list of module names.
    Preserves user-provided order and avoids duplicates.
    """
    module_list = []
    seen = set()

    for name in user_args:
        if not hasattr(process, name):
            print(f"[WARN] process has no attribute named '{name}'")
            continue

        obj_ = getattr(process, name)

        if hasattr(obj_, "moduleNames"):
            for mod in obj_.moduleNames():
                if mod not in seen:
                    module_list.append(mod)
                    seen.add(mod)
        else:
            if name not in seen:
                module_list.append(name)
                seen.add(name)

    return module_list


def _products_to_forward(consumed, process_name, portable):
    """
    The distinct data products behind a list of ConsumedProduct, as the product_names.json
    entries an MPISender/MPIReceiver pair carries. A data product read from the input is asked
    for with the process name its consumer uses, so that the sender finds the same one. Only
    the `portable` MPI modules carry device products.
    """
    products = {}
    registered = {}
    for c in consumed:
        product = c.product
        if is_device_product(product) and not portable:
            continue
        if product["process"] != process_name:
            product = dict(product, sender_process=c.tag[2])
        key = (product["module"], product["product_instance"], product["friendly_type_name"], product.get("sender_process", ""))
        if key in products:
            continue

        # a receiver can register a given type under a given product instance only once
        branch = alias_entry(product, "", grouped=True)[:2]
        if branch in registered:
            raise RuntimeError(
                f"the data products {registered[branch]} and {key} would be registered twice by the same MPIReceiver"
            )
        registered[branch] = key
        products[key] = product
    return list(products.values())


def split_remote(local_process, args, cpp_names_of_the_products, dependency_graph):
    # Are we using the portable MPI modules or the non-portable ones?
    portable = args.use_portable_mpi_modules

    modules_to_offload = _flatten_all_to_module_list(local_process, args.remote_modules)
    modules_to_run_on_both = _flatten_all_to_module_list(local_process, args.duplicate_modules)

    # list of all modules to run on remote
    modules_to_offload.extend(m for m in modules_to_run_on_both if m not in modules_to_offload)

    # the framework's own record of the consumes() dependencies and of the schedule
    analyzer = ModuleDependencyAnalyzer(dependency_graph, cpp_names_of_the_products)

    # the units to offload and to gate together
    groups = analyzer.dependency_groups(modules_to_offload)

    # the local producers the groups consume from, and the data products that have to be sent over
    producer_to_groups, consumed_by_producer = analyzer.external_dependencies_by_group(groups)

    # nothing forwards run, lumi or process block products, so warn about those dependencies
    for module, producers in sorted(analyzer.non_event_dependencies(modules_to_offload).items()):
        print(
            f"[WARN] offloaded module '{module}' consumes a run, lumi or process block product of "
            f"{', '.join(repr(producer) for producer in sorted(producers))}, which stay(s) in the local "
            f"process. Nothing forwards those products, so '{module}' will not see them on the remote "
            "process: offload or duplicate the producer(s), or keep this module local."
        )

    if args.verbose:
        print("Dependency groups: ", groups)
        print("Local producers - dependant groups correspondance: ", producer_to_groups)

    # get products whose data needs to be sent, excluding modules without local dependencies and modules which should run on both processes
    modules_to_send, modules_without_local_deps, consumed_back = analyzer.modules_to_send_back_by_group(
        groups, modules_to_run_on_both
    )

    if args.verbose:
        print("Offloaded modues whose products need to be sent: ", modules_to_send)
        print("Offloaded modules without local dependencies: ", modules_without_local_deps)

    # --- start editing ---

    controller_name = add_controller_to_local(local_process, args.remote_process_name)
    remote_process = create_remote_process(local_process, modules_to_offload, args.remote_process_name, local_process.name_())

    mpi_path_modules_local = [[controller_name] for _ in range(len(groups))]
    mpi_path_modules_remote = [[] for _ in range(len(groups))]

    instance = 1

    # the EDAliases the remote process needs, so that the offloaded modules find each data
    # product under the label they ask for: {label: {module label: {alias_entry()}}}
    remote_aliases = defaultdict(lambda: defaultdict(set))

    # send the data needed by offloaded modules from local to remote
    remote_filters_by_group = [[] for _ in range(len(groups))]
    for local_dependency, group_indices in producer_to_groups.items():
        # every member of every group that needs this dependency
        consumers_in_groups = [module_name for i in group_indices for module_name in groups[i]]
        # only the data products the offloaded modules read
        consumed = [c for c in consumed_by_producer[local_dependency] if portable or not is_device_product(c.product)]
        products_for_dependency = _products_to_forward(consumed, analyzer.process_name, portable)

        # the PathStateCapture whose token says the local paths reached those consumers
        capture_name = f"activityCaptureBefore{args.remote_process_name.title()}{local_dependency.title()}"
        # insert it in front of every one of them, always ahead of the filter that waits for it
        insert_path_state_capture_before(local_process, first_modules_in_a_group=consumers_in_groups, capture_name=capture_name)
        # send the products, or, when the token is missing, the news that the path was not reached
        sender = create_sender(
            products=products_for_dependency,
            instance=instance,
            upstream=controller_name,
            activity=capture_name,
            portable=portable,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}{local_dependency.title()}"
        setattr(local_process, sender_name, sender)

        if local_dependency == SOURCE:
            # "source" is a reserved cms.Source slot, so the receiver takes a label of its own and
            # carries the data products of the Source and of the input together
            receiver = create_receiver(
                products=products_for_dependency,
                instance=instance,
                upstream="source",
                activity=True,
                portable=portable,
                grouped=True,
            )
            receiver_name = f"mpiReceiver{args.remote_process_name.title()}{local_dependency.title()}"
            setattr(remote_process, receiver_name, receiver)

            # every label the offloaded modules ask for becomes an EDAlias of the receiver
            for c in consumed:
                remote_aliases[c.tag[0]][receiver_name].add(alias_entry(c.product, c.instance, grouped=True, portable=portable))
        else:
            # an ordinary producer: the receiver takes its module label over
            receiver = create_receiver(
                products=products_for_dependency,
                instance=instance,
                upstream="source",
                activity=True,
                portable=portable,
            )
            receiver_name = local_dependency
            setattr(remote_process, receiver_name, receiver)

            # the labels of the EDAliases of the producer become EDAliases of the receiver
            for c in consumed:
                if c.tag[0] != local_dependency:
                    remote_aliases[c.tag[0]][receiver_name].add(alias_entry(c.product, c.instance, grouped=False, portable=portable))

        # create filter for the path state
        filter_name = f"activityFilterAfter{local_dependency.title()}"
        add_activity_filter(remote_process, receiver_name, filter_name)
        for group_idx in group_indices:
            remote_filters_by_group[group_idx].append(filter_name)
            mpi_path_modules_local[group_idx].append(sender_name)
            mpi_path_modules_remote[group_idx].append(receiver_name)

        instance += 1

    # the EDAliases of offloaded producers, read by offloaded modules, move to the remote process as well
    offloaded = {m for group in groups for m in group}
    for module in offloaded:
        for c in analyzer.consumed_products(module):
            if c.origin in offloaded and c.tag[0] != c.origin:
                remote_aliases[c.tag[0]][c.origin].add(alias_entry(c.product, c.instance, grouped=False, portable=portable))

    for label, entries in sorted(remote_aliases.items()):
        # the Source slot of the remote process holds its MPISource, which an EDAlias can share
        if label != "source" and hasattr(remote_process, label):
            raise RuntimeError(f"the remote process already has a '{label}', so it cannot be an EDAlias too")
        setattr(remote_process, label, create_alias(entries))

    # the round trips above only say that a group's inputs were available, so every group also gets
    # a round trip of its own saying whether that group itself was reached
    for group_idx, group in enumerate(groups):
        # the PathStateCapture standing for this group's condition, captured at the original position of every member
        capture_name = f"activityCaptureBefore{args.remote_process_name.title()}Group{group_idx}"
        insert_path_state_capture_before(local_process, first_modules_in_a_group=group, capture_name=capture_name)
        # a sender carrying no products at all, only the token
        sender = create_sender(
            products=[],
            instance=instance,
            upstream=controller_name,
            activity=capture_name,
            portable=portable,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}Activity"
        setattr(local_process, sender_name, sender)

        receiver = create_receiver(
            products=[],
            instance=instance,
            upstream="source",
            activity=True,
            portable=portable,
        )
        receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}Activity"
        setattr(remote_process, receiver_name, receiver)

        # create filter for the path state
        filter_name = f"activityFilterBefore{args.remote_process_name.title()}Group{group_idx}"
        add_activity_filter(remote_process, receiver_name, filter_name)
        remote_filters_by_group[group_idx].append(filter_name)

        mpi_path_modules_local[group_idx].append(sender_name)
        mpi_path_modules_remote[group_idx].append(receiver_name)

        instance += 1

    # the remote PathStateCapture that reports back that a group actually ran
    per_group_remote_captures = [[] for _ in range(len(groups))]
    # the local receiver collecting each group's results, for the groups that send any
    group_receiver_local = [None] * len(groups)
    # the local EDAliases of the offloaded modules, pointed at the receivers instead
    local_aliases = defaultdict(lambda: defaultdict(set))

    # send the results from remote to local
    for group_idx, group in enumerate(modules_to_send):
        if len(group) == 0:
            continue

        remote_capture_name = f"activityCaptureAfter{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, remote_capture_name, cms.EDProducer("PathStateCapture"))
        per_group_remote_captures[group_idx].append(remote_capture_name)

        if len(mpi_path_modules_remote[group_idx]) != 0:
            sender_upstream = mpi_path_modules_remote[group_idx][-1]
        else:
            sender_upstream = "source"

        # the data products of this group's members that the local modules read
        group_products = _products_to_forward(consumed_back[group_idx], analyzer.process_name, portable)

        # the way back: this group's products, or the news that the group did not run
        sender = create_sender(
            products=group_products,
            instance=instance,
            upstream=sender_upstream,
            activity=remote_capture_name,
            portable=portable,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, sender_name, sender)

        receiver_upstream = mpi_path_modules_local[group_idx][-1]

        receiver = create_receiver(
            products=group_products,
            instance=instance,
            upstream=receiver_upstream,
            activity=True,
            portable=portable,
            grouped=True,
        )
        receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}"
        setattr(local_process, receiver_name, receiver)

        # every sender/receiver pair gets an MPI channel instance of its own
        instance += 1

        # hold each local path where a member of this group used to sit until the group's results arrive
        filter_name = f"activityFilterAfter{args.remote_process_name.title()}Group{group_idx}"
        add_activity_filter(local_process, receiver_name, filter_name)
        for offloaded_module in group:
            insert_modules_before(local_process, getattr(local_process, offloaded_module), getattr(local_process, filter_name))

        # the sender closes this group's MPI path on the remote
        mpi_path_modules_remote[group_idx].append(sender_name)
        # the receiver goes on a path of its own on local, see below
        group_receiver_local[group_idx] = receiver_name

        for offloaded_module in group:
            # the module label now names an EDAlias for what came back, so everything downstream is unchanged
            delattr(local_process, offloaded_module)
            module_alias = create_alias(
                {
                    receiver_name: {
                        alias_entry(p, p["product_instance"], grouped=True, portable=portable)
                        for p in group_products
                        if p["module"] == offloaded_module
                    }
                }
            )
            setattr(local_process, offloaded_module, module_alias)

        # the local modules reading through an EDAlias of a member
        for c in consumed_back[group_idx]:
            if c.tag[0] != c.origin and (portable or not is_device_product(c.product)):
                local_aliases[c.tag[0]][receiver_name].add(alias_entry(c.product, c.instance, grouped=True, portable=portable))

    # delete offloaded modules whose products are not needed on local from the local process:
    for product in modules_without_local_deps:
        delattr(local_process, product)

    # the local EDAliases naming a module that now runs only remotely point at the receivers instead
    remote_only = offloaded - set(modules_to_run_on_both)
    for alias_label, alias in list(local_process.aliases_().items()):
        names = alias.parameterNames_()
        if not remote_only.intersection(names):
            continue
        kept = {name: getattr(alias, name) for name in names if name not in remote_only}
        delattr(local_process, alias_label)
        if kept or alias_label in local_aliases:
            new_alias = create_alias(local_aliases.get(alias_label, {}))
            for name, entries in kept.items():
                setattr(new_alias, name, entries)
            setattr(local_process, alias_label, new_alias)

    # add all needed paths to the process and schedule them
    for i, group in enumerate(groups):
        if mpi_path_modules_local[i]:
            make_new_path(local_process, f"Offload{args.remote_process_name.title()}Group{i}", mpi_path_modules_local[i])
        # the receiver waits on its own path, so one group never orders anything against another group's sends
        if group_receiver_local[i] is not None:
            make_new_path(local_process, f"Offload{args.remote_process_name.title()}Group{i}Receive", [group_receiver_local[i]])
        if mpi_path_modules_remote[i]:
            make_new_path(remote_process, f"MPIPathGroup{i}", mpi_path_modules_remote[i])
        make_new_path(remote_process, args.remote_process_name.title()+"RemoteOffloadedSequence"+str(i), remote_filters_by_group[i]+group+per_group_remote_captures[i])

    # instance 0 is reserved for the MPIController/MPISource pair, and the MPI tag caps at 255
    if instance > 256:
        raise RuntimeError(
            f"splitting out '{args.remote_process_name}' needs {instance - 1} MPI channel instances, "
            "but only 255 are available; offload fewer modules, or split them over several remotes"
        )

    if args.verbose:
        print(f"Successfully split out remote config with name {args.remote_process_name}!")

    return remote_process
