"""
This module implements the logic to separate one remote process from the local one
"""

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import insert_modules_before

from HeterogeneousCore.MPICore.configuration_splitter.editor_functions import (
    add_controller_to_local,
    create_receiver,
    create_receiver_alias,
    create_remote_process,
    create_sender,
    make_new_path,
    products_of,
)
from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import (
    ModuleDependencyAnalyzer,
    flatten_all_to_module_list,
)
from HeterogeneousCore.MPICore.configuration_splitter.path_state_helpers import (
    add_activity_filter,
    insert_path_state_capture_before,
)


def split_remote(local_process, args, cpp_names_of_the_products, dependency_graph):
    # Are we using the portable MPI modules or the non-portable ones?
    portable = args.use_portable_mpi_modules

    modules_to_offload = flatten_all_to_module_list(local_process, args.remote_modules)
    modules_to_run_on_both = flatten_all_to_module_list(local_process, args.duplicate_modules)

    # list of all modules to run on remote
    modules_to_offload.extend(m for m in modules_to_run_on_both if m not in modules_to_offload)

    # everything the splitter knows about the configuration, from the framework's own
    # record of the consumes() dependencies and of the schedule
    analyzer = ModuleDependencyAnalyzer(dependency_graph)

    # the units to offload and gate together: modules that depend on each other and are
    # reached under the same condition
    groups = analyzer.dependency_groups(modules_to_offload)

    # the local producers the groups consume from, whose products have to be sent over
    producer_to_groups = analyzer.external_dependencies_by_group(groups)

    # warn about the dependencies the split would break: MPISender and MPIReceiver move
    # products event by event, so nothing forwards a run, lumi or process block one
    for module, producers in sorted(analyzer.non_event_dependencies(modules_to_offload).items()):
        print(f"[WARN] offloaded module '{module}' consumes a run, lumi or process block product of "
              f"{', '.join(repr(producer) for producer in sorted(producers))}, which stay(s) in the local "
              f"process. Nothing forwards those products, so '{module}' will not see them on the remote "
              "process: offload or duplicate the producer(s), or keep this module local.")

    if args.verbose:
        print("Dependency groups: ", groups)
        print("Local producers - dependant groups correspondance: ", producer_to_groups)

    # get products whose data needs to be sent, excluding modules without local dependencies and modules which should run on both processes
    modules_to_send, modules_without_local_deps = analyzer.modules_to_send_back_by_group(groups, modules_to_run_on_both)

    if args.verbose:
        print("Offloaded modues whose products need to be sent: ", modules_to_send)
        print("Offloaded modules without local dependencies: ", modules_without_local_deps)

    # --- start editing ---

    controller_name = add_controller_to_local(local_process, args.remote_process_name)
    remote_process = create_remote_process(local_process, modules_to_offload, args.remote_process_name, local_process.name_())

    mpi_path_modules_local = [[controller_name] for _ in range(len(groups))]
    mpi_path_modules_remote = [[] for _ in range(len(groups))]

    instance = 1

    # send the data needed by offloaded modules from local to remote
    remote_filters_by_group = [[] for _ in range(len(groups))]
    for local_dependency, group_indices in producer_to_groups.items():
        # every member of every group that needs this dependency, whose positions on the
        # local paths are where its activity has to be captured
        consumers_in_groups = [module_name for i in group_indices for module_name in groups[i]]
        # raw FED data is filed under the synthetic "rawDataCollector" label rather than
        # under the Source's own (issue 45137), so it travels along with the Source
        daq_raw_data_products = cpp_names_of_the_products["rawDataCollector"] if local_dependency == "source" else []
        # everything this dependency registers, which is what the round trip carries
        products_for_dependency = cpp_names_of_the_products[local_dependency] + daq_raw_data_products

        # the PathStateCapture whose token says the local paths reached those consumers
        capture_name = f"activityCaptureBefore{args.remote_process_name.title()}{local_dependency.title()}"
        # insert it in front of every one of them, so that the token exists whenever any
        # of them is reached, and always before any filter that will wait for it
        insert_path_state_capture_before(local_process, first_modules_in_a_group=consumers_in_groups, capture_name=capture_name)
        # the local end of the round trip: it sends the products, or, when the token is
        # missing, just the news that the path was not reached
        sender = create_sender(
            products=products_for_dependency,
            instance=instance,
            upstream=controller_name,
            activity=capture_name,
            portable=portable,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}{local_dependency.title()}"
        # add it to the local process
        setattr(local_process, sender_name, sender)

        if local_dependency == "source":
            # "source" is a reserved slot that has to stay a cms.Source, so the receiver
            # cannot take that label over; it gets one of its own, and carries the
            # products of the Source and of "rawDataCollector" together, which is why the
            # branches are labelled per module
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

            if daq_raw_data_products:
                # "rawDataCollector" exists only as a provenance convention, so recreate
                # it on the remote as an alias for what the receiver got
                alias = create_receiver_alias(
                    receiver_name=receiver_name,
                    products=daq_raw_data_products,
                    module_name="rawDataCollector",
                    portable=portable,
                )
                setattr(remote_process, "rawDataCollector", alias)

            # do the same for every local EDAlias that names this dependency, so that
            # the offloaded modules consuming through the alias still find its products
            for alias_label in local_process.aliases_().keys():
                if local_dependency in getattr(local_process, alias_label).parameterNames_():
                    alias = create_receiver_alias(
                        receiver_name=receiver_name,
                        products=cpp_names_of_the_products[local_dependency],
                        module_name=local_dependency,
                        portable=portable,
                    )
                    setattr(remote_process, alias_label, alias)
        else:
            # an ordinary producer: the receiver takes its module label over, so the
            # offloaded modules find the products under the name they always consumed
            receiver = create_receiver(
                products=products_for_dependency,
                instance=instance,
                upstream="source",
                activity=True,
                portable=portable,
            )
            receiver_name = local_dependency
            setattr(remote_process, receiver_name, receiver)

        # create filter for the path state
        filter_name = f"activityFilterAfter{local_dependency.title()}"
        add_activity_filter(remote_process, receiver_name, filter_name)
        for group_idx in group_indices:
            remote_filters_by_group[group_idx].append(filter_name)
            mpi_path_modules_local[group_idx].append(sender_name)
            mpi_path_modules_remote[group_idx].append(receiver_name)

        instance += 1

    # the round trips above only say that a group's inputs were available, and one local
    # product usually feeds several groups, so every group also gets a round trip of its
    # own saying whether that group itself was reached
    for group_idx, group in enumerate(groups):
        # the PathStateCapture standing for this group's condition
        capture_name = f"activityCaptureBefore{args.remote_process_name.title()}Group{group_idx}"
        # captured at the original position of every member: they all share one condition,
        # and the group's members are exactly where a filter will wait for this token
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

    # send the results from remote to local
    for group_idx, group in enumerate(modules_to_send):
        if len(group)==0:
            continue

        remote_capture_name = f"activityCaptureAfter{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, remote_capture_name, cms.EDProducer("PathStateCapture"))
        per_group_remote_captures[group_idx].append(remote_capture_name)

        if len(mpi_path_modules_remote[group_idx]) != 0:
            sender_upstream = mpi_path_modules_remote[group_idx][-1]
        else:
            sender_upstream = "source"

        # the way back: this group's products, or the news that the group did not run
        sender = create_sender(
            products=products_of(cpp_names_of_the_products, group),
            instance=instance,
            upstream=sender_upstream,
            activity=remote_capture_name,
            portable=portable,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, sender_name, sender)

        receiver_upstream = mpi_path_modules_local[group_idx][-1]

        receiver = create_receiver(
            products=products_of(cpp_names_of_the_products, group),
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

        # hold each local path where a member of this group used to sit until the group's
        # results arrive -- in front of every member, so that one with a local consumer of
        # its own is never left unguarded
        filter_name = f"activityFilterAfter{args.remote_process_name.title()}Group{group_idx}"
        add_activity_filter(local_process, receiver_name, filter_name)
        for offloaded_module in group:
            insert_modules_before(
                local_process, getattr(local_process, offloaded_module), getattr(local_process, filter_name)
            )

        # the sender closes this group's MPI path on the remote
        mpi_path_modules_remote[group_idx].append(sender_name)
        # the receiver goes on a path of its own on local, see below
        group_receiver_local[group_idx] = receiver_name


        for offloaded_module in group:
            # the module itself does not run in the local process any more
            delattr(local_process, offloaded_module)
            # its label now names an EDAlias for what came back, so that everything
            # downstream finds the products where it did before the split
            module_alias = create_receiver_alias(receiver_name=receiver_name,
                products=cpp_names_of_the_products[offloaded_module],
                module_name=offloaded_module,
                portable=portable,
            )
            setattr(local_process, offloaded_module, module_alias)


    # delete offloaded modules whose products are not needed on local from the local process:
    for product in modules_without_local_deps:
        delattr(local_process, product)

    # add all needed paths to the process and schedule them
    for i, group in enumerate(groups):
        if mpi_path_modules_local[i]:
            make_new_path(local_process, f"Offload{args.remote_process_name.title()}Group{i}", mpi_path_modules_local[i])
        if group_receiver_local[i] is not None:
            make_new_path(
                local_process, f"Offload{args.remote_process_name.title()}Group{i}Receive", [group_receiver_local[i]]
            )
        if mpi_path_modules_remote[i]:
            make_new_path(remote_process, f"MPIPathGroup{i}", mpi_path_modules_remote[i])

        make_new_path(remote_process, args.remote_process_name.title()+"RemoteOffloadedSequence"+str(i), remote_filters_by_group[i]+group+per_group_remote_captures[i])

    # instance 0 is reserved for the MPIController/MPISource pair, and MPISender rejects
    # anything above 255, which would not fit in the MPI tag
    if instance > 256:
        raise RuntimeError(
            f"splitting out '{args.remote_process_name}' needs {instance - 1} MPI channel instances, "
            "but only 255 are available; offload fewer modules, or split them over several remotes"
        )

    if args.verbose:
        print(f"Successfully split out remote config with name {args.remote_process_name}!")

    return remote_process
