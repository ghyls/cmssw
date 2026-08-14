"""
This module implements the logic to separate one remote process from the local one
"""

from itertools import groupby

from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import ModuleDependencyAnalyzer, flatten_all_to_module_list
from HeterogeneousCore.MPICore.configuration_splitter.editor_functions import *
from HeterogeneousCore.MPICore.configuration_splitter.path_state_helpers import *
from HeterogeneousCore.MPICore.configuration_splitter.multiple_remotes_option_parser import *
from HeterogeneousCore.MPICore.configuration_splitter.helper_jobs import DependencyGraphGetter

def split_remote(local_process, args, cpp_names_of_the_products):
    modules_to_offload = flatten_all_to_module_list(local_process, args.remote_modules)
    modules_to_run_on_both = flatten_all_to_module_list(local_process, args.duplicate_modules)

    # list of all modules to run on remote
    modules_to_offload.extend(m for m in modules_to_run_on_both if m not in modules_to_offload)

    graph = DependencyGraphGetter(local_process, reuse=args.reuse_dependency_graph).run()
    analyzer = ModuleDependencyAnalyzer(graph)

    groups = analyzer.dependency_groups(modules_to_offload)
    producer_to_groups = analyzer.external_dependencies_by_group(groups)

    first_dependency_in_a_group = [""]*len(groups)
    for i,group in enumerate(groups):
        elem = group[0]
        j=0
        while elem in modules_to_run_on_both and j<len(group):
            elem=group[j]
            j+=1
        first_dependency_in_a_group[i]=elem

    if args.verbose:
        print("Dependency groups: ", groups)
        print("Local producers - dependant groups correspondance: ", producer_to_groups)
        print("Modules to insert path state capture before for each group: ", first_dependency_in_a_group)

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

    # Whether a module ran in the original process has to be reproduced across the split,
    # since neither side sees the other's paths. That is done everywhere below with the
    # same three pieces: a PathStateCapture inserted at the position whose activity is
    # being reproduced, an MPISender/MPIReceiver pair carrying the resulting token, and a
    # PathStateRelease (an "activity filter") that stops the receiving path unless the
    # token arrived.

    # send the data needed by offloaded modules from local to remote
    remote_filters_by_group = [[] for _ in range(len(groups))]
    for local_dependency, group_indices in producer_to_groups.items():
        # at the group's topological root: a non-root member can depend on it too, and
        # gating the send on the root's position alone would send the data on the wrong
        # subset of events.
        direct_consumers_in_groups = [
            module_name
            for i in group_indices
            for module_name in groups[i]
            if local_dependency in analyzer.module_inputs.get(module_name, set())
        ]
        capture_name = f"activityCaptureBefore{local_dependency.title()}"
        insert_path_state_capture_before(local_process, first_modules_in_a_group=direct_consumers_in_groups, capture_name=capture_name)
        sender = create_sender(
                products=cpp_names_of_the_products[local_dependency],
                instance=instance,
                sender_upstream=controller_name,
                path_state_capture = capture_name
            )
        sender_name = f"mpiSender{args.remote_process_name.title()}{local_dependency.title()}"
        setattr(local_process, sender_name, sender)

        if local_dependency == "source":
            # "source" is a reserved Process slot that must stay a cms.Source, so unlike
            # an ordinary dependency below, the receiver cannot take over the name. Its
            # consumers reach it through an EDAlias (e.g. "rawDataCollector"), which
            # create_remote_process() does not copy over, so recreate that alias here.
            #
            # create_receiver_alias() expects the branch labelling that
            # create_group_receiver() produces (the module name embedded in the instance
            # name), which plain create_receiver() does not, hence the single-module group.
            receiver = create_group_receiver(
                group=[local_dependency],
                all_products=cpp_names_of_the_products,
                instance=instance,
                receiver_upstream="source",
                path_state_capture=True,
            )
            receiver_name = f"mpiReceiver{args.remote_process_name.title()}{local_dependency.title()}"
            setattr(remote_process, receiver_name, receiver)

            for alias_label in local_process.aliases_().keys():
                if local_dependency in getattr(local_process, alias_label).parameterNames_():
                    alias = create_receiver_alias(
                        receiver_name=receiver_name,
                        products=cpp_names_of_the_products[local_dependency],
                        module_name=local_dependency,
                    )
                    setattr(remote_process, alias_label, alias)
        else:
            receiver = create_receiver(
                    products=cpp_names_of_the_products[local_dependency],
                    instance=instance,
                    receiver_upstream="source",
                    path_state_capture=True,
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

        # One local product needed by several groups: the single capture above cannot
        # say which of them was actually reached, so add a per-group capture/sender,
        # receiver and filter on top. Groups that several such products feed still end
        # up with only one activation pair each.
        if len(group_indices) >= 2:
            for group_idx in group_indices:
                capture_name=f"activityCaptureBefore{args.remote_process_name.title()}Group{group_idx}"
                insert_path_state_capture_before(local_process, first_modules_in_a_group=[first_dependency_in_a_group[group_idx]], capture_name=capture_name)
                sender = create_sender(
                        products=[],
                        instance=instance,
                        sender_upstream=controller_name,
                        path_state_capture = capture_name
                    )
                sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}Activity"
                setattr(local_process, sender_name, sender)

                receiver = create_receiver(
                        products=[],
                        instance=instance,
                        receiver_upstream="source",
                        path_state_capture=True,
                    )
                receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}Activity"
                setattr(remote_process, receiver_name, receiver)

                # create filter for the path state
                filter_name = f"activityFilterBefore{args.remote_process_name.title()}Group{group_idx}"
                add_activity_filter(remote_process, receiver_name, filter_name)
                remote_filters_by_group[group_idx].append(filter_name)

                instance += 1
                mpi_path_modules_local[group_idx].append(sender_name)
                mpi_path_modules_remote[group_idx].append(receiver_name)

    # --- per-module activation gating ---
    #
    # dependency_groups() merges offloaded modules purely by data dependency, with no
    # notion of the Path or filter each one was originally gated by, so two members of
    # a group can come from completely differently-filtered Paths. The filters added
    # above only say that a needed external product was available somewhere in the
    # group; they say nothing about whether each member's own condition held. So give
    # every module back its own original activation, captured at its own position.
    #
    # Modules reached under a provably identical condition (equal reachability
    # signatures, e.g. a straight-line producer chain with no filter in between) share
    # one capture/sender/receiver/filter rather than each paying for an MPI round trip
    # per event. Modules on no Path at all get no gating object: there is no condition
    # to reproduce. See ModuleDependencyAnalyzer.reachability_signature.
    own_gating_filter = {}
    for group_idx, group in enumerate(groups):
        for eq_class in analyzer.reachability_classes(group):
            representative = eq_class[0]

            if analyzer.reachability_signature(representative) is None:
                for module_name in eq_class:
                    own_gating_filter[module_name] = None
                continue

            capture_name = f"activityCaptureOwn{representative.title()}"
            insert_path_state_capture_before(
                local_process, first_modules_in_a_group=[representative], capture_name=capture_name
            )
            sender = create_sender(
                module_name=representative,
                products=[],
                instance=instance,
                sender_upstream=mpi_path_modules_local[group_idx][-1],
                path_state_capture=capture_name,
            )
            sender_name = f"mpiSenderOwn{representative.title()}"
            setattr(local_process, sender_name, sender)
            mpi_path_modules_local[group_idx].append(sender_name)

            receiver_upstream = mpi_path_modules_remote[group_idx][-1] if mpi_path_modules_remote[group_idx] else "source"
            receiver = create_receiver(
                products=[],
                instance=instance,
                receiver_upstream=receiver_upstream,
                path_state_capture=True,
            )
            receiver_name = f"mpiReceiverOwn{representative.title()}"
            setattr(remote_process, receiver_name, receiver)
            mpi_path_modules_remote[group_idx].append(receiver_name)

            filter_name = f"activityFilterOwn{representative.title()}"
            add_activity_filter(remote_process, receiver_name, filter_name)

            for module_name in eq_class:
                own_gating_filter[module_name] = filter_name

            instance += 1

    per_group_remote_captures = [[] for _ in range(len(groups))]

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

        sender = create_group_sender(
            group=group,
            all_products=cpp_names_of_the_products,
            instance=instance,
            upstream_module=sender_upstream,
            path_state_capture=remote_capture_name,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, sender_name, sender)

        if len(mpi_path_modules_local[group_idx]) != 0:
            receiver_upstream = mpi_path_modules_local[group_idx][-1]
        else:
            receiver_upstream = controller_name

        receiver = create_group_receiver(
            group=group,
            all_products=cpp_names_of_the_products,
            instance=instance,
            receiver_upstream=receiver_upstream,
            path_state_capture=True,
        )
        receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}"
        setattr(local_process, receiver_name, receiver)

        instance += 1

        # insert filter on local before the first module which was supposed to run (is it correct?)
        filter_name = f"activityFilterAfter{args.remote_process_name.title()}Group{group_idx}"
        add_activity_filter(local_process, receiver_name, filter_name)
        insert_modules_before(local_process, getattr(local_process, group[0]), getattr(local_process, filter_name))

        mpi_path_modules_remote[group_idx].append(sender_name)
        mpi_path_modules_local[group_idx].append(receiver_name)


        for i, offloaded_module in enumerate(group):
            delattr(local_process, offloaded_module)
            module_alias = create_receiver_alias(receiver_name=receiver_name,
                products=cpp_names_of_the_products[offloaded_module],
                module_name=offloaded_module
            )
            setattr(local_process, offloaded_module, module_alias)


    # delete offloaded modules whose products are not needed on local from the local process:
    for product in modules_without_local_deps:
        delattr(local_process, product)

    # add all needed paths to the process and schedule them
    for i, group in enumerate(groups):
        if mpi_path_modules_local[i]:
            make_new_path(local_process, f"Offload{args.remote_process_name.title()}Group{i}", mpi_path_modules_local[i])
        if mpi_path_modules_remote[i]:
            make_new_path(remote_process, f"MPIPathGroup{i}", mpi_path_modules_remote[i])

        # Interleave each member with its own activation gate. A contiguous run of
        # members sharing a gate only needs it placed once; a gate that recurs later in
        # the group is simply referenced again, which costs nothing, since the framework
        # evaluates a module once per event and caches the result for any further
        # reference on the same Path.
        gated_group = []
        for filter_name, members in groupby(group, key=own_gating_filter.get):
            if filter_name is not None:
                gated_group.append(filter_name)
            gated_group.extend(members)

        make_new_path(remote_process, args.remote_process_name.title()+"RemoteOffloadedSequence"+str(i), remote_filters_by_group[i]+gated_group+per_group_remote_captures[i])

    if args.verbose:
        print(f"Successfully split out remote config with name {args.remote_process_name}!")

    return remote_process
