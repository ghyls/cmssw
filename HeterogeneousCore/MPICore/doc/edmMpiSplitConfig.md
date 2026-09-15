# What `edmMpiSplitConfig` actually does

This is a walk-through of everything that happens between the configuration you hand to
`edmMpiSplitConfig` and the two configurations it writes out. It follows one concrete
command, the one used to offload the GPU part of an HLT menu:

```bash
edmMpiSplitConfig hlt_ngt.py \
    --remote-modules hltBackend hltOnlineBeamSpotDevice \
                     hltSiPixelClustersSoA hltSiPixelRecHitsSoA hltPixelTracksSoA hltPixelVerticesSoA \
                     hltSiStripRawToClustersFacilityAlpaka \
                     hltEcalDigisSoA hltEcalUncalibRecHitSoA \
                     hltHcalDigisSoA hltHbheRecoSoA \
                     hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA \
    --duplicate-modules hltHcalDigis hltOnlineBeamSpot \
    --output-local  hlt_new_legacy_local.py \
    --output-remote hlt_new_legacy_remote.py \
    --use-portable-mpi-modules
```

and which is then run as

```bash
mpirun ... -np 1 cmsRun hlt_new_legacy_local.py : -np 1 cmsRun hlt_new_legacy_remote.py
```

The goal of the whole exercise is simple to state and fiddly to achieve: **the two
processes together must behave exactly like the one process did.** Every module must run
on the same events it ran on before, produce the same products, and no more often than
before.

---

## The shape of the result

Before the details, here is what the split looks like from far away.

The **local** process keeps the menu. Where an offloaded module used to sit, it now has
an `EDAlias` pointing at whatever came back over MPI, so that everything downstream still
finds the product under the label it always used. It gains an `MPIController` (which
drives the remote process through every run/lumi/event transition), a set of `MPISender`
modules that push the inputs the remote needs, and a set of `MPIReceiver` modules that
collect the results.

The **remote** process is a small, purpose-built process: an `MPISource` instead of a real
source, the offloaded modules themselves, and the mirror-image set of receivers and
senders. It has none of the menu's other 10 000 modules.

Both sides also carry the *gating* machinery — `PathStateCapture` producers and
`PathStateRelease` filters — which is what makes "run on the same events" true rather than
merely approximately true. That is the part most of this document is about.

---

## Stage 0 — parsing the command line

`multiple_remotes_option_parser.py` splits `sys.argv` on `:` into one group per remote
process. Options before the first `:` belong to the first remote, and so on. A handful of
options (`--output-local`, `--verbose`, `--reuse-dumps`, and the input file itself) are
*global*: they are picked out of the whole command line wherever they appear, so it does
not matter which group they land in.

Each group becomes one `argparse.Namespace` with all the defaults filled in — including
the remote process name (`REMOTE0`, `REMOTE1`, …) and the output file name — so that
everything downstream can just read the fields.

In the example there is no `:`, so there is exactly one remote, named `REMOTE0`.

## Stage 1 — loading the configuration

`processFromFile()` executes `hlt_ngt.py` and gives back the live `cms.Process` object.
From here on, "the configuration" means that Python object: a big bag of modules,
sequences, paths and services that can be inspected and modified in memory, and dumped
back to Python text at the end.

Two things are worth knowing about this object:

* It is **mutated in place**. The local output configuration *is* this object, edited.
* It knows almost nothing about *dependencies*. A module's parameters contain
  `cms.InputTag("hltSiPixelClusters")` values, but whether a given string is a module
  label, a product instance name, or just a string is not something the Python
  configuration can tell you. Which is why there is a Stage 2.

## Stage 2 — asking the framework what is really going on

This is the expensive stage, and the one that makes the tool trustworthy.

The splitter takes a **copy** of the process, attaches two dumping components to it, sets
`maxEvents.input = 0`, writes it to `.edmMpiSplitConfig/process_dump_cfg.py`, and runs
`cmsRun` on it. Zero events are processed — everything the splitter needs is known by the
time the framework has finished building the schedule — but the whole menu is still
constructed, which is why this single job dominates the run time of the tool.

The two components are:

**`DumpProductNames`** (an EDAnalyzer) writes `.edmMpiSplitConfig/product_names.json`: one
entry per product that any module registers, with the producing module's label, the
product instance name, the full C++ type, and the "friendly class name" that `EDAlias`
entries have to be written in. This is where the splitter learns *what* it would have to
send if it sent a given module's output.

**`DumpDependencyGraph`** (a Service, added to `FWCore/Services` by this work) writes
`.edmMpiSplitConfig/dependency_graph.json`: for every module, which modules it consumes
from, and for every Path and EndPath, the modules on it in schedule order.

Both are attached to the same process so the menu only runs once.

### Why the dependency graph comes from the framework and not from the Python

An earlier version of this tool worked out dependencies by walking each module's Python
parameters looking for `cms.InputTag` values and matching them against process attribute
names. That both **invents** dependencies (any string that happens to look like a module
label) and **misses** real ones: `mayConsumes`, `EDAlias` and `SwitchProducer`
indirection, and products made by the Source are all invisible that way.

`DumpDependencyGraph` instead reports what the framework itself resolved after building
the schedule, so all of the above come out right. For each module it writes three lists:

* **`consumes`** — the modules whose *event* products this module depends on. Aliases are
  already followed, so if a module consumes `hltEcalDigis` and that is an `EDAlias` for
  something else, the real producer is what appears here.
* **`consumesNonEvent`** — the same for the run, lumi and process-block transitions. These
  are kept separate because they matter to the splitter in a completely different way:
  `MPISender`/`MPIReceiver` move products **event by event**, so a dependency on a run
  product of a module that stays local simply cannot be forwarded. The splitter prints a
  warning when it sees one rather than quietly emitting a configuration whose remote never
  gets the product.
* **`consumesUnresolved`** — labels the module declared that name no module at all. These
  are not errors. The most important one in practice is `rawDataCollector`: raw FED data
  from a DAQ source is filed under that synthetic label rather than under the source's own
  label (a long-standing framework quirk, CMSSW issue #45137). The service reports such
  labels verbatim rather than guessing what they mean; interpreting them is the splitter's
  job, and `rawDataCollector` is the one it knows about.

### The `--reuse-dumps` shortcut

If you have run the splitter before on the same, unmodified input configuration, `-c` /
`--reuse-dumps` skips the `cmsRun` job entirely and reads back the JSON files from
`.edmMpiSplitConfig/`. This is the difference between a few minutes and a few seconds. It
is your responsibility to know the input has not changed: nothing checks.

When the dumps *are* regenerated, any previous copies are deleted first. That is
deliberate — if `cmsRun` were to fail, silently reading back a stale file would describe a
different configuration, or a different machine, than the one being split.

## Stage 3 — naming device products in a backend-independent way (portable mode only)

This stage only runs with `--use-portable-mpi-modules`, and it exists to solve one
specific problem.

`MPISenderPortable`/`MPIReceiverPortable` can move Alpaka **device** products directly,
without a host copy. To do so, each product in their configuration is named by its C++
type. But the type a product dump reports for a device product spells out the Alpaka
device it happens to live on, for example:

```
edm::DeviceProduct<SiPixelClustersDevice<alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt> > >
```

Writing that into a configuration pins the configuration to CUDA. A process running on a
machine with no GPU registers no serialiser under that name and has nothing to resolve it
with.

So the splitter runs a second, much cheaper `cmsRun` job (it does not read your
configuration at all) that attaches `DumpSerialiserTypes` twice: once on whichever backend
the machine would pick, and once forced to `serial_sync`. Each writes out the contents of
that backend's device serialiser registry.

The CPU backend is the useful one: there, each serialiser is also registered under the
type name spelled exactly as a configuration has to spell it, with the backend's namespace
replaced by the `ALPAKA_ACCELERATOR_NAMESPACE::` placeholder —
`ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection`. The two dumps are joined on a
key both backends agree on (the mangled type id of the serialiser's *host* flavour), which
is what lets the machine-specific device type be translated into the portable one.

`annotate_portable_types()` then walks the product dump and records that portable name on
every device product it recognises. Two fallbacks make this work on machines that are not
the machine the products were dumped on:

* a device product whose own type is unknown here is looked up through its **host mirror**
  — the host-side flavour of the same collection, which every backend spells identically.
  This is what lets a menu dumped on an NVIDIA machine be split on an AMD one, or on one
  with no GPU at all.
* a dump taken on a host-only backend contains **no device products at all** — an Alpaka
  module registers its collections under their host types there. The splitter falls back
  to the configuration itself: a module spelled `...@alpaka` and not pinned to
  `serial_sync` runs on whatever backend the job picks, so the collections it registers
  are exactly the ones that would be device products on a GPU.

A device product that still has no known portable name is left alone — most products are
never forwarded anyway. If one that *is* forwarded turns out to have no name, the splitter
raises an error telling you which product it was and that its type needs a
`DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN` registration.

---

Everything above happens **once**, from the configuration exactly as you wrote it. The
remaining stages run once per remote process, and they mutate the local process as they
go. That ordering is not incidental: grouping, gating and forwarding all exist to
reproduce the *original* schedule, and a dependency or a path position read off a
half-split process would describe the splitter's own earlier work instead.

---

## Stage 4 — resolving the module names you asked for

`--remote-modules` and `--duplicate-modules` take module labels, but they also take
**sequence and task names**: anything in the process that has a `moduleNames()` method is
expanded into its members, in order, without duplicates. A name that the process does not
have at all produces a warning and is skipped.

The duplicate modules are then appended to the offload list — a module that runs on both
sides still has to exist on the remote side.

In the example, the 13 `--remote-modules` are all plain labels, and `hltHcalDigis` and
`hltOnlineBeamSpot` join them, giving 15 modules to place on the remote.

### What "duplicate" means

`--duplicate-modules` marks modules that are **cheaper to recompute than to transfer**.
`hltHcalDigis` unpacks HCAL raw data; the remote needs its output as input to
`hltHcalDigisSoA`, but sending it would mean shipping the unpacked digis over MPI when the
raw data is being shipped anyway. So it is simply run on both sides. Its product is never
transferred in either direction, and — importantly — the local copy is *not* gated on
anything coming back from the remote, because it never needed to be.

## Stage 5 — grouping the offloaded modules

This is the heart of the tool, and it is done by `ModuleDependencyAnalyzer`.

The offloaded modules are not one lump. They are split into **groups**, and each group
gets its own gate and its own round trip. A group is defined by two things at once:

1. **the connected component** of the modules' dependencies on each other — if A consumes
   B and both are offloaded, they belong together; and
2. **the reachability signature** — the condition under which the module ran in the
   original schedule.

### Reachability signatures

A module's *signature* is the set of filter-free stretches of paths it sits in. Concretely:
walk each Path in schedule order, starting at "run 0"; every time you pass an `EDFilter`,
increment the run counter. Every module in one such stretch is reached under exactly the
same condition — all the filters before it on that path passed. A module on several paths
gets one entry per occurrence, and its real condition is the OR over them, since the
framework runs a module as soon as any path reaches its position.

Two modules whose signatures are **equal** are provably reached under identical
conditions, so one gate can serve both. Two modules whose signatures differ might or might
not actually differ in practice, but keeping them apart is always safe.

This matters enormously. Without it, offloading `A` (which runs on every event) together
with `B` (which runs only behind a filter, and happens to consume `A`) would put both
behind one gate — and that gate would be either A's condition, making B run too often, or
B's condition, making A run too rarely. Splitting by signature makes a single gate per
group exact rather than an approximation.

Modules that sit on **no path at all** — unscheduled, run on demand for whoever consumes
them — have no condition of their own. They inherit the key of a neighbour inside the
offloaded set, a consumer for preference.

Finally, groups that end up depending on each other in **both** directions are merged back
together, because there is no order in which to chain their gates. That is always safe,
just not always as tight as possible.

In the example this yields **10 groups**:

| group | members |
| --- | --- |
| 0 | `hltBackend` |
| 1 | `hltOnlineBeamSpot` |
| 2 | `hltSiPixelClustersSoA`, `hltOnlineBeamSpotDevice`, `hltSiPixelRecHitsSoA` |
| 3 | `hltPixelTracksSoA` |
| 4 | `hltPixelVerticesSoA` |
| 5 | `hltSiStripRawToClustersFacilityAlpaka` |
| 6 | `hltEcalDigisSoA`, `hltEcalUncalibRecHitSoA` |
| 7 | `hltHcalDigis` |
| 8 | `hltHcalDigisSoA`, `hltHbheRecoSoA` |
| 9 | `hltParticleFlowRecHitHBHESoA`, `hltParticleFlowClusterHBHESoA` |

All ten have distinct signatures, so no two of them could have been merged.

The groups come out ordered so that a group follows the ones it consumes from, and each
group's members come out in dependency order.

### What each group needs and produces

Two more questions are answered here:

* **`external_dependencies_by_group`** — for each module *outside* the offloaded set,
  which groups consume its products. These are exactly the products that have to be sent
  local → remote. A producer that is itself offloaded never appears: both ends run in the
  remote process, so nothing has to travel.
* **`modules_to_send_back_by_group`** — which offloaded modules have a consumer that is
  still running locally. Those products have to come back. Those that nothing local reads
  are simply dropped from the local process altogether.

In the example, the only external dependency is the **Source** — everything the offloaded
modules need comes from raw data.

## Stage 6 — building the two processes

`add_controller_to_local()` adds the `MPIController` (`mpiControllerRemote0`) and the MPI
services, and pins `numberOfConcurrentLuminosityBlocks` to 1, which is the current limit of
the controller.

`create_remote_process()` builds a fresh `cms.Process` containing: the global PSets and
EventSetup modules cloned from the local process (the remote needs the same conditions),
the MessageLogger, the MPI services, an `MPISource` in place of the real source, and clones
of the 15 offloaded modules. Nothing else.

## Stage 7 — forwarding inputs, local → remote

For each external dependency (here, just the Source), the splitter builds one round trip:

* an `MPISender` on local, configured with every product of that dependency;
* an `MPIReceiver` on remote, which **takes over the dependency's module label**, so that
  the offloaded modules, which consume `hltSomething`, find a module of exactly that name
  producing exactly those products. Downstream code cannot tell the difference.

The Source is the one case that needs special handling, because `source` is a reserved slot
that has to stay a `cms.Source`. So its receiver gets a name of its own and an `EDAlias` is
created pointing at it — including the `rawDataCollector` alias, which does not exist as a
config object in the original process at all (it is the provenance-level convention
mentioned in Stage 2) and so has to be recreated from scratch.

## Stage 8 — gating

Now the part that makes the result *correct* rather than merely functional.

Neither process can see the other's paths. The remote process has no idea whether the
event passed the filters that would have led to `hltHbheRecoSoA` running. So "did this run"
has to be transmitted, and that is done everywhere with the same three pieces:

* a **`PathStateCapture`** producer inserted into the local paths at exactly the position
  whose activity is being reproduced. It produces a `PathStateToken` — but only if the path
  actually reached it.
* an **`MPISender`/`MPIReceiver` pair** carrying that token. The token itself is not
  serialised; the sender signals "inactive" by setting the transferred product count to
  −1.
* a **`PathStateRelease`** filter (an "activity filter") on the receiving side, which stops
  its path unless the token arrived.

Two separate gates are set up:

**Per-dependency gates.** The capture for a dependency is placed before *every* member of
*every* group that needs it — not just before the members that consume it directly. The
product has to be sent whenever any of them is reached, and since all of a group's members
share one signature, those extra positions describe the very same condition anyway.

**Per-group activation gates.** The dependency gates only say that the inputs were
available somewhere; they say nothing about whether a particular group was reached, and
one local product typically feeds several groups. So every group also carries its own
activation round trip, captured at the position of each of its members. This is what
Stage 5's signature splitting bought: one capture per group is exact.

### Why the ordering is what stops deadlocks

This is the subtle bit, and it is worth spelling out because getting it wrong hangs both
processes on the first event.

A round trip that brings products *back* from the remote makes the local path **wait**. If
a group's activation gate were part of that same round trip, you could get: local is
waiting at a receiver for group N's products; remote cannot send group N's products
because it has not been told whether group M (further down the same local path, behind a
filter) runs; and local cannot reach the position that would tell it, because it is
blocked at the receiver. Both sides hang.

The rule that prevents this is: **every capture is inserted in front of every position
where a filter waiting for it ends up.** A local path therefore never arrives at a filter
whose captures it has not already produced, and every path can always make progress on its
own, whatever the other paths are doing.

Concretely, the per-group capture is inserted before every member of the group, *and* the
filter that waits for that group's products is inserted before every member too — so the
capture always precedes the filter at every single position. And a group with a different
signature has a gate of its own, captured at its own position, which is by construction
behind the point where the earlier group's products came back.

Three regression tests in `HeterogeneousCore/MPICore/test/` pin this down:
`testOffloadedModulesDoNotRunMoreThanNeeded.sh`, `testOffloadGatingDeduplication.sh` and
`testOffloadGatingDeadlock.sh`. Each splits a small configuration, runs both halves under
MPI, and checks that every module's execution count matches the unsplit baseline exactly.

## Stage 9 — sending results back, remote → local

For each group that has something to send back, another round trip is built: an `MPISender`
on remote and an `MPIReceiver` on local, carrying the products of every member of that
group that something local still reads.

Because one receiver carries several modules' products, the branches need the module label
in them to be told apart; the receiver registers each product under
`modulelabel@instancename` (or just `modulelabel` when the instance name is empty).

Then, on the local side:

* the offloaded module is **deleted** and replaced by an `EDAlias` pointing at the
  receiver, translating those `label@instance` branches back to the labels and instance
  names the rest of the menu expects. Everything downstream is untouched.
* the group's activity filter is inserted before **every** member of the group at its
  original position — not just before the last one — so that a member with a local
  consumer of its own is never left unguarded.

In portable mode, device products get one extra twist in the alias. A device product's
"friendly class name" spells out the Alpaka device (something like
`alpakaDevCudaRt128falseEcalDigiSoALayoutvoidPortableDeviceCollectionedmDeviceProduct`),
which would pin the configuration to one backend all over again. So a device product's
instance name is aliased with a single `*` wildcard entry, which takes whatever the
receiver actually registered there — and which also covers the host mirror that the
framework can rebuild from the device product on demand, for any legacy host-only consumer
further downstream.

Finally, offloaded modules that nothing local reads are simply deleted from the local
process.

## Stage 10 — paths and writing out

Each group gets up to four new paths:

* `Offload<Remote>Group<i>` on local — the controller and this group's senders;
* `Offload<Remote>Group<i>Receive` on local — this group's receiver, deliberately on a path
  of its own so that waiting for one group's products never orders anything against another
  group's sends;
* `MPIPathGroup<i>` on remote — this group's receivers and senders;
* `<Remote>RemoteOffloadedSequence<i>` on remote — the activity filters, the group's
  modules, and the capture that reports back.

A sanity check follows: MPI channel instance 0 is reserved for the controller/source pair
and the tag cannot exceed 255, so needing more than 255 channels is an error telling you to
offload fewer modules or spread them over several remotes. (The example uses 19.)

Both processes are then dumped back to Python text and written to the requested files.

---

## The files left behind

Everything the tool produces lives in `.edmMpiSplitConfig/` next to where you ran it:

| file | what it is |
| --- | --- |
| `process_dump_cfg.py` | your configuration plus the two dumping components, as actually run |
| `process_dump.log` | that job's `cmsRun` output — the first place to look when a dump is missing |
| `product_names.json` | every product every module registers: label, instance, C++ type, friendly name |
| `dependency_graph.json` | the resolved consumes() graph and the schedule |
| `serialiser_types_cfg.py`, `serialiser_types.log` | the portable-types job (portable mode only) |
| `serialiser_types.json`, `serialiser_types_serial_sync.json` | the two device serialiser registries |

Only the first four are affected by `--reuse-dumps`; the serialiser-type job is cheap
enough and independent of your configuration, so it always runs.

## Reading the result

A few things are worth recognising in the output configurations.

**`mpiController<Remote>`** drives the remote process. It is the first module on each
`Offload…` path.

**`activityCaptureBefore<Remote><Something>`** is a `PathStateCapture`. Seeing it appear
several times in a row inside one sequence is normal, not a bug: it was inserted before
several offloaded modules that sat in that sequence, and once those modules were replaced
by aliases the insertion points collapsed together. A module repeated in a path runs once.

**`activityFilterAfter<Remote>Group<i>`** is the `PathStateRelease` that stops a local path
unless group *i*'s results arrived.

**An `EDAlias` where a module used to be** is the substitution described in Stage 9. If you
want to know what is actually being transferred, read the `products` VPSet of the matching
`mpiReceiver…` instead.

**The remote configuration is small** — a few hundred kilobytes against the menu's five
megabytes — because it contains only what was offloaded and what is needed to feed it.
