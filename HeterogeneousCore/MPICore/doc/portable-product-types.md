# Naming device products so a configuration works on any backend

`edmMpiSplitConfig --use-portable-mpi-modules` writes configurations that move Alpaka
**device** products over MPI, without a host copy. Each such product has to be named in
the configuration by its C++ type, and that is where a problem appears: the C++ type of a
device product spells out the Alpaka device it lives on, so the obvious thing to write
would pin the configuration to one backend.

This document explains the two pieces that solve that — `SerialiserTypeGetter` (in
`helper_jobs.py`) and `annotate_portable_types()` (in `editor_functions.py`) — and why the
result is that **a machine with any backend, including a CPU-only one, can split a
configuration that then runs on any other backend.**

---

## 1. The problem

### What the sender needs

`MPISenderPortable` and `MPIReceiverPortable` are compiled once per Alpaka backend. At
construction each one reads a list of products:

```python
cms.PSet(
    name = cms.InputTag("hltSiPixelClustersSoA"),
    type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection"),
)
```

and looks `type` up in **this backend's device serialiser registry**
(`ngt::SerialiserFactoryDevice`) to find the object that knows how to read the product's
memory regions. `ALPAKA_ACCELERATOR_NAMESPACE::` is a literal placeholder: the module
substitutes its own backend's namespace into it before the lookup. If the lookup fails,
there is no way to move the product.

So: **the configuration must name the product the way that registry is keyed** — and the
keys differ from backend to backend.

### What a product dump reports

`DumpProductNames` reports the type as the framework sees it. On a CUDA machine, the
pixel-cluster SoA product comes out as

```
edm::DeviceProduct<SiPixelClustersDevice<alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt> > >
```

Writing *that* into the configuration would work on exactly one kind of machine. A process
on an AMD GPU, or on no GPU at all, registers nothing under that name and has nothing left
to resolve it with.

Worse, in the opposite direction: on a **CPU-only** machine the same product is dumped as

```json
{"module": "hltSiPixelClustersSoA", "product_instance": "",
 "type": "SiPixelClustersHost", "friendly_type_name": "SiPixelClustersHost"}
```

There are **no device products in the dump at all** — the host backend's "device" *is* the
host, so an Alpaka module registers its collections under their plain host types, and
nothing in the dump distinguishes them from products that are host-only on every backend.

### What the registry is actually keyed by

`DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(TYPE_HOST, TYPE_DEVICE)` (in
`HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h`)
registers each serialiser under two keys, and *which* two depends on the backend:

| key | GPU backends (CUDA, ROCm) | CPU backends (`serial_sync`, TBB) |
| --- | --- | --- |
| 1 | mangled typeid of `TYPE_HOST` | mangled typeid of `TYPE_HOST` |
| 2 | mangled typeid of `ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE` | — |
| 3 | — | the literal string `"<namespace>::TYPE_DEVICE"` |

Key 2 is omitted on CPU backends because there `ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE`
resolves to `TYPE_HOST`, which key 1 already covers. Key 3 exists only on CPU backends, as
a workaround for CMSSW issue #51427.

Three consequences follow, and everything below is a reaction to them:

* **Key 1 is the same on every backend.** The mangled typeid of the *host* flavour is the
  one thing all backends agree on. It is the join key.
* **Key 3 is the only key that is a human-writable name** — and it is exactly
  `TYPE_DEVICE` as the macro received it, which is exactly what a configuration has to
  write under the placeholder. Only CPU backends have it.
* **Key 2 names no backend but its own**, so it is useless in a portable configuration.

This is why the CPU backend is not the awkward case — it is the *authoritative* one. It is
the only backend that can tell you the portable spelling of a type.

---

## 2. `SerialiserTypeGetter`: learning the portable names

`SerialiserTypeGetter` is a `HelperJob`: it builds a tiny process, runs `cmsRun` on it over
zero events, and reads back JSON. Unlike `ProcessDump`, it **does not read your
configuration at all** — the serialiser registry is a property of the release, not of the
menu — so it runs over a bare `EmptySource` process and stays cheap. (That is also why
there is no flag to reuse its output.)

### The job

It attaches the `DumpSerialiserTypes` analyzer **twice**:

```python
process.dumpSerialiserTypes = cms.EDAnalyzer("DumpSerialiserTypes@alpaka", ...)

process.dumpSerialiserTypesSerialSync = cms.EDAnalyzer("DumpSerialiserTypes@alpaka",
    alpaka = cms.untracked.PSet(backend = cms.untracked.string("serial_sync")), ...)
```

* the first runs on **whatever backend this machine would pick** — CUDA on an NVIDIA node,
  ROCm on an AMD one, `serial_sync` on a CPU-only one;
* the second is **forced to `serial_sync`**, which every machine has, because that is where
  the portable names (key 3) live.

Each instance walks its own backend's `SerialiserFactoryDevice` registry, instantiates each
registered serialiser (the only way to learn which type it moves — the registry maps names
to makers, not to types), and writes one entry per serialiser:

```json
{
  "product_type": "SiPixelClustersHost",
  "keys": ["19SiPixelClustersHost",
           "alpaka_serial_sync::SiPixelClustersSoACollection"],
  "alias": "SiPixelClustersSoACollection"
}
```

* **`product_type`** — the C++ type of the product this serialiser moves, *as this backend
  resolves it*. On CUDA the same entry would read
  `edm::DeviceProduct<SiPixelClustersDevice<alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt> > >`.
* **`keys`** — every name the serialiser is registered under, from the table above.
* **`alias`** — present only when one of the keys starts with this backend's namespace
  prefix (key 3), with the prefix stripped. So it is present on CPU backends and absent on
  GPU ones. **This is the portable name.**

The entries are keyed internally by `product_type`, which is what collects the several
registration names of one serialiser back into a single entry.

### `parse()`: joining the two dumps

`parse()` returns one dictionary, `{C++ product type → portable name}`, built in three
steps:

```python
serial_data = self.load("serialiser_types_serial_sync")

aliases = {key: entry["alias"]
           for entry in serial_data if "alias" in entry
           for key in entry["keys"]}

portable_types = {entry["product_type"]: "ALPAKA_ACCELERATOR_NAMESPACE::" + entry["alias"]
                  for entry in serial_data if "alias" in entry}

for entry in self.load("serialiser_types"):
    alias = next((aliases[key] for key in entry["keys"] if key in aliases), None)
    if alias is not None:
        portable_types[entry["product_type"]] = "ALPAKA_ACCELERATOR_NAMESPACE::" + alias
```

1. **`aliases`** inverts the `serial_sync` dump: *every* registration key of a serialiser
   maps to that serialiser's portable name. In the example above, both
   `"19SiPixelClustersHost"` and `"alpaka_serial_sync::SiPixelClustersSoACollection"` map
   to `"SiPixelClustersSoACollection"`.

2. **`portable_types` is seeded from the `serial_sync` dump's own product types.** These
   are the *host* spellings — `SiPixelClustersHost`,
   `PortableHostCollection<reco::PFRecHitSoALayout<128,false> >` — and every backend spells
   them identically. This is the entry that makes the host-mirror fallback in §3 work.

3. **The machine's own dump is joined onto it through key 1.** For each entry in the first
   dump, look for any of its keys in `aliases`; the mangled host typeid is present in both
   dumps, so it always matches. The machine-specific `product_type` — the CUDA device type,
   say — is then recorded under the portable name too.

On a CPU-only machine the two dumps are byte-identical, step 3 is a no-op, and the result
is just step 2. That is fine; it is the case the rest of the design is built to handle.

The result for the running example:

```python
{
  "SiPixelClustersHost": "ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection",
  # and, on a CUDA machine, additionally:
  "edm::DeviceProduct<SiPixelClustersDevice<alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt> > >":
      "ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection",
}
```

---

## 3. `annotate_portable_types()`: applying them to the product dump

`SerialiserTypeGetter` says what the release *can* serialise. `annotate_portable_types()`
matches that against what the configuration being split *actually produces*, and writes the
portable name onto each product description, under a new `"portable_type"` key:

```python
annotate_portable_types(cpp_names_of_the_products,   # the product dump, mutated in place
                        SerialiserTypeGetter().run(),
                        local_process)                # the configuration itself
```

All type names are compared with whitespace stripped, so that `A<B<C> >` and `A<B<C>>`
match.

It walks module by module, and there are three ways a product can acquire a portable name.

### Path A — the product is a device product, looked up directly

The dump reports `edm::DeviceProduct<...>` and its type is in `portable_types`. This is the
straightforward case, and it is what happens when you split on the same kind of machine the
products were dumped on.

### Path B — the product is a device product, looked up through its host mirror

The dump reports `edm::DeviceProduct<...>` but that exact type is *not* in
`portable_types` — because the machine running the splitter has a different backend than
the machine the dump was taken on. Nothing here can resolve a CUDA device type on an AMD
node.

But the framework's automatic device→host transform means the *same* module also registers
a **host mirror** of the product, under the same `(module, product_instance)` key, and the
host spelling is in `portable_types` (step 2 of `parse()`). So `_host_mirror_of()` looks for
it:

* it must not itself be a device product;
* it must share the `(module, product_instance)` key;
* its outermost class name must be the device type's with `Device` replaced by `Host` —
  `SiPixelClustersDevice` → `SiPixelClustersHost`, `PortableDeviceCollection` →
  `PortableHostCollection`. That name test is what tells the mirror apart from unrelated
  products that merely share the key, such as `hltSiPixelClustersSoA`'s
  `std::map<unsigned int, std::vector<SiPixelRawDataError> >`, which also has an empty
  instance name.
* if several candidates survive — a module emitting two SoAs under one instance name, so
  both are called `PortableHostCollection` — `_portable_payload()` breaks the tie by
  comparing what the collections *hold*: the top-level template arguments minus the Alpaka
  device and minus the trailing `void` placeholder the device flavour carries and the host
  one does not. Both
  `PortableDeviceCollection<alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt>,reco::PFClusterSoALayout<128,false>,void>`
  and `PortableHostCollection<reco::PFClusterSoALayout<128,false> >` reduce to
  `("reco::PFClusterSoALayout<128,false>",)`.

If that is still ambiguous, the function gives up and returns `None` — forwarding only the
device flavour is still correct, just without restoring the mirror.

**This path is what lets a menu dumped on an NVIDIA machine be split on an AMD one.**

### Path C — the dump has no device products at all

The dump was taken on a **host-only backend**. There is nothing wearing an
`edm::DeviceProduct<...>` type to look up, and the module's collections are indistinguishable
from ordinary host products by type alone.

What breaks the tie is the **configuration itself**, which is why `annotate_portable_types()`
takes `process` as a third argument. `_produces_device_products()` asks:

* is the module's class spelled `...@alpaka`? and
* does its `alpaka` PSet leave the backend unset, or set to something other than
  `serial_sync`?

If both hold, the module runs on whichever backend the job picks, so the collections it
registers are exactly the ones a GPU job would register as device products. (The test
deliberately excludes the `SerialSync` clones an HLT menu carries next to its Alpaka
modules — `hltHbheRecoSoASerialSync` and friends. Those are pinned to the host backend on
purpose and really do produce host products everywhere.)

For such a module, every product whose type appears in `portable_types` gets the portable
name. Concretely, on the CPU-only machine, `hltSiPixelClustersSoA` dumps five products and
three of them are recognised:

| dumped type | portable name written |
| --- | --- |
| `SiPixelClustersHost` | `ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection` |
| `SiPixelDigiErrorsHost` | `ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsSoACollection` |
| `SiPixelDigisHost` | `ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisSoACollection` |
| `std::map<unsigned int,std::vector<SiPixelRawDataError> >` | — (no serialiser, stays a ROOT product) |
| `unsigned short` (instance `backend`) | — (ditto) |

**This path is what lets a CPU-only machine produce a configuration that moves device
products on a GPU.**

### Products left alone

A device product with no recognised spelling simply keeps no `portable_type`. That is not
an error: most products in a menu are never forwarded. It only becomes a problem if such a
product *is* selected for forwarding, and `_portable_type_of()` then raises with the
product's name, its type, and the advice to register a
`DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN` for it or to keep its producer out of
`--remote-modules`.

---

## 4. What gets written into the configuration

`_forwarded_products()` decides, per product, what the sender and receiver PSets say.

`_is_portable_product()` treats a product as a device product if *either* the dump reported
it as one (paths A and B) *or* `annotate_portable_types()` gave it a `portable_type`
(path C). For those, `_portable_type_of()` supplies the name.

Two further details:

* **Host mirrors are dropped** (`_drop_redundant_host_mirrors()`). When a device product is
  carried directly, its host mirror must not also be carried — that would register the same
  branch twice. Only actual mirrors are dropped; an unrelated host product sharing a key is
  left alone.
* **The `EDAlias` on the receiving side uses a wildcard.** An `EDAlias` names a product by
  its *friendly* class name, and a device product's friendly class name spells out the
  Alpaka device just as badly as its real name does
  (`alpakaDevCudaRt128falseEcalDigiSoALayoutvoidPortableDeviceCollectionedmDeviceProduct`).
  So `create_receiver_alias()` writes a single `type = "*"` entry for a device product's
  instance name, taking whatever the receiver actually registered there. The wildcard also
  covers the host mirror that the framework can rebuild from the device product on demand,
  which a legacy host-only consumer further downstream still needs — and a wildcard is
  required anyway, since aliasing the same product twice inside one `EDAlias` is an error.

Instance names with no device product under them keep their explicit friendly class names,
which name no backend.

---

## 5. Putting it together: the backend matrix

At run time, `MPISenderPortable` resolves `ALPAKA_ACCELERATOR_NAMESPACE::X` by substituting
its own namespace and then trying, in order:

1. `edm::TypeWithDict::byName("<namespace>::X")` → if ROOT resolves it, look up the mangled
   typeid;
2. failing that, look up the literal string `"<namespace>::X"`.

Which of the registry keys that lands on depends on the backend:

| backend running the config | substituted string | resolves via |
| --- | --- | --- |
| CUDA | `alpaka_cuda_async::SiPixelClustersSoACollection` | dict → mangled typeid of the CUDA device type = **key 2** |
| ROCm | `alpaka_rocm_async::SiPixelClustersSoACollection` | dict → mangled typeid of the ROCm device type = **key 2** |
| `serial_sync` | `alpaka_serial_sync::SiPixelClustersSoACollection` | dict → mangled typeid of `SiPixelClustersHost` = **key 1**, or the literal string = **key 3** |

One string; three different keys; the same serialiser everywhere.

And on the writing side, whichever backend the splitter ran on:

| splitter ran on | how the name was found |
| --- | --- |
| CUDA | first dump has CUDA device types; joined to the `serial_sync` dump through key 1 (path A). Products from a ROCm-dumped menu fall back to their host mirrors (path B). |
| ROCm | symmetric. |
| CPU only | both dumps identical; the product dump has no device products, so the configuration itself identifies the Alpaka modules (path C). Products from a GPU-dumped menu resolve through their host mirrors (path B). |

So the two axes are independent: **any backend can write the configuration, and any backend
can run it.** The reason is that the whole scheme is anchored on the two things that are
backend-invariant — the host type of a portable collection, and the `TYPE_DEVICE` name the
plugin macro was given — and never on the resolved device type, which is the only part that
varies.

---

## 6. Where to look when it goes wrong

| symptom | where to look |
| --- | --- |
| `no backend-independent type name is known for the device product '<module>:<instance>'` | the type has no `DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN`, or its host mirror could not be identified. Check `.edmMpiSplitConfig/serialiser_types_serial_sync.json` for an entry whose `alias` is the type you expect. |
| an Alpaka module's products are forwarded as ROOT products instead of device products | on a CPU-only machine, check `_produces_device_products()`: is the module spelled `...@alpaka`, and is its `alpaka.backend` unset or non-`serial_sync`? A `SerialSync` clone is excluded on purpose. |
| the receiver registers nothing under an instance name | compare the `type` strings in the generated `mpiSender…`/`mpiReceiver…` PSets against the `keys` in the serialiser dumps for that backend. |
| a product is carried twice, or a branch is registered twice | `_drop_redundant_host_mirrors()` / `_host_mirror_of()` — most likely a module emitting several portable collections under one instance name where `_portable_payload()` could not disambiguate. |

The two dumps are left in `.edmMpiSplitConfig/serialiser_types.json` and
`.edmMpiSplitConfig/serialiser_types_serial_sync.json`, with the job's configuration and log
next to them; they are the first thing to read.

See `doc/edmMpiSplitConfig.md` for the surrounding pipeline.
