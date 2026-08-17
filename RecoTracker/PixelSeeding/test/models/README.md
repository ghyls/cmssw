# Retraining the pixel-track machine-learned models

Everything needed to retrain, check and deploy the models used by the two pixel+outer-tracker
track reconstruction iterations and by the selector on their merged output. Three entry points
do the work:

```bash
./retrain_prompt.sh    <step> [options]     # hltPhase2PixelTracksSoAWithStubs
./retrain_displaced.sh <step> [options]     # hltPhase2PixelTracksSoADisplacedWithStubs
./retrain_merged.sh    <step> [options]     # hltPhase2PixelTrackHighPuritySelectorMerged
```

The first two take the same steps and the same options; the third has only the final selector.
Run any of them with `--help` for the full list. Every step prints each command before running
it, and `--dry-run` prints them without running anything.

## The models

Each iteration has three trainable models, trained in this order, and the merged collection has
one more:

| script | step | model | where it runs | deployed as | to deploy |
|---|---|---|---|---|---|
| prompt, displaced | `triplet` | per-triplet gate | inside the track-building kernel | `plugins/alpaka/CATripletDNNWeights_<arm>.h` | rebuild |
| prompt, displaced | `gate` | track DNN (loose -> tight) | inside the track classification kernel | `plugins/alpaka/CATrackDNNWeights_<arm>.h` | rebuild |
| prompt, displaced | `hp` | final high-purity selector of the iteration | a separate module after reconstruction | `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/<arm>_tree31_<tag>_<date>.bin` | copy the file |
| merged | `hp` | final high-purity selector of the merged collection | after the merger | `.../merged_tree42_<tag>_<date>.bin` | copy the file |

`<arm>` is `prompt` or `disp`. The first two models are compiled into the kernels, so deploying
them means rebuilding; the selectors are read at run time, so deploying one means copying a file
and pointing the module's configuration at it.

What each step does:

* `dump` reconstructs events and writes a training dataset (one row per track, or per triplet,
  with the truth). `--for triplet`, `--for gate` and `--for hp` choose the dataset; each training
  step produces its own dataset on the fly when it is not in `--work` yet.
* `triplet` trains the per-triplet gate on the triplet dataset and bakes it into its header.
* `gate` trains the track DNN on the `loose` dataset and bakes it into its header.
* `hp` builds a feature cache from the `tight` dataset, trains the final selector, exports it in
  the compact binary the selector loads and stages it in `--work`. `--deploy` also copies it into
  the data directory; the configuration is edited by hand, as printed at the end of the step.

## The final selectors

All three final selectors are gradient-boosted forests run by `PixelTrackForestHighPuritySelector@alpaka`
(`useHitFeatures=True`), reading a compact `.bin`. Every model consumes a prefix of the
`PixelTrackFeaturesSoA` column layout, in column order:

| selector | module | configuration file | model file it loads | features | values |
|---|---|---|---|---|---|
| prompt | `hltPhase2PixelTrackTorchHighPuritySelector` (the forest replaces the Torch model under `phase2CAStubs`) | `hltPhase2PixelTrackTorchHighPuritySelector_cfi.py` | `prompt_tree31_wp_20260914.bin` | 31: 17 fit and covariance, 10 hit and stub, `rzChi2`, `meanStubKappa`, `leverArm`, `rMax` | fp32 |
| displaced | `hltPhase2PixelTrackHighPuritySelectorDisplaced` | `hltPhase2PixelTrackHighPuritySelectorDisplaced_cfi.py` | `disp_tree31_wp_20260914.bin` | the same 31 | fp16 |
| merged | `hltPhase2PixelTrackHighPuritySelectorMerged` | `hltPhase2PixelTrackHighPuritySelectorMerged_cfi.py` | `merged_tree42_wp_20260914.bin` | 42: the 31, then `nAttached`, `nOTExtras`, `iterationId`, `ndof`, then 7 pixel-cluster columns | fp32 |

The configuration files are in `HLTrigger/Configuration/python/HLT_75e33/modules/`. One trainer,
`train_merged_forest.py`, produces all three; each `hp` step calls it with the recipe of the file
the chain runs today, and stages the result under the trainer's own name
`<arm>_tree<N>_<tag>_<date>.bin` in `--work`. The three chains, exactly as the scripts print them
(`$W` = `--work`, `$M` = this directory):

**Prompt** (`./retrain_prompt.sh hp --work $W ...`):

```bash
python3 $M/nano_loader.py cache-arm $W/prompt31_cache.npz $W/trackNano_<sample>_tight.root ... --prefix TrkPrompt --label mtv
python3 $M/make_profile.py --cache $W/prompt31_cache.npz --model <the deployed .bin> \
    --threshold <its scoreThreshold> --arm prompt --split test --label mtv --out $W/prompt_reference_profile.json
python3 $M/train_merged_forest.py --cache $W/prompt31_cache.npz --out $W/prompt_forest \
    --feats 31 --label mtv --recall 0.995 --arm prompt --tag wp --date <date> --threads 16 \
    --eff-weight 4.0 --no-fp16 --ntrees 1000 --es-rounds 0 --prune-tol 0.002 \
    --wp-rule profile --wp-profile $W/prompt_reference_profile.json --wp-margin 0
# staged: $W/prompt_forest/prompt_tree31_wp_<date>.bin
```

Update in `hltPhase2PixelTrackTorchHighPuritySelector_cfi.py`, on the
`_hltPhase2PixelTrackForestHighPuritySelector` producer: `model` (the new file name),
`scoreThreshold` (after the in-situ scan below). `useHitFeatures` stays `True`;
`scoreThresholdLowDxy` stays `-1.0` (no displacement ramp on this arm).

**Displaced** (`./retrain_displaced.sh hp --work $W ...`):

```bash
python3 $M/nano_loader.py cache-arm $W/disp31_cache.npz $W/trackNano_<sample>_tight.root ... --prefix TrkDisp --min-dxy 0.5 --label legacy
python3 $M/make_profile.py --cache $W/disp31_cache.npz --model <the deployed .bin> \
    --threshold <its scoreThreshold> --arm disp --split test --label legacy --out $W/disp_reference_profile.json
python3 $M/train_merged_forest.py --cache $W/disp31_cache.npz --out $W/disp_forest \
    --feats 31 --label legacy --recall 0.995 --arm disp --tag wp --date <date> --threads 16 \
    --ntrees 1000 --es-rounds 0 --prune-tol 0.002 \
    --wp-rule profile --wp-profile $W/disp_reference_profile.json --wp-margin 0
# staged: $W/disp_forest/disp_tree31_wp_<date>.bin
```

Update in `hltPhase2PixelTrackHighPuritySelectorDisplaced_cfi.py`: `model`, and the three
threshold parameters `scoreThreshold`, `scoreThresholdLowDxy`, `dxyRampKnee` together (see
"Working points").

**Merged** (`./retrain_merged.sh dump ...`, then `./retrain_merged.sh hp --work $W ...`; the dump
runs with `NANO_MERGED=1 NANO_CLUSTER=1` and both iterations' buffers enlarged):

```bash
python3 $M/nano_loader.py cache-merged $W/merged43_cache.npz $W/trackNano_<sample>_merged.root ...
python3 $M/train_merged_forest.py --cache $W/merged43_cache.npz --out $W/merged_forest \
    --feats 35 --abi-order nAttached,nOTExtra,iteration,ndof \
    --extra-cols minCharge,meanCharge,minChargeNorm,maxSizeY,meanSizeY,maxSizeX,nLowCharge \
    --label mtv --recall 0.995 --arm merged --recall-axis A_nodup --dup-as-fake 1.0 \
    --fake-region-weight abseta:1.2:1.8:1.5,absdxy:1:inf:1.5 --no-fp16 --seed 0 \
    --tag wp --date <date> --threads 16 --ntrees 1000 --es-rounds 0 --prune-tol 0.002 \
    --wp-rule profile --wp-profile $W/merged_reference_profile.json --wp-margin 0
# staged: $W/merged_forest/merged_tree42_wp_<date>.bin
```

Update in `hltPhase2PixelTrackHighPuritySelectorMerged_cfi.py`: `model`, and `scoreThreshold`,
`scoreThresholdLowDxy`, `dxyRampKnee` together.

Each `hp` step also leaves `result.json` (every metric, the working point, the feature list and
the full recipe) and `thrmap.txt` (score threshold against true-track recall, with fake
rejection) next to the staged file, and reads the `.bin` back through `read_compact_tree.py`,
re-scoring test rows against the trained booster.

The trainer's inputs and labels, in short: a cache carries the features in the selector's column
order and two labels per track, `y_mtv` (matched to any TrackingParticle: a true track, the
training target) and `y_eff` (matched to a TrackingParticle passing the efficiency selection: the
recall axis). Recall is quoted on the second, fake rejection on the tracks matched to nothing;
the true tracks that fail an efficiency cut are positives in training and on neither axis.
The displaced chain trains against the narrow label instead (`--label legacy`), as the deployed
displaced file was; `LABEL=mtv` switches it to the label of the other two.

### Working-point rules

`train_merged_forest.py --wp-rule` chooses how the trained scores are turned into a threshold.
The rule shapes the model, not only its cut, because the size curve below is read at that
threshold. `WP_RULE=profile|uniform|global` sets it in the entry scripts.

* `profile` (**the default of all three chains**): match or exceed a reference model bin by bin.
  `make_profile.py` measures the model the configuration loads today, at the threshold that
  configuration carries, on this run's own cache; the working point is then the threshold at
  which the new forest reaches that recall in every bin. It is the default because that is the
  question a retrain has to answer: is the new model at least as good as the one it replaces,
  everywhere, on the sample of this run. It ties the result to that model, so a cold start with
  nothing deployed needs one of the other two.
* `uniform`: the largest threshold at which the true-track
  recall is at least `--recall` (0.995) in every pT, |eta| and |dxyBS| bin of the trainer's
  standard edges (pT 0.9/1.5/3/10/30, |eta| 1.5/2.5, |dxyBS| 0.1/0.5/1/2/5, each axis open at
  the top). Bins with fewer than 100 true tracks do not constrain it; with no usable bin the
  global rule applies. It depends on nothing outside the run, so the same command means the same
  thing on any training sample, and it protects the sparse tails (high pT, forward, displaced)
  that a single overall quantile lets the populous bins pay for.
* `global`: the threshold at which the overall true-track recall is `--recall`.

Whatever the rule, the threshold in `result.json` is a **starting point** for the selectors: their
deployed values are chosen by running the reconstruction and the track validation (see "Working
points" below), which every `hp` step says at the end.

### Forest size

The three `hp` steps fit with a large budget and no early stopping (`--ntrees 1000
--es-rounds 0`) and then let the data choose the size: `--prune-tol 0.002` keeps the smallest
prefix of the forest whose fake rejection at the working point, on the validation split, is
within 0.002 of the best prefix. The whole curve is in `result.json` as `size_curve`, next to
`n_trees_used`, so the choice can be read back and argued with. A fixed budget with early
stopping answers a different question -- where the validation loss stops improving -- and the
selectors are judged on rejection at one threshold, not on the loss. `NTREES`, `ES_ROUNDS` and
`PRUNE_TOL` override the three; `PRUNE_TOL=0` switches the pruning off and keeps every tree.

A pruned size costs nothing to re-derive: `--from-full <out>/model_full.json` starts from the
saved full booster, so a different tolerance is a re-export rather than a refit.

### What a retrain reproduces

The defaults are the recipe: `./retrain_all.sh <src>` reproduces the shipped models end to end.
Per collection, what that means:

* Prompt: 31 features, true high-pT tracks up-weighted, fp32 export, the per-bin profile of the
  model the configuration loads.
* Displaced: the same 31 features, fp16 export, trained against the narrow efficiency-selected
  label (`LABEL=legacy` is the default of this chain, `mtv` switches it to the label of the
  other two), rows cut at `|dxyBS| > 0.5`.
* Merged: 42 columns, duplicates counted as negatives and kept off the recall axis, fakes in the
  |eta| ~ 1.5 band and at large |dxyBS| up-weighted in the loss, fp32 export.

All three fit with the large budget, let the data choose the size, and take their starting
threshold from the profile of the model they replace.

## Order

**Within an iteration: `triplet` -> `gate` -> `hp`.** Every model decides on the population the
previous one left behind, so each dataset must be produced with the upstream models deployed
and compiled in:

* the track DNN trains on quality **`loose`** tracks with the deployed track DNN switched off
  at dump time, because that is the population it decides on -- before it decided;
* the final selector trains on quality **`tight`** tracks, which are the ones the deployed
  track DNN promoted, so that dump runs the full chain;
* regenerate each step's dataset only after the previous step is deployed **and built**.

**Across collections: prompt, then displaced, then merged.** The displaced iteration
reconstructs the hits the prompt iteration did not use, so its input population changes whenever
a prompt model changes; the merged collection is produced by both, so its selector comes last
and its dataset needs a base configuration that runs both iterations and the merger.

**After any change to the track fit** (fit refactor, magnetic field, material description,
refit in the merger): re-run the `gate` step and the `hp` steps. Their features are fit outputs,
and a model trained on an older fit is quietly mismatched; `compare_track_dnn_banks.py` scores
the baked track-DNN header directly and makes that visible.

## Samples

Every model -- both iterations' three each, and the merged selector -- trains on the same five
PU200 samples, with the **same number of events from each**:

| tag | sample | what it brings |
|---|---|---|
| `ttbarPU` | top pairs | the bulk: dense jets, the whole momentum range, b-hadron displacement |
| `displacedPU` | long-lived displaced decays | tracks from centimetres out, which nothing else populates |
| `qcdPU` | jets | combinatorics: the highest triplet load per event |
| `bsPU` | a soft dimuon resonance | low-momentum tracks, and a second displacement scale |
| `zpPU` | a heavy dimuon resonance | high-momentum tracks, where the sparse tails live |

Equal event counts are how equal weight per sample is implemented; the trainers do not re-weight
per sample afterwards. The list is the default of all four entry points, so a retrain with no
`--sample` trains on all five; `SAMPLES` (or `--sample "..."`) overrides it, and every tag then
needs its own events under `--input`. Training on fewer means the model follows whichever
population dominates, which is visible first in the sparse bins -- forward, high pT, large
|dxy| -- that the per-bin working-point rules are there to protect.

## Inputs

Datasets are produced by reconstructing events and writing out per-track or per-triplet tables.
Pass the events with `--input`, and an already-produced dataset with `--dataset`:

```bash
--input /path/to/dir            # a directory of .root files
--input a.root,b.root           # a comma-separated list
--input files.txt               # a text file with one .root path per line
--dataset /work/trackNano_ttbarPU_loose.root    # skip the dump, use this
```

With several `--sample` names, give each one its own events (otherwise all of them are dumped
from the same files and the sample diversity the trainers assume is not there):

```bash
--sample "ttbarPU displacedPU qcdPU" --input ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd
--sample "ttbarPU displacedPU qcdPU" --input /data/ttbar --input /data/susy --input /data/qcd
```

The named form wins; repeated `--input` options are paired with the `--sample` names in order.
`--events` applies to the dump it is on, so a different event count per stage means one `dump`
per stage, after which the training steps pick the datasets up from `--work`:

```bash
W=/somewhere/work; S="ttbarPU displacedPU qcdPU"
IN="ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd"
./retrain_prompt.sh dump --for triplet --work $W --sample "$S" --input "$IN" --events 100
./retrain_prompt.sh triplet --work $W --sample "$S"       # then rebuild
./retrain_prompt.sh dump --for gate    --work $W --sample "$S" --input "$IN" --events 300
./retrain_prompt.sh gate    --work $W --sample "$S"       # then rebuild
./retrain_prompt.sh dump --for hp      --work $W --sample "$S" --input "$IN" --events 1000
./retrain_prompt.sh hp      --work $W --sample "$S"
```

Every training step is given all the per-sample datasets at once and trains on their union;
the files are named `trackNano_<sample>_<quality>.root`, `tripletNano_<sample>.root` and
`trackNano_<sample>_merged.root` in `--work`. `--chassis <module>` alone is also enough to
produce a dataset, since a chassis carries its own input file list.

A step with neither option looks for the standard dataset names in `--work`; if they are not
there it stops and says so.

**The base reconstruction configuration.** The two dump configurations
(`trackNano_config.py`, `triplet_dump_cfg.py`) only add tables to an existing reconstruction
process; they do not build one. With `--input` given, the entry scripts generate that base
configuration once into the working directory with `make_base_config.sh`, which is an ordinary
`cmsDriver.py` call:

```
cmsDriver.py step2 -s L1P2GT,HLT:75e33_trackingOnly --processName HLTX \
  --conditions auto:phase2_realistic_T35 --geometry ExtendedRun4D121 --era Phase2C22I13M9 \
  --procModifiers ngtScouting --filein <files> --no_exec \
  '--output={}' \
  '--inputCommands=keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
  --python_filename <out>
```

`--output={}` keeps ConfigBuilder from adding a `PoolOutputModule` and its `output_step`
EndPath: the dump configs write their own NanoAOD file, and without it every dump job would
also write a full GEN-SIM-DIGI-RAW copy of its input. `--inputCommands` drops the products of
an earlier HLT process that RelVal inputs carry, so unqualified `InputTag`s cannot resolve
against them; `make_base_config.sh --input-commands ''` omits it. `ngtScouting` (which
includes `phase2CAStubs` and `pixelTrackMask`) leaves the pixel chain every model is trained
on unchanged and replaces the iterative tracking behind it with a pass-through, so a dump is
cheap; `--modifiers` sets a different set verbatim.

The tracking-only menu runs the whole chain the training data comes from -- both iterations, the
stub merger, all three selectors -- and skips everything else. Use `--chassis <module>` (or
`NANO_CHASSIS` / `DS_CHASSIS`) to dump on top of a different configuration instead; its
directory must be on `PYTHONPATH`.

**Dataset provenance.** Every track-level dump prints one `[trackNano CHAIN-STATE]` line per
iteration recording the state it ran in: whether each in-kernel model was on, at which
threshold, the buffer sizes and the quality. A dataset is only valid for the state on that
line. To dump at a working point that is not in the configuration yet, use the
`NANO_PROMPT_*` / `NANO_DISP_*` overrides documented in `trackNano_config.py`; they are printed
on the same line.

## The build switch for the triplet step

The per-triplet dump is compiled out by default. Its single switch is the last line of
`RecoTracker/PixelSeeding/plugins/alpaka/CATripletDumpMacro.h`; with it commented out the build
is unchanged, which is how production must stay.

```
# before the dump: uncomment '#define CA_TRIPLET_DUMP', then
scram b code-format && scram b -j
#   ... run the triplet dataset step ...
# after the dump: comment it out again, then
scram b code-format && scram b -j
```

The `triplet` step checks the switch and stops with these instructions if it is off. It does
not edit the header.

## Working points: what the tooling derives, and what is deployed

**Every working point is stated against the model it replaces.** A retrain moves the model and
moves the threshold with it, to the point at which nothing the old model kept is given up
anywhere -- so the rules below need no options, and a retrain with no options is the recipe. The
two in-kernel banks are deployed at their rule's point directly; the three selector thresholds
get one more step, the in-situ scan below, because they are collections the track validation can
see and the offline scorer cannot measure the combinatorial load, container occupancy or the fake
load handed to the next stage. `--threshold <x>` (or `THRESHOLD=<x>`, `BAKE_THR=<x>`) bakes an
explicit value at any of them. Each step prints the value the configuration currently carries and
where it lives.

| model | the rule the step applies | deployed | configuration parameter |
|---|---|---|---|
| triplet gate | the largest threshold at which every populated (&#124;tip&#124;, pT) group's per-track survival matches or beats the outgoing bank's, measured on the same held-out events | at this point, no scan | `tripletDNNThreshold` on the CA producer; when it is not set there, the value baked into the header applies |
| track DNN | the largest threshold at which the new bank keeps at least the outgoing bank's fraction of matched tracks in every pT, &#124;eta&#124; and &#124;dxy&#124; bin (`track_dnn_working_point.py`) | at this point, no scan | `trackDNNThreshold` on the CA producer; same fallback |
| final selector | the profile of the model the configuration loads, bin by bin (the `profile` rule above) -- a starting point | after the in-situ scan | `scoreThreshold` on the selector; the displaced and merged selectors also have `scoreThresholdLowDxy` and `dxyRampKnee` |

The triplet gate is the one to be careful with. A track leaves about ten real triplets behind,
so a rule stated as "keep 99 % of real triplets" keeps only about 0.99^10 ~ 90 % of the tracks,
which is why the default rule works on per-track survival. The global-recall rule is available
for comparison with `THRESHOLD_RULE=global`; selecting it also prints what the default rule
would have chosen.

The displaced and merged final selectors have **two** thresholds joined by a ramp:
`scoreThresholdLowDxy` applies at zero transverse impact parameter, `scoreThreshold` from
`dxyRampKnee` centimetres outwards, and a negative `scoreThresholdLowDxy` disables the ramp and
leaves one flat cut. The merged selector runs with the ramp, the displaced one flat. A retrained
model moves whichever values are live -- re-scan them together over the displaced region.

### What is deployed today

| model | file or header | threshold |
|---|---|---|
| prompt triplet gate | `CATripletDNNWeights_prompt.h` | 7.218852e-03 |
| displaced triplet gate | `CATripletDNNWeights_displaced.h` | 4.957580e-02 |
| prompt track DNN | `CATrackDNNWeights_prompt.h` | 3.414618e-02 |
| displaced track DNN | `CATrackDNNWeights_displaced.h` | 1.517135e-02 |
| prompt selector | `prompt_tree31_wp_20260914.bin` | `scoreThreshold` 0.232, ramp off |
| displaced selector | `disp_tree31_wp_20260914.bin` | `scoreThreshold` 0.150, ramp off |
| merged selector | `merged_tree42_wp_20260914.bin` | `scoreThreshold` 0.0358, `scoreThresholdLowDxy` 0.0755, `dxyRampKnee` 1.0 cm |

The four headers carry the point their own step derived. The three selector thresholds are the
scan's.

### The in-situ scan of the three selectors

`forest_threshold_scan.py` runs the scan inside one validation job: per factor it clones each
selector at a scaled threshold, converts the result to reco tracks and validates it as its own
collection, and for the prompt and displaced points also runs its own merger, merged selector and
converter, so the effect on the MERGED collection is measured too. The nominal chain is untouched
and is the 1.0 point of every curve. Add it to any validation configuration that runs the whole
pixel chain:

```python
from forest_threshold_scan import add_scan
add_scan(process)                                # factors 0.5, 0.7, 1.4, 2.0
add_scan(process, factors=(0.8, 1.0, 1.25))      # a finer scan around the nominal point
```

(the directory of this file must be on `PYTHONPATH`; a configuration living next to it already
has that.) The collections appear in the MTV output under these names, `fff` being the factor
times 100 -- so factor 1.4 is `140`:

| collection | what it answers |
|---|---|
| `scanPromptOnly<fff>` | the prompt selector alone at that threshold |
| `scanMergedP<fff>` | the merged collection with the prompt selector moved and the displaced one nominal |
| `scanDisplacedOnly<fff>` | the displaced selector alone |
| `scanMergedD<fff>` | the merged collection with the displaced selector moved |
| `scanMergedM<fff>` | the merged selector's own two thresholds scaled together |

Reading them: per collection, plot efficiency and fake rate against the factor and take the
tightest point whose efficiency still matches the configuration the run started from, in every
region that matters for that collection (for the displaced and merged ones, over `dxy` and
`vertpos`, not only the integrated number). The two merged curves matter more than the
single-iteration ones, since the merged collection is what the rest of the trigger reads: a
prompt threshold that looks free on `scanPromptOnly` can still cost merged efficiency by handing
the merger fewer tracks to prefer. Pin one threshold per selector, then re-run the validation
once at the pinned set to confirm the combination.

## Deploying

Each step ends by printing what it produced, where it goes and what to update. In short:

* **Baked headers** (`CATripletDNNWeights_*.h`, `CATrackDNNWeights_*.h`): written in place by
  the step, then `scram b code-format && scram b -j`. Never build while a `cmsRun` job is
  running -- the shared libraries are replaced in place and running jobs crash. `code-format`
  reformats a freshly baked header, so its checksum changes while its numbers do not; compare
  banks with `compare_track_dnn_banks.py`, not by checksum.
* **Selector models** (`<arm>_tree<N>_<tag>_<date>.bin`): staged in `--work` with today's date
  (`MODEL_NAME` overrides it, `TAG` the tag), then copied into
  `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/` and named in the
  module's configuration. `--deploy` does the copy; editing the configuration is always manual.
  No rebuild. Keeping the date in the name is what makes a model swap visible in the
  configuration diff.

The configurations to edit are in `HLTrigger/Configuration/python/HLT_75e33/modules/`:
`hltPhase2PixelTracksSoAWithStubs_cfi.py`, `hltPhase2PixelTracksSoADisplacedWithStubs_cfi.py`,
`hltPhase2PixelTrackTorchHighPuritySelector_cfi.py` (upstream label; the prompt forest is a
`phase2CAStubs` `toReplaceWith` on it), `hltPhase2PixelTrackHighPuritySelectorDisplaced_cfi.py`,
`hltPhase2PixelTrackHighPuritySelectorMerged_cfi.py`.

## The whole retrain, as one command

```bash
./retrain_all.sh /path/to/CMSSW/src            # all seven models, in order
```

It runs the 25 stages of a full retrain: prompt (triplet gate -> track DNN -> forest), then
displaced, then the merged selector, with a rebuild after every baked header, each new forest
wired into its configuration before the next stage's dataset is produced, and the per-triplet
dump switch turned on and off around the triplet stages. The dumps of a stage run side by side,
one job per sample, spread over the available GPUs. Every stage writes a marker under the work
directory, so a run that failed or was killed is restarted with the same command and resumes
where it stopped; naming stages after the area runs only those. `RT_DRYRUN=1` prints the plan
and the settings without running anything, and `--help` lists the stages and every environment
override with the value this run would use:

```bash
SAMPLES="ttbarPU displacedPU qcdPU bsPU zpPU"      # the default; each tag needs its own events
INPUTS="ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd,bsPU=/data/bs,zpPU=/data/zp"
RT_DRIVER_WORK=/scratch/$USER/retrain_$(date +%Y%m%d)   # datasets and caches, node-local
TRACK_EV=1000 TRIPLET_EV=300 THREADS=16 GPUS="0 1" DUMP_JOBS=4
```

It ends with a summary of the seven models: the baked thresholds, the staged and deployed
forests, the working point each one derived, and the tree the run leaves behind. The only thing
left after it is the in-situ scan of the three selector thresholds.

It expects the checkout clean, the per-triplet dump switch in its production state and no
`cmsRun` running on the node; the preflight stage says so and stops if not.

## A full pass, one step at a time

```bash
W=/somewhere/work
EVENTS=/path/to/reconstructible/events      # directory, list, or file of paths

# 1. triplet gate  (uncomment CA_TRIPLET_DUMP and rebuild first)
#    2000 events covers the default scoring range [1400,2000) the trainer windows per file;
#    with fewer, retrain_triplet_gate.sh scales the train/score windows to the dataset.
./retrain_prompt.sh triplet --work $W --input $EVENTS --events 2000
#    comment CA_TRIPLET_DUMP out again, then:
scram b code-format && scram b -j

# 2. track DNN  (its dataset is produced under the new gate)
./retrain_prompt.sh gate --work $W --input $EVENTS --events 800
scram b code-format && scram b -j

# 3. final selector  (its dataset is produced under the new track DNN)
./retrain_prompt.sh hp --work $W --input $EVENTS --events 1000
#    copy the staged prompt_tree31_wp_<date>.bin, edit the configuration (model; scoreThreshold
#    after an in-situ scan)

# 4. the same three steps with ./retrain_displaced.sh (its selector: disp_tree31_wp_<date>.bin,
#    with the three threshold parameters re-scanned together)

# 5. the merged selector, once both iterations are deployed and built
./retrain_merged.sh dump --work $W --input $EVENTS --events 1000
./retrain_merged.sh hp   --work $W
#    copy the staged merged_tree42_wp_<date>.bin, edit the configuration (model; the three
#    threshold parameters re-scanned together)
```

`--dry-run` prints every command a step would run without running it. To produce a dataset
without training, use `dump`:

```bash
./retrain_prompt.sh dump --for gate --work $W --input $EVENTS
```

Datasets, caches, models, plots and logs all go to `--work`; nothing is written into the source
tree except the two baked headers, which is where they belong.

All entry scripts refuse to start next to a running `cmsRun` job, and the heavy passes are
watched against the memory limit of the machine (on the anonymous memory of the control group,
not its total, most of which is reclaimable file cache).

## Comparing model families (prompt)

`./retrain_prompt.sh hp --class both` (or `--class forest`, `--class mlp`) runs a comparison
instead of the deploy path: four candidates trained on one 27-feature cache and one event split
(`prompt_hp_bakeoff.py`), ranked by fake rejection at one recall:

| candidate | family | features |
|---|---|---|
| `mlp17` | neural network, four hidden layers of 64 | 17 fit and covariance |
| `mlp27` | neural network, same shape | 27 -- the same inputs as the forest, so the table separates "a forest is better" from "more features are better" |
| `forest17` | gradient-boosted forest, depth 8 | 17 |
| `forest27` | gradient-boosted forest, depth 8 | 27: the 17 plus 10 hit and stub features |

None of the four is the 31-feature family the chain runs, so the winner is staged for in-situ
checks only, and deploying it is a change of model, to be validated as one.

## Files here

| file | role |
|---|---|
| `retrain_all.sh` | the whole retrain as one resumable command: all seven models, the rebuilds and the dataset production in between |
| `retrain_prompt.sh` | entry point for the prompt iteration (triplet gate, track DNN, final selector) |
| `retrain_displaced.sh` | entry point for the displaced iteration (the same three) |
| `retrain_merged.sh` | entry point for the merged collection (its dataset and final selector) |
| `retrain_common.sh` | shared option parsing, environment, dataset dumps, deployed-value lookups, the shared part of the selector training, reports |
| `retrain_triplet_gate.sh` | the `triplet` step for either iteration: train, score on a disjoint event range, bake the header |
| `retrain_track_dnn.sh` | the `gate` step for either iteration: train, compare against the bank in use, bake the header |
| `make_base_config.sh` | generate the base reconstruction configuration the dumps run on top of |
| `trackNano_config.py` | reconstruction job writing the track-level dataset (features + truth, both iterations, optionally the merged collection with its cluster columns) |
| `triplet_dump_cfg.py` | reconstruction job writing the per-triplet dataset (needs the dump build) |
| `train_triplet_dnn.py` | trainer for the per-triplet gate, including the scoring tables and the header export |
| `train_disp_nano.py` | trainer for the track DNN (either iteration), including the header export |
| `track_dnn_working_point.py` | the track DNN's working point: the largest threshold holding the outgoing bank's matched-track fraction in every pT, &#124;eta&#124; and &#124;dxy&#124; bin |
| `forest_threshold_scan.py` | the in-situ scan of the three selector thresholds, added to any validation configuration with `add_scan(process)` |
| `compare_track_dnn_banks.py` | score a baked track-DNN header by reproducing its forward pass, and compare it with another bank or with the model it was baked from |
| `nano_loader.py` | build feature caches from the track-level datasets: `cache-arm` (31 features, per iteration), `cache-merged` (31 + provenance + cluster columns); `cache`, `cache-ext`, `cache-prompt`, `cache-prompt27` serve the comparison and studies |
| `train_merged_forest.py` | the final-selector trainer for all three collections: recipe, working-point rules, compact export and read-back |
| `make_profile.py` | measure a deployed selector on a cache, bin by bin, for the `profile` working-point rule |
| `read_compact_tree.py` | read a compact `.bin` and score rows with it, the way the selector kernel does |
| `export_compact_tree.py` | write a booster as the compact binary the selector loads (used by the trainer) |
| `prompt_hp_bakeoff.py`, `build_tree_model.py`, `train_prompt_hp_nano.py`, `nano_train_utils.py` | the prompt model-family comparison: the four candidates, their forest and neural-network trainers, shared helpers |
| `CATrackFeaturesTableProducer.cc`, `TripletFeaturesTableProducer.cc`, `HitTruthTableProducer.cc`, `TrackerLayerId.h` | the table producers the two dump configurations use; built with the package, not run from here |

## Requirements

The CMSSW environment is enough for every step: the entry scripts source it themselves, and it
carries everything the trainers import. Train the selectors in that environment and no other:
the deployed forests were trained with its xgboost (1.7.5), and a different xgboost version
gives a different tree count and a different rejection at the same recall.
