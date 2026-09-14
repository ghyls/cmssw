#!/bin/bash
# =============================================================================
# retrain_all.sh -- retrain all seven machine-learned models of the pixel+stub
# chain, in one nohup-able sequence: the whole recipe as one command.
#
#   nohup ./retrain_all.sh /path/to/CMSSW/src > retrain.out 2>&1 &
#
# The order is prompt (triplet gate -> track DNN -> high-purity forest), then
# displaced (the same three), then the merged selector, with a rebuild after
# every baked header and each new forest wired into its configuration before the
# next stage's dataset is produced.  Every stage writes a marker; a restart skips
# the stages whose marker is there, so the run is resumable after a failure or a
# kill.  Each step derives its own working point (README.md, "Working points"),
# so nothing here pins a threshold; the three selector thresholds are the one
# thing still chosen afterwards, by the in-situ scan of forest_threshold_scan.py.
#
#   ./retrain_all.sh --help                 what it runs and what it reads
#   ./retrain_all.sh <src> <stage> ...      run only these stages
#
# See README.md for the models, the datasets and the environment overrides.
# =============================================================================
set -uo pipefail

# ---- parameters -------------------------------------------------------------
# All of them can be overridden from the environment.

usage() {
  sed -n '/^# retrain_all.sh --/,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'
  echo "ENVIRONMENT (the value in brackets is what this run would use)"
  echo "  RT_DRIVER_WORK  datasets and caches; keep it node-local  [$RT_DRIVER_WORK]"
  echo "  SAMPLES         sample tags; every model trains on all of them"
  echo "                  [$SAMPLES]"
  echo "  INPUTS          <tag>=<directory of .root files> per sample, comma separated"
  echo "                  [${INPUTS:-required}]"
  echo "  TRACK_EV        events per sample, track-level dumps        [$TRACK_EV]"
  echo "  TRIPLET_EV      events per sample, per-triplet dump         [$TRIPLET_EV]"
  echo "  THREADS         reconstruction threads per dump job         [$THREADS]"
  echo "  GPUS            GPUs the dumps are spread over              [$GPUS]"
  echo "  TRIPLET_GPU     GPU for the triplet trainer alone           [the first of GPUS]"
  echo "  DUMP_JOBS       concurrent dump jobs                        [$DUMP_JOBS]"
  echo "  BACKENDS        native | all, what the rebuilds build       [$BACKENDS]"
  echo "  DEPLOY          copy each staged forest into the data directory  [$DEPLOY]"
  echo "  SWAP_CFI        point each selector cfi at the new model    [$SWAP_CFI]"
  echo "  MEM_GUARD_GB    memory guard of the retrain scripts         [$MEM_GUARD_GB]"
  echo "  RT_DRYRUN       1 = print what would run and stop"
  echo
  echo "STAGES"
  echo "  ${STAGES[*]}"
}

RT_HELP=0
AREA=${1:-}                                  # the CMSSW src directory to retrain in
case "$AREA" in -h|--help|help) AREA=""; RT_HELP=1 ;; esac
[ -n "$AREA" ] || [ "$RT_HELP" = 1 ] ||
  { echo "usage: $0 <CMSSW src area> [stage ...]   (see $0 --help)"; exit 2; }
[ "$RT_HELP" = 1 ] || [ -d "$AREA/RecoTracker/PixelSeeding/test/models" ] ||
  { echo "!! $AREA does not look like a checkout with the retrain tooling"; exit 2; }
[ "$RT_HELP" = 1 ] || AREA=$(cd "$AREA" && pwd)
shift || true
ONLY_STAGES=("$@")                           # optional: run only these stages

: "${RT_DRIVER_WORK:=/scratch/$USER/retrain_$(date +%Y%m%d)}"   # datasets -- node-local
: "${TRACK_EV:=1000}"        # events per sample, track-level dumps (gate / hp / merged)
: "${TRIPLET_EV:=300}"       # events per sample, per-triplet dump (GPU-resident matrix: keep low)
: "${THREADS:=16}"           # reconstruction threads per dump job
: "${GPUS:=0 1}"             # GPUs the dumps are spread over
: "${TRIPLET_GPU:=}"         # GPU for the triplet trainer alone (default: the first of GPUS)
: "${DUMP_JOBS:=4}"          # concurrent dump jobs (4 over 2 GPUs = the plan's wall times)
: "${BACKENDS:=native}"      # native | all -- what the rebuilds after a bake build
: "${DEPLOY:=1}"             # copy each staged forest .bin into the tree's data directory
: "${SWAP_CFI:=1}"           # point each selector cfi at the new .bin (threshold untouched)
: "${MEM_GUARD_GB:=300}"     # the retrain scripts' cgroup anonymous-memory guard
# The five samples every model trains on, with the same event count each: top
# pairs, long-lived displaced decays, jets, a soft dimuon resonance and a heavy
# one.  Together they cover the momentum and displacement range the chain has to
# work over; trained on fewer, a model follows whichever population dominates.
# INPUTS gives every tag its own directory of reconstructible events.
: "${SAMPLES:=ttbarPU displacedPU qcdPU bsPU zpPU}"
: "${INPUTS:?set INPUTS=<tag>=<directory>,... for every tag in SAMPLES (see --help)}"
export MEM_GUARD_GB

W=$RT_DRIVER_WORK
MARK=${RT_DRIVER_MARKERS:-$W/markers}
LOGS=${RT_DRIVER_LOGS:-$W/logs}
# One working directory PER ARM.  The retrain scripts name their datasets after
# the sample and the quality only -- trackNano_<sample>_{loose,tight}.root and
# tripletNano_<sample>.root -- so the prompt and the displaced arm write the SAME
# file names, while the contents differ (which arm's track DNN is switched off,
# whose combinatorics buffers are enlarged, and which iteration the per-triplet
# dump comes from).  Sharing one --work therefore makes the second arm silently
# reuse the first arm's datasets and train against the wrong population.
W_PROMPT=$W/prompt
W_DISP=$W/displaced
W_MERGED=$W/merged
MODELS=$AREA/RecoTracker/PixelSeeding/test/models
MACRO=$AREA/RecoTracker/PixelSeeding/plugins/alpaka/CATripletDumpMacro.h
DATADIR=$AREA/RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector
CFIDIR=$AREA/HLTrigger/Configuration/python/HLT_75e33/modules
CFI_PROMPT=$CFIDIR/hltPhase2PixelTrackTorchHighPuritySelector_cfi.py
CFI_DISP=$CFIDIR/hltPhase2PixelTrackHighPuritySelectorDisplaced_cfi.py
CFI_MERGED=$CFIDIR/hltPhase2PixelTrackHighPuritySelectorMerged_cfi.py
HDR=$AREA/RecoTracker/PixelSeeding/plugins/alpaka

[ "$RT_HELP" = 1 ] || mkdir -p "$W" "$MARK" "$LOGS" "$W_PROMPT" "$W_DISP" "$W_MERGED"

# The run owns the node: it starts several dumps side by side on purpose, which
# would trip the retrain scripts' own node-wide `pgrep -x cmsRun` guard.  The
# node-exclusivity check is therefore done once, in the preflight stage, and the
# guard is then handed to the children as allowed.  RT_ALLOW_CONCURRENT_CMSRUN=1
# in this script's own environment waives the preflight check too.
ALLOW_CONCURRENT=${RT_ALLOW_CONCURRENT_CMSRUN:-}
export RT_ALLOW_CONCURRENT_CMSRUN=1

say() { echo "[$(date '+%F %T')] $*"; }

# ---- stage machinery --------------------------------------------------------
STAGES=(
  preflight
  prompt_triplet_build_on  prompt_triplet_dump  prompt_triplet_train  prompt_triplet_build_off
  prompt_gate_dump         prompt_gate_train    prompt_gate_build
  prompt_hp_dump           prompt_hp_train      prompt_hp_wire
  disp_triplet_build_on    disp_triplet_dump    disp_triplet_train    disp_triplet_build_off
  disp_gate_dump           disp_gate_train      disp_gate_build
  disp_hp_dump             disp_hp_train        disp_hp_wire
  merged_dump              merged_hp_train      merged_hp_wire
  summary
)

if [ "$RT_HELP" = 1 ]; then usage; exit 0; fi

wanted() {  # $1 = stage name
  [ ${#ONLY_STAGES[@]} -eq 0 ] && return 0
  local s; for s in "${ONLY_STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1
}

run_stage() {  # $1 = stage name
  local name=$1 rc t0 t1
  wanted "$name" || return 0
  if [ -f "$MARK/$name.done" ]; then
    say "STAGE $name SKIP (done: $(cat "$MARK/$name.done"))"
    return 0
  fi
  t0=$(date +%s)
  say "STAGE $name START"
  ( set -o pipefail; "stage_$name" ) > "$LOGS/$name.log" 2>&1
  rc=$?
  t1=$(date +%s)
  say "STAGE $name DONE rc=$rc  ($(( (t1 - t0) / 60 )) min $(( (t1 - t0) % 60 )) s, log $LOGS/$name.log)"
  if [ $rc -eq 0 ]; then
    echo "rc=0 $(date '+%F %T') $(( t1 - t0 ))s" > "$MARK/$name.done"
  else
    say "!! ABORTED at stage $name (rc=$rc). Read $LOGS/$name.log, fix, and start it again:"
    say "!! the stages already marked in $MARK are skipped, so it resumes here."
    tail -30 "$LOGS/$name.log" || true
    exit $rc
  fi
  return 0
}

# ---- helpers ----------------------------------------------------------------
cmsenv() {
  # shellcheck disable=SC1091
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  export SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc13}
  eval "$(cd "$AREA" && scramv1 runtime -sh)"
}

rebuild() {  # the gate after every baked header; BACKENDS=native keeps it short
  ( cd "$AREA" || exit 1
    if [ "$BACKENDS" = native ]; then scram b enable-gpus:native > /dev/null 2>&1 || true; fi
    scram b code-format || exit 1
    scram b -j || exit 1 )
}

macro_dump() {  # $1 = on | off
  if [ "$1" = on ]; then
    sed -i 's|^// *#define CA_TRIPLET_DUMP|#define CA_TRIPLET_DUMP|' "$MACRO"
    grep -Eq '^[[:space:]]*#define[[:space:]]+CA_TRIPLET_DUMP' "$MACRO" ||
      { echo "!! could not switch the per-triplet dump ON in $MACRO"; return 1; }
  else
    sed -i 's|^#define CA_TRIPLET_DUMP|// #define CA_TRIPLET_DUMP|' "$MACRO"
    grep -Eq '^[[:space:]]*#define[[:space:]]+CA_TRIPLET_DUMP' "$MACRO" &&
      { echo "!! could not switch the per-triplet dump OFF in $MACRO"; return 1; }
  fi
  echo ">>> $MACRO: per-triplet dump $1"
  return 0
}

# One dump job per sample, DUMP_JOBS at a time, spread over GPUS.  The retrain
# scripts dump their samples one after another inside a single invocation; one
# invocation per sample is what makes them run side by side, and the dataset
# file names (trackNano_<sample>_<quality>.root) never collide.  The base
# reconstruction configuration is generated once, under the tooling's own flock.
# The dataset file each dump stage produces, per sample.
dump_output() {  # $1 = work dir, $2 = --for value ("" = merged), $3 = sample
  case "$2" in
    triplet) echo "$1/tripletNano_$3.root" ;;
    gate)    echo "$1/trackNano_$3_loose.root" ;;
    hp)      echo "$1/trackNano_$3_tight.root" ;;
    "")      echo "$1/trackNano_$3_merged.root" ;;
    *)       echo "$1/trackNano_$3_$2.root" ;;
  esac
}

par_dump() {  # $1 = script, $2 = --for value ("" for the merged dump), $3 = events, $4 = tag, $5 = work dir
  local script=$1 forwhat=$2 nev=$3 tag=$4 wd=$5
  local -a gpus=($GPUS) pids=() names=()
  local i=0 s rc=0 n out gpu
  for s in $SAMPLES; do
    out=$(dump_output "$wd" "$forwhat" "$s")
    # Per-sample markers, not just per stage: a dump stage that was interrupted
    # or that failed on one sample must not be redone from scratch, and -- more
    # important -- must not REUSE what it left behind. The retrain scripts skip
    # a dataset file that already exists, so a truncated .root from a killed job
    # would be trained on silently. A sample without its marker therefore has
    # its dataset deleted before the job is relaunched.
    if [ -f "$MARK/${tag}_${s}.done" ] && [ -s "$out" ]; then
      echo ">>> [$tag/$s] SKIP (done: $(cat "$MARK/${tag}_${s}.done"))"
      continue
    fi
    if [ -e "$out" ]; then
      echo ">>> [$tag/$s] discarding $out ($(stat -c %s "$out") B): no completion marker, so it cannot be trusted"
      rm -f "$out"
    fi
    gpu=${gpus[$(( i % ${#gpus[@]} ))]}
    local -a cmd=("$MODELS/$script" dump --work "$wd" --sample "$s" --input "$INPUTS"
                  --events "$nev" --threads "$THREADS")
    [ -n "$forwhat" ] && cmd+=(--for "$forwhat")
    echo ">>> [$tag/$s] GPU $gpu : ${cmd[*]}"
    CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOGS/${tag}_${s}.log" 2>&1 &
    pids+=($!); names+=("$s")
    i=$((i + 1))
    # throttle: never more than DUMP_JOBS reconstruction jobs at once
    while [ "$(jobs -rp | wc -l)" -ge "$DUMP_JOBS" ]; do sleep 20; done
  done
  for ((n = 0; n < ${#pids[@]}; n++)); do
    wait "${pids[$n]}"; local r=$?
    out=$(dump_output "$wd" "$forwhat" "${names[$n]}")
    if [ $r -eq 0 ] && [ -s "$out" ]; then
      echo "rc=0 $(date '+%F %T') $(stat -c %s "$out") B" > "$MARK/${tag}_${names[$n]}.done"
    else
      [ $r -eq 0 ] && { echo "!! [$tag/${names[$n]}] exited 0 but left no dataset at $out"; r=1; }
      rm -f "$out"
    fi
    echo ">>> [$tag/${names[$n]}] rc=$r  (log $LOGS/${tag}_${names[$n]}.log)"
    [ $r -eq 0 ] || rc=$r
  done
  [ $rc -eq 0 ] || { echo "!! at least one $tag dump failed"; tail -25 "$LOGS/${tag}"_*.log; }
  return $rc
}

dataset_args() {  # $1 = work dir, $2 = suffix (loose|tight|merged|triplet)
  local wd=$1 q=$2 s
  for s in $SAMPLES; do
    if [ "$q" = triplet ]; then printf ' --dataset %s' "$wd/tripletNano_${s}.root"
    else printf ' --dataset %s' "$wd/trackNano_${s}_${q}.root"; fi
  done
}

newest_bin() {  # $1 = work dir, $2 = subdirectory, $3 = name stem
  ls -t "$1/$2/$3"*.bin 2>/dev/null | head -1
}

# Point a selector configuration at a new model file, leaving every threshold
# alone: the models move first and the selector working points move afterwards,
# one in-situ scan per selector.  This matters between
# stages, not only at the end: the displaced iteration reconstructs the hits the
# PROMPT high-purity selector left free, and the merged collection is built from
# both selectors' output, so a downstream dataset produced with the old model
# still loaded would train the next model against a chain that no longer exists.
wire_cfi() {  # $1 = cfi file, $2 = new .bin path, $3 = basename pattern of the old model
  local cfi=$1 newbin=$2 pat=$3
  [ "$SWAP_CFI" = 1 ] || { echo ">>> SWAP_CFI=0: $cfi left pointing at $(grep -o "$pat" "$cfi" | head -1)"; return 0; }
  [ -n "$newbin" ] && [ -f "$newbin" ] || { echo "!! no staged model to wire into $cfi"; return 1; }
  local new; new=$(basename "$newbin")
  local old; old=$(grep -oE "$pat" "$cfi" | head -1)
  [ -n "$old" ] || { echo "!! no model file name matching /$pat/ in $cfi"; return 1; }
  if [ "$old" = "$new" ]; then echo ">>> $cfi already loads $new"; return 0; fi
  sed -i "s|$old|$new|g" "$cfi" || return 1
  grep -q "$new" "$cfi" || { echo "!! the swap did not take in $cfi"; return 1; }
  echo ">>> $cfi: $old -> $new  (thresholds untouched)"
  grep -nE "scoreThreshold|dxyRampKnee|$new" "$cfi"
  return 0
}

# ---- stages -----------------------------------------------------------------
stage_preflight() {
  echo "area         : $AREA"
  echo "work         : $W"
  echo "samples      : $SAMPLES"
  echo "inputs       : $INPUTS"
  echo "events       : track-level $TRACK_EV / triplet $TRIPLET_EV per sample"
  echo "threads      : $THREADS   dump jobs: $DUMP_JOBS   GPUs: $GPUS   triplet GPU: ${TRIPLET_GPU:-$(set -- $GPUS; echo "$1")}"
  echo "backends     : $BACKENDS   deploy: $DEPLOY   wire cfi: $SWAP_CFI"
  cmsenv
  echo "CMSSW_BASE   : ${CMSSW_BASE:-unset}"
  [ "${CMSSW_BASE:-}/src" = "$AREA" ] || { echo "!! cmsenv gave CMSSW_BASE=${CMSSW_BASE:-unset}, not $AREA"; return 1; }

  # node exclusivity: the only check of it in the whole run (this script then
  # runs its own jobs side by side and tells the tooling to allow that)
  if [ -z "$ALLOW_CONCURRENT" ] && pgrep -x cmsRun > /dev/null; then
    echo "!! a cmsRun job is running on this node:"
    pgrep -ax cmsRun | head
    echo "!! the retrain wants the node to itself (both GPUs for the dumps, one GPU alone for the"
    echo "!! triplet trainer). Wait for it, or set RT_ALLOW_CONCURRENT_CMSRUN=1 for a dry run."
    return 1
  fi

  # the production state of the per-triplet dump switch
  if grep -Eq '^[[:space:]]*#define[[:space:]]+CA_TRIPLET_DUMP' "$MACRO"; then
    echo "!! $MACRO has the per-triplet dump switched ON; the tree must start from the production state"
    return 1
  fi

  # the tree must be clean: a model trained against a tree that then changes is
  # the failure the tooling's README warns about
  local dirty; dirty=$(cd "$AREA" && git status --short | grep -v '^??' || true)
  if [ -n "$dirty" ] && [ -z "${RT_ALLOW_DIRTY:-}" ]; then
    echo "!! the checkout has uncommitted changes; commit or stash them first (RT_ALLOW_DIRTY=1 to override):"
    echo "$dirty"
    return 1
  fi
  ( cd "$AREA" && git log -1 --format='tree at %h %s' )

  # the samples
  local s p ok=0
  for s in $SAMPLES; do
    p=$(echo "$INPUTS" | tr ',' '\n' | grep "^$s=" | cut -d= -f2-)
    [ -n "$p" ] || { echo "!! no --input entry for sample $s"; return 1; }
    local n; n=$(ls "$p"/*.root 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "!! no .root files under $p (sample $s)"; return 1; }
    echo "sample $s: $n files under $p"
    ok=$((ok + 1))
  done
  echo "$ok samples, the same event count each ($TRACK_EV) -- that is how equal share per gate is implemented"

  df -h "$W" | tail -1
  echo "free disk above should be >= ~15 GB of datasets and caches per sample and arm"
  return 0
}

stage_prompt_triplet_build_on()  { cmsenv && macro_dump on  && rebuild; }
stage_prompt_triplet_build_off() { cmsenv && macro_dump off && rebuild; }
stage_disp_triplet_build_on()    { cmsenv && macro_dump on  && rebuild; }
stage_disp_triplet_build_off()   { cmsenv && macro_dump off && rebuild; }
stage_prompt_gate_build()        { cmsenv && rebuild; }
stage_disp_gate_build()          { cmsenv && rebuild; }

stage_prompt_triplet_dump() { cmsenv && par_dump retrain_prompt.sh    triplet "$TRIPLET_EV" prompt_triplet_dump "$W_PROMPT"; }
stage_disp_triplet_dump()   { cmsenv && par_dump retrain_displaced.sh triplet "$TRIPLET_EV" disp_triplet_dump   "$W_DISP"; }
stage_prompt_gate_dump()    { cmsenv && par_dump retrain_prompt.sh    gate    "$TRACK_EV"   prompt_gate_dump    "$W_PROMPT"; }
stage_disp_gate_dump()      { cmsenv && par_dump retrain_displaced.sh gate    "$TRACK_EV"   disp_gate_dump      "$W_DISP"; }
stage_prompt_hp_dump()      { cmsenv && par_dump retrain_prompt.sh    hp      "$TRACK_EV"   prompt_hp_dump      "$W_PROMPT"; }
stage_disp_hp_dump()        { cmsenv && par_dump retrain_displaced.sh hp      "$TRACK_EV"   disp_hp_dump        "$W_DISP"; }
stage_merged_dump()         { cmsenv && par_dump retrain_merged.sh    ""      "$TRACK_EV"   merged_dump         "$W_MERGED"; }

# The per-triplet trainer holds the whole feature matrix GPU-resident (~30 MB
# per event per sample), so it gets a GPU to itself.
triplet_train() {  # $1 = script, $2 = work dir
  local gpu=${TRIPLET_GPU:-$(set -- $GPUS; echo "$1")}
  cmsenv || return 1
  echo ">>> triplet trainer on GPU $gpu (alone: the feature matrix is GPU-resident)"
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$gpu "$MODELS/$1" triplet --work "$2" --sample "$SAMPLES" --device cuda:0 \
    $(dataset_args "$2" triplet)
}
stage_prompt_triplet_train() { triplet_train retrain_prompt.sh    "$W_PROMPT"; }
stage_disp_triplet_train()   { triplet_train retrain_displaced.sh "$W_DISP"; }

# The track DNN: train, compare against the bank in use, derive the working point
# against that bank bin by bin, bake.  No --threshold here: the step's own rule
# is the recipe.
gate_train() {  # $1 = script, $2 = bank, $3 = work dir
  local gpu; gpu=$(set -- $GPUS; echo "$1")
  cmsenv || return 1
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$gpu "$MODELS/$1" gate --work "$3" --sample "$SAMPLES" --device cuda:0 \
    $(dataset_args "$3" loose) || return 1
  echo "=============== track DNN comparison, $2 ==============="
  cat "$3/track_dnn_comparison_$2.txt" 2>/dev/null || echo "(no comparison file)"
}
stage_prompt_gate_train() { gate_train retrain_prompt.sh    prompt    "$W_PROMPT"; }
stage_disp_gate_train()   { gate_train retrain_displaced.sh displaced "$W_DISP"; }

# The high-purity forests.  The hp steps fit with a large tree budget, let the
# data choose the size inside it, and derive a starting threshold by matching the
# model the configuration loads today, bin by bin.  The deployed thresholds come
# from the in-situ scan afterwards, so nothing here edits one.
hp_train() {  # $1 = script, $2 = arm, $3 = work dir
  cmsenv || return 1
  local -a extra=(); [ "$DEPLOY" = 1 ] && extra+=(--deploy)
  local kind=tight; [ "$2" = merged ] && kind=merged
  # shellcheck disable=SC2046
  "$MODELS/$1" hp --work "$3" --sample "$SAMPLES" "${extra[@]}" $(dataset_args "$3" "$kind")
}
stage_prompt_hp_train() { hp_train retrain_prompt.sh    prompt    "$W_PROMPT"; }
stage_disp_hp_train()   { hp_train retrain_displaced.sh displaced "$W_DISP"; }
stage_merged_hp_train() { hp_train retrain_merged.sh    merged    "$W_MERGED"; }

stage_prompt_hp_wire() {
  cmsenv || return 1
  wire_cfi "$CFI_PROMPT" "$(newest_bin "$W_PROMPT" prompt_forest prompt_tree31_)" 'prompt_tree31_[A-Za-z0-9_]*\.bin' || return 1
  rebuild
}
stage_disp_hp_wire() {
  cmsenv || return 1
  wire_cfi "$CFI_DISP" "$(newest_bin "$W_DISP" disp_forest disp_tree31_)" 'disp_tree31_[A-Za-z0-9_]*\.bin' || return 1
  rebuild
}
stage_merged_hp_wire() {
  cmsenv || return 1
  wire_cfi "$CFI_MERGED" "$(newest_bin "$W_MERGED" merged_forest merged_tree42_)" 'merged_tree42_[A-Za-z0-9_]*\.bin' || return 1
  rebuild
}

stage_summary() {
  local h b
  echo "==============================================================================="
  echo " THE SEVEN RETRAINED MODELS"
  echo "==============================================================================="
  echo
  echo "-- in-kernel banks (baked headers; NEITHER CA cfi sets a threshold, so the baked"
  echo "   kDefaultThreshold is the value the chain runs) -------------------------------"
  for h in CATripletDNNWeights_prompt.h CATripletDNNWeights_displaced.h \
           CATrackDNNWeights_prompt.h   CATrackDNNWeights_displaced.h; do
    printf '%-34s %s\n' "$h" "$(grep -oE 'kDefaultThreshold[^;]*' "$HDR/$h" | head -1)"
    printf '%-34s   %s\n' "" "$(cd "$AREA" && git log -1 --format='last commit %h %ad' --date=short -- "RecoTracker/PixelSeeding/plugins/alpaka/$h" 2>/dev/null)"
    printf '%-34s   %s\n' "" "$(cd "$AREA" && git status --short -- "RecoTracker/PixelSeeding/plugins/alpaka/$h")"
  done
  echo
  echo "-- high-purity forests (data files + the cfi that loads them) -------------------"
  for b in "$W_PROMPT:prompt_forest:prompt_tree31_:$CFI_PROMPT" \
           "$W_DISP:disp_forest:disp_tree31_:$CFI_DISP" \
           "$W_MERGED:merged_forest:merged_tree42_:$CFI_MERGED"; do
    # one assignment per statement: bash expands every word of a `local` before
    # it assigns any of them, so a later word cannot see an earlier one
    local wd r1 dir r2 stem cfi staged
    wd=${b%%:*}; r1=${b#*:}
    dir=${r1%%:*}; r2=${r1#*:}
    stem=${r2%%:*}; cfi=${r2#*:}
    staged=$(newest_bin "$wd" "$dir" "$stem")
    echo "staged   : ${staged:-none}"
    echo "deployed : $(ls -l "$DATADIR/$(basename "${staged:-none}")" 2>/dev/null || echo 'NOT in the data directory (DEPLOY=0?)')"
    echo "config   : $cfi"
    grep -nE "\.bin|scoreThreshold|dxyRampKnee" "$cfi" | sed 's/^/           /'
    echo "result   : $wd/$dir/result.json"
    [ -f "$wd/$dir/result.json" ] && python3 -c "
import json
d = json.load(open('$wd/$dir/result.json'))
def show(k, label=None):
    if k in d and not isinstance(d[k], (dict, list)):
        print('           %-22s %s' % (label or k, d[k]))
for k in ('wp_mode', 'working_point', 'n_trees_used', 'n_feats', 'label',
          'auc_AC', 'thr@0.9950', 'rej@0.9950', 'rows', 'train_s'):
    show(k)
wp = d.get('at_wp')
if isinstance(wp, dict):
    print('           at the working point:')
    for k in sorted(wp):
        if not isinstance(wp[k], (dict, list)):
            print('             %-20s %s' % (k, wp[k]))
print('           (offline starting points -- the deployed value comes from the in-situ scan)')
" 2>/dev/null
    echo
  done
  echo "-- working points -------------------------------------------------------------"
  echo "The in-kernel gates and track DNNs are deployed at the point their own step derived:"
  echo "each was held to the bank it replaced, group by group and bin by bin, so the headers"
  echo "above need nothing further. The three SELECTOR thresholds still do. Run a validation"
  echo "job with the in-situ scan and pin, per collection, the point that matches the"
  echo "efficiency of the configuration this run started from:"
  echo "    from forest_threshold_scan import add_scan; add_scan(process)"
  echo "The displaced and merged selectors have three threshold parameters each and must be"
  echo "scanned together. README.md, 'Working points', says how to read the curves."
  echo
  echo "-- the tree ------------------------------------------------------------------"
  ( cd "$AREA" && git status --short )
  echo
  echo "work directory : $W  (prompt / displaced / merged datasets in its three subdirectories)"
  echo "stage markers  : $MARK"
  echo "stage logs     : $LOGS"
  return 0
}

# ---- run --------------------------------------------------------------------
say "retrain start: area $AREA, work $W"
say "samples: $SAMPLES"
say "stages: ${STAGES[*]}"
if [ "${RT_DRYRUN:-0}" != 0 ]; then
  say "RT_DRYRUN: nothing is run."
  say "RT_DRYRUN: inputs $INPUTS"
  say "RT_DRYRUN: events per sample -- track-level $TRACK_EV, triplet $TRIPLET_EV"
  say "RT_DRYRUN: markers $MARK, logs $LOGS"
  exit 0
fi
T0=$(date +%s)
for s in "${STAGES[@]}"; do run_stage "$s"; done
T1=$(date +%s)
say "RETRAIN COMPLETE rc=0  total $(( (T1 - T0) / 3600 )) h $(( ((T1 - T0) % 3600) / 60 )) min"
