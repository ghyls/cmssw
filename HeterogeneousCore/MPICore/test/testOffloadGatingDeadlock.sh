#! /bin/bash

# Takes configuration_for_filter_position_deadlock_cfg.py, which offloads moduleX
# and moduleY: moduleY consumes moduleX, but the two sit on opposite sides of
# evenNumberFilter, so they are reached under different conditions. Gating them
# together would put a capture behind a filter that waits for it, and hang both
# processes on the first event -- see the configuration for the step by step.
#
# This script splits the configuration, runs both the single-process and the
# local/remote versions, and counts the times each module runs in each of them.
# The two results should match, and the split run should reach the end at all.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
SINGLE_PROCESS_CONFIG="$HERE/configuration_for_filter_position_deadlock_cfg.py"
RESULT_DIR="./gating_deadlock_result"
mkdir -p "$RESULT_DIR"

module_executed_count() {
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

# how many events the job processed, as the framework summary reports it
events_total() {
    awk '/^TrigReport Events total =/ { print $5; exit }' "$1"
}

echo "Baseline (unsplit, single process) module execution counts"
BASELINE_LOG="$RESULT_DIR/baseline.log"
cmsRun "$SINGLE_PROCESS_CONFIG" > "$BASELINE_LOG" 2>&1
EVENTS=$(events_total "$BASELINE_LOG")
declare -A BASELINE
for m in moduleX moduleY localReadX localReadY; do
    BASELINE[$m]=$(module_executed_count "$BASELINE_LOG" "$m")
    echo "baseline: $m Executed = ${BASELINE[$m]} / $EVENTS"
done

# Split the config. Offload moduleX and moduleY.
LOCAL_PATH="$RESULT_DIR/local.py"
REMOTE_PATH="$RESULT_DIR/remote.py"
edmMpiSplitConfig "$SINGLE_PROCESS_CONFIG" \
    --remote-modules moduleX moduleY \
    -l "$LOCAL_PATH" -r "$REMOTE_PATH"

echo 'process.MessageLogger.files.local_report = cms.untracked.PSet()' >> "$LOCAL_PATH"
echo 'process.MessageLogger.files.remote_report = cms.untracked.PSet()' >> "$REMOTE_PATH"

# Run the local and remote configs
if ! (cd "$RESULT_DIR" && "$HERE/testMPICommWorld.sh" local.py remote.py) > "$RESULT_DIR/mpi_run.log" 2>&1; then
    echo "error: the split processes did not run to completion; see $RESULT_DIR/mpi_run.log"
    exit 1
fi

FAIL=0
# the offloaded modules now run in the remote process, everything else still in
# the local one
for spec in "moduleX:remote" "moduleY:remote" "localReadX:local" "localReadY:local"; do
    m="${spec%%:*}"
    SPLIT=$(module_executed_count "$RESULT_DIR/${spec#*:}_report.log" "$m")
    echo "post-split: $m Executed = $SPLIT / $EVENTS (baseline ${BASELINE[$m]} / $EVENTS)"
    if [ "$SPLIT" != "${BASELINE[$m]}" ]; then
        echo "  error: mismatch for $m"
        FAIL=1
    fi
done

if [ "$FAIL" == "0" ]; then
    exit 0
else
    exit 1
fi
