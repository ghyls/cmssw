#! /bin/bash

# Takes offloaded_modules_do_not_run_more_than_needed.py, which defines the
# following modules:
#
# - producerA, which is consumed by moduleX
# - moduleX, which is consumed by moduleY.
# - evenEventIDFilter, who sits between moduleX and moduleY, and passes only if
#   eventID is even.
#
# moduleY should run only when evenEventIDFilter passes. moduleX runs always.
#
# moduleY consumes moduleX's products, and thus the two should still be
# offloaded together. The two should be gated by moduleY's own condition, not
# moduleX's. Otherwise moduleY would run even when it does not need to.
#
# This scripts splits the configuration, offloading moduleX and moduleY to the
# remote process. Then runs both the single-process and the local/remote
# versions and counts the times each module runs in each configuration. The two
# results should match.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
SINGLE_PROCESS_CONFIG="$HERE/offloaded_modules_do_not_run_more_than_needed.py"
RESULT_DIR="./offloaded_modules_do_not_run_more_than_needed_result"
mkdir -p "$RESULT_DIR"

module_executed_count() {
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

# how many events the job processed, as the framework summary reports it
events_total() {
    awk '/^TrigReport Events total =/ { print $5; exit }' "$1"
}

BASELINE_LOG="$RESULT_DIR/baseline.log"
cmsRun "$SINGLE_PROCESS_CONFIG" > "$BASELINE_LOG" 2>&1
EVENTS=$(events_total "$BASELINE_LOG")
BASELINE_MODULEY=$(module_executed_count "$BASELINE_LOG" moduleY)
echo "baseline (single-process): moduleY Executed = $BASELINE_MODULEY / $EVENTS (gated by evenEventIDFilter)"

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

SPLIT_MODULEY=$(module_executed_count "$RESULT_DIR/remote_report.log" moduleY)
echo "Two-processes: moduleY ran remotely = $SPLIT_MODULEY / $EVENTS times"

echo
if [ "$BASELINE_MODULEY" == "$SPLIT_MODULEY" ]; then
    exit 0
else
    echo "FAIL: moduleY ran $SPLIT_MODULEY times after splitting vs
    $BASELINE_MODULEY times in the single-process configuration."
    exit 1
fi
