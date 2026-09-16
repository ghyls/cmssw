#! /bin/bash


# Uses offloaded_modules_do_not_run_more_than_needed_cfg.py. Splits it in local
# and remote parts, and runs them. Performs some checks on the split configs,
# and checks that modules offloaded to the remote process only ran when they had
# to.


set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
SINGLE_PROCESS_CONFIG="$HERE/offloaded_modules_do_not_run_more_than_needed_cfg.py"
RESULT_DIR="./offloaded_modules_do_not_run_more_than_needed_result"
mkdir -p "$RESULT_DIR"

# How many times a module was executed
module_executed_count() {
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

REMOTE_MODULES="producerB producerC producerD producerE"
LOCAL_MODULES="producerF"

# Two groups are expected to be formed in the remote config
EXPECTED_GROUPS=2

# Baseline: Single process case
BASELINE_LOG="$RESULT_DIR/baseline.log"
cmsRun "$SINGLE_PROCESS_CONFIG" > "$BASELINE_LOG" 2>&1
declare -A BASELINE
for module in $REMOTE_MODULES $LOCAL_MODULES; do
    BASELINE[$module]=$(module_executed_count "$BASELINE_LOG" "$module")
    echo "baseline: $module executed ${BASELINE[$module]} times"
done

# Split config
LOCAL_PATH="$RESULT_DIR/local.py"
REMOTE_PATH="$RESULT_DIR/remote.py"
edmMpiSplitConfig "$SINGLE_PROCESS_CONFIG" \
    --remote-modules producerB producerC producerD producerE \
    -l "$LOCAL_PATH" -r "$REMOTE_PATH"


FAIL=0

# Each offload group gets: a capture and a sender in the local process, and a
# receiver and a filter in the remote one. So every one of these modules has to
# have been generated exactly $EXPECTED_GROUPS times.
for name in "activityCaptureBeforeRemote0Group[0-9]*" "mpiSenderRemote0Group[0-9]*Activity"; do
    COUNT=$(grep -c "process\.${name} = cms\." "$LOCAL_PATH" || true)
    echo "$name: $COUNT object(s) (expected $EXPECTED_GROUPS)"
    if [ "$COUNT" != "$EXPECTED_GROUPS" ]; then
        FAIL=1
    fi
done

for name in "mpiReceiverRemote0Group[0-9]*Activity" "activityFilterBeforeRemote0Group[0-9]*"; do
    COUNT=$(grep -c "process\.${name} = cms\." "$REMOTE_PATH" || true)
    echo "$name: $COUNT object(s) (expected $EXPECTED_GROUPS)"
    if [ "$COUNT" != "$EXPECTED_GROUPS" ]; then
        FAIL=1
    fi
done

# To get the per-process summary with the amount of times each module ran
echo 'process.MessageLogger.files.local_report = cms.untracked.PSet()' >> "$LOCAL_PATH"
echo 'process.MessageLogger.files.remote_report = cms.untracked.PSet()' >> "$REMOTE_PATH"

# Run the split configs
if ! (cd "$RESULT_DIR" && "$HERE/testMPICommWorld.sh" local.py remote.py) > "$RESULT_DIR/mpi_run.log" 2>&1; then
    echo "FAIL: the split processes did not run to completion; see $RESULT_DIR/mpi_run.log"
    exit 1
fi

# compare every module's execution count against its baseline. A module missing
# from the logs counts as a mismatch, since it did not end up where the split
# was supposed to put it
compare_to_baseline() {
    local report=$1
    shift
    for module in "$@"; do
        SPLIT=$(module_executed_count "$report" "$module")
        echo "post-split: $module executed = $SPLIT times (baseline ${BASELINE[$module]})"
        if [ "$SPLIT" != "${BASELINE[$module]}" ]; then
            echo "  FAIL: mismatch for $module"
            FAIL=1
        fi
    done
}

compare_to_baseline "$RESULT_DIR/remote_report.log" $REMOTE_MODULES
compare_to_baseline "$RESULT_DIR/local_report.log" $LOCAL_MODULES

echo
if [ "$FAIL" == "0" ]; then
    rm -rf "$RESULT_DIR"
    exit 0
else
    exit 1
fi
