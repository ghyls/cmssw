#! /bin/bash

# Regression test for a bug in the config splitter's offload grouping: a whole group
# used to share one activation signal, captured only before the group's topological root
# module, so every other member silently inherited that root's gating instead of its own.
#
# configuration_for_grouping_bug_cfg.py sets up exactly that. moduleY only runs when
# gatingFilter passes (half the events), and separately consumes moduleX, which is
# unconditional. Offloading them together puts them in one group, so before the fix
# moduleY ended up gated by moduleX's unconditional Path and over-ran on the remote side.
# It is fixed by split_remote.py's per-module activation gating.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
WHOLE_CONFIG="$HERE/configuration_for_grouping_bug_cfg.py"
RESULT_DIR="./grouping_bug_result"
mkdir -p "$RESULT_DIR"

module_executed_count() {
    # Prints the "Executed" column of a module's line in the TrigReport Module Summary.
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

echo "=== Step 1: baseline (unsplit) execution counts ==="
BASELINE_LOG="$RESULT_DIR/baseline.log"
cmsRun "$WHOLE_CONFIG" > "$BASELINE_LOG" 2>&1
BASELINE_MODULEY=$(module_executed_count "$BASELINE_LOG" moduleY)
echo "baseline: moduleY Executed = $BASELINE_MODULEY / 20 (gated by gatingFilter)"

echo
echo "=== Step 2: split the config, offloading moduleX and moduleY together ==="
LOCAL_PATH="$RESULT_DIR/local.py"
REMOTE_PATH="$RESULT_DIR/remote.py"
edmMpiSplitConfig "$WHOLE_CONFIG" \
    --remote-modules moduleX moduleY \
    -l "$LOCAL_PATH" -r "$REMOTE_PATH"

# Both processes otherwise write to the same mpirun stdout and interleave mid-line.
echo 'process.MessageLogger.files.local_report = cms.untracked.PSet()' >> "$LOCAL_PATH"
echo 'process.MessageLogger.files.remote_report = cms.untracked.PSet()' >> "$REMOTE_PATH"

echo
echo "=== Step 3: run the split local/remote processes over MPI ==="
(cd "$RESULT_DIR" && "$HERE/testMPICommWorld.sh" local.py remote.py) > "$RESULT_DIR/mpi_run.log" 2>&1

SPLIT_MODULEY=$(module_executed_count "$RESULT_DIR/remote_report.log" moduleY)
echo "post-split: moduleY Executed (remote process) = $SPLIT_MODULEY / 20"

echo
if [ "$BASELINE_MODULEY" == "$SPLIT_MODULEY" ]; then
    echo "PASS: moduleY ran the same number of times before and after splitting."
    exit 0
else
    echo "FAIL: moduleY ran $SPLIT_MODULEY times after splitting vs $BASELINE_MODULEY times in the baseline."
    echo "It inherited moduleX's unconditional activation instead of its own gatingFilter."
    exit 1
fi
