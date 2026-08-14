#! /bin/bash

# Regression test for the per-module activation gating optimization: modules with equal
# reachability signatures must share one activation capture/sender/receiver/filter chain
# rather than each getting their own.
#
# configuration_for_gating_collapse_cfg.py offloads a filter-free chain of three
# producers together with a fourth module that consumes the chain but sits behind a
# filter on another Path, all in one offload group. So the split must produce 2 gating
# chains, not 4, and -- as a guard against merging modules that are not actually
# equivalent -- every module must still run exactly as often as it did unsplit.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
WHOLE_CONFIG="$HERE/configuration_for_gating_collapse_cfg.py"
RESULT_DIR="./gating_collapse_result"
mkdir -p "$RESULT_DIR"

module_executed_count() {
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

echo "=== Step 1: baseline (unsplit) execution counts ==="
BASELINE_LOG="$RESULT_DIR/baseline.log"
cmsRun "$WHOLE_CONFIG" > "$BASELINE_LOG" 2>&1
declare -A BASELINE
for m in chainA chainB chainC gatedD; do
    BASELINE[$m]=$(module_executed_count "$BASELINE_LOG" "$m")
    echo "baseline: $m Executed = ${BASELINE[$m]} / 20"
done

echo
echo "=== Step 2: split the config, offloading the whole chain together ==="
LOCAL_PATH="$RESULT_DIR/local.py"
REMOTE_PATH="$RESULT_DIR/remote.py"
edmMpiSplitConfig "$WHOLE_CONFIG" \
    --remote-modules chainA chainB chainC gatedD \
    -l "$LOCAL_PATH" -r "$REMOTE_PATH"

echo
echo "=== Step 3: check the generated configs collapsed to 2 gating classes ==="
FAIL=0
# the capture and sender live in the local process, the receiver and filter in the remote one
for spec in "activityCaptureOwn:$LOCAL_PATH" "mpiSenderOwn:$LOCAL_PATH" \
            "mpiReceiverOwn:$REMOTE_PATH" "activityFilterOwn:$REMOTE_PATH"; do
    kind="${spec%%:*}"
    file="${spec#*:}"
    COUNT=$(grep -c "process\.${kind}[A-Za-z]* = cms\." "$file" || true)
    echo "$kind*: $COUNT object(s) (expected 2)"
    if [ "$COUNT" != "2" ]; then
        FAIL=1
    fi
done

echo 'process.MessageLogger.files.local_report = cms.untracked.PSet()' >> "$LOCAL_PATH"
echo 'process.MessageLogger.files.remote_report = cms.untracked.PSet()' >> "$REMOTE_PATH"

echo
echo "=== Step 4: run the split local/remote processes over MPI ==="
(cd "$RESULT_DIR" && "$HERE/testMPICommWorld.sh" local.py remote.py) > "$RESULT_DIR/mpi_run.log" 2>&1

echo
for m in chainA chainB chainC gatedD; do
    SPLIT=$(module_executed_count "$RESULT_DIR/remote_report.log" "$m")
    echo "post-split: $m Executed = $SPLIT / 20 (baseline ${BASELINE[$m]} / 20)"
    if [ "$SPLIT" != "${BASELINE[$m]}" ]; then
        echo "  FAIL: mismatch for $m"
        FAIL=1
    fi
done

echo
if [ "$FAIL" == "0" ]; then
    echo "PASS: gating collapsed to 2 classes and every module's execution count matches baseline."
    exit 0
else
    echo "FAIL: see mismatches above."
    exit 1
fi
