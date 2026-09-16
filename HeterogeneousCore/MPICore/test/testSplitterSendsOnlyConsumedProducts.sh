#! /bin/bash


# Uses split_only_consumed_products_cfg.py. Splits it in local and remote parts,
# checks that each MPISender carries only the data products consumed on the
# other side, and runs both parts: the local IntTestAnalyzers check the values
# computed remotely from the data products that were sent over.


set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
SINGLE_PROCESS_CONFIG="$HERE/split_only_consumed_products_cfg.py"
RESULT_DIR="./split_only_consumed_products_result"
mkdir -p "$RESULT_DIR"

# How many times a module was executed
module_executed_count() {
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

REMOTE_MODULES="fromSource viaSourceAlias fromLocalOther viaLocalAlias thingsViaMixed remoteSum remoteViaAlias"
LOCAL_MODULES="checkRemoteSum checkAliasOfRemoteSum checkRemoteViaAlias"

# Baseline: Single process case
cmsRun "$SINGLE_PROCESS_CONFIG" > "$RESULT_DIR/baseline.log" 2>&1

# Split config
edmMpiSplitConfig "$SINGLE_PROCESS_CONFIG" --remote-modules $REMOTE_MODULES \
    -l "$RESULT_DIR/local.py" -r "$RESULT_DIR/remote.py"

# The (label, instance) of the data products carried by each MPISender of a
# configuration, excluding the activity-only ones
senders() {
    python3 - "$1" <<'EOF'
import runpy
import sys

process = runpy.run_path(sys.argv[1])["process"]
for name, module in sorted(process.producers_().items()):
    if module.type_() == "MPISender" and len(module.products):
        print(name, sorted((p.name.getModuleLabel(), p.name.getProductInstanceLabel()) for p in module.products))
EOF
}

FAIL=0
check_senders() {
    local actual
    actual=$(senders "$1")
    echo "$actual"
    if [ "$actual" != "$2" ]; then
        echo "  FAIL: expected"
        echo "$2"
        FAIL=1
    fi
}

# Only "other" of localSum, the Event ThingCollection of things (not the run and
# lumi ones), and the IntProduct of the Source, read directly and through EDAliases
check_senders "$RESULT_DIR/local.py" "\
mpiSenderRemote0Localsum [('localSum', 'other')]
mpiSenderRemote0Source [('source', '')]
mpiSenderRemote0Things [('things', '')]"

# Only what the local modules read: "other" of remoteSum, directly and through
# an EDAlias, and "" of remoteViaAlias
check_senders "$RESULT_DIR/remote.py" "\
mpiSenderRemote0Group0 [('remoteSum', 'other'), ('remoteViaAlias', '')]"

# thingsViaMixed reads an EDAlias of data products of both the Source and
# things, but only the one of things: nothing of the Source is sent
edmMpiSplitConfig "$SINGLE_PROCESS_CONFIG" --remote-modules thingsViaMixed \
    -l "$RESULT_DIR/local_mixed.py" -r "$RESULT_DIR/remote_mixed.py"
check_senders "$RESULT_DIR/local_mixed.py" "\
mpiSenderRemote0Things [('things', '')]"

# To get the per-process summary with the amount of times each module ran
echo 'process.MessageLogger.files.local_report = cms.untracked.PSet()' >> "$RESULT_DIR/local.py"
echo 'process.MessageLogger.files.remote_report = cms.untracked.PSet()' >> "$RESULT_DIR/remote.py"

# Run the split configs
if ! (cd "$RESULT_DIR" && "$HERE/testMPICommWorld.sh" local.py remote.py) > "$RESULT_DIR/mpi_run.log" 2>&1; then
    echo "FAIL: the split processes did not run to completion; see $RESULT_DIR/mpi_run.log"
    exit 1
fi

# every module has to run as often as in the single process
compare_to_baseline() {
    local report=$1
    shift
    for module in "$@"; do
        BASELINE=$(module_executed_count "$RESULT_DIR/baseline.log" "$module")
        SPLIT=$(module_executed_count "$report" "$module")
        echo "$module executed $SPLIT times (baseline $BASELINE)"
        if [ -z "$SPLIT" ] || [ "$SPLIT" != "$BASELINE" ]; then
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
