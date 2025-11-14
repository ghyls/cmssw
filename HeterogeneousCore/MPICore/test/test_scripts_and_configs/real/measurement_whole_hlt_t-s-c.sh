#!/bin/bash
# set -ex

# Number of runs per test (first can be treated as warm-up)
runs=6

# Threads/Streams combinations to test
# thread_stream_combos=("1:1" "2:2" "4:4" "8:8" "16:16" "24:24" "32:32")
thread_stream_combos=("8:6" "16:12" "32:24")

# Script to run
script_name="hlt_test.py"

# Base directory for logs
BASE_DIR="../../test_results/whole_hlt_t-s-c-ngtworkshop"
mkdir -p "$BASE_DIR"

LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

for combo in "${thread_stream_combos[@]}"; do
    IFS=':' read -r threads streams <<< "$combo"

    end_core=$((32 + threads - 1))
    TEST_DIR="$BASE_DIR/test_t${threads}s${streams}"
    mkdir -p "$TEST_DIR"

    echo "=== Running tests with ${threads} threads, ${streams} streams, CPUs: 32-$end_core ==="

    for i in $(seq 1 $runs); do
        echo "Run #$i for t${threads}s${streams} on CPU list: 32-$end_core"
        LOG_FILE="$LOG_DIR/log_t${threads}s${streams}_run${i}.log"

        export RUN_ID=$i
        export EXPERIMENT_THREADS=$threads
        export EXPERIMENT_STREAMS=$streams
        export EXPERIMENT_NAME="local_t${threads}s${streams}_r${i}"
        export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"
        export THROUGHPUT_LOG_FILE="$BASE_DIR/throughputs.txt"

        # Run pinned to the CPU list
        numactl --physcpubind=32-$end_core cmsRun "$script_name" &> "$LOG_FILE"
    done

    echo "Completed tests for threads=$threads, streams=$streams"
done

echo "All local (non-offload) tests completed!"
