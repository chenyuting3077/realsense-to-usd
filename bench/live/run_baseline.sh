#!/bin/bash
# n profiled replays of the same rec/ into <out>_1 .. <out>_n, then their median table.
# Usage: bench/live/run_baseline.sh <rec dir> <out prefix> [n=3]
set -e
cd "$(dirname "$0")/../.."
n=${3:-3}
for i in $(seq 1 "$n"); do
    bench/live/run_replay.sh "$1" "$2_$i" > "$2_$i.log" 2>&1
done
PATH=$HOME/miniforge3/envs/realsense-to-usd/bin:$PATH python -m bench.live.summarize $(seq -f "$2_%g" 1 "$n")
