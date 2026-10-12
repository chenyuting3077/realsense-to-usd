#!/bin/bash
# One profiled replay of a rec/: main.py --no-view with tegrastats every 100 ms beside it.
# Usage: bench/run_replay.sh <rec dir> <out dir> [nsys]   (nsys: under nsys profile, with NVTX ranges)
set -e
cd "$(dirname "$0")/.."
ENV=$HOME/miniforge3/envs/realsense-to-usd
export PATH=$ENV/bin:$PATH
rec=$1 out=$2
mkdir -p "$out"
tegrastats --interval 100 --logfile "$out/tegrastats.log" &
TS=$!
trap 'kill $TS 2>/dev/null' EXIT
if [ "$3" = view ]; then  # with the window (main.py finds the desktop); it waits for the window: ended once live.json is there
    rm -f "$out/live.json"
    setsid python main.py "$rec" -o "$out" &
    PID=$!
    until [ -f "$out/live.json" ]; do sleep 1; done
    sleep 3
    kill -- -"$PID" 2>/dev/null || true
elif [ "$3" = nsys ]; then
    PROFILE=1 /usr/local/cuda/bin/nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-memory-usage=true \
        -o "$out/nsys" --force-overwrite true python main.py "$rec" --no-view -o "$out"
else
    python main.py "$rec" --no-view -o "$out"
fi
python -m bench.summarize "$out"
