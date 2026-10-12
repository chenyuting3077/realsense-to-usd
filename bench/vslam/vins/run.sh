#!/bin/bash
# VINS-Fusion on one recorded session, in docker (docker build -t vins:noetic bench/vslam/vins).
#   bench/vslam/vins/run.sh <session dir with rec/> <work dir>
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
SESSION=$(realpath "$1")
mkdir -p "$2"
WORK=$(realpath "$2")
docker run --rm --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e RATE="${RATE:-1.0}" \
    -v "$HERE":/scripts:ro -v "$SESSION/rec":/rec:ro -v "$WORK":/work vins:noetic bash /scripts/vins/run_session.sh
