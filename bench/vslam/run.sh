#!/bin/bash
# OpenVINS + RTAB-Map on one recorded session, in docker (see run_session.sh for MODE, SKIP_OV).
#   [MODE=stereo|rgbd] [SKIP_OV=1] bench/vslam/run.sh <session dir with rec/ and frames/> <work dir>
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SESSION=$(realpath "$1")
mkdir -p "$2"
WORK=$(realpath "$2")
docker run --rm --net=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e RATE="${RATE:-1.0}" -e MODE="${MODE:-stereo}" -e SKIP_OV="${SKIP_OV:-0}" \
    -e RTAB_EXTRA="${RTAB_EXTRA:-}" -e OV_EXTRA="${OV_EXTRA:-}" -v "$HERE":/scripts:ro -v "$SESSION/rec":/rec:ro -v "$SESSION/frames":/frames:ro \
    -v "$WORK":/work ov-rtabmap:jazzy bash /scripts/run_session.sh
