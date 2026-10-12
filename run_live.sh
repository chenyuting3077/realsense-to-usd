#!/usr/bin/env bash
# Live reconstruction from the D455 (window: Stop, New recording, 3DGS -> USD).
#   ./run_live.sh [-o output/live_room] [--duration 120]    (options: python recon/live.py -h)
# Hold the camera still for a second at the start (IMU leveling), then move it.
set -eo pipefail
cd "$(dirname "$0")"
[[ ${CONDA_DEFAULT_ENV:-base} != base ]] || { echo "conda activate the env first (docs/install)" >&2; exit 1; }
export DISPLAY=${DISPLAY:-:1}  # the desktop's display when started over ssh
exec python recon/live.py "$@"
