#!/usr/bin/env bash
# Live reconstruction from the D455 in one command: ORB-SLAM3 + FFS + nvblox with the window
# (map, camera, depth, status, Stop), and every frameset saved for offline steps.
#
#   ./run_live.sh                      # until Stop in the window or Ctrl+C -> output/live_<date>_<time>/
#   ./run_live.sh --duration 120       # other options go to recon/live.py (python recon/live.py -h)
#   ./run_live.sh -o output/live_room --no-record
#   ./run_live.sh --no-imu             # cameras only (recon/live.py's own default)
#
# Defaults here: --projector alternate (dots on every other frameset for depth, the others tracked
# at 30 Hz: ORB-SLAM3 does not keep up with 60 Hz on Thor anyway, 18-22 ms a frame) and --imu
# (ORB-SLAM3 with the D455's IMU, which carries blurred or blank seconds). With --imu hold the
# camera still for a second (leveling), then move it: tracking starts once the IMU sees some motion.
#
# Before starting: max clocks (jetson_clocks, set at boot on this machine), a D455 on USB 3, and
# the free disk space (rec/ takes ~40 MB/s, 2.4 GB a minute).
# Output (see recon/live.py): mesh_live.ply, poses_live.npy, live.json, frames/ (the fused
# frames with FFS depth), rec/ (every frameset: color, D455 depth, IR, poses.npy).
# The window shows the nvblox map (depth to 2.5 m) while recording; after Stop, mesh_live.ply is
# made from every stored frame with depth to 4 m, near observations weighted higher
# (--final-depth-max, --final-weight-ref).
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
source env.sh

free_gb=$(df -BG --output=avail output 2>/dev/null | tail -1 | tr -dc 0-9)
if [[ -n $free_gb && $free_gb -lt 20 ]]; then
  echo "warning: ${free_gb} GB free: rec/ fills ~2.4 GB a minute (--no-record to skip it)"
fi
export DISPLAY=${DISPLAY:-:1}  # the desktop's display when started over ssh; --no-view for none

python recon/live.py --projector alternate --imu "$@"  # options given later win
