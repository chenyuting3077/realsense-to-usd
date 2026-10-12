#!/bin/bash
# cuVSLAM (stereo + IMU, and stereo only; both with SLAM) and VINS-Fusion (stereo + IMU, loop_fusion) on every
# session of the README's tables. cuVSLAM runs in the realsense-to-usd env; VINS-Fusion in the vins:noetic image.
#   docs/vslam/run_all_cuvslam_vins.sh [recordings dir, default output] [work dir, default output/vslam]
HERE=$(cd "$(dirname "$0")" && pwd)
REC=${1:-output}
WORK=${2:-output/vslam}
for s in live_20261012_152105 live_20261012_160515 live_20261012_171428 live_20261012_182911 live_20261012_184220; do
    echo "== $s"
    python "$HERE/run_cuvslam.py" "$REC/$s/rec" "$WORK/$s/cuvslam_vio" 2>&1 | grep '^{'
    python "$HERE/run_cuvslam.py" "$REC/$s/rec" "$WORK/$s/cuvslam_stereo" --no-imu 2>&1 | grep '^{'
    "$HERE/vins/run.sh" "$REC/$s" "$WORK/$s/vins" 2>&1 | grep -E "^VINS|poses ->|rror"
done
