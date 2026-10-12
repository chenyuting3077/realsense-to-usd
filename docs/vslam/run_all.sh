#!/bin/bash
# Every session of the README's tables: OpenVINS, then RTAB-Map with stereo IR and with color + FFS depth.
#   docs/vslam/run_all.sh [recordings dir, default output] [work dir, default output/vslam]
HERE=$(cd "$(dirname "$0")" && pwd)
REC=${1:-output}
WORK=${2:-output/vslam}
for s in live_20261012_152105 live_20261012_160515 live_20261012_171428 live_20261012_182911 live_20261012_184220; do
    [ -f "$WORK/$s/ov_rtab_rgbd.tum" ] && continue  # done
    echo "== $s"
    MODE=stereo "$HERE/run.sh" "$REC/$s" "$WORK/$s" 2>&1 | grep -E "^OpenVINS|^RTAB-Map"
    MODE=rgbd SKIP_OV=1 "$HERE/run.sh" "$REC/$s" "$WORK/$s" 2>&1 | grep -E "^RTAB-Map"
done
