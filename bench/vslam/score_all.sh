#!/bin/bash
# The README's tables from run_all.sh and run_all_cuvslam_vins.sh: stability (from the IMU), trajectories vs
# ORB-SLAM3 (the live poses.npy), the revisit check and its noise floor (ICP: NO_ICP=1 skips it), RTAB-Map's
# loop closures and per-update time. In the realsense-to-usd env.
#   bench/vslam/score_all.sh [recordings dir, default output] [work dir, default output/vslam]
cd "$(dirname "$0")/../.."  # the repo root: python -m bench.vslam.*
REC=${1:-output}
WORK=${2:-output/vslam}
for s in live_20261012_152105 live_20261012_160515 live_20261012_171428 live_20261012_182911 live_20261012_184220; do
    F=$REC/$s/frames
    W=$WORK/$s
    T="orb=$F/poses.npy"
    for x in ov=ov.tum ov_rtab_stereo=ov_rtab_stereo.tum ov_rtab_rgbd=ov_rtab_rgbd.tum \
             cuv_vio=cuvslam_vio/cuvslam_slam.tum cuv_stereo=cuvslam_stereo/cuvslam_slam.tum \
             vins=vins/vins.tum vins_loop=vins/vins_loop.tum; do
        [ -s "$W/${x#*=}" ] && T="$T ${x%%=*}=$W/${x#*=}"
    done
    echo "== $s"
    echo "-- stability"
    python -m bench.vslam.stability "$REC/$s/rec" $T --json "$W/stability.json"
    echo "-- vs ORB-SLAM3"
    python -m bench.vslam.compare_traj "$F" "$W/traj.png" $T
    if [ "$NO_ICP" != 1 ]; then
        echo "-- revisits (> 20 s apart)"
        python -m bench.vslam.eval_revisits "$F" $T --json "$W/revisits.json"
        echo "-- noise floor (0.1-0.5 s apart)"
        python -m bench.vslam.eval_revisits "$F" $T --min-dt 0.1 --max-dt 0.5 --json "$W/floor.json"
    fi
    python -m bench.vslam.log_times "$W"
done
