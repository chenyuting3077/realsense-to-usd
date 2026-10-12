#!/bin/bash
# One recording through OpenVINS (stereo + IMU VIO, real time) and then RTAB-Map (loop closure + graph
# optimization on OpenVINS' odometry). Runs inside the ov-rtabmap:jazzy image, see run.sh:
#   /rec    the session's rec/ (read only)       /frames  its frames/ (read only, MODE=rgbd)
#   /work   output: ov.tum (OpenVINS, left-IR poses), rtab_<mode>.tum (RTAB-Map's graph nodes, left-IR),
#           ov_rtab_<mode>.tum (every OpenVINS pose with the graph's correction), rtabmap_<mode>.db, logs
# MODE=stereo: RTAB-Map's loop closures from the projector-off IR pair; MODE=rgbd: from color + FFS depth.
# SKIP_OV=1 reuses ov.tum. OV_EXTRA / RTAB_EXTRA: more OpenVINS settings (key=value ...) / RTAB-Map arguments.
set -e
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77} ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
RATE=${RATE:-1.0}
MODE=${MODE:-stereo}
cd /work

stop() {  # SIGINT, then whatever is left
    kill -INT "$@" 2>/dev/null || true
    for _ in $(seq 20); do kill -0 "$@" 2>/dev/null || return 0; sleep 1; done
    kill -9 "$@" 2>/dev/null || true
}

# --- pass 1: OpenVINS in real time
if [ "$SKIP_OV" != 1 ]; then
    python3 /scripts/make_ov_config.py /rec/meta.json /ws/src/open_vins/config/rs_d455 ovcfg
    # factory-calibrated, hardware-synced D455: no online calibration (with it on, the first update on the
    # 250 s session turned both cameras' extrinsics by 8 deg and the time offset to 43 ms, and the filter
    # diverged to km); FAST 15 for the dark projector-off IR; a hand-held start is gentler than OpenVINS'
    # default 1.5 m/s^2 "jerk" for its static initialization; zero-velocity updates while the camera is held
    # still (without them the 42 s session sank 35 cm in its last 5 s, set down: 14.4 -> 3.6 cm RMSE vs ORB-SLAM3)
    sed -i -E 's/^(try_zupt): .*/\1: true/;
               s/^(calib_cam_intrinsics|calib_cam_extrinsics|calib_cam_timeoffset): .*/\1: false/;
               s/^(fast_threshold): .*/\1: 15/; s/^(num_pts): .*/\1: 250/;
               s/^(verbosity): .*/\1: "WARNING"/; s/^(init_imu_thresh): .*/\1: 0.25/' ovcfg/estimator_config.yaml
    for kv in $OV_EXTRA; do  # OV_EXTRA="try_zupt=true ...": more estimator_config.yaml settings
        sed -i -E "s/^(${kv%%=*}): .*/\1: ${kv#*=}/" ovcfg/estimator_config.yaml
    done
    [ -d bag_raw ] || python3 /scripts/rec_to_bag.py /rec bag_raw
    IMU_FROM_CAM=$(python3 -c "
import json, numpy as np
m = json.load(open('/rec/meta.json'))
print(' '.join(f'{x:.9f}' for x in np.linalg.inv(np.array(m['imu']['ir1_from_imu'])).ravel()))")
    rm -f ov.tum
    ros2 run ov_msckf run_subscribe_msckf --ros-args -r __ns:=/ov_msckf \
        -p config_path:=/work/ovcfg/estimator_config.yaml > ov.log 2>&1 &
    OV=$!
    python3 /scripts/record_tum.py /ov_msckf/poseimu ov.tum --imu-from-cam $IMU_FROM_CAM > rec_ov.log 2>&1 &
    REC=$!
    sleep 4
    ros2 bag play bag_raw --rate $RATE > play1.log 2>&1
    sleep 2
    stop $OV $REC
    pkill -9 -f run_subscribe_msckf || true
    echo "OpenVINS: $(wc -l < ov.tum) poses"
fi

# --- pass 2: RTAB-Map on OpenVINS' odometry
rm -rf bag_$MODE rtabmap_$MODE.db
if [ "$MODE" = rgbd ]; then
    python3 /scripts/rec_to_bag.py /rec bag_$MODE --odom ov.tum --frames /frames
    ros2 run tf2_ros static_transform_publisher $(python3 -c "
import json, numpy as np; from scipy.spatial.transform import Rotation
C = np.array(json.load(open('/rec/meta.json'))['stereo']['ir1_from_color'])
q = Rotation.from_matrix(C[:3, :3]).as_quat()
print(f'--x {C[0,3]} --y {C[1,3]} --z {C[2,3]} --qx {q[0]} --qy {q[1]} --qz {q[2]} --qw {q[3]}')") \
        --frame-id cam0 --child-frame-id color --ros-args -p use_sim_time:=true > tf_static.log 2>&1 &
    INPUT="-p subscribe_depth:=true -p subscribe_stereo:=false -p approx_sync:=true -p approx_sync_max_interval:=0.02
        -r rgb/image:=/color/image_raw -r depth/image:=/depth/image_raw -r rgb/camera_info:=/color/camera_info"
else
    python3 /scripts/rec_to_bag.py /rec bag_$MODE --odom ov.tum
    INPUT="-p subscribe_stereo:=true -p subscribe_depth:=false -p approx_sync:=false
        -r left/image_rect:=/cam0/image_raw -r right/image_rect:=/cam1/image_raw
        -r left/camera_info:=/cam0/camera_info -r right/camera_info:=/cam1/camera_info"
fi
ros2 run rtabmap_slam rtabmap --ros-args -p use_sim_time:=true $INPUT -r odom:=/odom \
    -p frame_id:=cam0 -p odom_frame_id:=odom -p database_path:=/work/rtabmap_$MODE.db -p publish_tf:=false \
    -p Rtabmap/DetectionRate:="'2'" -p Mem/IncrementalMemory:="'true'" -p RGBD/OptimizeMaxError:="'3.0'" \
    -p Vis/MinInliers:="'20'" -p Reg/Force3DoF:="'false'" -p RGBD/LinearUpdate:="'0.05'" \
    -p RGBD/AngularUpdate:="'0.05'" -p Stereo/MaxDisparity:="'200'" $RTAB_EXTRA > rtab_$MODE.log 2>&1 &
RT=$!
sleep 5
ros2 bag play bag_$MODE --clock --rate $RATE > play_$MODE.log 2>&1
sleep 5
stop $RT
pkill -f static_transform_publisher || true
pkill -INT -f "rtabmap_slam/rtabmap" || true
sleep 3
echo "RTAB-Map ($MODE): $(grep -c 'Rejected loop closure' rtab_$MODE.log || true) loop closures rejected"

# Optimized poses of every node (frame_id = cam0, the left IR) -> TUM with node ids
rtabmap-export --poses --poses_format 11 --output_dir /work --output rtab_$MODE rtabmap_$MODE.db > export_$MODE.log 2>&1
grep -v '^#' rtab_${MODE}_poses.txt | awk '{print $1, $2, $3, $4, $5, $6, $7, $8}' > rtab_$MODE.tum
echo "RTAB-Map ($MODE): $(wc -l < rtab_$MODE.tum) optimized poses"
python3 /scripts/correct_odom.py ov.tum rtab_$MODE.tum ov_rtab_$MODE.tum  # every OpenVINS pose, corrected
