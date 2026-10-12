#!/bin/bash
# One recording through VINS-Fusion (stereo + IMU, real time) with loop_fusion. Runs inside the vins:noetic image,
# see run.sh: /rec the session's rec/ (read only), /work output, /scripts docs/vslam (read only).
# /work gets: vins.tum (VIO, every frame), vins_loop_nodes.tum (keyframes on the final pose graph),
# vins_loop.tum (every VIO pose with the graph's correction), logs.
set -e
source /opt/ros/noetic/setup.bash
source /ws/devel/setup.bash
RATE=${RATE:-1.0}
cd /work
python3 /scripts/vins/make_config.py /rec/meta.json cfg
[ -f raw.bag ] || python3 /scripts/vins/rec_to_bag1.py /rec raw.bag
rm -rf vins_out && mkdir -p vins_out/pose_graph
export ROS_MASTER_URI=http://localhost:${ROS_PORT:-11411} ROS_HOME=/work/.ros
roscore -p ${ROS_PORT:-11411} > roscore.log 2>&1 &
sleep 4
rosrun vins vins_node /work/cfg/vins.yaml > vins.log 2>&1 &
VINS=$!
rosrun loop_fusion loop_fusion_node /work/cfg/vins.yaml > loop.log 2>&1 &
LOOP=$!
sleep 4
rosbag play raw.bag --quiet -r $RATE > play.log 2>&1
sleep 5
kill -INT $VINS $LOOP 2>/dev/null || true
sleep 3
kill -9 $VINS $LOOP 2>/dev/null || true
pkill -INT roscore || true
python3 /scripts/vins/vins_to_tum.py vins_out/vio.csv /rec/meta.json vins.tum
python3 /scripts/vins/vins_to_tum.py vins_out/vio_loop.csv /rec/meta.json vins_loop_nodes.tum
python3 /scripts/correct_odom.py vins.tum vins_loop_nodes.tum vins_loop.tum
echo "VINS-Fusion: $(wc -l < vins.tum) VIO poses, $(wc -l < vins_loop_nodes.tum) keyframes," \
     "$(grep -ci 'loop' loop.log || true) loop lines in the log"
