"""VINS-Fusion config for a recording (its meta.json): stereo + IMU, the factory-rectified IR pair (pinhole, no
distortion) and the factory IMU extrinsics, no online extrinsic / time-offset estimation (as for OpenVINS:
the D455 is calibrated and hardware-synced). IMU noise: VINS-Fusion's own RealSense D435i config (the same
BMI055 IMU).

    python3 make_config.py rec/meta.json out_dir
"""
import json
import os
import sys

import numpy as np


def mat(T):
    return ("!!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: ["
            + ", ".join(f"{x:.9f}" for x in T.ravel()) + "]")


meta_path, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
meta = json.load(open(meta_path))
st = meta["stereo"]
imu_from_ir1 = np.linalg.inv(np.array(meta["imu"]["ir1_from_imu"]))
imu_from_ir2 = imu_from_ir1 @ np.array(st["ir1_from_ir2"])
for name in ("cam0", "cam1"):
    open(os.path.join(out, f"{name}.yaml"), "w").write(f"""%YAML:1.0
---
model_type: PINHOLE
camera_name: {name}
image_width: {st['width']}
image_height: {st['height']}
distortion_parameters:
   k1: 0.0
   k2: 0.0
   p1: 0.0
   p2: 0.0
projection_parameters:
   fx: {st['fx']}
   fy: {st['fy']}
   cx: {st['cx']}
   cy: {st['cy']}
""")
open(os.path.join(out, "vins.yaml"), "w").write(f"""%YAML:1.0

imu: 1
num_of_cam: 2
imu_topic: "/imu0"
image0_topic: "/cam0/image_raw"
image1_topic: "/cam1/image_raw"
output_path: "/work/vins_out/"
cam0_calib: "cam0.yaml"
cam1_calib: "cam1.yaml"
image_width: {st['width']}
image_height: {st['height']}

estimate_extrinsic: 0
body_T_cam0: {mat(imu_from_ir1)}
body_T_cam1: {mat(imu_from_ir2)}

multiple_thread: 1
max_cnt: 150
min_dist: 30
freq: 10
F_threshold: 1.0
show_track: 0
flow_back: 1

max_solver_time: 0.04
max_num_iterations: 8
keyframe_parallax: 10.0

acc_n: 0.1
gyr_n: 0.01
acc_w: 0.001
gyr_w: 0.0001
g_norm: 9.805

estimate_td: 0
td: 0.0

load_previous_pose_graph: 0
pose_graph_save_path: "/work/vins_out/pose_graph/"
save_image: 0
""")
print("VINS-Fusion config ->", out)
