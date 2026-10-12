"""A session's rec/ -> ROS 1 bag for VINS-Fusion: the projector-off IR pair as /cam0, /cam1 (mono8) and the IMU
as /imu0, in time order (the ROS 2 version for OpenVINS / RTAB-Map is ../rec_to_bag.py).

    python3 rec_to_bag1.py output/live_x/rec raw.bag
"""
import json
import os
import sys

import cv2
import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import Image, Imu

rec, out = sys.argv[1:3]
meta = json.load(open(os.path.join(rec, "meta.json")))
ir_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["stereo"]["frames"]]
imu = np.loadtxt(os.path.join(rec, meta["imu"]["file"]), delimiter=",", skiprows=1)
events = sorted([(int(r[0]), 0, i) for i, r in enumerate(imu)] + [(ns, 1, k) for k, ns in enumerate(ir_ns)])
with rosbag.Bag(out, "w") as bag:
    for ns, kind, k in events:
        t = rospy.Time(ns // 1_000_000_000, ns % 1_000_000_000)
        if kind == 0:
            m = Imu()
            m.header.stamp, m.header.frame_id = t, "imu0"
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, imu[k, 1:4])
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, imu[k, 4:7])
            bag.write("/imu0", m, t)
            continue
        for cam, d in (("cam0", "ir1"), ("cam1", "ir2")):
            img = cv2.imread(os.path.join(rec, d, f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE)
            m = Image()
            m.header.stamp, m.header.frame_id = t, cam
            m.height, m.width = img.shape
            m.encoding, m.step, m.data = "mono8", img.shape[1], img.tobytes()
            bag.write(f"/{cam}/image_raw", m, t)
print(f"{len(ir_ns)} IR pairs, {len(imu)} IMU samples -> {out}")
