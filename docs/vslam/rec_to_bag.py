"""A session's rec/ -> ROS 2 bag for OpenVINS and RTAB-Map: the projector-off IR pair as /cam0 and /cam1
(mono8, with camera_info: already rectified, so RTAB-Map reads the baseline from P) and the IMU as /imu0.
With --odom (a TUM file of left-IR poses, e.g. OpenVINS'), also /odom + /tf odom->cam0 at every IR pair:
RTAB-Map's input for the second pass. With --frames (the session's frames/), also the color image and its
FFS depth (people zeroed) as /color/image_raw, /depth/image_raw, /color/camera_info (the transform
cam0 -> color is published by run_session.sh: a /tf_static in a bag reaches no one, its QoS differs).

    python3 rec_to_bag.py output/live_x/rec /bags/live_x [--odom ov.tum] [--frames output/live_x/frames]
"""
import argparse
import json
import os

import cv2
import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_msgs.msg import TFMessage


def stamp(ns):
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def camera_info(K, ns, frame, tx):
    m = CameraInfo()
    m.header.stamp, m.header.frame_id = stamp(ns), frame
    m.width, m.height, m.distortion_model = K["width"], K["height"], "plumb_bob"
    m.d = [0.0] * 5
    m.k = [K["fx"], 0.0, K["cx"], 0.0, K["fy"], K["cy"], 0.0, 0.0, 1.0]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [K["fx"], 0.0, K["cx"], -K["fx"] * tx, 0.0, K["fy"], K["cy"], 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


def image(img, ns, frame, encoding):
    m = Image()
    m.header.stamp, m.header.frame_id = stamp(ns), frame
    m.height, m.width = img.shape[:2]
    m.encoding, m.step, m.data = encoding, img.strides[0], img.tobytes()
    return m


def transform(ns, parent, child, T):
    tf = TransformStamped()
    tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp(ns), parent, child
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, T[:3, 3])
    r = tf.transform.rotation
    r.x, r.y, r.z, r.w = map(float, Rotation.from_matrix(T[:3, :3]).as_quat())
    return tf


class Tum:
    """Poses of a TUM file, interpolated; None outside it."""

    def __init__(self, path):
        a = np.loadtxt(path)
        self.t, self.xyz = (a[:, 0] * 1e9).astype(np.int64), a[:, 1:4]
        self.slerp = Slerp(self.t, Rotation.from_quat(a[:, 4:8]))

    def __call__(self, ns):
        if not self.t[0] <= ns <= self.t[-1]:
            return None
        j = int(np.clip(np.searchsorted(self.t, ns), 1, len(self.t) - 1))
        s = (ns - self.t[j - 1]) / max(self.t[j] - self.t[j - 1], 1)
        T = np.eye(4)
        T[:3, :3] = self.slerp([ns]).as_matrix()[0]
        T[:3, 3] = (1 - s) * self.xyz[j - 1] + s * self.xyz[j]
        return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec")
    p.add_argument("bag")
    p.add_argument("--odom", help="TUM file (s, xyz, qxyzw) of left-IR poses: adds /odom and /tf")
    p.add_argument("--frames", help="the session's frames/: adds color + FFS depth")
    a = p.parse_args()
    meta = json.load(open(os.path.join(a.rec, "meta.json")))
    st = meta["stereo"]
    baseline = float(np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3]))
    ir_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in st["frames"]]
    imu = np.loadtxt(os.path.join(a.rec, meta["imu"]["file"]), delimiter=",", skiprows=1)
    events = [(int(r[0]), 0, r) for r in imu] + [(ns, 1, k) for k, ns in enumerate(ir_ns)]

    topics = {"/cam0/image_raw": "sensor_msgs/msg/Image", "/cam1/image_raw": "sensor_msgs/msg/Image",
              "/cam0/camera_info": "sensor_msgs/msg/CameraInfo", "/cam1/camera_info": "sensor_msgs/msg/CameraInfo",
              "/imu0": "sensor_msgs/msg/Imu"}
    odom = Tum(a.odom) if a.odom else None
    if odom:
        topics.update({"/odom": "nav_msgs/msg/Odometry", "/tf": "tf2_msgs/msg/TFMessage"})
    if a.frames:
        fmeta = json.load(open(os.path.join(a.frames, "meta.json")))
        events += [(int(round(f["timestamp_ms"] * 1e6)), 2, k) for k, f in enumerate(fmeta["frames"])]
        topics.update({"/color/image_raw": "sensor_msgs/msg/Image", "/depth/image_raw": "sensor_msgs/msg/Image",
                       "/color/camera_info": "sensor_msgs/msg/CameraInfo"})
    events.sort(key=lambda e: (e[0], e[1]))

    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    for i, (name, typ) in enumerate(topics.items()):
        w.create_topic(rosbag2_py.TopicMetadata(id=i, name=name, type=typ, serialization_format="cdr"))

    n_odom = 0
    for ns, kind, x in events:
        if kind == 0:
            m = Imu()
            m.header.stamp, m.header.frame_id = stamp(ns), "imu0"
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, x[1:4])
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, x[4:7])
            m.orientation_covariance[0] = -1.0
            w.write("/imu0", serialize_message(m), ns)
        elif kind == 2:
            name = f"{x:06d}.png"
            color = cv2.imread(os.path.join(a.frames, "color", name))
            depth = cv2.imread(os.path.join(a.frames, "depth_ffs", name), cv2.IMREAD_UNCHANGED)
            mask = os.path.join(a.frames, "mask", name)
            if os.path.exists(mask):
                depth[cv2.imread(mask, cv2.IMREAD_GRAYSCALE) == 0] = 0
            w.write("/color/image_raw", serialize_message(image(color, ns, "color", "bgr8")), ns)
            w.write("/depth/image_raw", serialize_message(image(depth, ns, "color", "16UC1")), ns)
            w.write("/color/camera_info", serialize_message(camera_info(fmeta, ns, "color", 0.0)), ns)
        else:
            for cam, d, tx in (("cam0", "ir1", 0.0), ("cam1", "ir2", baseline)):
                img = cv2.imread(os.path.join(a.rec, d, f"{x:06d}.png"), cv2.IMREAD_GRAYSCALE)
                w.write(f"/{cam}/image_raw", serialize_message(image(img, ns, cam, "mono8")), ns)
                w.write(f"/{cam}/camera_info", serialize_message(camera_info(st, ns, cam, tx)), ns)
            T = odom(ns) if odom else None
            if T is None:
                continue
            tf = transform(ns, "odom", "cam0", T)
            o = Odometry()
            o.header, o.child_frame_id = tf.header, "cam0"
            o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, T[:3, 3])
            o.pose.pose.orientation = tf.transform.rotation
            cov = np.diag([1e-4] * 3 + [1e-5] * 3).ravel().tolist()  # RTAB-Map: the odometry links' information
            o.pose.covariance, o.twist.covariance = cov, cov
            w.write("/odom", serialize_message(o), ns)
            w.write("/tf", serialize_message(TFMessage(transforms=[tf])), ns)
            n_odom += 1
    del w
    print(f"{len(ir_ns)} IR pairs, {len(imu)} IMU samples, {n_odom} odometry poses"
          + (f", {len(fmeta['frames'])} color + depth" if a.frames else "") + f" -> {a.bag}")


if __name__ == "__main__":
    main()
