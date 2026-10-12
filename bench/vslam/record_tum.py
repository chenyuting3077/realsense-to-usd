"""Record a PoseWithCovarianceStamped / Odometry topic as a TUM file of left-IR (cam0) poses.
OpenVINS publishes the IMU's pose; with --imu-from-cam (16 numbers, row-major) it is moved to the camera.

    python3 record_tum.py /ov_msckf/poseimu out.tum --imu-from-cam 1 0 0 -0.0302 ...
"""
import argparse

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("topic")
    p.add_argument("out")
    p.add_argument("--imu-from-cam", type=float, nargs=16)
    a = p.parse_args()
    I_C = np.array(a.imu_from_cam).reshape(4, 4) if a.imu_from_cam else np.eye(4)
    rclpy.init()
    node = Node("record_tum")
    f = open(a.out, "w")

    def cb(m):
        q, t = m.pose.pose.orientation, m.pose.pose.position
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        T = T @ I_C
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        f.write(f"{s:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {x:.9f} {y:.9f} {z:.9f} {w:.9f}\n")
        f.flush()

    node.create_subscription(PoseWithCovarianceStamped, a.topic, cb, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    f.close()


if __name__ == "__main__":
    main()
