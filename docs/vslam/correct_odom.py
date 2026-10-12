"""Dense corrected trajectory: RTAB-Map's optimized graph has a node every 0.5 s or so; every OpenVINS pose
between two nodes gets the correction (node's optimized pose x inverse of its odometry pose) of both,
interpolated (translation linearly, rotation by slerp).

    python3 correct_odom.py ov.tum rtab_nodes.tum out.tum
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if not hasattr(Rotation, "as_matrix"):  # SciPy < 1.4 (Ubuntu 20.04, ROS Noetic)
    Rotation.as_matrix, Rotation.from_matrix = Rotation.as_dcm, Rotation.from_dcm


def load(path):
    a = np.loadtxt(path, comments="#")
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()
    T[:, :3, 3] = a[:, 1:4]
    return a[:, 0], T


def main():
    t_ov, T_ov = load(sys.argv[1])
    t_n, T_n = load(sys.argv[2])
    # each node's odometry pose: the OpenVINS pose at its stamp (nodes are made from odometry messages)
    j = np.clip(np.searchsorted(t_ov, t_n), 0, len(t_ov) - 1)
    j = np.where((j > 0) & (np.abs(t_ov[j - 1] - t_n) < np.abs(t_ov[j] - t_n)), j - 1, j)
    D = T_n @ np.linalg.inv(T_ov[j])  # correction per node: world <- odometry world
    slerp = Slerp(t_n, Rotation.from_matrix(D[:, :3, :3]))
    with open(sys.argv[3], "w") as f:
        for t, T in zip(t_ov, T_ov):
            tc = np.clip(t, t_n[0], t_n[-1])
            k = int(np.clip(np.searchsorted(t_n, tc), 1, len(t_n) - 1))
            s = (tc - t_n[k - 1]) / max(t_n[k] - t_n[k - 1], 1e-9)
            C = np.eye(4)
            C[:3, :3] = slerp([tc]).as_matrix()[0]
            C[:3, 3] = (1 - s) * D[k - 1, :3, 3] + s * D[k, :3, 3]
            X = C @ T
            q = Rotation.from_matrix(X[:3, :3]).as_quat()
            f.write(f"{t:.9f} {X[0, 3]:.6f} {X[1, 3]:.6f} {X[2, 3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


if __name__ == "__main__":
    main()
