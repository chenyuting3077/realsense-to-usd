"""VINS-Fusion's result CSV (vio.csv: every frame; vio_loop.csv: keyframes on loop_fusion's pose graph) ->
TUM file of left-IR poses. Rows: time in ns, x, y, z, qw, qx, qy, qz[, velocity] of the IMU (VINS' body).

    python3 vins_to_tum.py vio.csv rec/meta.json out.tum
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

if not hasattr(Rotation, "as_matrix"):  # SciPy < 1.4 (Ubuntu 20.04, ROS Noetic)
    Rotation.as_matrix, Rotation.from_matrix = Rotation.as_dcm, Rotation.from_dcm

csv, meta_path, out = sys.argv[1:4]
imu_from_ir1 = np.linalg.inv(np.array(json.load(open(meta_path))["imu"]["ir1_from_imu"]))
rows = [r for r in (line.strip().rstrip(",").split(",") for line in open(csv)) if len(r) >= 8]
a = np.array([[float(x) for x in r[:8]] for r in rows])
a = a[np.argsort(a[:, 0], kind="stable")]
a = a[np.r_[True, np.diff(a[:, 0]) > 0]]  # loop_fusion rewrites the file: keep one row per time
with open(out, "w") as f:
    for t, x, y, z, qw, qx, qy, qz in a:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = x, y, z
        T = T @ imu_from_ir1
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        f.write(f"{t / 1e9:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
print(f"{len(a)} poses -> {out}")
