"""frames/ + poses.npy + the TSDF point cloud -> a COLMAP text dataset for 3DGS (known poses, no SfM):
<out>/images/, <out>/sparse/0/{cameras,images,points3D}.txt, and <out>/masks/ (person_mask.py's, no
loss where 0 in the patched FastGS) when frames/mask/ exists.

Usage: python gs/export_colmap.py <frames> <pointcloud.ply> <out>
"""
import os
import shutil
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recon"))
from frames import intrinsic_matrix, load_meta

MAX_POINTS = 300_000


def qvec(R):
    """Rotation matrix -> COLMAP quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def main():
    frames, cloud, out = sys.argv[1:4]
    sparse = os.path.join(out, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    shutil.copytree(os.path.join(frames, "color"), os.path.join(out, "images"), dirs_exist_ok=True)
    shutil.rmtree(os.path.join(out, "masks"), ignore_errors=True)
    if os.path.isdir(os.path.join(frames, "mask")):
        shutil.copytree(os.path.join(frames, "mask"), os.path.join(out, "masks"))
    meta, poses = load_meta(frames), np.load(os.path.join(frames, "poses.npy"))
    K = intrinsic_matrix(meta)
    with open(os.path.join(sparse, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {meta['width']} {meta['height']} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
    with open(os.path.join(sparse, "images.txt"), "w") as f:
        for i, T in enumerate(poses):
            T_cw = np.linalg.inv(T)  # COLMAP stores world-to-camera
            f.write(f"{i + 1} {' '.join(map(str, qvec(T_cw[:3, :3])))} {' '.join(map(str, T_cw[:3, 3]))} 1 "
                    f"{i:06d}.png\n\n")
    pcd = o3d.io.read_point_cloud(cloud)
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    if len(pts) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[idx], cols[idx]
    rgb = (np.clip(cols, 0, 1) * 255).round().astype(int)
    with open(os.path.join(sparse, "points3D.txt"), "w") as f:
        for j, (x, c) in enumerate(zip(pts, rgb)):
            f.write(f"{j + 1} {x[0]} {x[1]} {x[2]} {c[0]} {c[1]} {c[2]} 0\n")
    print(f"{len(poses)} images, {len(pts)} points -> {out}")


if __name__ == "__main__":
    main()
