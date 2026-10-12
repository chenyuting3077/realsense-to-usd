"""Export a frames dir + its poses as a COLMAP text-format dataset
for 3D Gaussian Splatting.

Layout:
  <out>/images/000000.png ...
  <out>/depths/000000.png ...          (uint16 millimeters, filtered like reconstruct.py)
  <out>/sparse/0/{cameras,images,points3D}.txt

Poses are known (from the SLAM trajectory), so no SfM is needed; the fused TSDF
point cloud initializes the Gaussians.

Usage:
  python gs/export_colmap.py output/room/frames output/room/recon -o output/room/colmap --every 1 \
      --depth-dir depth_ffs --depth-max 4
"""
import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recon"))
from frames import filter_depth, intrinsic_matrix, load_meta


def rotmat_to_qvec(R):
    """Rotation matrix -> COLMAP quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def depth_to_mm(depth_u16, meta, K, depth_max, edge_thresh, grazing_deg):
    """Filter sensor depth like reconstruct.py, convert to uint16 mm, drop beyond depth_max."""
    d = filter_depth(depth_u16, K, edge_thresh, grazing_deg)
    d_mm = d.astype(np.float64) * (1000.0 / meta["depth_scale"])
    d_mm[d_mm > depth_max * 1000] = 0
    return np.round(d_mm).astype(np.uint16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", help="frames dir")
    p.add_argument("recon_dir", help="reconstruct.py output dir (poses.npy, pointcloud.ply)")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--every", type=int, default=2, help="keep every N-th frame")
    p.add_argument("--max-points", type=int, default=300000)
    p.add_argument("--depth-max", type=float, default=2.5)
    p.add_argument("--init", choices=["tsdf", "depth"], default="tsdf",
                   help="initial points: the TSDF point cloud (recon_dir/pointcloud.ply), or back-projected "
                        "from the exported depth maps (denser; every 10th image, every 4th pixel, 5 mm voxels)")
    p.add_argument("--depth-dir", default="depth", help="depth subdirectory of frames, e.g. depth_ffs")
    p.add_argument("--edge-thresh", type=float, default=0.05)
    p.add_argument("--grazing-deg", type=float, default=0.0)
    args = p.parse_args()

    img_dir = os.path.join(args.out, "images")
    depth_dir = os.path.join(args.out, "depths")
    os.makedirs(depth_dir, exist_ok=True)
    sparse_dir = os.path.join(args.out, "sparse", "0")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    poses = np.load(os.path.join(args.recon_dir, "poses.npy"))
    meta = load_meta(args.frames)
    K = intrinsic_matrix(meta)
    assert len(poses) == len(meta["frames"]), "poses.npy does not match this frames dir"

    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        f.write(f"1 PINHOLE {meta["width"]} {meta["height"]} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")

    # Only the kept frames are read. Color PNGs are copied as is (they are already the
    # extracted images); depth is filtered on the GPU and PNG-encoded on a thread pool
    # (cv2 releases the GIL), so reading, filtering and writing overlap.
    keep = range(0, len(poses), args.every)
    read = lambda i: cv2.imread(os.path.join(args.frames, args.depth_dir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
    lines = []
    with ThreadPoolExecutor(8) as pool:
        depths = [pool.submit(read, i) for i in keep]
        writes = []
        for i, depth in zip(keep, depths):
            name = f"{i:06d}.png"
            writes.append(pool.submit(shutil.copyfile, os.path.join(args.frames, "color", name),
                                      os.path.join(img_dir, name)))
            d_mm = depth_to_mm(depth.result(), meta, K, args.depth_max, args.edge_thresh, args.grazing_deg)
            writes.append(pool.submit(cv2.imwrite, os.path.join(depth_dir, name), d_mm))
            # COLMAP stores world-to-camera.
            T_cw = np.linalg.inv(poses[i])
            q = rotmat_to_qvec(T_cw[:3, :3])
            t = T_cw[:3, 3]
            lines.append(f"{len(lines) + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {name}\n\n")
        for w in writes:
            w.result()
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] (empty)\n")
        f.writelines(lines)

    if args.init == "depth":
        pcd = o3d.geometry.PointCloud()
        v, u = np.mgrid[0:meta["height"]:4, 0:meta["width"]:4]
        for i in list(keep)[::10]:
            d = cv2.imread(os.path.join(depth_dir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)[v, u] / 1000.0
            c = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"{i:06d}.png")), cv2.COLOR_BGR2RGB)[v, u] / 255.0
            ok = d > 0
            X = np.stack([(u[ok] - K[0, 2]) * d[ok] / K[0, 0], (v[ok] - K[1, 2]) * d[ok] / K[1, 1], d[ok]], 1)
            part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X @ poses[i][:3, :3].T + poses[i][:3, 3]))
            part.colors = o3d.utility.Vector3dVector(c[ok])
            pcd += part
        pcd = pcd.voxel_down_sample(0.005)
    else:
        pcd = o3d.io.read_point_cloud(os.path.join(args.recon_dir, "pointcloud.ply"))
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    if len(pts) > args.max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    rgb = (np.clip(cols, 0, 1) * 255).round().astype(int)
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for j, (x, c) in enumerate(zip(pts, rgb)):
            f.write(f"{j + 1} {x[0]} {x[1]} {x[2]} {c[0]} {c[1]} {c[2]} 0\n")

    print(f"{len(lines)} images, {len(pts)} points -> {args.out}")


if __name__ == "__main__":
    main()
