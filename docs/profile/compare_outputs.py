"""Two runs of main.py on the same rec/ -> how far apart their outputs are: the accuracy gate of
docs/profile (a speed-up keeps the outputs within the spread of the baseline's own re-runs).

- trajectory: ATE (cm) of the color-camera poses of the frames in both (frames/meta.json times),
  after a rigid alignment (Umeyama, no scale); tracked frames of each
- mesh: Chamfer distance (mm, mean of both directions, and p90) of mesh_live.ply, sampled, B moved
  into A's world by the trajectory alignment (the world itself may shift between runs); triangles
- depth: per pixel |difference| (mm) of frames/depth_ffs at the same timestamps, where both have depth

Usage: python -m docs.profile.compare_outputs <run A> <run B> [<run C> ...]   (every pair against A)
"""
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

N_SAMPLES = 300_000


def load(run):
    frames = os.path.join(run, "frames")
    with open(os.path.join(frames, "meta.json")) as f:
        meta = json.load(f)
    t = np.array([round(fr["timestamp_ms"] * 1000) for fr in meta["frames"]], np.int64)  # us
    with open(os.path.join(run, "live.json")) as f:
        live = json.load(f)
    return {"frames": frames, "t": t, "poses": np.load(os.path.join(frames, "poses.npy")), "live": live,
            "mesh": o3d.io.read_triangle_mesh(os.path.join(run, "mesh_live.ply"))}


def umeyama(A, B):
    """R, t minimizing |R B + t - A|."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    return R, ma - R @ mb


def chamfer(m1, m2):
    o3d.utility.random.seed(0)
    p1 = np.asarray(m1.sample_points_uniformly(N_SAMPLES).points)
    p2 = np.asarray(m2.sample_points_uniformly(N_SAMPLES).points)
    c1, c2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p1)), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p2))
    d = np.concatenate([np.asarray(c1.compute_point_cloud_distance(c2)), np.asarray(c2.compute_point_cloud_distance(c1))])
    return 1000 * float(d.mean()), 1000 * float(np.percentile(d, 90))


def compare(a, b):
    common, ia, ib = np.intersect1d(a["t"], b["t"], return_indices=True)
    A, B = a["poses"][ia, :3, 3], b["poses"][ib, :3, 3]
    R, t = umeyama(A, B)
    ate = 100 * np.linalg.norm(B @ R.T + t - A, axis=1)
    diffs = []
    for i, j in list(zip(ia, ib))[::4]:
        da = cv2.imread(os.path.join(a["frames"], "depth_ffs", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        db = cv2.imread(os.path.join(b["frames"], "depth_ffs", f"{j:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        both = (da > 0) & (db > 0)
        diffs.append(np.abs(da - db)[both])
    d = np.concatenate(diffs) if diffs else np.zeros(1)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    cd, cd90 = chamfer(a["mesh"], o3d.geometry.TriangleMesh(b["mesh"]).transform(M))  # in A's world
    return {"common frames": len(common), "frames": (len(a["t"]), len(b["t"])),
            "ATE cm (rmse / max)": (float(np.sqrt((ate ** 2).mean())), float(ate.max())),
            "depth mm (median / p90 / p99)": tuple(float(np.percentile(d, q)) for q in (50, 90, 99)),
            "chamfer mm (mean / p90)": (cd, cd90),
            "triangles": (len(a["mesh"].triangles), len(b["mesh"].triangles))}


def main():
    runs = [load(r) for r in sys.argv[1:]]
    for name, r in zip(sys.argv[2:], runs[1:]):
        print(f"{sys.argv[1]} vs {name}")
        for k, v in compare(runs[0], r).items():
            print(f"  {k}: " + (" / ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in v)
                                if isinstance(v, tuple) else str(v)))


if __name__ == "__main__":
    main()
