"""Global consistency of trajectories without ground truth: frames that see the same place far apart in
time (a revisit) should have depth that lines up. For each revisit pair, ICP (point-to-plane, FFS depth)
starts from the trajectory's relative pose; how far ICP moves it is that trajectory's error there.

    python docs/vslam/eval_revisits.py output/live_x/frames orb=orb.tum ov=ov.tum ovrtab=rtab.tum

Trajectories are TUM files (s, xyz, qxyzw) of left-IR poses; `orb` may also be given as poses.npy
(color-camera poses of frames/, as main.py saves them).
"""
import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp


def load_traj(path, frame_ns, ir1_from_color):
    """Color-camera-to-world 4x4 at frame_ns (None outside the trajectory)."""
    if path.endswith(".npy"):
        P = np.load(path, allow_pickle=True)
        return [None if T is None or not np.isfinite(np.asarray(T, float)).all() else np.asarray(T, float) for T in P]
    a = np.loadtxt(path)
    t = (a[:, 0] * 1e9).astype(np.int64)
    slerp = Slerp(t, Rotation.from_quat(a[:, 4:8]))
    out = []
    for ns in frame_ns:
        if ns < t[0] or ns > t[-1]:
            out.append(None)
            continue
        j = int(np.clip(np.searchsorted(t, ns), 1, len(t) - 1))
        if t[j] - t[j - 1] > 200_000_000:  # a gap: no pose
            out.append(None)
            continue
        s = (ns - t[j - 1]) / max(t[j] - t[j - 1], 1)
        T = np.eye(4)
        T[:3, :3] = slerp([ns]).as_matrix()[0]
        T[:3, 3] = (1 - s) * a[j - 1, 1:4] + s * a[j, 1:4]
        out.append(T @ ir1_from_color)
    return out


def cloud(frames, k, K, depth_max=3.0, stride=4):
    d = cv2.imread(os.path.join(frames, "depth_ffs", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)[::stride, ::stride] / 1000.0
    mask_path = os.path.join(frames, "mask", f"{k:06d}.png")
    if os.path.exists(mask_path):
        d = d * (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[::stride, ::stride] > 0)
    v, u = np.nonzero((d > 0.2) & (d < depth_max))
    z = d[v, u]
    x = (u * stride - K[0, 2]) / K[0, 0] * z
    y = (v * stride - K[1, 2]) / K[1, 1] * z
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.stack([x, y, z], 1)))
    pc = pc.voxel_down_sample(0.02)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    return pc


def revisit_pairs(trajs, frame_ns, min_dt_s, max_dist, max_deg, per_place, max_dt_s=np.inf):
    """Pairs (i, j), j - i > min_dt_s apart, close in position and view direction under any trajectory;
    thinned so a place gives at most per_place pairs."""
    pairs = set()
    t = np.array(frame_ns) / 1e9
    for T in trajs.values():
        ok = [k for k, X in enumerate(T) if X is not None]
        C = np.array([T[k][:3, 3] for k in ok])
        Z = np.array([T[k][:3, 2] for k in ok])
        for a, i in enumerate(ok):
            far = (t[ok] - t[i] > min_dt_s) & (t[ok] - t[i] < max_dt_s)
            near = (np.linalg.norm(C - C[a], axis=1) < max_dist) & (Z @ Z[a] > np.cos(np.radians(max_deg)))
            for b in np.nonzero(far & near)[0]:
                pairs.add((i, ok[b]))
    pairs = sorted(pairs)
    kept, seen = [], {}
    for i, j in pairs:  # thin: one pair per (i // 10, j // 10) cell, at most per_place per i // 10
        key = (i // 10, j // 10)
        if key in seen or sum(1 for k in seen if k[0] == i // 10) >= per_place:
            continue
        seen[key] = True
        kept.append((i, j))
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames")
    p.add_argument("traj", nargs="+", help="name=path")
    p.add_argument("--min-dt", type=float, default=20.0)
    p.add_argument("--max-dt", type=float, default=np.inf, help="with --min-dt 0: the noise floor (no drift)")
    p.add_argument("--max-dist", type=float, default=0.7)
    p.add_argument("--max-deg", type=float, default=25.0)
    p.add_argument("--per-place", type=int, default=3)
    p.add_argument("--json")
    a = p.parse_args()
    meta = json.load(open(os.path.join(a.frames, "meta.json")))
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])
    frame_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["frames"]]
    ir1_from_color = np.array(meta["stereo"]["ir1_from_color"])
    trajs = dict(s.split("=", 1) for s in a.traj)
    trajs = {n: load_traj(path, frame_ns, ir1_from_color) for n, path in trajs.items()}

    pairs = revisit_pairs(trajs, frame_ns, a.min_dt, a.max_dist, a.max_deg, a.per_place, a.max_dt)
    clouds = {}
    get = lambda k: clouds.setdefault(k, cloud(a.frames, k, K))  # noqa: E731
    res = {n: [] for n in trajs}
    for i, j in pairs:
        if any(T[i] is None or T[j] is None for T in trajs.values()):
            continue
        src, dst = get(j), get(i)
        if len(src.points) < 500 or len(dst.points) < 500:
            continue
        for n, T in trajs.items():
            init = np.linalg.inv(T[i]) @ T[j]  # frame j in frame i, per this trajectory
            r = o3d.pipelines.registration.registration_icp(
                src, dst, 0.15, init, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            D = np.linalg.inv(init) @ r.transformation
            cm = 100 * np.linalg.norm(D[:3, 3])
            deg = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
            res[n].append((i, j, cm, deg, r.fitness, r.inlier_rmse))
    # A pair counts only if ICP converged for every trajectory to the same answer (fitness > 0.3):
    # then the correction measures the trajectory, not ICP.
    n0 = next(iter(res))
    good = [k for k in range(len(res[n0])) if all(res[n][k][4] > 0.3 for n in res)]
    summary = {"pairs": len(good), "frames": len(frame_ns)}
    print(f"{len(pairs)} revisit pairs, {len(good)} with ICP fitness > 0.3 under every trajectory")
    print(f"{'trajectory':<12}{'median cm':>10}{'p90 cm':>9}{'median deg':>11}{'p90 deg':>9}{'> 5 cm':>8}")
    for n in res:
        cm = np.array([res[n][k][2] for k in good])
        deg = np.array([res[n][k][3] for k in good])
        if not len(cm):
            continue
        s = {"median_cm": float(np.median(cm)), "p90_cm": float(np.percentile(cm, 90)),
             "median_deg": float(np.median(deg)), "p90_deg": float(np.percentile(deg, 90)),
             "over_5cm": float((cm > 5).mean())}
        summary[n] = s
        print(f"{n:<12}{s['median_cm']:>10.1f}{s['p90_cm']:>9.1f}{s['median_deg']:>11.2f}{s['p90_deg']:>9.2f}"
              f"{100 * s['over_5cm']:>7.0f}%")
    if a.json:
        json.dump({"summary": summary, "pairs": {n: [res[n][k] for k in good] for n in res}}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
