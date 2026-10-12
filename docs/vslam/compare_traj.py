"""Trajectories of one session side by side: each rigidly aligned (rotation + translation, no scale) to the
first one over the frames both pose, top view and height, and how far apart they are after alignment.

    python docs/vslam/compare_traj.py output/live_x/frames out.png orb=frames/poses.npy ov=ov.tum ...
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_revisits import load_traj  # noqa: E402


def umeyama(A, B):
    """R, t with R @ B + t ~ A (rigid, least squares)."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, ma - R @ mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames")
    p.add_argument("png")
    p.add_argument("traj", nargs="+", help="name=path; the first is the reference frame")
    a = p.parse_args()
    meta = json.load(open(os.path.join(a.frames, "meta.json")))
    ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["frames"]]
    C = np.array(meta["stereo"]["ir1_from_color"])
    trajs = {n: load_traj(path, ns, C) for n, path in (s.split("=", 1) for s in a.traj)}
    names = list(trajs)
    ref = trajs[names[0]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    t = (np.array(ns) - ns[0]) / 1e9
    out = {}
    for n in names:
        T = trajs[n]
        both = [k for k in range(len(ns)) if T[k] is not None and ref[k] is not None]
        A = np.array([ref[k][:3, 3] for k in both])
        B = np.array([T[k][:3, 3] for k in both])
        R, s = umeyama(A, B)
        P = B @ R.T + s
        err = np.linalg.norm(P - A, axis=1)
        L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        out[n] = {"frames": len(both), "path_m": float(L), "rmse_to_ref_cm": float(100 * np.sqrt((err ** 2).mean())),
                  "max_to_ref_cm": float(100 * err.max())}
        ax[0].plot(P[:, 0], P[:, 1], lw=1, label=f"{n} ({L:.1f} m)")
        ax[1].plot(t[both], P[:, 2], lw=1, label=n)
        print(f"{n:<10} {len(both):5d} frames  path {L:6.1f} m  vs {names[0]}: rmse {out[n]['rmse_to_ref_cm']:6.1f} cm"
              f"  max {out[n]['max_to_ref_cm']:6.1f} cm")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].legend()
    ax[0].set_title(f"top view, aligned to {names[0]}")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("z (m)")
    ax[1].set_title("height")
    fig.tight_layout()
    fig.savefig(a.png, dpi=110)
    return out


if __name__ == "__main__":
    main()
