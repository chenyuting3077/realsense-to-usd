"""How stable trajectories are, without ground truth, from the session's own IMU:
  coverage   share of the recording (first to last IR pair) with a pose, and when the first pose comes
  jumps      steps faster than a hand can move the camera (> 2 m/s or > 6 rad/s between poses at most
             0.25 s apart), and the largest such step
  still      while the IMU says the camera is held still (1 s windows: |gyro| < 0.03 rad/s, accel spread
             < 0.05 m/s^2), how fast the pose moves anyway (cm/s): drift and jitter at rest
The camera's real motion is never that fast and never moves while the IMU is still, so all of it is error.

    python -m bench.vslam.stability output/live_x/rec orb=output/live_x/frames/poses.npy ov=ov.tum ...
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

MAX_SPEED, MAX_RATE, MAX_DT = 2.0, 6.0, 0.25


def load(path, rec):
    """(t_s, positions, rotations) of a TUM file, or of frames/poses.npy (color camera, timestamps from its meta)."""
    if path.endswith(".npy"):
        frames = os.path.dirname(path)
        meta = json.load(open(os.path.join(frames, "meta.json")))
        t = np.array([f["timestamp_ms"] for f in meta["frames"]]) / 1e3
        P = np.load(path, allow_pickle=True)
        ok = [k for k, T in enumerate(P) if T is not None and np.isfinite(np.asarray(T, float)).all()]
        P = np.array([P[k] for k in ok], float)
        C = np.array(meta["stereo"]["ir1_from_color"])
        P = P @ np.linalg.inv(C)  # left-IR poses, like the TUM files
        return t[ok], P[:, :3, 3], Rotation.from_matrix(P[:, :3, :3])
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1:4], Rotation.from_quat(a[:, 4:8])


def still_windows(rec):
    meta = json.load(open(os.path.join(rec, "meta.json")))
    imu = np.loadtxt(os.path.join(rec, meta["imu"]["file"]), delimiter=",", skiprows=1)
    t = imu[:, 0] / 1e9
    out = []
    for a in np.arange(t[0], t[-1] - 1, 0.5):
        s = (t >= a) & (t < a + 1)
        if s.sum() > 50 and np.abs(imu[s, 4:7]).max() < 0.03 and imu[s, 1:4].std(0).max() < 0.05:
            out.append((a, a + 1))
    return out, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec")
    p.add_argument("traj", nargs="+", help="name=path")
    p.add_argument("--json")
    a = p.parse_args()
    still, meta = still_windows(a.rec)
    ir = np.array([f["timestamp_ms"] for f in meta["stereo"]["frames"]]) / 1e3
    span = ir[-1] - ir[0]
    res = {"still_s": 0.5 * len(still) + 0.5 if still else 0.0}
    print(f"{span:.0f} s recorded, {res['still_s']:.0f} s held still")
    print(f"{'trajectory':<16}{'coverage':>9}{'1st pose':>9}{'jumps':>6}{'largest':>9}{'still cm/s':>11}{'still max':>10}")
    for name, path in (s.split("=", 1) for s in a.traj):
        t, x, R = load(path, a.rec)
        cover = float(np.clip((np.minimum(t[-1], ir[-1]) - np.maximum(t[0], ir[0])) / span, 0, 1))
        gaps = np.diff(t)
        cover -= float(gaps[gaps > MAX_DT].sum() / span)  # holes in the trajectory
        dt = np.diff(t)
        step = np.linalg.norm(np.diff(x, axis=0), axis=1)
        rot = (R[:-1].inv() * R[1:]).magnitude()
        ok = (dt > 0) & (dt <= MAX_DT)
        jump = ok & ((step / np.maximum(dt, 1e-6) > MAX_SPEED) | (rot / np.maximum(dt, 1e-6) > MAX_RATE))
        speeds = []
        for w0, w1 in still:
            s = (t >= w0) & (t <= w1)
            if s.sum() >= 3:
                speeds.append(np.linalg.norm(x[s][-1] - x[s][0]) / (t[s][-1] - t[s][0]))
        r = {"coverage": cover, "first_pose_s": float(t[0] - ir[0]), "jumps": int(jump.sum()),
             "largest_jump_m": float(step[jump].max()) if jump.any() else 0.0,
             "still_cm_s": float(100 * np.median(speeds)) if speeds else None,
             "still_max_cm_s": float(100 * np.max(speeds)) if speeds else None}
        res[name] = r
        st = "-" if r["still_cm_s"] is None else f"{r['still_cm_s']:.2f}"
        sm = "-" if r["still_max_cm_s"] is None else f"{r['still_max_cm_s']:.1f}"
        print(f"{name:<16}{100 * cover:>8.1f}%{r['first_pose_s']:>8.1f}s{r['jumps']:>6}{r['largest_jump_m']:>8.2f}m"
              f"{st:>11}{sm:>10}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
