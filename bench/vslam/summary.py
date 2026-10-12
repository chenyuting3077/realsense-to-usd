"""The README's summary table, from score_all.sh's stability.json (and revisits.json when ICP was run) per session,
plus the RMSE against ORB-SLAM3 over each session, up to --until-end s before its end (ORB-SLAM3 itself fails in
the last seconds of the 250 s session).

    python -m bench.vslam.summary output output/vslam
"""
import argparse
import json
import os

import numpy as np

from bench.vslam.compare_traj import umeyama
from bench.vslam.eval_revisits import load_traj

SESSIONS = ["live_20261012_152105", "live_20261012_160515", "live_20261012_171428", "live_20261012_182911",
            "live_20261012_184220"]
FILES = {"orb": None, "ov": "ov.tum", "ov_rtab_stereo": "ov_rtab_stereo.tum", "ov_rtab_rgbd": "ov_rtab_rgbd.tum",
         "cuv_vio": "cuvslam_vio/cuvslam_slam.tum", "cuv_stereo": "cuvslam_stereo/cuvslam_slam.tum",
         "vins": "vins/vins.tum", "vins_loop": "vins/vins_loop.tum"}


def rmse_vs_orb(frames, path, until_end_s):
    meta = json.load(open(os.path.join(frames, "meta.json")))
    ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["frames"]]
    C = np.array(meta["stereo"]["ir1_from_color"])
    ref = load_traj(os.path.join(frames, "poses.npy"), ns, C)
    T = load_traj(path, ns, C)
    end = ns[-1] - until_end_s * 1e9
    both = [k for k in range(len(ns)) if ns[k] <= end and T[k] is not None and ref[k] is not None]
    A = np.array([ref[k][:3, 3] for k in both])
    B = np.array([T[k][:3, 3] for k in both])
    R, s = umeyama(A, B)
    return float(100 * np.sqrt((np.linalg.norm(B @ R.T + s - A, axis=1) ** 2).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec", help="recordings dir (output)")
    p.add_argument("work", help="output/vslam")
    p.add_argument("--until-end", type=float, default=15.0)
    a = p.parse_args()
    rows = {}
    for s in SESSIONS:
        st = json.load(open(os.path.join(a.work, s, "stability.json")))
        rv = os.path.join(a.work, s, "revisits.json")
        rv = json.load(open(rv))["summary"] if os.path.exists(rv) and s.endswith("184220") else {}
        for n, f in FILES.items():
            if n not in st:
                continue
            r = rows.setdefault(n, {"cover": [], "first": [], "jumps": 0, "largest": 0.0, "still": [], "still_max": 0.0,
                                    "rmse_short": [], "rmse_long": None, "revisit": None})
            x = st[n]
            r["cover"].append(x["coverage"])
            r["first"].append(x["first_pose_s"])
            r["jumps"] += x["jumps"]
            r["largest"] = max(r["largest"], x["largest_jump_m"])
            if x["still_cm_s"] is not None:
                r["still"].append(x["still_cm_s"])
                r["still_max"] = max(r["still_max"], x["still_max_cm_s"])
            if f:
                e = rmse_vs_orb(os.path.join(a.rec, s, "frames"), os.path.join(a.work, s, f),
                                a.until_end if s.endswith("184220") else 0.0)
                (r["rmse_short"].append(e) if not s.endswith("184220") else r.update(rmse_long=e))
            if n in rv:
                r["revisit"] = (rv[n]["median_cm"], rv[n]["p90_cm"])
    print("| | coverage | first pose | jumps (largest) | still: median / max cm/s | vs ORB-SLAM3, 4 short | "
          f"vs ORB-SLAM3, 250 s (to {a.until_end:.0f} s before end) | revisits 250 s, median / p90 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for n, r in rows.items():
        short = f"{min(r['rmse_short']):.1f}–{max(r['rmse_short']):.1f} cm" if r["rmse_short"] else "–"
        long = f"{r['rmse_long']:.1f} cm" if r["rmse_long"] is not None else "–"
        if r["rmse_long"] is not None and r["rmse_long"] > 1000:
            long = f"{r['rmse_long'] / 100:.0f} m"
        rev = f"{r['revisit'][0]:.1f} / {r['revisit'][1]:.1f} cm" if r["revisit"] else "–"
        print(f"| {n} | {100 * np.mean(r['cover']):.0f}% | {np.mean(r['first']):.1f} s | {r['jumps']} "
              f"({r['largest']:.2f} m) | {np.median(r['still']):.2f} / {r['still_max']:.1f} | {short} | {long} | {rev} |")


if __name__ == "__main__":
    main()
