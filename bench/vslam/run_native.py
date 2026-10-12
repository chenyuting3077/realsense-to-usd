"""recon/vio.py (the native OpenVINS + RTAB-Map that main.py tracks with) on a session's rec/, frame by frame as
the tracker process feeds it, as fast as it goes but waiting for RTAB-Map as if live. Writes vio.tum (every IR
pair's final pose: left IR, world as recon/vio.py), vio_odom.tum (OpenVINS alone) and vio.json (per-pair time,
loop closures), to compare with the docker runs
(run_all.sh) by compare_traj.py / stability.py.

    python -m bench.vslam.run_native output/live_x/rec output/vslam/live_x/native
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from recon.vio import IMU_AHEAD_NS, Vio  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec")
    p.add_argument("out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.rec, "meta.json")))
    ir_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["stereo"]["frames"]]
    imu = np.loadtxt(os.path.join(a.rec, meta["imu"]["file"]), delimiter=",", skiprows=1)
    vio = Vio(meta)
    t0, j, ms, events = ir_ns[0], 0, [], []
    read = lambda k: [cv2.imread(os.path.join(a.rec, d, f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE)  # noqa: E731
                      for d in ("ir1", "ir2")]
    with ThreadPoolExecutor(4) as pool:
        pending = [pool.submit(read, k) for k in range(min(16, len(ir_ns)))]
        for k, ns in enumerate(ir_ns):
            left, right = pending[k].result()
            pending[k] = None
            if k + 16 < len(ir_ns):
                pending.append(pool.submit(read, k + 16))
            i0 = j
            while j < len(imu) and imu[j, 0] <= ns + IMU_AHEAD_NS:
                j += 1
            rows = imu[max(i0 - (10 if k == 0 else 0), 0):j].copy()
            rows[:, 0] = (rows[:, 0] - t0) / 1e9
            t = time.perf_counter()
            vio.track((ns - t0) / 1e9, left, right, rows)
            ms.append(1000 * (time.perf_counter() - t))
            vio.finish()  # faster than real time here: let RTAB-Map take every keyframe it would live
            ch = vio.changed()
            if ch:
                events.append(((ns - t0) / 1e9, ch[0]))
    vio.finish()
    T_all = [vio.pose(k) for k in range(len(ir_ns))]
    odom = [None if T is None else vio.world @ T for T in vio.odom]
    for name, P in (("vio.tum", T_all), ("vio_odom.tum", odom)):
        with open(os.path.join(a.out, name), "w") as f:
            for ns, T in zip(ir_ns, P):
                if T is not None:
                    q = Rotation.from_matrix(T[:3, :3]).as_quat()
                    f.write(f"{ns / 1e9:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} "
                            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
    ms = np.array(ms)
    info = {"frames": len(ir_ns), "posed": sum(T is not None for T in T_all), "loop_closures": vio.loops,
            "proximity_links": vio.proximity, "nodes": len(vio.nodes),
            "track_ms": {"median": float(np.median(ms)), "p90": float(np.percentile(ms, 90)), "max": float(ms.max())},
            "events": events[:20] + (["..."] if len(events) > 20 else [])}
    json.dump(info, open(os.path.join(a.out, "vio.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "events"}), f"{len(events)} trajectory updates")


if __name__ == "__main__":
    main()
