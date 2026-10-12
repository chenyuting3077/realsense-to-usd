"""Estimate camera poses for a frames dir (rec/ of a live session) with ORB-SLAM3 (stereo).

ORB-SLAM3 tracks the left/right IR pairs (projector off when it alternates, so the dot pattern
does not move with the camera), closes loops and runs global bundle adjustment.

The IR images are rectified and distortion-free, so ORB-SLAM3 runs in "Rectified" stereo mode
with the IR intrinsics and the recorded baseline. It is driven through its stereo_euroc example
(real-time playback, viewer off; ~2 min for a 2 min recording): the IR pairs are linked into an
EuRoC-style folder in a temporary directory. Its trajectory is of the left IR camera; poses are
moved to the color camera with the recorded ir1_from_color extrinsics and interpolated (slerp +
lerp) from the two IR frames around each color frame.
Frames tracked while a loop closure is being corrected can come out referenced to the not yet
corrected keyframes (room_1005, one run: two frames 35 cm off, the next ones fine again). IR
poses more than --outlier-cm from the median of their 7 neighbours are therefore dropped and
interpolated from the surrounding frames; smooth hand-held motion (1-2 cm per frame) is left
untouched.

Output:
  <out>/poses.npy                    Nx4x4 color camera-to-world (frame 0 = identity)
  <out>/trajectory.json              intrinsics, poses, mode, per-frame tracked flag, loop count
  <out>/orbslam3_ir1_trajectory.txt  ORB-SLAM3's own output (EuRoC: t_ns x y z qx qy qz qw)
  <out>/orbslam3.log

Needs ORB-SLAM3 built with patches/orbslam3-thor.patch in --orb-root and Pangolin, see README.

Usage (after `source env.sh`):
  python recon/track_orbslam3.py output/room/rec -o output/room/orbslam3
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

from frames import fill_untracked, interpolate, intrinsic_matrix, load_meta

SETTINGS = """%YAML:1.0
File.version: "1.0"
Camera.type: "Rectified"
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}
Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps}
Stereo.b: {baseline}
Camera.RGB: 0
Stereo.ThDepth: 40.0
ORBextractor.nFeatures: {features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: {ini_fast}
ORBextractor.minThFAST: {min_fast}
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


def drop_outliers(track, t_ns, outlier_cm):
    """Poses (4x4 or None) more than outlier_cm from the median of their 7 neighbours are
    replaced by interpolating the others. Returns (poses, indices replaced)."""
    ok = [i for i, T in enumerate(track) if T is not None]
    if outlier_cm <= 0 or len(ok) <= 7:
        return track, []
    pos = np.stack([track[i][:3, 3] for i in ok])
    med = np.stack([np.median(pos[max(0, k - 3):k + 4], axis=0) for k in range(len(ok))])
    bad = [ok[k] for k in np.where(np.linalg.norm(pos - med, axis=1) > outlier_cm / 100)[0]]
    if not bad:
        return track, []
    track = list(track)
    for i in bad:
        track[i] = None
    good = [i for i, T in enumerate(track) if T is not None]
    fixed = interpolate([track[i] for i in good], [t_ns[i] for i in good], [t_ns[i] for i in bad],
                        max_gap_ns=1_000_000_000)
    for i, T in zip(bad, fixed):
        track[i] = T
    return track, bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", help="frames dir with IR (rec/ of a live session)")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--orb-root", default=os.path.expanduser("~/ORB_SLAM3"))
    p.add_argument("--pangolin-lib", default=os.path.expanduser("~/.local/pangolin/lib"))
    p.add_argument("--features", type=int, default=1200, help="ORB features per image")
    p.add_argument("--fast", type=int, nargs=2, default=[12, 5], metavar=("INI", "MIN"),
                   help="FAST corner thresholds, the second for cells where the first finds none (ORB-SLAM3's "
                        "examples use 20 7); lower keeps more corners in dark, low-texture IR images")
    p.add_argument("--outlier-cm", type=float, default=5.0,
                   help="drop IR poses this far from the median of their neighbours (0: keep all)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = load_meta(args.frames)
    if "stereo" not in meta:
        p.error("needs IR images (rec/ of a live session)")
    st = meta["stereo"]
    ir_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in st["frames"]]
    color_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["frames"]]
    baseline = float(np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3]))
    fps = round(1e9 / np.median(np.diff(ir_ns)))

    with tempfile.TemporaryDirectory() as tmp:
        seq = os.path.join(tmp, "seq")
        for cam, d in (("cam0", "ir1"), ("cam1", "ir2")):
            os.makedirs(os.path.join(seq, "mav0", cam, "data"))
            for i, t in enumerate(ir_ns):
                os.symlink(os.path.abspath(os.path.join(args.frames, d, f"{i:06d}.png")),
                           os.path.join(seq, "mav0", cam, "data", f"{t}.png"))
        with open(os.path.join(tmp, "times.txt"), "w") as f:
            f.write("\n".join(map(str, ir_ns)) + "\n")
        with open(os.path.join(tmp, "settings.yaml"), "w") as f:
            f.write(SETTINGS.format(fx=st["fx"], fy=st["fy"], cx=st["cx"], cy=st["cy"], width=st["width"],
                                    height=st["height"], fps=fps, baseline=baseline, features=args.features,
                                    ini_fast=args.fast[0], min_fast=args.fast[1]))
        env = dict(os.environ)
        libs = [args.pangolin_lib, os.path.join(args.orb_root, "lib"),
                os.path.join(args.orb_root, "Thirdparty/DBoW2/lib"), os.path.join(args.orb_root, "Thirdparty/g2o/lib")]
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
        cmd = [os.path.join(args.orb_root, "Examples/Stereo/stereo_euroc"),
               os.path.join(args.orb_root, "Vocabulary/ORBvoc.txt"), "settings.yaml", "seq", "times.txt", "run"]
        print(f"ORB-SLAM3 stereo on {len(ir_ns)} IR pairs ({fps} fps, baseline {baseline * 1000:.1f} mm), "
              f"real-time playback ...")
        t0 = time.time()
        with open(os.path.join(args.out, "orbslam3.log"), "w") as log:
            ret = subprocess.run(cmd, cwd=tmp, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        traj = os.path.join(tmp, "f_run.txt")
        if ret != 0 or not os.path.exists(traj):
            raise SystemExit(f"ORB-SLAM3 failed (exit {ret}), see {args.out}/orbslam3.log")
        shutil.copy(traj, os.path.join(args.out, "orbslam3_ir1_trajectory.txt"))
    log = open(os.path.join(args.out, "orbslam3.log")).read()
    n_maps = len(re.findall(r"Creation of new map with id", log))
    n_loops = len(re.findall(r"\*Loop detected", log))
    print(f"ORB-SLAM3 finished in {time.time() - t0:.0f}s: {n_maps} map(s), {n_loops} loop closure(s)")
    if n_maps > 1:
        print("warning: tracking was lost and ORB-SLAM3 started a new map; the frames of the other "
              "maps are in their own coordinate frames")

    # EuRoC trajectory of the left IR camera -> color camera, at the IR frame times
    A = np.loadtxt(os.path.join(args.out, "orbslam3_ir1_trajectory.txt"), ndmin=2)
    by_stamp = {}
    for row in A:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()  # qx qy qz qw
        T[:3, 3] = row[1:4]
        by_stamp[int(round(row[0]))] = T @ np.array(st["ir1_from_color"])
    track = [by_stamp.get(t) for t in ir_ns]
    print(f"tracked {sum(T is not None for T in track)}/{len(ir_ns)} IR frames")
    track, bad = drop_outliers(track, ir_ns, args.outlier_cm)
    if bad:
        print(f"{len(bad)} IR poses off their neighbours by > {args.outlier_cm:g} cm, interpolated: {bad[:20]}")
    poses = interpolate(track, ir_ns, color_ns)
    tracked = [T is not None for T in poses]
    untracked = [i for i, ok in enumerate(tracked) if not ok]
    if untracked:
        print(f"warning: {len(untracked)} frames untracked, holding previous pose: "
              f"{untracked[:20]}{' ...' if len(untracked) > 20 else ''}")
    poses = fill_untracked(poses)
    poses = np.linalg.inv(poses[tracked.index(True)]) @ poses  # first tracked frame = world origin

    np.save(os.path.join(args.out, "poses.npy"), poses)
    with open(os.path.join(args.out, "trajectory.json"), "w") as f:
        json.dump({"intrinsic": intrinsic_matrix(meta).tolist(), "mode": "orbslam3-stereo", "poses": poses.tolist(),
                   "tracked": tracked, "maps": n_maps, "loop_closures": n_loops}, f)
    length = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()
    print(f"trajectory length {length:.2f} m -> {args.out}/poses.npy")


if __name__ == "__main__":
    main()
