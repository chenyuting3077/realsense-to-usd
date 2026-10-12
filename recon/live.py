"""Live reconstruction: ORB-SLAM3 poses, Fast-FoundationStereo depth and nvblox TSDF at once,
with the map rebuilt in the background whenever a loop closure moves the trajectory.

Frames come from a D455 (--camera: 848x480 at 60 fps, projector alternating) or from a recorded
session (rec/ of an earlier --camera run), replayed at its own clock:
  tracker  (own process, recon/live_workers.py) the newest projector-off IR pair -> ORB-SLAM3
           stereo (recon/orbslam3_py), 30 Hz; pairs that arrive while it is busy are skipped
  depth    the newest projector-on frameset (IR pair + color) -> FFS (TensorRT) -> depth in the
           color camera, ~9 Hz on Jetson Thor (the others are skipped)
  mapper   (main) pose of each depth frame interpolated from the IR poses around it -> nvblox
           depth + color integration; the depth and color are kept for rebuilds
ORB-SLAM3 reports loop closures, global BA and map merges as "big changes". Each one bumps
the pose version: the corrected trajectory is read back (recon/orbslam3_py trajectory(), the
frames tracked while the loop was being corrected filtered as in track_orbslam3.py), and the
mapper re-integrates every stored frame with the new poses into a new nvblox map, between
live frames. Live frames keep going into the current map meanwhile; when the rebuild has
caught up the maps are swapped. A newer version restarts the rebuild. At the end (recording
over, --duration, or Ctrl+C) the last trajectory is applied the same way, and that map is the
result.

Output in -o:
  mesh_live.ply   final mesh: every stored frame in reconstruct.py's distance-weighted TSDF
                  (--final-depth-max, --final-weight-ref; 0 for the nvblox map instead)
  poses_live.npy  Nx4x4 color camera-to-world (the mesh's world) of every frame of the replayed
                  recording, or with --camera of the frames in frames/. The world is leveled from
                  the accelerometer at the start (Z up, X the first camera's forward direction,
                  origin at it): the D455's with --camera, a recording's "up_ir1" (meta.json)
                  when replayed; without it the world is the first camera's frame.
  frames/         --camera: the frames given to the mapper in the frames-dir layout
                  (color/, depth_ffs/ in mm, meta.json, poses.npy) for offline steps (3DGS, ...)
  rec/            --camera (unless --no-record): every frameset of the session in the frames-dir layout
                  (color/, depth/ = the D455's depth aligned to color, ir1_on/ ir2_on/,
                  ir1/ ir2/, meta.json) and poses.npy of its color frames from the live trajectory:
                  the input of offline depth (stereo_depth.py, depth completion) and tracking. ~40 MB/s
  live.json       timings: tracking, depth, integration, latency, rebuilds, mesh updates

With the window, one run can hold several sessions: Stop (or Ctrl+C) ends one and saves it, New
recording in the window starts the next (-o gets _2, _3, ...; the FFS engine is kept), Discard &
restart deletes the running one and starts again, Re-level (IMU) re-levels the world from the D455.

ORB-SLAM3 and the --view window (recon/live_view.py: the map with the trajectory and the
current camera, the tracked IR image with its ORB features, depth, status, session buttons) run in
their own processes (recon/live_workers.py):
in a thread here they would wait for the GIL behind nvblox calls and fall behind.

Usage (after `source env.sh` and `sudo jetson_clocks`; build recon/orbslam3_py first, see README):
  python recon/live.py                              # D455 until Ctrl+C -> output/live_<date>_<time>,
                                                    # with a window when there is a display ($DISPLAY)
  python recon/live.py --no-view --duration 120 -o output/live_room
  python recon/live.py output/live_room/rec         # replay -> output/live_rec_<date>_<time>
"""
import argparse
import gc
import itertools
import json
import multiprocessing as mp
import os

os.environ.setdefault("GLOG_minloglevel", "1")  # nvblox's info logs (layer and hash allocations)
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d
import torch
from nvblox_torch.sensor import Sensor

from ffs import FFS
from frames import FloorDepth, fill_untracked, filter_depth, interpolate, intrinsic_matrix
from live_view import view_process
from live_workers import track_process
from reconstruct import WeightedTSDF, extract_mesh
from reconstruct_nvblox import add_mapper_args, clean_mesh, make_mapper
from stereo_depth import DisparityToColorDepth


def default_engine(st):
    """recon/export_ffs_trt.py's fp32 + TF32 engine for this IR size (width padded to a multiple of 32)."""
    return f"output/ffs/ffs_{st['height']}x{-(-st['width'] // 32) * 32}_it8_fp32_tf32.engine"


class Tracker(threading.Thread):
    """Mirror of live_workers.track_process in this process: ir_ns / T are the times and left-IR
    camera-to-world poses (None: not tracked) of the IR pairs tracked so far, `version` the number
    of trajectory corrections (each replaces T as a whole)."""

    def __init__(self, source, args, view_q=None):
        super().__init__(daemon=True)
        self.ir_ns, self.T, self._ns = [], [], np.zeros(0, np.int64)
        self.version, self.finished = 0, False
        self.track_ms, self.lag_ms, self.events, self.skipped, self.notes = [], [], [], 0, []
        ctx = mp.get_context("spawn")
        self.ctrl_q, self.q, self.req_q, self.frames_q = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=track_process, daemon=True,
                                args=(source, {"features": args.features, "fast": args.fast, "imu": args.imu, "orb_root": args.orb_root,
                                               "outlier_cm": args.outlier_cm}, self.ctrl_q, self.q, self.req_q,
                                      self.frames_q, view_q))
        self.proc.start()
        while True:  # the child's traceback is already printed if it dies (e.g. no camera)
            try:
                msg = self.q.get(timeout=1)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise SystemExit("ORB-SLAM3 / the camera did not start")
        if msg[0] != "ready":
            raise SystemExit("ORB-SLAM3 / the camera did not start")
        self.meta = msg[1]

    def go(self):
        self.ctrl_q.put(("go", time.perf_counter()))
        self.start()

    def stop(self):
        self.ctrl_q.put(("stop",))

    def relevel(self):
        self.ctrl_q.put(("level",))

    def run(self):
        while True:
            msg = self.q.get()
            if msg[0] == "pose":
                _, t, T, track_ms, lag_ms, skipped = msg
                self.ir_ns.append(t)
                self.T.append(T)
                self.track_ms.append(track_ms)
                self.lag_ms.append(lag_ms)
                self.skipped += skipped
            elif msg[0] == "traj":
                _, version, T, event = msg
                self.T, self.version = T + self.T[len(T):], version
                self.events.append(event)
                if event.get("up_ir1"):  # changed by a re-level
                    self.meta["up_ir1"] = event["up_ir1"]
            elif msg[0] == "note":
                print(f"[tracker] {msg[1]}", flush=True)
                self.notes.append(msg[1])
            else:
                self.finished = True
                return

    def alive(self):
        return self.finished or self.proc.is_alive()

    def last_ns(self):
        return self.ir_ns[-1] if self.ir_ns else -1

    def pose(self, t_ns, table=None):
        """Left-IR camera-to-world at time t_ns, or None."""
        table = self.T if table is None else table
        n = min(len(table), len(self.ir_ns))
        if len(self._ns) != n:
            self._ns = np.array(self.ir_ns[:n], np.int64)
        if n == 0:
            return None
        j = int(np.searchsorted(self._ns, t_ns))
        a, b = max(j - 1, 0), min(j, n - 1)
        return interpolate([table[a], table[b]], self._ns[[a, b]], [t_ns])[0]


class Depth(threading.Thread):
    """FFS on the newest projector-on frameset (IR pair + color) from the source."""

    def __init__(self, meta, args, ffs, req_q, in_q, out_q):
        super().__init__(daemon=True)
        self.req_q, self.in_q, self.q, self.args = req_q, in_q, out_q, args
        self.K = intrinsic_matrix(meta)
        self.dev = torch.device("cuda")
        self.ffs = ffs  # the TensorRT engine, kept from one session to the next
        self.to_color = DisparityToColorDepth(meta, self.dev)
        self.ffs_ms, self.skipped, self.received, self.finished = [], 0, 0, False
        # Own stream: synchronizing it does not wait for nvblox, and the other way round.
        self.stream = torch.cuda.Stream()
        z = torch.zeros(meta["stereo"]["height"], meta["stereo"]["width"], dtype=torch.uint8, device=self.dev)
        with torch.cuda.stream(self.stream):
            self.ffs(z, z)  # warm up
        self.stream.synchronize()

    def run(self):
        with torch.cuda.stream(self.stream):
            self.loop()
        self.finished = True

    def loop(self):
        last = -1
        while True:
            self.req_q.put(1)  # the newest frameset from the source; the ones before are skipped
            item = self.in_q.get()
            if item is None:
                return
            i, t_ns, wall, ir1, ir2, color = item
            self.skipped += i - last - 1
            last = i
            t = time.perf_counter()
            disp = self.ffs(torch.as_tensor(ir1, device=self.dev), torch.as_tensor(ir2, device=self.dev))
            # Wait for FFS here, where the GIL is released. The warp's boolean indexing syncs too, but
            # inside a torch op that holds the GIL: the mapper would stall for the whole inference.
            self.stream.synchronize()
            depth = filter_depth(self.to_color(disp), self.K, self.args.edge_thresh, self.args.grazing_deg,
                                 as_tensor=True)
            color = torch.from_numpy(color).to(self.dev)
            self.stream.synchronize()  # nvblox reads them on its own stream
            self.ffs_ms.append((time.perf_counter() - t) * 1000)
            self.q.put((t_ns, wall, depth, color))


def mesh_for_view(mapper, tracker, status, frame, mesh=None):
    """The window's map message (live_view.py): mesh (the mapper's, or an Open3D mesh), left-IR
    camera centers, status, and the newest (color, depth) at half resolution."""
    if mesh is None:
        m = mapper.get_color_mesh()
        v, f, c = m.vertices().cpu().numpy(), m.triangles().cpu().numpy(), m.vertex_colors().cpu().numpy()
    else:
        v, f = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)
        c = np.asarray(mesh.vertex_colors) * 255
    cams = np.array([T[:3, 3] for T in list(tracker.T) if T is not None])
    if frame is not None:
        _, _, d, c_ = frame
        frame = (c_[::2, ::2].cpu().numpy(), d[::2, ::2].cpu().numpy())
    return ("map", v, f, c, cams.reshape(-1, 3), dict(status, triangles=len(f)), frame)


def final_mesh(args, K, stored, poses):
    """The session's mesh from every stored frame: reconstruct.py's distance-weighted TSDF (Triton),
    depth up to --final-depth-max, each observation weighted min(1, (--final-weight-ref / z)^2).
    Better than the nvblox map out to 4 m, and its GPU memory is returned."""
    tsdf = WeightedTSDF(args.voxel_size, args.trunc_vox, args.final_depth_max, args.final_weight_ref, 1_000_000)
    for (_, _, d, c), T in zip(stored, poses):
        tsdf.integrate(d, c, K, T, 1.0)  # depth in meters, on the GPU
    mesh = extract_mesh(tsdf.voxel_grid(), args.voxel_size, WeightedTSDF.RES, args.mesh_min_weight)
    del tsdf
    torch.cuda.empty_cache()
    return clean_mesh(mesh, args.min_cluster)


def save_frames(out, meta, stored, poses):
    """The frames given to the mapper as a frames dir (color/, depth_ffs/ in mm,
    meta.json), so reconstruct*.py, gs/export_colmap.py etc. can run on a live session."""
    for d in ("color", "depth_ffs"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    def write(k):
        _, _, d, c = stored[k]
        cv2.imwrite(os.path.join(out, "color", f"{k:06d}.png"), cv2.cvtColor(c.cpu().numpy(), cv2.COLOR_RGB2BGR))
        mm = torch.clamp(torch.round(d * meta["depth_scale"]), 0, 65535).to(torch.int32).cpu().numpy()
        cv2.imwrite(os.path.join(out, "depth_ffs", f"{k:06d}.png"), mm.astype(np.uint16))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(write, range(len(stored))))
    m = dict(meta, frames=[{"timestamp_ms": t / 1e6} for t, _, _, _ in stored])
    m["stereo"] = {k: v for k, v in meta["stereo"].items() if k != "frames"}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(m, f, indent=1)
    np.save(os.path.join(out, "poses.npy"), poses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", nargs="?", help="frames dir to replay (rec/ of a --camera session; "
                                             "default: live from the D455")
    p.add_argument("-o", "--out", help="output dir (default: output/live_<recording>_<date>_<time>)")
    p.add_argument("--camera", action="store_true",
                   help="live from a D455 (the default without a frames dir)")
    p.add_argument("--serial", help="D455 serial number (default: the first one)")
    p.add_argument("--duration", type=float, help="seconds to run with --camera (default: until Ctrl+C)")
    p.add_argument("--projector", choices=["alternate", "off"], default="alternate",
                   help="with the D455: alternate (dots on every other frameset for depth, the others tracked "
                        "at 30 Hz) or off (no dots: every frameset tracked at 60 Hz and used for depth too; FFS "
                        "depth is barely worse, the D455's own depth much sparser on plain surfaces)")
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=True,
                   help="with --camera, save every frameset (color, D455 depth, IR) to <out>/rec")
    p.add_argument("--engine", help="FFS TensorRT engine (recon/export_ffs_trt.py; default: the fp32 + TF32 "
                                    "engine for the IR size in output/ffs)")
    p.add_argument("--orb-root", default=os.path.expanduser("~/ORB_SLAM3"))
    p.add_argument("--features", type=int, default=1200, help="ORB features per image")
    p.add_argument("--fast", type=int, nargs=2, default=[12, 5], metavar=("INI", "MIN"),
                   help="FAST corner thresholds (as track_orbslam3.py)")
    p.add_argument("--imu", action=argparse.BooleanOptionalAction, default=False,
                   help="ORB-SLAM3 stereo-inertial with the D455's gyroscope and accelerometer (a replay needs "
                        "imu.csv): the IMU carries the pose through blurred or textureless seconds")
    p.add_argument("--outlier-cm", type=float, default=5.0, help="as track_orbslam3.py")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed (1: as recorded)")
    p.add_argument("--rebuild-chunk", type=int, default=8,
                   help="stored frames re-integrated between two checks for a new live frame")
    p.add_argument("--mesh-every", type=float, default=1.0, help="seconds between mesh updates")
    p.add_argument("--final-depth-max", type=float, default=4.0,
                   help="the final mesh (after Stop) integrates the depth up to this far with recon/reconstruct.py's "
                        "distance-weighted TSDF; <= 0: the final mesh is the nvblox map's (--depth-max)")
    p.add_argument("--final-weight-ref", type=float, default=2.0,
                   help="final mesh: observations within this depth (m) weigh 1, farther ones (ref / z)^2")
    p.add_argument("--view", action=argparse.BooleanOptionalAction,
                   help="show the map, the camera and the status in a window (recon/live_view.py; "
                        "default: when there is a display, $DISPLAY)")
    add_mapper_args(p)
    args = p.parse_args()
    if args.view is None:
        args.view = bool(os.environ.get("DISPLAY"))
    if args.frames and args.camera:
        p.error("give a frames dir to replay, or --camera, not both")
    args.camera = not args.frames
    ctx = mp.get_context("spawn")
    ui = {"viewer": None, "ffs": None, "map_q": None, "cam_q": None, "ui_q": None, "error": p.error}
    if args.view:
        ui.update(map_q=ctx.Queue(), cam_q=ctx.Queue(maxsize=2), ui_q=ctx.Queue())
    for n in itertools.count():
        if args.out:
            out = args.out if n == 0 else f"{args.out}_{n + 1}"
        else:
            name = "" if args.camera else os.path.basename(os.path.normpath(args.frames)) + "_"
            out = os.path.join("output", f"live_{name}{time.strftime('%Y%m%d_%H%M%S')}")
        how = run_session(args, out, ui)
        viewer = ui["viewer"]
        if viewer is None or not viewer.is_alive():
            return
        if how != "restart":
            # The window stays with the result: "new" starts the next session, closing it ends.
            print("New recording in the window for another session; close the window to exit", flush=True)
            try:
                while viewer.is_alive():
                    try:
                        if ui["ui_q"].get(timeout=0.2) == "new":
                            break
                    except queue.Empty:
                        pass
                else:
                    return
            except KeyboardInterrupt:
                viewer.terminate()
                return
        ui["map_q"].put(("reset",))
        gc.collect()  # the last session's nvblox maps and frames
        torch.cuda.empty_cache()


def run_session(args, out, ui):
    """One recording (or replay) into out. Returns "done", or "restart" when the window's Discard
    & restart threw it away (out deleted)."""
    os.makedirs(out, exist_ok=True)
    rec_dir = os.path.join(out, "rec") if args.camera and args.record else None
    source = ("camera", args.serial, args.duration, rec_dir, args.projector) if args.camera \
        else ("replay", args.frames, args.speed)
    ctx = mp.get_context("spawn")
    map_q, cam_q, ui_q, viewer = ui["map_q"], ui["cam_q"], ui["ui_q"], ui["viewer"]
    tracker = Tracker(source, args, cam_q)
    meta = tracker.meta
    st = meta["stereo"]
    if st.get("projector") not in ("alternate", "off"):
        ui["error"]("needs IR pairs without the dot pattern: a projector-alternating recording ("
                    "ir1/ir2 and ir1_on/ir2_on) or one with the projector off")
    args.engine = args.engine or default_engine(st)
    if not os.path.exists(args.engine):
        ui["error"](f"{args.engine} not found: build it with recon/export_ffs_trt.py (--width for 848x480)")
    if ui["ffs"] is None:
        ui["ffs"] = FFS(engine=args.engine)
    ir1_from_color = np.array(st["ir1_from_color"])
    K = intrinsic_matrix(meta)
    sensor = Sensor.from_camera_matrix(torch.tensor(K, dtype=torch.float32), meta["width"], meta["height"])
    # nvblox kernels ahead of FFS on the GPU: at equal priority every one of the many short steps of
    # an integration waits behind TensorRT, and a rebuild during the recording crawls (5 vs 80 frames/s).
    os.environ["NVBLOX_STREAM_PRIORITY"] = str(torch.cuda.Stream.priority_range()[1])
    active = make_mapper(args)  # the map being built live and shown
    # Depth beyond --depth-max kept on the floor (up to --floor-depth-max); the world is Z-up when leveled.
    floor = FloorDepth(K, meta["width"], meta["height"], [0, 0, 1] if meta.get("up_ir1") else None,
                       args.depth_max, args.floor_depth_max, args.floor_band)
    active_table = tracker.T  # the trajectory the active map was built with
    frames_q = queue.Queue()
    depth = Depth(meta, args, ui["ffs"], tracker.req_q, tracker.frames_q, frames_q)
    if args.view and viewer is None:
        viewer = ui["viewer"] = ctx.Process(target=view_process, args=(map_q, cam_q, ui_q, st, bool(meta.get("up_ir1")),
                                                                        args.camera))
        viewer.start()
    if viewer is not None:
        map_q.put(("session", out))
    t_start = time.perf_counter()
    tracker.go()
    depth.start()
    print("camera " + meta["camera"] + " " + meta["serial"] if args.camera else f"replaying {args.frames}",
          f"({st['width']}x{st['height']}), FFS {os.path.basename(args.engine)}"
          + f" -> {out}" + (" - Ctrl+C to stop" if args.camera and not args.duration else ""), flush=True)

    def color_pose(t_ns, table=None):
        T = tracker.pose(t_ns, table)
        return None if T is None else T @ ir1_from_color

    def integrate(mapper, fl, d, c, T):
        d = fl.apply(d, T)  # stored frames keep their full depth: the floor may be found later
        Tt = torch.from_numpy(T.astype(np.float32))
        mapper.add_depth_frame(d, Tt, sensor)
        mapper.add_color_frame(c, Tt, sensor, depth_frame=None if args.color_sphere_trace else d)

    stored = []  # (time, capture wall time, depth, color) of every frame given to the mapper
    built = 0  # pose version the active map was built with
    rebuild = None
    stats = {"integrate_ms": [], "latency_ms": [], "no_pose": 0, "rebuilds": [], "mesh_ms": [], "view_ms": [],
             "chunks": []}
    last_mesh, last_report = time.perf_counter(), time.perf_counter()
    t_end, stopping, discard = None, False, False

    def view_status(done=None, stage=None):
        now = time.perf_counter() - t_start
        if stage is None:
            stage = "done" if done else "trajectory" if stopping or (depth.finished and tracker.finished) \
                else "record"
        last = lambda x: x[-1] if x else 0  # noqa: E731
        return {"t": now, "tracked": len(tracker.ir_ns), "skipped": tracker.skipped, "lag_ms": last(tracker.lag_ms),
                "depth": len(depth.ffs_ms), "depth_hz": len(depth.ffs_ms) / max(now, 1e-3),
                "ffs_ms": last(depth.ffs_ms), "latency_ms": last(stats["latency_ms"]), "map_version": built,
                "stored": len(stored), "stopping": stopping and not done, "done": done, "stage": stage,
                "notes": tracker.notes[-2:],
                "rebuild": f"v{rebuild['version']}: {rebuild['next']}/{len(stored)}" if rebuild else None,
                "events": [(e["kind"], e["t_s"], e["version"]) for e in tracker.events]}

    def show(done=None, stage=None, mesh=None):
        if viewer is not None and viewer.is_alive():  # closed: nothing more to send
            t = time.perf_counter()
            map_q.put(mesh_for_view(active, tracker, view_status(done, stage), stored[-1] if stored else None, mesh))
            stats["view_ms"].append((time.perf_counter() - t) * 1000)

    while True:
        try:
            cmd = None
            try:
                cmd = ui_q.get_nowait() if ui_q is not None else None
            except queue.Empty:
                pass
            if cmd == "level" and not stopping:
                tracker.relevel()
            elif cmd == "discard" and not discard:
                print("discarding this session ...", flush=True)
                discard = True
                if not stopping:
                    tracker.stop()
                    stopping = True
            elif cmd == "stop" and not stopping:
                raise KeyboardInterrupt  # the window's Stop button
            if discard:  # drain until the tracker and the source are done (the camera free again)
                try:
                    frames_q.get(timeout=0.02)
                except queue.Empty:
                    pass
                if depth.finished and tracker.finished and frames_q.empty():
                    break
                continue
            busy = rebuild is not None or tracker.version != built
            try:
                item = frames_q.get_nowait() if busy else frames_q.get(timeout=0.02)
            except queue.Empty:
                item = None
            if item is not None:
                t_ns, wall, d, c = item
                while not tracker.finished and tracker.last_ns() < t_ns:
                    if not tracker.alive():
                        raise SystemExit("ORB-SLAM3 process died")
                    time.sleep(0.001)  # the IR pair after this frame is not tracked yet
                T = color_pose(t_ns)
                t = time.perf_counter()
                if T is None:
                    stats["no_pose"] += 1
                else:
                    floor.update(d, T)
                    integrate(active, floor, d, c, T)
                    stats["integrate_ms"].append((time.perf_counter() - t) * 1000)
                    stats["latency_ms"].append((time.perf_counter() - wall) * 1000)
                stored.append(item)
            elif tracker.version != built or rebuild is not None:
                if rebuild is None or rebuild["version"] != tracker.version:
                    restart = rebuild is not None
                    # Always a new map: nvblox's clear() leaves most blocks of a large map out of later meshes.
                    fl = floor.copy()
                    if any("re-level" in e["kind"] for e in tracker.events if e["version"] > built):
                        # A rigid turn of the world: move the floor estimate with it.
                        k = next((k for k in range(min(len(active_table), len(tracker.T)) - 1, -1, -1)
                                  if active_table[k] is not None and tracker.T[k] is not None), None)
                        if k is not None:
                            fl = floor.moved(tracker.T[k] @ np.linalg.inv(active_table[k]))
                    rebuild = {"map": make_mapper(args), "floor": fl, "version": tracker.version, "table": tracker.T,
                               "next": 0,
                               "start": rebuild["start"] if restart else time.perf_counter(),
                               "restarts": rebuild["restarts"] + 1 if restart else 0}
                t_chunk, n_chunk = time.perf_counter(), 0
                for _ in range(args.rebuild_chunk):
                    if rebuild["next"] >= len(stored):
                        break
                    t_ns, _, d, c = stored[rebuild["next"]]
                    # The table is the corrected trajectory; frames tracked after the correction are
                    # appended to it live, already in the corrected frame.
                    T = color_pose(t_ns, rebuild["table"])
                    if T is not None:
                        integrate(rebuild["map"], rebuild["floor"], d, c, T)
                        n_chunk += 1
                    rebuild["next"] += 1
                stats["chunks"].append(((time.perf_counter() - t_chunk) * 1000, n_chunk, time.perf_counter()))
                if rebuild["next"] >= len(stored) and frames_q.empty():
                    rebuild["map"].update_color_mesh()
                    rebuild_frames = sum(n for _, n, w in stats["chunks"] if w >= rebuild["start"])
                    active, built = rebuild["map"], rebuild["version"]  # the old map is freed
                    floor, active_table = rebuild["floor"], rebuild["table"]
                    ev = next(e for e in tracker.events if e["version"] == built)
                    stats["rebuilds"].append({"version": built, "kind": ev["kind"], "t_s": ev["t_s"],
                                              "frames": len(stored), "integrated": rebuild_frames,
                                              "restarts": rebuild["restarts"],
                                              "s_since_change": time.perf_counter() - ev["wall"],
                                              "s_rebuilding": time.perf_counter() - rebuild["start"]})
                    print(f"[mapper] map rebuilt with trajectory v{built} ({ev['kind']} at {ev['t_s']:.1f}s): "
                          f"{len(stored)} frames, {stats['rebuilds'][-1]['s_since_change']:.1f}s after the change",
                          flush=True)
                    rebuild = None
            elif depth.finished and tracker.finished and frames_q.empty():
                break
            if not tracker.alive() or not (depth.is_alive() or depth.finished):
                raise SystemExit("the tracker or the depth thread died")

            now = time.perf_counter()
            if now - last_mesh >= args.mesh_every:
                last_mesh = now
                active.update_color_mesh()
                stats["mesh_ms"].append((time.perf_counter() - now) * 1000)
                show()
            if now - last_report >= 10:
                last_report = now
                print(f"[{now - t_start:5.1f}s] tracked {len(tracker.ir_ns)} ({tracker.skipped} skipped, "
                      f"lag {tracker.lag_ms[-1] if tracker.lag_ms else 0:.0f} ms), depth {len(depth.ffs_ms)} "
                      f"({depth.skipped} skipped), queue {frames_q.qsize()}, map v{built}"
                      + (f", rebuilding v{rebuild['version']} {rebuild['next']}/{len(stored)}" if rebuild else ""),
                      flush=True)
            if t_end is None and depth.finished and tracker.finished:
                t_end = time.perf_counter()
        except KeyboardInterrupt:
            if not stopping:
                print("stopping: final trajectory and map ...", flush=True)
                tracker.stop()
                stopping = True

    if discard:
        tracker.proc.join(30)
        shutil.rmtree(out, ignore_errors=True)
        print(f"discarded {out}", flush=True)
        return "restart"
    ir_ns = np.array(tracker.ir_ns)
    to_color = [None if T is None else T @ ir1_from_color for T in tracker.T]
    tracked = any(T is not None for T in to_color)
    # the stored frames with the final trajectory
    frame_poses = fill_untracked(interpolate(to_color, ir_ns, [t for t, _, _, _ in stored])) \
        if tracked and stored else None
    t_mesh = time.perf_counter()
    triton = args.final_depth_max > 0 and frame_poses is not None
    if triton:
        show(stage="mesh")
        mesh = final_mesh(args, K, stored, frame_poses)
    else:
        active.update_color_mesh()
        mesh = clean_mesh(active.get_color_mesh().to_open3d(), args.min_cluster)
    o3d.io.write_triangle_mesh(os.path.join(out, "mesh_live.ply"), mesh)
    t_done = time.perf_counter()
    show(stage="save", mesh=mesh)
    if tracked:
        if args.camera:  # the session's frames, for offline steps (and the poses of those frames)
            poses = frame_poses if frame_poses is not None else np.zeros((0, 4, 4))
            save_frames(os.path.join(out, "frames"), meta, stored, poses)
        else:  # poses of every frame of the recording
            poses = fill_untracked(interpolate(to_color, ir_ns, [int(round(f["timestamp_ms"] * 1e6))
                                                                 for f in meta["frames"]]))
        np.save(os.path.join(out, "poses_live.npy"), poses)
        if rec_dir and os.path.exists(os.path.join(rec_dir, "meta.json")):  # written when the source stopped
            with open(os.path.join(rec_dir, "meta.json")) as f:
                rec_ns = [int(round(fr["timestamp_ms"] * 1e6)) for fr in json.load(f)["frames"]]
            np.save(os.path.join(rec_dir, "poses.npy"), fill_untracked(interpolate(to_color, ir_ns, rec_ns)))
    else:
        print("warning: no frame was tracked")
    run_s = (t_end or t_done) - t_start

    def pct(x):
        return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(np.max(x))} \
            if len(x) else None
    report = {
        "source": "camera" if args.camera else args.frames, "run_s": run_s, "speed": args.speed,
        "tracking": {"frames": len(tracker.ir_ns), "skipped": tracker.skipped,
                     "tracked": sum(T is not None for T in tracker.T),
                     "ms": pct(tracker.track_ms), "lag_ms": pct(tracker.lag_ms)},
        "depth": {"frames": len(depth.ffs_ms), "skipped": depth.skipped,
                  "hz": len(depth.ffs_ms) / run_s, "ffs_ms": pct(depth.ffs_ms)},
        "integration": {"frames": len(stats["integrate_ms"]), "no_pose": stats["no_pose"],
                        "ms": pct(stats["integrate_ms"])},
        "latency_ms": pct(stats["latency_ms"]),
        "trajectory_changes": [{k: v for k, v in e.items() if k != "wall"} for e in tracker.events],
        "rebuilds": stats["rebuilds"],
        "mesh_update_ms": pct(stats["mesh_ms"]), "view_update_ms": pct(stats["view_ms"]),
        "s_after_end_to_final_mesh": t_done - (t_end or t_done),
        "final_mesh": {"s": t_done - t_mesh, "triton": triton,
                       "depth_max": args.final_depth_max if triton else args.depth_max,
                       "weight_ref": args.final_weight_ref if triton else None},
        "mesh_triangles": len(mesh.triangles),
        "stored_frames_ms": [t / 1e6 for t, _, _, _ in stored],
    }
    with open(os.path.join(out, "live.json"), "w") as f:
        json.dump(report, f, indent=1)
    def ms(x, key="median"):
        return f"{x[key]:.0f}" if x else "-"
    d, tr, ig = report["depth"], report["tracking"], report["integration"]
    print(f"{run_s:.0f} s: tracking {tr['tracked']}/{tr['frames']} ({tr['skipped']} skipped), "
          f"{ms(tr['ms'])} ms/frame (p90 {ms(tr['ms'], 'p90')}), lag p90 {ms(tr['lag_ms'], 'p90')} ms")
    print(f"depth {d['frames']} framesets ({d['hz']:.1f} Hz, {d['skipped']} skipped), FFS {ms(d['ffs_ms'])} ms; "
          f"integration {ms(ig['ms'])} ms; capture -> in map {ms(report['latency_ms'])} ms "
          f"(p90 {ms(report['latency_ms'], 'p90')})")
    for r in stats["rebuilds"]:
        print(f"rebuild v{r['version']} ({r['kind']} at {r['t_s']:.1f}s): {r['frames']} frames, "
              f"{r['s_since_change']:.1f}s after the change ({r['restarts']} restarts)")
    print(f"mesh update {ms(report['mesh_update_ms'])} ms (max {ms(report['mesh_update_ms'], 'max')}); "
          f"final mesh {report['s_after_end_to_final_mesh']:.1f}s after the end, "
          f"{len(mesh.triangles)} triangles -> {out}/mesh_live.ply"
          + (f"; frames -> {out}/frames" if args.camera and os.path.isdir(os.path.join(out, "frames")) else "")
          + (f"; every frameset -> {rec_dir}" if rec_dir else ""))
    tracker.proc.join(30)  # the camera free for the next session
    if viewer is not None and viewer.is_alive():
        show(done=f"{out}/mesh_live.ply", mesh=mesh)
    return "done"


if __name__ == "__main__":
    main()
