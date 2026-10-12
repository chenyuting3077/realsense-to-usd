"""Live reconstruction from a D455 (or a replayed rec/): ORB-SLAM3 stereo + IMU poses (own process),
Fast-FoundationStereo depth (thread), nvblox TSDF (main). A loop closure or another trajectory
correction re-integrates every stored frame into a new map in the background, swapped in when it
has caught up. At the end the final trajectory gives the final mesh (reconstruct.py's TSDF).

Output in -o: mesh_live.ply, live.json (timings), frames/ (color, depth_ffs/ in mm, meta.json,
poses.npy: the input of gs/offline.py), rec/ (every frameset, for replay).
In the window: Stop ends a session, New recording starts the next (-o gets _2, _3, ...),
Discard & restart throws the running one away, 3DGS -> USD runs gs/offline.py on the last one.

Usage: python recon/live.py [-o output/live_room] [--duration 120]
       python recon/live.py output/live_room/rec      # replay
"""
import argparse
import gc
import itertools
import json
import multiprocessing as mp
import os

os.environ.setdefault("GLOG_minloglevel", "1")  # nvblox's info logs
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d
import torch
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, MeshIntegratorParams, ProjectiveIntegratorParams
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor

from ffs import FFS, DisparityToColorDepth
from frames import FloorDepth, fill_untracked, filter_depth, interpolate, intrinsic_matrix
from live_view import view_process
from live_workers import track_process
from reconstruct import WeightedTSDF

VOXEL = 0.01
DEPTH_MAX, FLOOR_DEPTH_MAX = 2.5, 4.0  # the live map; on the floor depth is kept farther (FloorDepth)
REBUILD_CHUNK = 8  # stored frames re-integrated between two checks for a live frame
MESH_EVERY = 1.0  # s between mesh updates
OFFLINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gs", "offline.py")
sys.path.insert(0, os.path.dirname(OFFLINE))
from offline import estimate  # noqa: E402


def make_mapper():
    # Free space is cleared only 16 voxels in front of the surface (patches/nvblox-thor.patch):
    # along the whole ray noisy views erase surfaces seen well from elsewhere.
    os.environ["NVBLOX_FREE_SPACE_VOX"] = "16"
    integrator = ProjectiveIntegratorParams()
    integrator.projective_integrator_max_integration_distance_m = FLOOR_DEPTH_MAX
    integrator.projective_integrator_max_weight = 1e6  # average every frame (nvblox's 5 follows moving things)
    mesh = MeshIntegratorParams()
    mesh.mesh_integrator_min_weight = 3.0  # no surfaces seen once (edge noise)
    params = MapperParams()
    params.set_projective_integrator_params(integrator)
    params.set_mesh_integrator_params(mesh)
    return Mapper(voxel_sizes_m=[VOXEL], integrator_types=[ProjectiveIntegratorType.TSDF], mapper_parameters=params)


class Tracker(threading.Thread):
    """live_workers.track_process seen from here: ir_ns / T are the times and left-IR camera-to-world
    poses (None: not tracked) so far, `version` the number of trajectory corrections."""

    def __init__(self, source, imu, view_q=None):
        super().__init__(daemon=True)
        self.ir_ns, self.T, self._ns = [], [], np.zeros(0, np.int64)
        self.version, self.finished = 0, False
        self.track_ms, self.lag_ms, self.events, self.skipped, self.notes = [], [], [], 0, []
        ctx = mp.get_context("spawn")
        self.ctrl_q, self.q, self.req_q, self.frames_q = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=track_process, daemon=True,
                                args=(source, imu, self.ctrl_q, self.q, self.req_q, self.frames_q, view_q))
        self.proc.start()
        while True:
            try:
                msg = self.q.get(timeout=1)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise SystemExit("ORB-SLAM3 / the camera did not start")
                continue
            if msg[0] == "ready":
                break
            print(f"[tracker] {msg[1]}", flush=True)
            self.notes.append(msg[1])
        self.meta = msg[1]

    def go(self):
        self.ctrl_q.put(("go", time.perf_counter()))
        self.start()

    def stop(self):
        self.ctrl_q.put(("stop",))

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
                self.events.append(event)  # before the version: the mapper looks the event up by it
                self.T, self.version = T + self.T[len(T):], version
                if event.get("up_ir1"):
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
        """Left-IR camera-to-world at t_ns (from `table`, default the latest trajectory), or None."""
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
    """FFS on the newest projector-on frameset -> filtered depth (m) in the color camera."""

    def __init__(self, meta, ffs, req_q, in_q, out_q):
        super().__init__(daemon=True)
        self.req_q, self.in_q, self.q, self.ffs = req_q, in_q, out_q, ffs
        self.dev = torch.device("cuda")
        self.to_color = DisparityToColorDepth(meta, self.dev)
        self.ffs_ms, self.skipped, self.finished = [], 0, False
        self.stream = torch.cuda.Stream()  # syncing it does not wait for nvblox
        z = torch.zeros(meta["stereo"]["height"], meta["stereo"]["width"], dtype=torch.uint8, device=self.dev)
        with torch.cuda.stream(self.stream):
            self.ffs(z, z)  # warm up
        self.stream.synchronize()

    def run(self):
        with torch.cuda.stream(self.stream):
            last = -1
            while True:
                self.req_q.put(1)
                item = self.in_q.get()
                if item is None:
                    break
                i, t_ns, wall, ir1, ir2, color = item
                self.skipped += i - last - 1
                last = i
                t = time.perf_counter()
                disp = self.ffs(torch.as_tensor(ir1, device=self.dev), torch.as_tensor(ir2, device=self.dev))
                # Wait here, where the GIL is released: the warp's indexing would sync holding it.
                self.stream.synchronize()
                depth = filter_depth(self.to_color(disp), as_tensor=True)
                color = torch.from_numpy(color).to(self.dev)
                self.stream.synchronize()  # nvblox reads them on its own stream
                self.ffs_ms.append((time.perf_counter() - t) * 1000)
                self.q.put((t_ns, wall, depth, color))
        self.finished = True


def view_map(mapper, tracker, status, frame, mesh=None):
    """The window's map message: mesh, camera centers, status, newest (color, depth) at half size."""
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


def save_frames(out, meta, stored, poses):
    """The frames given to the mapper as a frames dir, with their final poses."""
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


def export_usd(out, map_q):
    """gs/offline.py on a session, its progress lines to the window."""
    p = subprocess.Popen([sys.executable, OFFLINE, out], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.startswith("progress "):
            _, frac, left, text = line.rstrip("\n").split(" ", 3)
            left = float(left)
            map_q.put(("progress", float(frac), f"{text}, " + (f"about {left / 60:.0f} min left" if left >= 90
                                                               else "under 2 min left"), False))
        else:
            print(line, end="", flush=True)
    ok = p.wait() == 0
    map_q.put(("progress", 1.0, f"saved: {out}/scene.usdc", True) if ok else
              ("progress", 0.0, "3DGS -> USD failed, see the terminal", True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", nargs="?", help="rec/ of a session to replay (default: live from the D455)")
    p.add_argument("-o", "--out", help="output dir (default: output/live_<date>_<time>)")
    p.add_argument("--duration", type=float, help="seconds to record (default: until Stop / Ctrl+C)")
    p.add_argument("--projector", choices=["alternate", "off"], default="alternate",
                   help="alternate: dots on every other frameset for depth, the others tracked at 30 Hz; "
                        "off: every frameset tracked at 60 Hz and used for depth")
    p.add_argument("--no-record", dest="record", action="store_false", help="do not save rec/")
    p.add_argument("--no-imu", dest="imu", action="store_false", help="track with the cameras only")
    p.add_argument("--no-view", dest="view", action="store_false", help="no window")
    args = p.parse_args()
    args.view = args.view and bool(os.environ.get("DISPLAY"))
    if args.frames and not os.path.isdir(os.path.join(args.frames, "ir1_on")):
        p.error(f"{args.frames} is not a rec/ of a live session (no ir1_on/)")
    ctx = mp.get_context("spawn")
    ui = {"viewer": None, "ffs": None, "map_q": None, "cam_q": None, "ui_q": None}
    if args.view:
        ui.update(map_q=ctx.Queue(), cam_q=ctx.Queue(maxsize=2), ui_q=ctx.Queue())
    export = None
    try:
        for n in itertools.count():
            if args.out:
                out = args.out if n == 0 else f"{args.out}_{n + 1}"
            else:
                out = os.path.join("output", f"live_{time.strftime('%Y%m%d_%H%M%S')}")
            how = run_session(args, out, ui)
            viewer = ui["viewer"]
            if viewer is None or not viewer.is_alive():
                return
            if how != "restart":
                print("New recording in the window for another session; close the window to exit", flush=True)
                while viewer.is_alive():
                    try:
                        cmd = ui["ui_q"].get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if cmd == "usd" and (export is None or not export.is_alive()):
                        export = threading.Thread(target=export_usd, args=(out, ui["map_q"]), daemon=True)
                        export.start()
                    elif cmd == "new":
                        break
                else:
                    return
            ui["map_q"].put(("reset",))
            gc.collect()  # the last session's maps and frames
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        if ui["viewer"] is not None:
            ui["viewer"].terminate()
    finally:
        if export is not None and export.is_alive():
            print("waiting for the USD export to finish ...", flush=True)
            export.join()


def run_session(args, out, ui):
    """One recording (or replay) into out. Returns "done", or "restart" after Discard & restart."""
    os.makedirs(out, exist_ok=True)
    camera = not args.frames
    rec_dir = os.path.join(out, "rec") if camera and args.record else None
    source = ("camera", args.duration, rec_dir, args.projector) if camera else ("replay", args.frames)
    map_q, cam_q, ui_q, viewer = ui["map_q"], ui["cam_q"], ui["ui_q"], ui["viewer"]
    tracker = Tracker(source, args.imu, cam_q)
    meta = tracker.meta
    st = meta["stereo"]
    if ui["ffs"] is None:
        ui["ffs"] = FFS()
    ir1_from_color = np.array(st["ir1_from_color"])
    K = intrinsic_matrix(meta)
    sensor = Sensor.from_camera_matrix(torch.tensor(K, dtype=torch.float32), meta["width"], meta["height"])
    # nvblox kernels ahead of FFS on the GPU: else every short integration step waits behind TensorRT
    os.environ["NVBLOX_STREAM_PRIORITY"] = str(torch.cuda.Stream.priority_range()[1])
    active = make_mapper()  # the map being built live and shown
    floor = FloorDepth(K, meta["width"], meta["height"], [0, 0, 1] if meta.get("up_ir1") else None,
                       DEPTH_MAX, FLOOR_DEPTH_MAX)
    active_table = tracker.T  # the trajectory the active map was built with
    frames_q = queue.Queue()
    depth = Depth(meta, ui["ffs"], tracker.req_q, tracker.frames_q, frames_q)
    if args.view and viewer is None:
        viewer = ui["viewer"] = mp.get_context("spawn").Process(
            target=view_process, args=(map_q, cam_q, ui_q, st, bool(meta.get("up_ir1"))))
        viewer.start()
    if viewer is not None:
        map_q.put(("session", out))
    t_start = time.perf_counter()
    tracker.go()
    depth.start()
    print(f"camera {meta['camera']} {meta['serial']}" if camera else f"replaying {args.frames}",
          f"-> {out}" + (" - Ctrl+C to stop" if camera and not args.duration else ""), flush=True)

    def color_pose(t_ns, table=None):
        T = tracker.pose(t_ns, table)
        return None if T is None else T @ ir1_from_color

    def integrate(mapper, fl, d, c, T):
        d = fl.apply(d, T)  # stored frames keep their full depth: the floor may be found later
        Tt = torch.from_numpy(T.astype(np.float32))
        mapper.add_depth_frame(d, Tt, sensor)
        mapper.add_color_frame(c, Tt, sensor, depth_frame=d)  # voxels to color from the depth (patch)

    stored = []  # (time, capture wall time, depth, color) of every frame given to the mapper
    built = 0  # pose version of the active map
    rebuild = None
    stats = {"integrate_ms": [], "latency_ms": [], "rebuilds": []}
    last_mesh, last_report = time.perf_counter(), time.perf_counter()
    t_end, stopping, discard = None, False, False

    def status(done=None, stage=None):
        now = time.perf_counter() - t_start
        if stage is None:
            stage = "done" if done else "trajectory" if stopping or (depth.finished and tracker.finished) else "record"
        last = lambda x: x[-1] if x else 0  # noqa: E731
        return {"t": now, "tracked": len(tracker.ir_ns), "skipped": tracker.skipped, "lag_ms": last(tracker.lag_ms),
                "depth": len(depth.ffs_ms), "depth_hz": len(depth.ffs_ms) / max(now, 1e-3),
                "ffs_ms": last(depth.ffs_ms), "latency_ms": last(stats["latency_ms"]), "map_version": built,
                "stored": len(stored), "stopping": stopping and not done, "done": done, "stage": stage,
                "notes": tracker.notes[-2:],
                "rebuild": f"v{rebuild['version']}: {rebuild['next']}/{len(stored)}" if rebuild else None,
                "events": [(e["kind"], e["t_s"], e["version"]) for e in tracker.events]}

    def show(done=None, stage=None, mesh=None):
        if viewer is not None and viewer.is_alive():
            map_q.put(view_map(active, tracker, status(done, stage), stored[-1] if stored else None, mesh))

    while True:
        try:
            cmd = None
            try:
                cmd = ui_q.get_nowait() if ui_q is not None else None
            except queue.Empty:
                pass
            if cmd == "discard" and not discard:
                print("discarding this session ...", flush=True)
                discard = True
                if not stopping:
                    tracker.stop()
                    stopping = True
            elif cmd == "stop" and not stopping:
                raise KeyboardInterrupt
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
                if T is not None:
                    floor.update(d, T)
                    integrate(active, floor, d, c, T)
                    stats["integrate_ms"].append((time.perf_counter() - t) * 1000)
                    stats["latency_ms"].append((time.perf_counter() - wall) * 1000)
                stored.append(item)
            elif busy:
                if rebuild is None or rebuild["version"] != tracker.version:
                    # Always a new map: nvblox's clear() leaves most blocks of a large map out of later meshes.
                    fl = floor.copy()
                    if any("re-level" in e["kind"] for e in tracker.events if e["version"] > built):
                        k = next((k for k in range(min(len(active_table), len(tracker.T)) - 1, -1, -1)
                                  if active_table[k] is not None and tracker.T[k] is not None), None)
                        if k is not None:  # the world turned: move the floor estimate with it
                            fl = floor.moved(tracker.T[k] @ np.linalg.inv(active_table[k]))
                    rebuild = {"map": make_mapper(), "floor": fl, "version": tracker.version, "table": tracker.T,
                               "next": 0, "start": rebuild["start"] if rebuild else time.perf_counter()}
                for _ in range(REBUILD_CHUNK):
                    if rebuild["next"] >= len(stored):
                        break
                    t_ns, _, d, c = stored[rebuild["next"]]
                    # frames tracked after the correction are appended to its table, already corrected
                    T = color_pose(t_ns, rebuild["table"])
                    if T is not None:
                        integrate(rebuild["map"], rebuild["floor"], d, c, T)
                    rebuild["next"] += 1
                if rebuild["next"] >= len(stored) and frames_q.empty():
                    rebuild["map"].update_color_mesh()
                    active, built = rebuild["map"], rebuild["version"]
                    floor, active_table = rebuild["floor"], rebuild["table"]
                    ev = next(e for e in tracker.events if e["version"] == built)
                    stats["rebuilds"].append({"version": built, "kind": ev["kind"], "t_s": ev["t_s"],
                                              "frames": len(stored), "s_after_change": time.perf_counter() - ev["wall"]})
                    print(f"[mapper] map rebuilt with trajectory v{built} ({ev['kind']} at {ev['t_s']:.1f}s): "
                          f"{len(stored)} frames, {stats['rebuilds'][-1]['s_after_change']:.1f}s after the change",
                          flush=True)
                    rebuild = None
            elif depth.finished and tracker.finished and frames_q.empty():
                break
            if not tracker.alive() or not (depth.is_alive() or depth.finished):
                raise SystemExit("the tracker or the depth thread died")

            now = time.perf_counter()
            if now - last_mesh >= MESH_EVERY:
                last_mesh = now
                active.update_color_mesh()
                show()
            if now - last_report >= 10:
                last_report = now
                print(f"[{now - t_start:5.1f}s] tracked {len(tracker.ir_ns)} ({tracker.skipped} skipped), "
                      f"depth {len(depth.ffs_ms)} ({depth.skipped} skipped), map v{built}", flush=True)
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
    to_color = [None if T is None else T @ ir1_from_color for T in tracker.T]
    poses = interpolate(to_color, np.array(tracker.ir_ns), [t for t, _, _, _ in stored])
    if not any(T is not None for T in poses):
        print("warning: no stored frame was tracked, nothing saved", flush=True)
        tracker.proc.join(30)
        if viewer is not None and viewer.is_alive():
            show(done="Nothing tracked, nothing saved")
            map_q.put(("estimate", None))
        return "done"
    poses = fill_untracked(poses)
    show(stage="mesh")
    t_mesh = time.perf_counter()
    tsdf = WeightedTSDF()
    for (_, _, d, c), T in zip(stored, poses):
        tsdf.integrate(d, c, K, T, 1.0)  # depth in meters, on the GPU
    mesh = tsdf.mesh()
    del tsdf
    torch.cuda.empty_cache()
    o3d.io.write_triangle_mesh(os.path.join(out, "mesh_live.ply"), mesh)
    t_done = time.perf_counter()
    show(stage="save", mesh=mesh)
    save_frames(os.path.join(out, "frames"), meta, stored, poses)

    def pct(x):
        return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90))} if len(x) else None
    run_s = (t_end or t_done) - t_start
    report = {
        "run_s": run_s,
        "tracking": {"frames": len(tracker.ir_ns), "skipped": tracker.skipped,
                     "tracked": sum(T is not None for T in tracker.T), "ms": pct(tracker.track_ms)},
        "depth": {"frames": len(depth.ffs_ms), "hz": len(depth.ffs_ms) / run_s, "ffs_ms": pct(depth.ffs_ms)},
        "integrate_ms": pct(stats["integrate_ms"]), "latency_ms": pct(stats["latency_ms"]),
        "rebuilds": stats["rebuilds"], "final_mesh_s": t_done - t_mesh, "mesh_triangles": len(mesh.triangles),
    }
    with open(os.path.join(out, "live.json"), "w") as f:
        json.dump(report, f, indent=1)
    tr, d, lat = report["tracking"], report["depth"], report["latency_ms"]
    print(f"{run_s:.0f} s: tracked {tr['tracked']}/{tr['frames']}, {tr['ms']['median']:.0f} ms/frame; "
          f"depth {d['frames']} frames ({d['hz']:.1f} Hz), FFS {d['ffs_ms']['median']:.0f} ms; "
          f"capture -> map {lat['median'] if lat else 0:.0f} ms; final mesh {len(mesh.triangles)} triangles "
          f"in {t_done - t_mesh:.1f}s -> {out}", flush=True)
    tracker.proc.join(30)  # the camera free for the next session
    if viewer is not None and viewer.is_alive():
        show(done=f"Saved {out}/mesh_live.ply", mesh=mesh)
        map_q.put(("estimate", estimate(os.path.join(out, "frames"))))
    return "done"


if __name__ == "__main__":
    main()
