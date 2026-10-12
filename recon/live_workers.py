"""Processes of main.py. nvblox and Open3D hold the GIL for tens of ms, so the tracker in a
thread of the mapper would fall behind and lose track; the frame source with tracking
(track_process: OpenVINS + RTAB-Map, recon/vio.py) and the window (live_view.py) run in their own
processes and use queues.
Child processes ignore SIGINT: Ctrl+C reaches the main process, which asks the source to stop.
"""
import ssl  # noqa: F401  conda's OpenSSL first: recon/vio_py links the older system libcrypto
import json
import os
import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from recon.prof import span, summary

# 848x480 at 60 fps; color exposure, gain and white balance fixed. The IR exposure is short
# (less blur when turning) with the gain set by IrExposure.
CAMERA = {"width": 848, "height": 480, "fps": 60, "ir_exposure_ms": 4, "ir_exposure_max_ms": 8,
          "ir_brightness": 90, "color_exposure_ms": 6, "color_gain": 80, "white_balance": 3900}

class IrExposure:
    """Auto gain at a short IR exposure: every PERIOD frames the mean brightness is moved towards
    the target by the gain; only at full gain does the exposure rise (to max_ms). Changes over 8%
    are sent from a thread of their own: a control transfer takes up to ~40 ms."""
    PERIOD = 15

    def __init__(self, sensor, exposure_ms, max_ms, target):
        import pyrealsense2 as rs
        self.rs, self.s, self.target = rs, sensor, target
        self.exp = self.min_exp = exposure_ms * 1000
        self.max_exp, self.gain, self.n = max_ms * 1000, 64, 0
        self.applied = (self.exp, self.gain)
        self.s.set_option(rs.option.enable_auto_exposure, 0)
        self._apply()
        self.wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def _apply(self, tries=20):
        exp, gain = int(self.exp), int(self.gain)
        for k in range(tries):  # right after the stream starts the camera may refuse for a moment
            try:
                self.s.set_option(self.rs.option.exposure, exp)
                self.s.set_option(self.rs.option.gain, gain)
                self.applied = (exp, gain)
                return
            except RuntimeError:
                if k == tries - 1:
                    raise
                time.sleep(0.05)

    def _worker(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            try:
                self._apply(tries=3)
            except RuntimeError:
                pass  # the next update tries again

    def update(self, ir):
        self.n += 1
        if self.n % self.PERIOD:
            return
        b = max(float(ir[::8, ::8].mean()), 1.0)
        f = min(max((self.target / b) ** 0.7, 0.7), 1.4)
        if f > 1 and self.gain >= 248 and self.exp < self.max_exp:
            self.exp = min(self.exp * f, self.max_exp)
        elif f < 1 and self.exp > self.min_exp:
            self.exp = max(self.exp * f, self.min_exp)
        else:
            self.gain = min(max(self.gain * f, 16), 248)
        e0, g0 = self.applied
        if abs(self.exp / e0 - 1) > 0.08 or abs(self.gain / g0 - 1) > 0.08:
            self.wake.set()


class Latest:
    """Slot holding the newest item; older ones are overwritten (skipped)."""

    def __init__(self):
        self.cond, self.item, self.n, self.closed = threading.Condition(), None, 0, False

    def put(self, item):
        with self.cond:
            self.item, self.n = item, self.n + 1
            self.cond.notify()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify()

    def take(self, last_n):
        """(item, n) newer than last_n, or (None, n) once closed."""
        with self.cond:
            self.cond.wait_for(lambda: self.n > last_n or self.closed)
            return (self.item, self.n) if self.n > last_n else (None, self.n)


def replay_source(frames, meta, w0, off, on, stop):
    """A recording (rec/) at its own clock, from wall time w0 (perf_counter, the same in every process)."""
    ir_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["stereo"]["frames"]]
    color_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["frames"]]
    t0 = min(ir_ns[0], color_ns[0])
    wall = lambda t: w0 + (t - t0) / 1e9  # noqa: E731
    read = lambda d, k: cv2.imread(os.path.join(frames, d, f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE)  # noqa: E731
    for t, is_on, k in sorted([(t, 0, k) for k, t in enumerate(ir_ns)] + [(t, 1, i) for i, t in enumerate(color_ns)]):
        if stop.is_set():
            break
        time.sleep(max(wall(t) - time.perf_counter(), 0))
        if is_on:
            color = cv2.cvtColor(cv2.imread(os.path.join(frames, "color", f"{k:06d}.png")), cv2.COLOR_BGR2RGB)
            on.put((k, t, wall(t), read("ir1_on", k), read("ir2_on", k), color))
        else:
            off.put((t, wall(t), read("ir1", k), read("ir2", k)))
    off.close()
    on.close()


class ImuStream:
    """The D455's IMU, on the frames' clock. between(t_a, t_b) -> Nx7 (t_ns, accel xyz, gyro xyz) with
    the accelerometer interpolated to the gyroscope's times; up() -> gravity's up in the left-IR frame."""
    GYRO_HZ = 200

    def __init__(self, dev, ir1):
        import pyrealsense2 as rs
        self.motion = next(s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == "Motion Module")
        profiles = self.motion.get_stream_profiles()
        accel = max((p for p in profiles if p.stream_type() == rs.stream.accel), key=lambda p: p.fps())
        gyro = min((p for p in profiles if p.stream_type() == rs.stream.gyro), key=lambda p: abs(p.fps() - self.GYRO_HZ))
        e = accel.get_extrinsics_to(ir1)
        self.ir1_from_imu = np.eye(4)
        self.ir1_from_imu[:3, :3] = np.array(e.rotation).reshape(3, 3, order="F")
        self.ir1_from_imu[:3, 3] = e.translation
        self.hz = gyro.fps()
        self.acc, self.gyr, self.rows, self.lock = [], [], [], threading.Lock()
        self._g, self._a = 0, 0  # next gyroscope sample to sync, accelerometer search position

        def on_frame(f):
            m = f.as_motion_frame()
            d = m.get_motion_data()
            with self.lock:
                (self.acc if m.get_profile().stream_type() == rs.stream.accel else self.gyr).append(
                    (int(round(m.get_timestamp() * 1e6)), d.x, d.y, d.z))
        self.motion.open([accel, gyro])
        self.motion.start(on_frame)

    def _sync(self):
        with self.lock:
            acc, gyr = self.acc, self.gyr
            while self._g < len(gyr) and acc and gyr[self._g][0] <= acc[-1][0]:
                t = gyr[self._g][0]
                while self._a + 1 < len(acc) and acc[self._a + 1][0] < t:
                    self._a += 1
                a0, a1 = acc[self._a], acc[min(self._a + 1, len(acc) - 1)]
                w = 0.0 if a1[0] == a0[0] else min(max((t - a0[0]) / (a1[0] - a0[0]), 0.0), 1.0)
                self.rows.append((t, *[(1 - w) * a0[k] + w * a1[k] for k in (1, 2, 3)], *gyr[self._g][1:]))
                self._g += 1

    def between(self, t_a, t_b, wait_s=0.05):
        """Samples with t_a < t <= t_b; waits up to wait_s for those up to t_b."""
        end = time.perf_counter() + wait_s
        while True:
            self._sync()
            if (self.rows and self.rows[-1][0] >= t_b) or time.perf_counter() > end:
                break
            time.sleep(0.002)
        ts = [r[0] for r in self.rows[-2000:]]
        lo = len(self.rows) - len(ts)
        i, j = np.searchsorted(ts, t_a, "right"), np.searchsorted(ts, t_b, "right")
        return np.array(self.rows[lo + i:lo + j], float).reshape(-1, 7)

    def up(self, seconds=0.5):
        """(up in the left-IR frame, accelerometer spread in m/s^2: ~0.02 at rest) over the next seconds."""
        with self.lock:
            n = len(self.acc)
        time.sleep(seconds)
        with self.lock:
            a = np.array([x[1:] for x in self.acc[n:]])
        up = self.ir1_from_imu[:3, :3] @ a.mean(0)
        return up / np.linalg.norm(up), float(np.linalg.norm(np.std(a, axis=0)))

    def save(self, path):
        """imu.csv of a recording; returns meta["imu"]."""
        self._sync()
        with open(path, "w") as f:
            f.write("t_ns,ax,ay,az,gx,gy,gz\n")
            for r in self.rows:
                f.write(f"{r[0]}," + ",".join(f"{x:.6f}" for x in r[1:]) + "\n")
        return {"file": os.path.basename(path), "hz": self.hz, "ir1_from_imu": self.ir1_from_imu.tolist()}

    def stop(self):
        try:
            self.motion.stop()
            self.motion.close()
        except RuntimeError:
            pass


class ImuFile:
    """A recording's imu.csv with ImuStream's between()."""

    def __init__(self, frames, meta):
        self.rows = np.loadtxt(os.path.join(frames, meta["imu"]["file"]), delimiter=",", skiprows=1)
        self.ir1_from_imu = np.array(meta["imu"]["ir1_from_imu"])
        self.hz = meta["imu"]["hz"]

    def between(self, t_a, t_b):
        t = self.rows[:, 0]
        return self.rows[np.searchsorted(t, t_a, "right"):np.searchsorted(t, t_b, "right")]


def open_camera(projector):
    """D455 -> (pipeline, meta in the frames-dir layout with "up_ir1", ImuStream, IrExposure).
    projector "alternate": dots on every other frameset (depth), the others clean (tracking);
    "off": every frameset clean, tracked at 60 Hz and used for depth too."""
    import pyrealsense2 as rs
    c = CAMERA
    pipe, cfg = rs.pipeline(), rs.config()
    for i in (1, 2):
        cfg.enable_stream(rs.stream.infrared, i, c["width"], c["height"], rs.format.y8, c["fps"])
    cfg.enable_stream(rs.stream.color, c["width"], c["height"], rs.format.yuyv, c["fps"])
    profile = pipe.start(cfg)
    dev = profile.get_device()
    stereo = dev.first_depth_sensor()
    stereo.set_option(rs.option.emitter_enabled, int(projector == "alternate"))
    # Manual exposure before alternating: afterwards switching auto exposure off is refused.
    exposure = IrExposure(stereo, c["ir_exposure_ms"], c["ir_exposure_max_ms"], c["ir_brightness"])
    try:  # FW 5.17 reports an error although the projector does alternate
        stereo.set_option(rs.option.emitter_on_off, int(projector == "alternate"))
    except RuntimeError:
        pass
    color = next(s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == "RGB Camera")
    for o, v in ((rs.option.enable_auto_exposure, 0), (rs.option.exposure, c["color_exposure_ms"] * 10),
                 (rs.option.gain, c["color_gain"]), (rs.option.enable_auto_white_balance, 0),
                 (rs.option.white_balance, c["white_balance"]), (rs.option.auto_exposure_priority, 0)):
        color.set_option(o, v)

    def extr(a, b):  # p_b = T @ p_a
        e = a.get_extrinsics_to(b)
        T = np.eye(4)
        T[:3, :3] = np.array(e.rotation).reshape(3, 3, order="F")
        T[:3, 3] = e.translation
        return T.tolist()
    ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    col = profile.get_stream(rs.stream.color).as_video_stream_profile()
    imu = ImuStream(dev, ir1)
    ci, ii = col.get_intrinsics(), ir1.get_intrinsics()
    meta = {"width": ci.width, "height": ci.height, "fx": ci.fx, "fy": ci.fy, "cx": ci.ppx, "cy": ci.ppy,
            "depth_scale": 1000.0, "fps": c["fps"] // 2 if projector == "alternate" else c["fps"], "frames": [],
            "camera": dev.get_info(rs.camera_info.name), "serial": dev.get_info(rs.camera_info.serial_number),
            "up_ir1": imu.up()[0].tolist(),
            "stereo": {"width": ii.width, "height": ii.height, "fx": ii.fx, "fy": ii.fy, "cx": ii.ppx, "cy": ii.ppy,
                       "projector": projector, "ir1_from_ir2": extr(ir2, ir1), "ir1_from_color": extr(col, ir1),
                       "frames": []}}
    return pipe, meta, imu, exposure


class Recorder:
    """Every frameset of a session as a frames dir, for replay: color/, ir1_on/, ir2_on/ (projector
    on, at most 30 Hz), ir1/, ir2/ (projector off), imu.csv, meta.json. PNGs are encoded by a
    low-priority thread pool so they never delay ORB-SLAM3; when it falls behind, whole framesets
    are dropped, so the streams stay numbered together."""

    def __init__(self, out, meta, workers=6, max_pending=90):
        self.out, self.meta = out, meta
        for d in ("color", "ir1", "ir2", "ir1_on", "ir2_on"):
            os.makedirs(os.path.join(out, d), exist_ok=True)
        self.pool = ThreadPoolExecutor(workers, initializer=lambda: os.setpriority(
            os.PRIO_PROCESS, threading.get_native_id(), 10))
        self.sem = threading.Semaphore(max_pending)
        self.on, self.off, self.dropped = [], [], 0

    def _write(self, d, k, img):
        try:
            cv2.imwrite(os.path.join(self.out, d, f"{k:06d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        finally:
            self.sem.release()

    def _put(self, images):
        for n in range(len(images)):
            if not self.sem.acquire(blocking=False):
                for _ in range(n):
                    self.sem.release()
                self.dropped += 1
                return False
        for d, k, img in images:
            self.pool.submit(self._write, d, k, img)
        return True

    def add_on(self, t_ms, color, ir1, ir2):
        k = len(self.on)
        if self._put([("color", k, cv2.cvtColor(color, cv2.COLOR_RGB2BGR)), ("ir1_on", k, ir1),
                      ("ir2_on", k, ir2)]):
            self.on.append({"timestamp_ms": t_ms})

    def add_off(self, t_ms, ir1, ir2):
        if self._put([("ir1", len(self.off), ir1), ("ir2", len(self.off), ir2)]):
            self.off.append({"timestamp_ms": t_ms})

    def close(self, imu):
        self.pool.shutdown(wait=True)
        m = dict(self.meta, frames=self.on, imu=imu.save(os.path.join(self.out, "imu.csv")))
        m["stereo"] = dict(self.meta["stereo"], frames=self.off)
        with open(os.path.join(self.out, "meta.json"), "w") as f:
            json.dump(m, f, indent=1)
        print(f"[recorder] {len(self.on)} color + {len(self.off)} IR framesets -> {self.out}"
              + (f" ({self.dropped} dropped: the disk fell behind)" if self.dropped else ""), flush=True)


def camera_source(pipe, off, on, stop, duration, rec, imu, exposure, alternate):
    """Framesets from the D455 until stop or duration (s): projector-off IR pairs to `off`
    (tracking), projector-on framesets to `on` (depth); without alternation each goes to both."""
    import pyrealsense2 as rs
    mode = rs.frame_metadata_value.frame_emitter_mode
    i, t_end, t_last = 0, time.perf_counter() + duration if duration else None, -1.0
    try:
        while not stop.is_set() and (t_end is None or time.perf_counter() < t_end):
            with span("camera/wait"):
                fs = pipe.wait_for_frames(5000)
            a, b, c = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_color_frame()
            if not (a and b) or not a.supports_frame_metadata(mode):
                continue
            t_ms = a.get_timestamp()
            if t_ms <= t_last:  # the same frameset again
                continue
            t_last = t_ms
            t = int(round(t_ms * 1e6))
            wall = time.perf_counter() - (time.time() * 1000 - t_ms) / 1000
            ir1, ir2 = np.asanyarray(a.get_data()).copy(), np.asanyarray(b.get_data()).copy()
            projector_on = a.get_frame_metadata(mode) if alternate else True
            if not alternate or not projector_on:
                off.put((t, wall, ir1, ir2))
                exposure.update(ir1)
                if rec:
                    rec.add_off(t_ms, ir1, ir2)
            if projector_on:
                if c:
                    with span("camera/yuyv to rgb"):
                        yuyv = np.asanyarray(c.get_data()).view(np.uint8).reshape(c.get_height(), c.get_width(), 2)
                        color = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
                    on.put((i, t, wall, ir1, ir2, color))
                    if rec and (alternate or i % 2 == 0):
                        rec.add_on(t_ms, color, ir1, ir2)
                i += 1
    finally:
        pipe.stop()
        off.close()
        on.close()
        if rec:
            rec.close(imu)
        imu.stop()


def serve_newest(on, req_q, frames_q):
    """For every request, the newest projector-on frameset not sent yet (None at the end): a queue
    fed at 30 Hz would hand the depth thread stale frames."""
    n = 0
    while req_q.get() is not None:
        item, n = on.take(n)
        frames_q.put(item)
        if item is None:
            return

def track_process(source, ctrl_q, out_q, req_q, frames_q, view_q=None):
    """OpenVINS + RTAB-Map (recon/vio.py) on the frames of `source`, in a process of its own.
    source: ("replay", frames dir) or ("camera", duration, rec dir or None, projector). Sends ("ready", meta),
    waits for ("go", w0), then tracks until the source ends or ("stop",): ("pose", t_ns, T, track ms, lag ms,
    skipped) per IR pair, ("traj", version, T list, event) when RTAB-Map's graph moves or the start is filled in,
    ("note", text), ("prof", stage timings), ("done",)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from recon.frames import load_meta
    from recon.vio import IMU_AHEAD_NS, Vio

    pipe = imu = None
    if source[0] == "camera":
        pipe, meta, imu, exposure = open_camera(source[3])
    else:
        meta = load_meta(source[1])
        if meta.get("imu") and os.path.exists(os.path.join(source[1], meta["imu"]["file"])):
            imu = ImuFile(source[1], meta)
    if imu is None:
        out_q.put(("note", "this recording has no imu.csv: OpenVINS needs the IMU"))
        out_q.put(("done",))
        return
    meta.setdefault("imu", {"hz": imu.hz, "ir1_from_imu": imu.ir1_from_imu.tolist()})
    vio = Vio(meta)
    out_q.put(("ready", meta))
    msg = ctrl_q.get()
    if msg[0] != "go":
        return
    off, on, stop = Latest(), Latest(), threading.Event()
    if pipe:
        rec = Recorder(source[2], meta) if source[2] else None
        src = threading.Thread(target=camera_source, daemon=True,
                               args=(pipe, off, on, stop, source[1], rec, imu, exposure, source[3] == "alternate"))
    else:
        src = threading.Thread(target=replay_source, args=(source[1], meta, msg[1], off, on, stop), daemon=True)
    src.start()
    server = threading.Thread(target=serve_newest, args=(on, req_q, frames_q), daemon=True)
    server.start()

    def control():  # ("stop",); ("level", ...) is ORB-SLAM3's: OpenVINS' world is leveled by gravity
        while ctrl_q.get()[0] != "stop":
            pass
        stop.set()
    threading.Thread(target=control, daemon=True).start()

    version, n, t0, t_prev, t_imu, last_view = 0, 0, None, None, None, 0.0
    rot = {"T": None, "t": None, "dps": 0.0}  # turning speed (deg/s, smoothed) for the window
    while True:
        item, n_new = off.take(n)
        if item is None:
            break
        skipped, n = n_new - n - 1, n_new
        t, wall, left, right = item
        if t_prev is not None and t <= t_prev:
            continue
        t0 = t if t0 is None else t0
        start = time.perf_counter()
        with span("track/imu"):
            rows = imu.between(t - 50_000_000 if t_imu is None else t_imu, t + IMU_AHEAD_NS).copy()
        if len(rows):
            t_imu = int(rows[-1, 0])
        rows[:, 0] = (rows[:, 0] - t0) / 1e9
        t_prev = t
        T = vio.track((t - t0) / 1e9, left, right, rows)
        track_ms = (time.perf_counter() - start) * 1000
        out_q.put(("pose", t, T, track_ms, (time.perf_counter() - wall) * 1000, skipped))
        if T is None and len(vio.t) == 90:
            out_q.put(("note", "OpenVINS: hold the camera still, then move it to start tracking"))
        if T is not None:
            if rot["T"] is not None and t > rot["t"]:
                R = rot["T"][:3, :3].T @ T[:3, :3]
                dps = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) / ((t - rot["t"]) / 1e9)
                rot["dps"] += 0.15 * (dps - rot["dps"])
            rot.update(T=T, t=t)
        if view_q is not None and time.perf_counter() - last_view >= 1 / 15:
            last_view = time.perf_counter()
            with span("track/to window"):
                try:
                    view_q.put_nowait(("cam", left, vio.keypoints(), T, track_ms, T is not None, rot["dps"]))
                except queue.Full:
                    pass
        changed = vio.changed()
        if changed:
            kind, T_all = changed
            version += 1
            T_first = next((X for X in T_all if X is not None), None)
            up = (T_first[:3, :3].T @ [0.0, 0, 1]).tolist() if T_first is not None else None
            event = {"kind": kind, "version": version, "t_s": (t - t0) / 1e9, "wall": time.perf_counter(),
                     "up_ir1": up}
            out_q.put(("traj", version, T_all, event))
            print(f"[tracker] {kind} at {event['t_s']:.1f}s: trajectory v{version}", flush=True)
    stop.set()
    src.join()
    vio.finish()
    changed = vio.changed(final=True)
    if changed:  # the last keyframe's loop closure
        version += 1
        out_q.put(("traj", version, changed[1], {"kind": "final", "version": version,
                                                  "t_s": (t_prev - t0) / 1e9 if t0 else 0.0,
                                                  "wall": time.perf_counter(), "up_ir1": None}))
    print(f"[tracker] OpenVINS + RTAB-Map: {vio.loops} loop closures, {vio.proximity} proximity links", flush=True)
    server.join(60)  # until the depth thread has got the end (None)
    out_q.put(("prof", summary()))
    out_q.put(("done",))
