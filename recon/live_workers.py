"""Processes of recon/live.py that must not share a GIL with the mapper.

nvblox and Open3D calls hold the GIL for tens of milliseconds (a mesh update, converting a
mesh for display); ORB-SLAM3 in a thread of the same process then waits for it after every
frame, falls behind, skips frames and loses track. So the frame source with tracking, and the
viewer, run in their own processes and talk to the mapper over queues:
  track_process   frames from a recording replayed at its own clock (replay_source) or from a
                  D455 (camera_source). Projector-off IR pairs -> ORB-SLAM3, the newest one
                  whenever the previous is tracked -> ("pose", t_ns, T, ...) per frame and
                  ("traj", version, T list, event) per correction on ctrl; projector-on
                  framesets (IR pair + color) -> the newest one whenever the depth thread asks.
                  With a D455 every frameset can also be saved (Recorder).
  view_process    the window (recon/live_view.py): mesh, trajectory and current camera in the
                  map, the tracked IR image with its ORB features, depth, status
Child processes ignore SIGINT: Ctrl+C reaches the main process, which asks the source to stop.
"""
import ssl  # noqa: F401  conda's OpenSSL first: ORB-SLAM3 links the older system libcrypto
import json
import os
import queue
import signal
import sys
import tempfile
import threading
import time

import numpy as np

# Camera settings (848x480 at 60 fps, projector alternating, color exposure / gain / white balance
# fixed). The IR exposure is short and fixed with the gain set by IrExposure:
# the D455 (FW 5.17) ignores its auto exposure limit, and its auto exposure sat at 15 ms indoors,
# which blurs a turning camera (100 deg/s: 11 px; 4 ms: 3 px).
CAMERA = {"width": 848, "height": 480, "fps": 60, "ir_exposure_ms": 4, "ir_exposure_max_ms": 8,
          "ir_brightness": 90, "color_exposure_ms": 6, "color_gain": 80, "white_balance": 3900}


class IrExposure:
    """Auto gain at a short, fixed IR exposure: every PERIOD projector-off frames the mean
    brightness is brought towards the target by the gain (16..248); only at full gain does the
    exposure go up (to max_ms), and back down first when it gets bright again. A change is sent
    to the camera from a thread of its own, and only when it is over 8%: each control transfer
    takes up to ~40 ms, and made from the frame loop it held frames up (69 gaps in 8 s)."""
    PERIOD = 15

    def __init__(self, sensor, exposure_ms, max_ms, target):
        import pyrealsense2 as rs
        self.rs, self.s, self.target = rs, sensor, target
        self.exp, self.min_exp, self.max_exp, self.gain, self.n = exposure_ms * 1000, exposure_ms * 1000, max_ms * 1000, 64, 0
        self.applied = (self.exp, self.gain)
        self.s.set_option(rs.option.enable_auto_exposure, 0)
        self._apply()
        self.wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def _apply(self, tries=20):
        # Right after the stream starts the camera may refuse the controls for a moment: try again.
        exp, gain = int(self.exp), int(self.gain)
        for k in range(tries):
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
        f = min(max((self.target / b) ** 0.7, 0.7), 1.4)  # damped: a step at most -30% / +40%
        if f > 1 and self.gain >= 248 and self.exp < self.max_exp:
            self.exp = min(self.exp * f, self.max_exp)
        elif f < 1 and self.exp > self.min_exp:
            self.exp = max(self.exp * f, self.min_exp)
        else:
            self.gain = min(max(self.gain * f, 16), 248)
        e0, g0 = self.applied
        if abs(self.exp / e0 - 1) > 0.08 or abs(self.gain / g0 - 1) > 0.08:
            self.wake.set()


class Clock:
    """Recording time (ns) running at `speed` x wall time from w0 (time.perf_counter, which is
    CLOCK_MONOTONIC and so the same in every process)."""

    def __init__(self, t0_ns, speed, w0=None):
        self.t0, self.speed, self.w0 = t0_ns, speed, w0

    def start(self):
        self.w0 = time.perf_counter()

    def now(self):
        return self.t0 + int((time.perf_counter() - self.w0) * 1e9 * self.speed)

    def wall(self, t_ns):
        """Wall time (perf_counter) at which recording time t_ns is reached."""
        return self.w0 + (t_ns - self.t0) / 1e9 / self.speed

    def wait_until(self, t_ns):
        dt = (t_ns - self.now()) / 1e9 / self.speed
        if dt > 0:
            time.sleep(dt)


class Latest:
    """Slot holding the newest projector-off IR pair; older ones are overwritten (skipped)."""

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
        """(item, n) newer than last_n, or (None, n) once closed and drained."""
        with self.cond:
            self.cond.wait_for(lambda: self.n > last_n or self.closed)
            return (self.item, self.n) if self.n > last_n else (None, self.n)


def replay_source(frames, meta, speed, w0, off, on, stop):
    """A frames dir at its recording clock (from wall time w0)."""
    import cv2
    st = meta["stereo"]
    ir_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in st["frames"]]
    color_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["frames"]]
    clock = Clock(min(ir_ns[0], color_ns[0]), speed, w0)
    events = sorted([(t, 0, k) for k, t in enumerate(ir_ns)] + [(t, 1, i) for i, t in enumerate(color_ns)])
    read = lambda d, k: cv2.imread(os.path.join(frames, d, f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE)  # noqa: E731
    # Without alternation each color frame's IR pair is ir1/ir2 with its number.
    on_dirs = ("ir1_on", "ir2_on") if os.path.isdir(os.path.join(frames, "ir1_on")) else ("ir1", "ir2")
    for t, is_on, k in events:
        if stop.is_set():
            break
        clock.wait_until(t)
        if is_on:
            color = cv2.cvtColor(cv2.imread(os.path.join(frames, "color", f"{k:06d}.png")), cv2.COLOR_BGR2RGB)
            on.put((k, t, clock.wall(t), read(on_dirs[0], k), read(on_dirs[1], k), color))
        else:
            off.put((t, clock.wall(t), read("ir1", k), read("ir2", k)))
    off.close()
    on.close()


class ImuStream:
    """The D455's gyroscope and accelerometer for a whole session (the Motion Module started on its
    own, beside the pipeline). Samples are on the same global clock as the frames. Accelerometer
    readings are interpolated to the gyroscope's times (as realsense-ros unite_imu_method 2):
    between(t_a, t_b) -> Nx7 (t_ns, ax, ay, az, gx, gy, gz) in the IMU frame, for ORB-SLAM3's
    stereo-inertial mode; up() -> gravity's up direction in the left-IR frame."""
    GYRO_HZ = 200

    def __init__(self, dev, ir1):
        import pyrealsense2 as rs
        self.motion = next(s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == "Motion Module")
        profiles = self.motion.get_stream_profiles()
        accel = max((p for p in profiles if p.stream_type() == rs.stream.accel), key=lambda p: p.fps())
        gyro = min((p for p in profiles if p.stream_type() == rs.stream.gyro),
                   key=lambda p: abs(p.fps() - self.GYRO_HZ))
        e = accel.get_extrinsics_to(ir1)
        self.ir1_from_imu = np.eye(4)
        self.ir1_from_imu[:3, :3] = np.array(e.rotation).reshape(3, 3, order="F")
        self.ir1_from_imu[:3, 3] = e.translation
        self.hz = gyro.fps()
        self.acc, self.gyr, self.rows, self.lock = [], [], [], threading.Lock()
        self._g, self._a = 0, 0  # next gyroscope sample to sync, accelerometer search position
        accel_t = rs.stream.accel

        def on_frame(f):
            m = f.as_motion_frame()
            d = m.get_motion_data()
            sample = (int(round(m.get_timestamp() * 1e6)), d.x, d.y, d.z)
            with self.lock:
                (self.acc if m.get_profile().stream_type() == accel_t else self.gyr).append(sample)
        self.motion.open([accel, gyro])
        self.motion.start(on_frame)

    def _sync(self):
        """Gyroscope samples with accelerometer readings on both sides -> self.rows."""
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
        """Samples with t_a < t <= t_b (ns); waits up to wait_s for those up to t_b to arrive."""
        end = time.perf_counter() + wait_s
        while True:
            self._sync()
            if (self.rows and self.rows[-1][0] >= t_b) or time.perf_counter() > end:
                break
            time.sleep(0.002)
        ts = [r[0] for r in self.rows[-2000:]]  # 10 s back is plenty for one frame
        lo = len(self.rows) - len(ts)
        i, j = np.searchsorted(ts, t_a, "right"), np.searchsorted(ts, t_b, "right")
        return np.array(self.rows[lo + i:lo + j], float).reshape(-1, 7)

    def up(self, seconds=0.5, spread=None):
        """Up direction in the left-IR frame from the accelerometer over the next `seconds` (at rest
        it reads the reaction to gravity, i.e. up). With spread (a list), the samples' standard
        deviation (m/s^2, ~0.02 at rest) is appended to it."""
        with self.lock:
            n = len(self.acc)
        time.sleep(seconds)
        with self.lock:
            a = np.array([x[1:] for x in self.acc[n:]])
        if spread is not None:
            spread.append(float(np.linalg.norm(np.std(a, axis=0))))
        up = self.ir1_from_imu[:3, :3] @ a.mean(0)
        return up / np.linalg.norm(up)

    def save(self, path):
        """Every synced sample as a frames dir's imu.csv; returns meta["imu"]."""
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
    """A recording's imu.csv (rec/ of a live session) with ImuStream's between()."""

    def __init__(self, frames, meta):
        self.rows = np.loadtxt(os.path.join(frames, meta["imu"]["file"]), delimiter=",", skiprows=1)
        self.ir1_from_imu = np.array(meta["imu"]["ir1_from_imu"])
        self.hz = meta["imu"]["hz"]

    def between(self, t_a, t_b, wait_s=0):
        t = self.rows[:, 0]
        return self.rows[np.searchsorted(t, t_a, "right"):np.searchsorted(t, t_b, "right")]


IMU_SETTINGS = """
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [{T}]
IMU.InsertKFsWhenLost: 0
IMU.NoiseGyro: 1e-3
IMU.NoiseAcc: 1e-2
IMU.GyroWalk: 1e-6
IMU.AccWalk: 1e-4
IMU.Frequency: {hz:.1f}
"""  # noise and random walks as ORB-SLAM3's RealSense_D435i.yaml (the same Bosch BMI055 IMU)


def rotation_to_z(u):
    """4x4 rotation taking direction u to +Z by the shortest way (keeps the heading)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    axis, c = np.cross(u, [0.0, 0, 1]), u[2]
    s = np.linalg.norm(axis)
    R = np.eye(3)
    if s > 1e-9:
        k = axis / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    return T


def level_transform(up):
    """4x4 from the first camera's frame (x right, y down, z forward) to a Z-up world: Z against
    gravity, X the camera's forward direction made horizontal, Y left (ROS), origin at the camera."""
    z = np.asarray(up, float) / np.linalg.norm(up)
    x = np.array([0.0, 0.0, 1.0]) - z[2] * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z])
    return T


def open_camera(serial=None, depth=False, projector="alternate"):
    """D455 at 848x480, 60 fps, projector alternating -> (pipeline, meta in the frames-dir format, with
    "up_ir1" from the accelerometer, ImuStream, IrExposure). projector: "alternate" (on for every
    other frameset: dots for depth, clean IR for tracking) or "off" (every frameset clean, tracked at
    60 Hz and given to FFS as well). depth: also the D455's own depth stream (for Recorder)."""
    import pyrealsense2 as rs
    c = CAMERA
    pipe, cfg = rs.pipeline(), rs.config()
    if serial:
        cfg.enable_device(serial)
    for i in (1, 2):
        cfg.enable_stream(rs.stream.infrared, i, c["width"], c["height"], rs.format.y8, c["fps"])
    cfg.enable_stream(rs.stream.color, c["width"], c["height"], rs.format.rgb8, c["fps"])
    if depth:
        cfg.enable_stream(rs.stream.depth, c["width"], c["height"], rs.format.z16, c["fps"])
    profile = pipe.start(cfg)
    dev = profile.get_device()
    stereo = dev.first_depth_sensor()
    stereo.set_option(rs.option.emitter_enabled, int(projector == "alternate"))
    # Manual exposure first: with the projector alternating, switching auto exposure off is refused
    # (changing exposure and gain afterwards is not).
    exposure = IrExposure(stereo, c["ir_exposure_ms"], c["ir_exposure_max_ms"], c["ir_brightness"])
    try:  # the D455 (FW 5.17) reports an error here although the projector does alternate
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
        T[:3, :3] = np.array(e.rotation).reshape(3, 3, order="F")  # column-major
        T[:3, 3] = e.translation
        return T.tolist()
    ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    imu = ImuStream(dev, ir1)
    up = imu.up()
    ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    col = profile.get_stream(rs.stream.color).as_video_stream_profile()
    ci, ii = col.get_intrinsics(), ir1.get_intrinsics()
    meta = {"width": ci.width, "height": ci.height, "fx": ci.fx, "fy": ci.fy, "cx": ci.ppx, "cy": ci.ppy,
            "depth_scale": round(1 / stereo.get_depth_scale(), 3), "fps": c["fps"] // 2 if projector == "alternate" else c["fps"], "frames": [], "camera": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number), "up_ir1": up.tolist(),
            "stereo": {"width": ii.width, "height": ii.height, "fx": ii.fx, "fy": ii.fy, "cx": ii.ppx, "cy": ii.ppy,
                       "projector": projector, "ir1_from_ir2": extr(ir2, ir1), "ir1_from_color": extr(col, ir1),
                       "frames": []}}
    return pipe, meta, imu, exposure


class Recorder:
    """Every frameset of a D455 session in the frames-dir layout, so that the offline steps
    (stereo_depth.py, track_orbslam3.py, depth completion, ...) run on it as on a decoded bag:
      color/, depth/ (the D455's depth aligned to color, mm), ir1_on/, ir2_on/   projector on
      ir1/, ir2/                                                                  projector off
    With the projector off every IR pair goes to ir1/ir2 (60 Hz, tracking) and every other
    frameset to color/depth/ir1_on/ir2_on (30 Hz): all of them at 60 Hz took the tracker from
    16 to 21 ms a frame (over the 16.7 ms between frames: 240 of 1127 skipped in 20 s).
      meta.json (written by close())
    PNGs are written by a thread pool (the encoder releases the GIL) at a lower CPU priority
    (nice NICE), so that encoding never delays ORB-SLAM3. When it falls behind, whole framesets
    are dropped and counted, so color/depth/IR stay numbered together."""
    NICE = 10

    def __init__(self, out, meta, workers=6, max_pending=90):
        from concurrent.futures import ThreadPoolExecutor
        import cv2
        self.cv2, self.out, self.meta = cv2, out, meta
        for d in ("color", "depth", "ir1", "ir2", "ir1_on", "ir2_on"):
            os.makedirs(os.path.join(out, d), exist_ok=True)
        # Linux: a thread's nice value is its own (setpriority on its thread id).
        self.pool = ThreadPoolExecutor(workers, initializer=lambda: os.setpriority(
            os.PRIO_PROCESS, threading.get_native_id(), self.NICE))
        self.sem = threading.Semaphore(max_pending)
        self.on, self.off, self.dropped = [], [], 0

    def _write(self, d, k, img):
        try:
            self.cv2.imwrite(os.path.join(self.out, d, f"{k:06d}.png"), img, [self.cv2.IMWRITE_PNG_COMPRESSION, 1])
        finally:
            self.sem.release()

    def _put(self, images):
        if not all(self.sem.acquire(blocking=False) or self._undo(n) for n in range(len(images))):
            self.dropped += 1
            return False
        for d, k, img in images:
            self.pool.submit(self._write, d, k, img)
        return True

    def _undo(self, n):  # release what a frameset got before the pool ran out
        for _ in range(n):
            self.sem.release()
        return False

    def add_on(self, t_ms, color_frame, color, depth, ir1, ir2, color_t_ms):
        """A frameset for color/depth, with its own IR pair in ir1_on/ir2_on (numbered like color)."""
        k = len(self.on)
        if self._put([("color", k, self.cv2.cvtColor(color, self.cv2.COLOR_RGB2BGR)), ("depth", k, depth),
                      ("ir1_on", k, ir1), ("ir2_on", k, ir2)]):
            self.on.append({"timestamp_ms": t_ms, "color_frame_number": color_frame,
                            "depth_dt_ms": abs(color_t_ms - t_ms)})

    def add_off(self, t_ms, ir1, ir2):
        if self._put([("ir1", len(self.off), ir1), ("ir2", len(self.off), ir2)]):
            self.off.append({"timestamp_ms": t_ms})

    def close(self, imu=None):
        self.pool.shutdown(wait=True)
        m = dict(self.meta, frames=self.on, recorded_dropped=self.dropped)
        if imu is not None:
            m["imu"] = imu.save(os.path.join(self.out, "imu.csv"))
        m["stereo"] = dict(self.meta["stereo"], frames=self.off)
        with open(os.path.join(self.out, "meta.json"), "w") as f:
            json.dump(m, f, indent=1)
        print(f"[recorder] {len(self.on)} color/depth + {len(self.off)} IR framesets -> {self.out}"
              + (f" ({self.dropped} dropped: the disk or the encoder fell behind)" if self.dropped else ""),
              flush=True)


def camera_source(pipe, off, on, stop, duration, rec=None, imu=None, exposure=None, alternate=True):
    """Framesets from the D455 until stop or duration (s); emitter state from frame metadata
    (alternate), else every frameset goes to both the tracker and the depth thread.
    Times are the frames' global (system clock) timestamps; capture wall time is mapped from it.
    rec: a Recorder that gets every frameset (with the D455 depth aligned to color)."""
    import pyrealsense2 as rs
    mode = rs.frame_metadata_value.frame_emitter_mode
    align = rs.align(rs.stream.color) if rec else None
    i, t_end, t_last = 0, time.perf_counter() + duration if duration else None, -1.0
    try:
        while not stop.is_set() and (t_end is None or time.perf_counter() < t_end):
            fs = pipe.wait_for_frames(5000)
            a, b, c = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_color_frame()
            if not (a and b) or not a.supports_frame_metadata(mode):
                continue
            t_ms = a.get_timestamp()
            if t_ms <= t_last:  # the same frameset again (seen a few times a minute): nothing new
                continue
            t_last = t_ms
            t = int(round(t_ms * 1e6))
            wall = time.perf_counter() - (time.time() * 1000 - t_ms) / 1000
            ir1, ir2 = np.asanyarray(a.get_data()).copy(), np.asanyarray(b.get_data()).copy()
            projector_on = a.get_frame_metadata(mode) if alternate else True
            if not alternate:
                off.put((t, wall, ir1, ir2))
                if exposure:
                    exposure.update(ir1)
                if rec:
                    rec.add_off(t_ms, ir1, ir2)
            if projector_on:
                if c:
                    color = np.asanyarray(c.get_data()).copy()
                    on.put((i, t, wall, ir1, ir2, color))
                    d = align.process(fs).get_depth_frame() if rec and (alternate or i % 2 == 0) else None
                    if d:
                        rec.add_on(t_ms, c.get_frame_number(), color, np.asanyarray(d.get_data()).copy(), ir1, ir2,
                                   c.get_timestamp())
                i += 1
            else:
                off.put((t, wall, ir1, ir2))
                if exposure:
                    exposure.update(ir1)
                if rec:
                    rec.add_off(t_ms, ir1, ir2)
    finally:
        pipe.stop()
        off.close()
        on.close()
        if rec:
            rec.close(imu)
        if imu:
            imu.stop()


def serve_newest(on, req_q, frames_q):
    """For every request on req_q, the newest projector-on frameset not sent yet (None at the end).
    The depth thread asks when it is free: a queue fed at 30 Hz would hand it stale frames (the
    1.5 MB items wait in the pipe one by one, so the receiver cannot skip to the newest)."""
    n = 0
    while req_q.get() is not None:
        item, n = on.take(n)
        frames_q.put(item)
        if item is None:
            return


def track_process(source, args, ctrl_q, out_q, req_q, frames_q, view_q=None):
    """source: ("replay", frames dir, speed) or ("camera", serial, duration, dir to save every
    frameset in (Recorder) or None). Sends ("ready", meta),
    waits for ("go", w0) on ctrl_q, then runs until the source ends or ("stop",) arrives.
    ("level",) re-levels the world from the D455 accelerometer (the camera held still for 0.5 s):
    the world is turned about its origin so that up is +Z again (heading kept), which is sent as a
    trajectory change; notes for the user go out as ("note", text). With the D455 this also runs
    once tracking starts (retried every second while the camera moves): the reading taken when the
    camera opened is seconds older than the first tracked frame (ORB-SLAM3 loads its vocabulary in
    between), and a camera turned meanwhile tilts the whole world.
    Projector-on framesets go to frames_q on request (serve_newest). view_q (bounded): the tracked
    left image with its ORB features and pose for the window, ~15 Hz, dropped when it is full."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbslam3_py"))
    import orbslam3_py
    from frames import load_meta
    from track_orbslam3 import SETTINGS, drop_outliers

    pipe = imu = None
    if source[0] == "camera":
        pipe, meta, imu, exposure = open_camera(source[1], depth=bool(source[3]), projector=source[4])
    else:
        meta = load_meta(source[1])
        if meta.get("imu") and os.path.exists(os.path.join(source[1], meta["imu"]["file"])):
            imu = ImuFile(source[1], meta)
    inertial = bool(args.get("imu")) and imu is not None
    if args.get("imu") and imu is None:
        out_q.put(("note", "--imu: this recording has no imu.csv, tracking with the cameras only"))
    st = meta["stereo"]
    baseline = float(np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3]))
    settings = SETTINGS.format(fx=st["fx"], fy=st["fy"], cx=st["cx"], cy=st["cy"], width=st["width"],
                               height=st["height"], fps=meta["fps"],
                               baseline=baseline, features=args["features"],
                               ini_fast=args["fast"][0], min_fast=args["fast"][1])
    if inertial:  # ORB-SLAM3's body is the IMU: T_b_c1 = imu_from_ir1
        T = np.linalg.inv(imu.ir1_from_imu)
        settings += IMU_SETTINGS.format(T=", ".join(f"{x:.6f}" for x in T.ravel()), hz=float(imu.hz))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        f.write(settings)
        f.flush()
        slam = orbslam3_py.Stereo(os.path.join(args["orb_root"], "Vocabulary/ORBvoc.txt"), f.name, inertial)
    # Leveled (Z up) from the accelerometer at the start: the camera's, or the recording's up_ir1
    # (meta.json); without it the world is the first camera's frame.
    up = meta.get("up_ir1")
    level = level_transform(up) if up else np.eye(4)
    meta["world"] = "z-up (accelerometer), x forward, origin at the first camera" if up \
        else "first left-IR camera (x right, y down, z forward)"
    out_q.put(("ready", meta))
    msg = ctrl_q.get()
    if msg[0] != "go":
        return
    off, on, stop = Latest(), Latest(), threading.Event()
    if pipe:
        rec = Recorder(source[3], meta) if source[3] else None
        src = threading.Thread(target=camera_source, args=(pipe, off, on, stop, source[2], rec, imu, exposure,
                                     source[4] == "alternate"),
                               daemon=True)
    else:
        src = threading.Thread(target=replay_source, args=(source[1], meta, source[2], msg[1], off, on, stop),
                               daemon=True)
    src.start()
    server = threading.Thread(target=serve_newest, args=(on, req_q, frames_q), daemon=True)
    server.start()
    ir_ns, T_all, version, t0, main_map = [], [], 0, None, None
    # Frames ORB-SLAM3 only carried on the IMU (stereo-inertial, RECENTLY_LOST: no image features):
    # their poses are a few cm off (3 s of black frames on room_1005: 7 cm median, 16 cm max), so the
    # mapper does not get them (None), while tracking goes on without a new map.
    bridged = set()
    releveled = {}  # from control(): "up" (left-IR frame) and "frame" (the IR pair index it was measured at)

    def control():
        while True:
            msg = ctrl_q.get()
            if msg[0] == "stop":
                stop.set()
                return
            if msg[0] == "level":
                if not pipe:
                    out_q.put(("note", "re-level needs the D455 (a replay keeps the recording's up_ir1)"))
                    continue
                retries = msg[1] if len(msg) > 1 else 0
                spread = []
                up = imu.up(spread=spread)
                if spread[0] <= 0.15 and ir_ns:
                    releveled.update(up=up, frame=len(ir_ns) - 1)
                elif retries > 0 and not stop.is_set():
                    if retries % 5 == 0:
                        out_q.put(("note", "leveling: hold the camera still for a moment"))
                    threading.Timer(1.0, ctrl_q.put, (("level", retries - 1),)).start()
                else:
                    out_q.put(("note", f"re-level: the camera moved ({spread[0]:.2f} m/s^2), hold it still"))
    threading.Thread(target=control, daemon=True).start()
    if pipe:  # level again now that tracking starts (see the docstring)
        threading.Timer(0.5, ctrl_q.put, (("level", 30),)).start()

    def correct(kind):
        """Re-read the trajectory. Only the map with the most frames is kept: after tracking is
        lost ORB-SLAM3 starts a new map in a frame of its own, until it merges the two."""
        nonlocal T_all, version, main_map
        t, P, maps = slam.trajectory()
        cur = int(np.bincount(maps).argmax()) if len(maps) else slam.current_map_id()
        main_map = cur
        by_us = {int(round(x * 1e6)): T for x, T, m in zip(t, P, maps) if m == cur}
        new = [by_us.get((ns - t0) // 1000) for ns in ir_ns]
        T_all, bad = drop_outliers([None if T is None or k in bridged else level @ T for k, T in enumerate(new)],
                                   np.array(ir_ns), args["outlier_cm"])
        version += 1
        event = {"kind": kind, "version": version, "t_s": (ir_ns[-1] - t0) / 1e9, "wall": time.perf_counter(),
                 "map": cur, "state": list(slam.state()), "up_ir1": meta.get("up_ir1"),
                 "frames": len(by_us), "other_maps": int(sum(m != cur for m in maps)), "outliers": len(bad)}
        out_q.put(("traj", version, T_all, event))
        print(f"[tracker] {kind} at {event['t_s']:.1f}s: trajectory v{version} ({len(by_us)} frames, "
              f"{len(bad)} outliers)" + (f", {event['other_maps']} frames in other maps dropped"
                                        if event["other_maps"] else ""), flush=True)

    state, recheck, n, last_view = slam.state(), None, 0, 0.0
    rot = {"T": None, "t": None, "dps": 0.0}  # turning speed (deg/s, smoothed over ~0.2 s) for the window
    t_prev, imu_ready = None, False

    def first_up():
        """Up in the first tracked camera, for meta.json's up_ir1 (the recorder writes it at the end)."""
        T_first = next((T for T in T_all if T is not None), None)
        if T_first is not None:
            meta["up_ir1"] = (T_first[:3, :3].T @ [0.0, 0, 1]).tolist()
    while True:
        item, n_new = off.take(n)
        if item is None:
            break
        skipped, n = n_new - n - 1, n_new
        t, wall, left, right = item
        if t_prev is not None and t <= t_prev:  # no time between the frames: nothing to track (or integrate)
            continue
        t0 = t if t0 is None else t0
        start = time.perf_counter()
        # Microseconds since the first frame: absolute ns do not fit a double, and the
        # trajectory's times map back to these keys exactly.
        imu_rows = None
        if inertial:  # the IMU samples since the last tracked frame (all of them, also over skipped frames)
            imu_rows = imu.between(t - 33_000_000 if t_prev is None else t_prev, t).copy()
            imu_rows[:, 0] = (imu_rows[:, 0] - t0) / 1e9
        t_prev = t
        ok, T = slam.track(left, right, (t - t0) // 1000 * 1e-6, imu_rows)
        track_ms = (time.perf_counter() - start) * 1000
        ir_ns.append(t)
        # A pose in another map than the kept one is in a frame of its own: not used until a merge.
        in_main = main_map is None or slam.current_map_id() == main_map
        imu_only = ok and inertial and slam.tracking_state() == 3  # Tracking::RECENTLY_LOST
        if inertial and not ok and not any(T is not None for T in T_all[-90:]) and len(ir_ns) % 90 == 0:
            out_q.put(("note", "IMU mode: move the camera to start tracking (it waits for some motion)"))
        if imu_only:
            bridged.add(len(ir_ns) - 1)
        T_all.append(level @ T if ok and in_main and not imu_only else None)
        out_q.put(("pose", t, T_all[-1], track_ms, (time.perf_counter() - wall) * 1000, skipped))
        if ok:
            if rot["T"] is not None and t > rot["t"]:
                R = rot["T"][:3, :3].T @ T[:3, :3]
                dps = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) / ((t - rot["t"]) / 1e9)
                rot["dps"] += 0.15 * (dps - rot["dps"])
            rot.update(T=T, t=t)
        if view_q is not None and time.perf_counter() - last_view >= 1 / 15:
            last_view = time.perf_counter()
            try:
                view_q.put_nowait(("cam", left, slam.keypoints(), T_all[-1], track_ms,
                                   "imu" if imu_only else ok, rot["dps"]))
            except queue.Full:
                pass
        if inertial and imu_ready and not slam.imu_initialized():
            # ORB-SLAM3 dropped the map (after its IMU initialization the camera barely moved: "Not
            # enough motion for initializing"); the new one is in a frame of its own until the IMU
            # is initialized again. The frames before are lost to the map.
            imu_ready = False
            correct("map reset (move the camera for the IMU)")
        if inertial and not imu_ready and slam.imu_initialized():
            # ORB-SLAM3 has estimated gravity and turned its world to it (Z up): use that world as is.
            imu_ready, level = True, np.eye(4)
            correct("IMU initialized")
            first_up()
        if releveled:
            up, k = releveled.pop("up"), releveled.pop("frame")
            T_k = next((T for T in T_all[k::-1] if T is not None), None) if k >= 0 else None
            if imu_ready:
                out_q.put(("note", "re-level: not needed, ORB-SLAM3 keeps the world level from the IMU"))
            elif T_k is None:
                out_q.put(("note", "re-level: no tracked pose yet"))
            else:
                # Up measured in the camera, expressed in the world: turn the world so that it is +Z.
                level = rotation_to_z(T_k[:3, :3] @ up) @ level
                correct("re-level (IMU)")
                first_up()
        new_state = slam.state()
        if new_state != state:
            names = ("loop / global BA", "map merge", "new map")
            correct(", ".join(n_ for n_, a, b in zip(names, state, new_state) if a != b))
            state, recheck = new_state, len(ir_ns) + 15  # the map may still be moving: look again
        elif recheck is not None and len(ir_ns) >= recheck:
            correct("after the change")
            recheck = None
    if ir_ns:
        correct("final")
    stop.set()
    src.join()
    slam.shutdown()
    server.join(60)  # until the depth thread has asked once more and got the end (None)
    out_q.put(("done",))
