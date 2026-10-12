"""Live tracking with OpenVINS (stereo + IMU VIO, every IR pair) and RTAB-Map (loop closures and pose graph
on OpenVINS' odometry, a keyframe every KEYFRAME_S in a thread of its own), for the tracker process.

OpenVINS needs, on the D455 (docs/vslam): no online camera-IMU calibration (it diverged with it), zero-velocity
updates (it drifted while held still without them), a lower jerk threshold for its static initialization (a
hand-held start is gentle). Before it initializes there is no pose: the camera is still then (that is what
the initialization waits for), so the IR pairs during which the IMU says so get the first pose afterwards.

World: Z up (OpenVINS' gravity), the first pose's camera at the origin looking along +X.
"""
import os
import re
import sys
import tempfile
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vio_py"))
import vio_py  # noqa: E402

from recon.prof import span  # noqa: E402

OPENVINS_CONFIG = os.path.expanduser("~/open_vins/config/rs_d455")
KEYFRAME_S = 0.5  # RTAB-Map: 2 nodes per second (110-200 ms each on the Thor's CPU)
MOVING_GYRO, MOVING_ACCEL = 0.1, 0.5  # rad/s, m/s^2 spread: the camera moved (not bumps of the table under it)
WINDOW_S = 0.25  # the IMU judged in windows this long
IMU_AHEAD_NS = 10_000_000  # IMU past each IR pair first: OpenVINS propagates to the pair (as its ROS node waits)
JERK_S = 2.0  # OpenVINS initializes up to this long after the IMU first sees motion (1.4 s on a recording)
# The whole trajectory goes to main.py (which rebuilds the map on it) when the graph moved a node past
# MOVE_CM / MOVE_DEG, at most every UPDATE_S: RTAB-Map links a revisited place every keyframe. New poses always
# get the latest correction.
MOVE_CM, MOVE_DEG, UPDATE_S = 2.0, 1.0, 5.0
ODOM_VAR = [1e-4] * 3 + [1e-5] * 3  # odometry links' variances for RTAB-Map (m^2, rad^2)
RTABMAP_PARAMS = {
    "Mem/IncrementalMemory": "true", "RGBD/OptimizeMaxError": "3.0", "Vis/MinInliers": "20",
    "Reg/Force3DoF": "false", "RGBD/LinearUpdate": "0.05", "RGBD/AngularUpdate": "0.05",
    "Stereo/MaxDisparity": "200",
}


def openvins_config(meta, out):
    """estimator_config.yaml + kalibr chains for this D455 in out (from OpenVINS' rs_d455)."""
    st = meta["stereo"]
    imu_from_ir1 = np.linalg.inv(np.array(meta["imu"]["ir1_from_imu"]))
    cams = ""
    for k, T in enumerate((imu_from_ir1, imu_from_ir1 @ np.array(st["ir1_from_ir2"]))):
        rows = "\n".join(f"    - [{', '.join(f'{x:.9f}' for x in r)}]" for r in T)
        cams += (f"cam{k}:\n  T_imu_cam:\n{rows}\n  cam_overlaps: [{1 - k}]\n  camera_model: pinhole\n"
                 f"  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n  distortion_model: radtan\n"
                 f"  intrinsics: [{st['fx']}, {st['fy']}, {st['cx']}, {st['cy']}]\n"
                 f"  resolution: [{st['width']}, {st['height']}]\n  rostopic: /cam{k}/image_raw\n"
                 f"  timeshift_cam_imu: 0.0\n")
    with open(os.path.join(out, "kalibr_imucam_chain.yaml"), "w") as f:
        f.write("%YAML:1.0\n\n" + cams)
    imu = open(os.path.join(OPENVINS_CONFIG, "kalibr_imu_chain.yaml")).read()
    imu = re.sub(r"update_rate: .*", f"update_rate: {float(meta['imu']['hz'])}", imu)
    with open(os.path.join(out, "kalibr_imu_chain.yaml"), "w") as f:
        f.write(imu)
    est = open(os.path.join(OPENVINS_CONFIG, "estimator_config.yaml")).read()
    for key, val in {"use_stereo": "true", "max_cameras": "2", "calib_cam_intrinsics": "false",
                     "calib_cam_extrinsics": "false", "calib_cam_timeoffset": "false", "try_zupt": "true",
                     "init_imu_thresh": "0.25", "fast_threshold": "15", "num_pts": "250", "use_mask": "false",
                     "verbosity": '"WARNING"', "record_timing_information": "false",
                     "save_total_state": "false"}.items():
        est = re.sub(rf"^{key}: .*$", f"{key}: {val}", est, flags=re.M)
    path = os.path.join(out, "estimator_config.yaml")
    with open(path, "w") as f:
        f.write(est)
    return path


class Vio:
    """track() per IR pair -> world_from_left IR (or None before OpenVINS initializes); changed() -> the whole
    trajectory again when RTAB-Map's graph moved (loop closure / proximity link) or the start was filled in."""

    def __init__(self, meta):
        self._dir = tempfile.TemporaryDirectory()
        self.ov = vio_py.OpenVins(openvins_config(meta, self._dir.name))
        st = meta["stereo"]
        self.rtab = vio_py.RtabMap(RTABMAP_PARAMS, st["fx"], st["fy"], st["cx"], st["cy"],
                                   float(np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3])),
                                   st["width"], st["height"])
        self.imu_from_ir1 = np.linalg.inv(np.array(meta["imu"]["ir1_from_imu"]))
        self.t, self.odom = [], []  # per IR pair: time (s), OpenVINS pose (or None)
        self.imu = []  # the IMU rows until OpenVINS has started (for the still IR pairs before it)
        self.world = None  # world <- OpenVINS' global frame, from the first pose
        self.nodes = {}  # RTAB-Map node id -> index of its IR pair
        self.D = self.sent = None  # corrections at the nodes: (indices, 4x4 world <- odometry); the last sent
        self.lock = threading.Lock()
        self.event = self.pending = None  # to send now (the start); to send when UPDATE_S has passed
        self.last_sent = -1e9
        self.busy = threading.Event()
        self.last_kf = -1e9
        self.loops = self.proximity = 0

    def track(self, t, left, right, imu_rows):
        """t: seconds; imu_rows: Nx7 (t s, accel, gyro) since the last pair."""
        with span("track/openvins"):
            if len(imu_rows):
                self.ov.imu(imu_rows)
            ok, T = self.ov.track(t, left, right)
        T = T @ self.imu_from_ir1 if ok else None
        with self.lock:
            self.t.append(t)
            self.odom.append(T)
            if self.world is None:
                self.imu.append(imu_rows)
            k = len(self.t) - 1
            if T is not None and self.world is None:
                self._start(k, T)
        if T is not None and t - self.last_kf >= KEYFRAME_S and not self.busy.is_set():
            self.last_kf = t
            self.busy.set()
            threading.Thread(target=self._keyframe, args=(k, t, left, right, T), daemon=True).start()
        return self.pose(k)

    def _start(self, k, T):
        """The first pose: the world (X along the camera's forward, horizontal; origin at the camera), and the
        IR pairs before it, back to the last one in motion, get it too."""
        f = T[:3, 2].copy()
        yaw = np.arctan2(f[1], f[0])
        W = np.eye(4)
        W[:3, :3] = Rotation.from_euler("z", -yaw).as_matrix()
        W[:3, 3] = -W[:3, :3] @ T[:3, 3]
        self.world = W
        # OpenVINS starts on the first jerk after holding still. Going back from it in WINDOW_S windows of the
        # IMU: past the motion (at most JERK_S), then until the camera moved before. Those IR pairs get the first
        # pose; the ones in motion stay unposed.
        H = np.vstack([r for r in self.imu if len(r)]) if any(len(r) for r in self.imu) else np.zeros((0, 7))
        self.imu = []

        def still(a, b):
            w = H[(H[:, 0] >= a) & (H[:, 0] < b)]
            return len(w) > 10 and np.abs(w[:, 4:7]).max() < MOVING_GYRO and w[:, 1:4].std(0).max() < MOVING_ACCEL
        t_init = self.t[k]
        hi = t_init
        while hi > t_init - JERK_S and not still(hi - WINDOW_S, hi):
            hi -= WINDOW_S / 2
        lo = hi
        while lo - WINDOW_S >= self.t[0] - WINDOW_S and still(lo - WINDOW_S, lo):
            lo -= WINDOW_S / 2
        filled = 0
        if still(hi - WINDOW_S, hi):
            for j in range(k):
                if lo - WINDOW_S / 2 <= self.t[j] <= hi:
                    self.odom[j] = T
                    filled += 1
        self.event = "OpenVINS initialized" + (f" ({filled} still IR pairs before it posed)" if filled else "")

    def _keyframe(self, k, t, left, right, T):
        try:
            with span("rtabmap/keyframe"):
                node, loop, prox = self.rtab.add(t, left, right, T, ODOM_VAR)
            if not node:
                return
            with self.lock:
                self.nodes[node] = k
            if loop or prox:
                self.loops += bool(loop)
                self.proximity += bool(prox)
                poses = self.rtab.poses()
                with self.lock:
                    ids = sorted(i for i in poses if i in self.nodes)
                    idx = np.array([self.nodes[i] for i in ids])
                    self.D = (idx, np.stack([poses[i] @ np.linalg.inv(self.odom[self.nodes[i]]) for i in ids]))
                    if self._moved():
                        kind = "loop closure (RTAB-Map)" if loop else "proximity link (RTAB-Map)"
                        self.pending = kind if self.pending is None or loop else self.pending
        finally:
            self.busy.clear()

    def _moved(self):
        """Whether a node's pose on the graph moved past MOVE_CM / MOVE_DEG from the trajectory sent last."""
        for k in self.D[0]:
            A = self._correction(k) @ self.odom[k]
            B = self._correction(k, self.sent) @ self.odom[k]
            deg = np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))
            if 100 * np.linalg.norm(A[:3, 3] - B[:3, 3]) > MOVE_CM or deg > MOVE_DEG:
                return True
        return False

    def _correction(self, k, D=False):
        """world <- odometry at IR pair k from D (default the latest): the nodes' corrections, interpolated in
        between, the last one after."""
        D = self.D if D is False else D
        if D is None:
            return np.eye(4)
        idx, D = D
        if k <= idx[0] or len(idx) == 1:
            return D[0]
        if k >= idx[-1]:
            return D[-1]
        j = int(np.searchsorted(idx, k))
        s = (k - idx[j - 1]) / (idx[j] - idx[j - 1])
        C = np.eye(4)
        C[:3, :3] = Slerp([0, 1], Rotation.from_matrix(D[[j - 1, j], :3, :3]))([s]).as_matrix()[0]
        C[:3, 3] = (1 - s) * D[j - 1, :3, 3] + s * D[j, :3, 3]
        return C

    def pose(self, k):
        with self.lock:
            T = self.odom[k]
            return None if T is None or self.world is None else self.world @ self._correction(k) @ T

    def changed(self, final=False):
        """(event, every IR pair's pose) once after the start was filled in, or the graph moved and UPDATE_S has
        passed since the last (final: at once), else None."""
        with self.lock:
            event, self.event = self.event, None
            if event is None and self.pending and (final or self.t[-1] - self.last_sent >= UPDATE_S):
                event, self.pending = self.pending, None
            if event is None:
                return None
            self.sent, self.last_sent = self.D, self.t[-1]
        return event, [self.pose(k) for k in range(len(self.t))]

    def finish(self, timeout_s=5.0):
        """Wait for the keyframe RTAB-Map is working on."""
        t_end = time.perf_counter() + timeout_s
        while self.busy.is_set() and time.perf_counter() < t_end:
            time.sleep(0.01)

    def keypoints(self):
        """Nx3 (x, y, 1): the left camera's tracked points, in the window's (x, y, matched) layout."""
        kp = self.ov.keypoints()
        return np.hstack([kp, np.ones((len(kp), 1), np.float32)])
