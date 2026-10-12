"""cuVSLAM (stereo + IMU, with SLAM: loop closures and pose graph) on a session's rec/: the projector-off IR
pair and the IMU, in order, as fast as cuVSLAM takes them (each track() call is timed; images are decoded
ahead on a thread pool). Writes left-IR poses as TUM files:
  cuvslam_odom.tum  the odometry, per frame, as track() returned it
  cuvslam_slam.tum  every frame's pose on the final pose graph (get_all_slam_poses)
and cuvslam.json: track() time per frame, frames without a pose, loop closures.

    python docs/vslam/run_cuvslam.py output/live_x/rec output/vslam/live_x
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cuvslam as vslam
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# the D455's BMI055, as cuVSLAM's RealSense examples (examples/realsense/camera_utils.py)
GYRO_NOISE, GYRO_WALK = 6.0673370376614875e-03, 3.6211951458325785e-05
ACCEL_NOISE, ACCEL_WALK = 3.3621979208052800e-02, 9.8256589971851467e-04


def pose(T):
    return vslam.Pose(rotation=Rotation.from_matrix(T[:3, :3]).as_quat(), translation=T[:3, 3])


def rig(meta):
    """Left IR is the rig frame (OpenCV axes); the right IR and the IMU from the factory extrinsics."""
    st = meta["stereo"]
    cams = []
    for T in (np.eye(4), np.array(st["ir1_from_ir2"])):
        c = vslam.Camera()
        c.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
        c.focal = st["fx"], st["fy"]
        c.principal = st["cx"], st["cy"]
        c.size = st["width"], st["height"]
        c.rig_from_camera = pose(T)
        cams.append(c)
    imu = vslam.ImuCalibration()
    imu.rig_from_imu = pose(np.array(meta["imu"]["ir1_from_imu"]))
    imu.gyroscope_noise_density, imu.gyroscope_random_walk = GYRO_NOISE, GYRO_WALK
    imu.accelerometer_noise_density, imu.accelerometer_random_walk = ACCEL_NOISE, ACCEL_WALK
    imu.frequency = meta["imu"]["hz"]
    r = vslam.Rig()
    r.cameras, r.imus = cams, [imu]
    return r


def write_tum(path, rows):
    with open(path, "w") as f:
        for ns, p in rows:
            t, q = p.translation, p.rotation
            f.write(f"{ns / 1e9:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec")
    p.add_argument("out")
    p.add_argument("--no-imu", action="store_true", help="stereo only (Multicamera mode)")
    p.add_argument("--denoise", action="store_true", help="cuVSLAM's image denoising (noisy, dark images)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.rec, "meta.json")))
    ir_ns = [int(round(f["timestamp_ms"] * 1e6)) for f in meta["stereo"]["frames"]]
    imu = np.loadtxt(os.path.join(a.rec, meta["imu"]["file"]), delimiter=",", skiprows=1)

    odom_cfg = vslam.Tracker.OdometryConfig(
        async_sba=False, rectified_stereo_camera=True, enable_observations_export=False,
        enable_landmarks_export=False, use_denoising=a.denoise,
        odometry_mode=vslam.Tracker.OdometryMode.Multicamera if a.no_imu else vslam.Tracker.OdometryMode.Inertial)
    slam_cfg = vslam.Tracker.SlamConfig(sync_mode=True)  # loop closures in track(): the same result every run
    tracker = vslam.Tracker(rig(meta), odom_cfg, slam_cfg)

    read = lambda k: [cv2.imread(os.path.join(a.rec, d, f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE)  # noqa: E731
                      for d in ("ir1", "ir2")]
    odom, ms, lost = [], [], []
    j = 0
    with ThreadPoolExecutor(4) as pool:
        pending = [pool.submit(read, k) for k in range(min(16, len(ir_ns)))]
        for k, ns in enumerate(ir_ns):
            images = pending[k].result()
            pending[k] = None
            if k + 16 < len(ir_ns):
                pending.append(pool.submit(read, k + 16))
            if not a.no_imu:
                while j < len(imu) and imu[j, 0] <= ns:
                    m = vslam.ImuMeasurement()
                    m.timestamp_ns = int(imu[j, 0])
                    m.linear_accelerations = tuple(map(float, imu[j, 1:4]))
                    m.angular_velocities = tuple(map(float, imu[j, 4:7]))
                    tracker.register_imu_measurement(0, m)
                    j += 1
            t = time.perf_counter()
            est, _ = tracker.track(ns, images)
            ms.append(1000 * (time.perf_counter() - t))
            if est.world_from_rig is None:
                lost.append(k)
            else:
                odom.append((ns, est.world_from_rig.pose))
    slam = [(s.timestamp_ns, s.pose) for s in tracker.get_all_slam_poses()]
    loops = tracker.get_loop_closure_poses()
    write_tum(os.path.join(a.out, "cuvslam_odom.tum"), odom)
    write_tum(os.path.join(a.out, "cuvslam_slam.tum"), slam)
    ms = np.array(ms)
    info = {"frames": len(ir_ns), "odom_poses": len(odom), "slam_poses": len(slam), "no_pose": len(lost),
            "first_pose_s": (odom[0][0] - ir_ns[0]) / 1e9 if odom else None, "loop_closures": len(loops),
            "track_ms": {"median": float(np.median(ms)), "p90": float(np.percentile(ms, 90)),
                         "max": float(ms.max())}, "mode": "stereo" if a.no_imu else "stereo+imu", "denoise": a.denoise}
    json.dump(info, open(os.path.join(a.out, "cuvslam.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
