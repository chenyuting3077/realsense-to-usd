"""OpenVINS config for a recording: kalibr_imucam_chain.yaml / kalibr_imu_chain.yaml from its meta.json
(the D455's rectified IR pair and its factory IMU extrinsics), estimator_config.yaml from OpenVINS'
config/rs_d455 with stereo on.

    python3 make_ov_config.py rec/meta.json /ws/src/open_vins/config/rs_d455 out_dir
"""
import json
import os
import re
import sys

import numpy as np


def mat(T):
    return "\n".join(f"    - [{', '.join(f'{x:.9f}' for x in row)}]" for row in T)


def main():
    meta_path, template, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(meta_path))
    st = meta["stereo"]
    imu_from_ir1 = np.linalg.inv(np.array(meta["imu"]["ir1_from_imu"]))
    imu_from_ir2 = imu_from_ir1 @ np.array(st["ir1_from_ir2"])
    cams = ""
    for k, T in enumerate((imu_from_ir1, imu_from_ir2)):
        cams += f"""cam{k}:
  T_imu_cam: #rotation from camera to IMU R_CtoI, position of camera in IMU p_CinI
{mat(T)}
  cam_overlaps: [{1 - k}]
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  distortion_model: radtan
  intrinsics: [{st['fx']}, {st['fy']}, {st['cx']}, {st['cy']}]
  resolution: [{st['width']}, {st['height']}]
  rostopic: /cam{k}/image_raw
  timeshift_cam_imu: 0.0
"""
    open(os.path.join(out, "kalibr_imucam_chain.yaml"), "w").write("%YAML:1.0\n\n" + cams)

    imu = open(os.path.join(template, "kalibr_imu_chain.yaml")).read()
    imu = re.sub(r"rostopic: .*", "rostopic: /imu0", imu)
    imu = re.sub(r"update_rate: .*", f"update_rate: {float(meta['imu']['hz'])}", imu)
    open(os.path.join(out, "kalibr_imu_chain.yaml"), "w").write(imu)

    est = open(os.path.join(template, "estimator_config.yaml")).read()
    for key, val in {"use_stereo": "true", "max_cameras": "2", "use_mask": "false",
                     "record_timing_information": "false", "save_total_state": "false"}.items():
        est = re.sub(rf"^{key}: .*$", f"{key}: {val}", est, flags=re.M)
    open(os.path.join(out, "estimator_config.yaml"), "w").write(est)
    print("OpenVINS config ->", out)


if __name__ == "__main__":
    main()
