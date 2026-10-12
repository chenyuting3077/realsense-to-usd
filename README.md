# realsense-to-usd

[繁體中文](README.zh-TW.md)

Hand-held Intel RealSense D455 → live 3D reconstruction → 3D Gaussian Splatting → a USD stage for NVIDIA Isaac Sim.

- **Live** (`run_live.sh`): ORB-SLAM3 stereo + IMU poses, Fast-FoundationStereo depth (TensorRT) and an nvblox TSDF preview, all at once,
  in a window with the map, the tracked image, depth and status. On Stop, a distance-weighted TSDF mesh is built from every stored frame.
- **Offline** (`run_offline.sh`): the session's frames → TSDF mesh → FastGS 3DGS → `scene.usdc`
  (Z-up, meters, leveled; Gaussians for display, the decimated mesh as a static collider).

Developed and tested on a Jetson AGX Thor (JetPack 7.1, CUDA 13).
Isaac Sim needs RTX ray-tracing hardware to render the result, so open the USD on an x86_64 + RTX machine.

## Example: a 3-minute loop around a lab

Hand-held D455 (848x480, projector alternating), 187 s, 19.4 m of camera path, Jetson AGX Thor.

| Step | Result | Time |
|---|---|---|
| Tracking (ORB-SLAM3 stereo + IMU) | 4811 / 4829 IR pairs tracked; 35 ms each (p90 43) | live |
| Depth (FFS, TensorRT fp32 + TF32) | 934 frames, 5.0 Hz; 188 ms each | live |
| Capture → in the preview map | 263 ms (p90 460) | live |
| Stop → final mesh (4 m, distance-weighted TSDF) | 3.3 M triangles | 31 s |
| `run_offline.sh`: TSDF + COLMAP export + FastGS (31k steps) + USD | test views PSNR 25.45, SSIM 0.838, LPIPS 0.272; 364 k Gaussians; `scene.usdc` 169 MB | ~11 min |

![Final mesh](docs/mesh_views.png)

![3DGS test views](docs/3dgs_test_views.png)

## Pipeline

```
D455 ──► tracker process: projector-off IR pair ──► ORB-SLAM3 stereo (+ IMU), 30 Hz
     │                                                     │ poses (loop closures → map rebuilt with the corrected trajectory)
     └─► depth thread: newest projector-on IR pair ──► FFS (TensorRT) ──► depth in the color camera
                                                           │
                                     mapper: nvblox TSDF preview (depth ≤ 2.5 m) ──► window
Stop ──► final trajectory ──► distance-weighted TSDF of every stored frame (depth ≤ 4 m) ──► mesh_live.ply
     └─► frames/ (color + FFS depth + poses), rec/ (every frameset: color, IR, IMU)

3DGS -> USD (window) or run_offline.sh: frames/ ──► TSDF + point cloud ──► COLMAP text ──► FastGS ──► scene.usdc
```

## Installation

See [INSTALL.md](docs/INSTALL.md): the conda env, nvblox, ORB-SLAM3 and its Python binding, Fast-FoundationStereo + TensorRT, FastGS.

## Usage

```bash
conda activate realsense-to-usd
./run_live.sh -o output/live_room   # Stop in the window or Ctrl+C to end
./run_offline.sh output/live_room   # -> scene.usdc, or 3DGS -> USD in the window
```

Hold the camera still for a second at the start (leveling with the IMU), then move it. Keep the D455 on USB 3.

## References

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
- [nvblox](https://github.com/nvidia-isaac/nvblox)
- [FastGS](https://github.com/fastgs/FastGS)

## Licenses

GPL-3.0. Since the pipeline includes Inria 3DGS and NVIDIA Fast-FoundationStereo, it is for **non-commercial research only**.
