# realsense-to-usd

[繁體中文](README.zh-TW.md)

Hand-held Intel RealSense D455 → live 3D reconstruction → 3D Gaussian Splatting → a USD stage for NVIDIA Isaac Sim.

- **Live** (`run_live.sh`): ORB-SLAM3 stereo + IMU poses, Fast-FoundationStereo depth (TensorRT) and an nvblox TSDF preview, all at once,
  in a window with the map, the tracked image, depth and status. On Stop, a distance-weighted TSDF mesh is built from every stored frame.
- **Offline** (`run_offline.sh`): the session's frames → TSDF mesh → FastGS 3DGS → `scene.usdc`
  (Z-up, meters, leveled; Gaussians for display, the decimated mesh as a static collider).

Developed and tested on a Jetson AGX Thor (JetPack 7.1, CUDA 13). The scripts also run on x86_64 + an NVIDIA RTX GPU;
Isaac Sim needs RTX ray-tracing hardware to render the result, so open the USD on an x86_64 + RTX machine.

![Four frames of the recording](docs/frames.jpg)

## Example: a 3-minute loop around a lab

Hand-held D455 (848x480, projector alternating), 187 s, 19.4 m of camera path, Jetson AGX Thor.

| Step | Result | Time |
|---|---|---|
| Tracking (ORB-SLAM3 stereo + IMU) | 4811 / 4829 IR pairs tracked; 35 ms each (p90 43) | live |
| Depth (FFS, TensorRT fp32 + TF32) | 934 frames, 5.0 Hz; 188 ms each | live |
| Capture → in the preview map | 263 ms (p90 460) | live |
| Stop → final mesh (4 m, distance-weighted TSDF) | 3.3 M triangles | 31 s |
| `run_offline.sh`: TSDF + COLMAP export + FastGS (31k steps) + USD | test views PSNR 25.45, SSIM 0.838, LPIPS 0.272; 364 k Gaussians; `scene.usdc` 169 MB | ~11 min |

![Final mesh](docs/mesh_views.jpg)

*`mesh_live.ply` rendered from the recorded poses. Far glass and background have no surface (black).*

![3DGS test views](docs/3dgs_test_views.jpg)

*Left: captured, right: 3DGS. Screens whose content changed and people walking by are blurred.*

## Pipeline

```
D455 ──► tracker process: projector-off IR pair ──► ORB-SLAM3 stereo (+ IMU), 30 Hz
     │                                                     │ poses (loop closures → map rebuilt with the corrected trajectory)
     └─► depth thread: newest projector-on IR pair ──► FFS (TensorRT) ──► depth in the color camera
                                                           │
                                     mapper: nvblox TSDF preview (depth ≤ 2.5 m) ──► window
Stop ──► final trajectory ──► distance-weighted TSDF of every stored frame (depth ≤ 4 m) ──► mesh_live.ply
     └─► frames/ (color + FFS depth + poses), rec/ (every frameset: color, D455 depth, IR, IMU)

run_offline.sh: frames/ ──► TSDF + point cloud ──► COLMAP text ──► FastGS ──► scene.usdc
```

| Path | What |
|---|---|
| `run_live.sh` | live session from the D455 (defaults `--projector alternate --imu`, other options go to `recon/live.py`) |
| `run_offline.sh` | a session → `recon/`, `colmap/`, `fastgs/`, `scene.usdc` |
| `recon/live.py` | live reconstruction (D455 or replay of a recorded `rec/`) |
| `recon/live_workers.py` | D455 capture, IR exposure control, IMU, recording, ORB-SLAM3 tracker process |
| `recon/live_view.py` | the window (Open3D GUI) |
| `recon/orbslam3_py/` | pybind11 binding of ORB-SLAM3 used by the tracker |
| `recon/ffs.py`, `recon/export_ffs_trt.py` | Fast-FoundationStereo inference / TensorRT engine export |
| `recon/stereo_depth.py` | offline FFS depth for a frames dir; disparity → color-camera depth used live too |
| `recon/reconstruct.py` | Open3D TSDF, and the distance-weighted TSDF (Triton) used for the final mesh |
| `recon/reconstruct_nvblox.py` | nvblox TSDF (live preview; also offline) |
| `recon/track_orbslam3.py` | offline ORB-SLAM3 poses for a recorded `rec/` |
| `gs/export_colmap.py`, `gs/train_fastgs.sh` | COLMAP-format export with known poses, FastGS training + evaluation |
| `gs/export_usd.py` | mesh + 3DGS → USD |
| `patches/` | fixes to nvblox, ORB-SLAM3 and FastGS (below) |
| `frames.py` | frames-dir I/O, depth filters, pose helpers |

## Installation

Everything runs in one conda env (default name `realsense-to-usd`; set `RS2USD_ENV` to use another).
`source env.sh` activates it, points `CUDA_HOME` at the CUDA 13 toolkit pip-installed into the env and puts the repo root on `PYTHONPATH`.
Run all commands from the repo root.

```bash
conda create -n realsense-to-usd python=3.12 "numpy>=2" imageio matplotlib pip
source env.sh
pip install torch==2.9.1+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130/
pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*" "nvidia-npp==13.*" ninja open3d==0.20.0 pyrealsense2 opencv-python-headless \
    scipy plyfile pybind11
C=$CUDA_HOME && ln -sf lib $C/lib64 && ln -sf libcudart.so.13 $C/lib/libcudart.so
conda install -c conda-forge openusd=26.03   # USD export (PyPI usd-core has no aarch64 build)
```

`nvidia-nvvm` must stay at 13.0: newer versions emit PTX that ptxas does not accept (`Unsupported .version`).

### nvblox

Built from source with `patches/nvblox-thor.patch` (`live.py` uses the patched `add_color_frame`, so the release wheel does not work).
On x86_64 set `CMAKE_CUDA_ARCHITECTURES` to your GPU (e.g. `89`) and `CMAKE_CUDA_COMPILER` to your CUDA 13 `nvcc`.

```bash
git clone --branch v0.0.10 https://github.com/nvidia-isaac/nvblox.git ~/nvblox
git -C ~/nvblox apply $PWD/patches/nvblox-thor.patch
cmake -S ~/nvblox -B ~/nvblox/build -DCMAKE_BUILD_TYPE=Release -DBUILD_RENDERER=0 -DBUILD_TESTING=0 \
    -DCMAKE_CUDA_ARCHITECTURES=110 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DTorch_DIR=$(python -c "import torch,os; print(os.path.dirname(torch.__file__))")/share/cmake/Torch \
    -DPython3_EXECUTABLE=$(which python)
cmake --build ~/nvblox/build -j12
(cd ~/nvblox/nvblox_torch && CUDA_VERSION=13.0 pip install --no-build-isolation --editable .)
```

`BUILD_RENDERER=0`: the renderer needs the Vulkan SDK and is not used. The patch: correct CUDA architecture for sm_110, `numpy>=2` / `opencv-python-headless` instead of the pinned versions,
`NVBLOX_FREE_SPACE_VOX` (limit how far free space is cleared), `add_color_frame(..., depth_frame)` (color fused with the aligned depth),
`NVBLOX_STREAM_PRIORITY` (`live.py` runs integration ahead of FFS on the GPU).

### ORB-SLAM3 and the Python binding

```bash
git clone -b v0.9.3 https://github.com/stevenlovegrove/Pangolin.git ~/Pangolin
cmake -S ~/Pangolin -B ~/Pangolin/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$HOME/.local/pangolin
cmake --build ~/Pangolin/build -j12 && cmake --install ~/Pangolin/build
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ~/ORB_SLAM3 && git -C ~/ORB_SLAM3 checkout 4452a3c
git -C ~/ORB_SLAM3 apply $PWD/patches/orbslam3-thor.patch
(cd ~/ORB_SLAM3 && for d in Thirdparty/DBoW2 Thirdparty/g2o; do
     cmake -S $d -B $d/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 && cmake --build $d/build -j12; done
 tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
 cmake -B build -DCMAKE_BUILD_TYPE=Release -DPangolin_DIR=$HOME/.local/pangolin/lib/cmake/Pangolin
 cmake --build build -j8 --target ORB_SLAM3 stereo_euroc)

cmake -S recon/orbslam3_py -B recon/orbslam3_py/build -DPython3_EXECUTABLE=$(which python) \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build recon/orbslam3_py/build -j8   # -> recon/orbslam3_py/orbslam3_py.cpython-312-*.so
```

The patch: C++17 (Pangolin 0.9), `mnFullBAIdx` as `int`, no segfault printing settings of a rectified stereo rig, no viewer in
`stereo_euroc`, `GetTracker()` / `GetAtlas()` accessors for the binding, stereo + IMU skips the IMU prediction for a frame without
preintegration instead of dereferencing a null pointer, and the "not enough motion" map reset only for monocular.
`--orb-root` (default `~/ORB_SLAM3`) and the CMake variable `ORB_SLAM3_ROOT` point elsewhere if needed.

### Fast-FoundationStereo + TensorRT

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git ~/Fast-FoundationStereo
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/c-fast-foundationstereo', local_dir='$HOME/Fast-FoundationStereo/weights/c-ffs')"
pip install scikit-image omegaconf "tensorrt-cu13==11.3.0.99"
python recon/export_ffs_trt.py -o output/ffs --width 864   # the 848x480 engine live.py looks for (~5 min)
```

TensorRT is pip-installed into the env, leaving a system TensorRT alone. An engine is tied to its input size and TensorRT version.

### FastGS

```bash
git clone https://github.com/fastgs/FastGS.git ~/FastGS
git -C ~/FastGS apply $PWD/patches/fastgs-thor.patch
pip install websockets lpips
pip install --no-build-isolation ~/FastGS/submodules/simple-knn ~/FastGS/submodules/fused-ssim \
    ~/FastGS/submodules/diff-gaussian-rasterization_fastgs
```

`FASTGS_ROOT` if it is not in `~/FastGS`. The patch: CUDA 13 build fixes, a tile-list out-of-bounds write in the rasterizer
(`cudaErrorIllegalAddress` mid-training), and an off-center principal point in the projection matrix (the D455 color camera's is
off by several pixels; without it the Gaussians sit ~9 mm off the TSDF mesh).

## Usage

```bash
sudo jetson_clocks   # Jetson: max clocks (otherwise the GPU idles down between FFS calls)
./run_live.sh                        # until Stop in the window or Ctrl+C -> output/live_<date>_<time>/
./run_live.sh --duration 120 -o output/live_room
./run_live.sh --no-imu --no-record   # cameras only, without the full recording
./run_offline.sh output/live_room    # -> output/live_room/scene.usdc
```

With `--imu` (the `run_live.sh` default) hold the camera still for a second at the start (leveling), then move it:
tracking starts once the IMU sees some motion. Keep the D455 on USB 3; `rec/` takes ~40 MB/s (2.4 GB a minute).

The window shows the nvblox map with the trajectory and current camera (Free / Follow / Camera views), the tracked IR image
with its ORB features, the FFS depth over color, and tracking / depth / map status. Buttons: **Stop** (end and save; then
**New recording** starts the next session in `<out>_2`, ...), **Discard & restart**, **Re-level (IMU)**.
Without a display (`--no-view`) the numbers go to `live.json`. `DISPLAY` defaults to `:1` in `run_live.sh`.

A recorded session can be replayed through the same live pipeline: `python recon/live.py output/live_room/rec -o output/replay --imu`.

### Session output

```
output/live_<date>_<time>/
  mesh_live.ply     final mesh (world: Z up from the accelerometer, origin at the first camera)
  poses_live.npy    Nx4x4 color camera-to-world of the frames in frames/
  live.json         timings: tracking, depth, integration, latency, rebuilds
  frames/           fused frames: color/, depth_ffs/ (uint16 mm), meta.json, poses.npy   <- run_offline.sh input
  rec/              every frameset: color/, depth/ (D455, aligned to color), ir1_on/ ir2_on/ (projector on),
                    ir1/ ir2/ (projector off), imu.csv, meta.json, poses.npy
  recon/ colmap/ fastgs/ scene.usdc     after run_offline.sh
```

A *frames dir* (`frames/` or `rec/`) is what every offline script takes: numbered PNGs per stream and `meta.json`
(color intrinsics, per-frame timestamps, and for IR the stereo intrinsics, baseline, `ir1_from_color` and the projector mode).
Offline tools for `rec/`: `recon/stereo_depth.py` (FFS depth at 30 Hz instead of live's ~5 Hz), `recon/track_orbslam3.py`.

### USD stage

```
/World                default prim
    DomeLight
    Scene             leveled (largest horizontal plane of the mesh)
        Mesh          full mesh with vertex colors, hidden (--show-mesh to show)
        Collision     decimated to 300k triangles, hidden, static collider (UsdPhysics)
        Gaussians     UsdVol.ParticleField3DGaussianSplat, SH degree 3
/PhysicsScene         gravity -Z (outside the default prim, so referencing the asset does not bring it along)
```

`ParticleField3DGaussianSplat` is the OpenUSD 26.03 3DGS schema; Isaac Sim 6.0 renders it. Tools without it (e.g. `usdview`)
show nothing unless `--show-mesh`.

## Licenses

This repository is GPL-3.0 (the live tracker links ORB-SLAM3, which is GPL-3.0). Third-party components keep their own terms:
ORB-SLAM3 GPL-3.0; nvblox Apache-2.0; Fast-FoundationStereo code under NVIDIA's own source license (see its LICENSE.txt), the C-FFS
weights under the NVIDIA Open Model Agreement; FastGS MIT, but its Inria 3DGS code is for non-commercial research only.
