# Installation

[繁體中文](README.zh-TW.md)

Everything runs in one conda env (`realsense-to-usd` below; any name works). Run all commands from the repo root.
The `export` line is only for the builds on this page (it points them at the CUDA 13 toolkit pip-installed into the env):
keep this terminal for the whole installation. Afterwards `conda activate realsense-to-usd` is all a session needs.

```bash
conda create -n realsense-to-usd python=3.12 "numpy>=2" imageio matplotlib pip
conda activate realsense-to-usd
export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13 && export PATH=$CUDA_HOME/bin:$PATH
pip install torch==2.9.1+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130/
pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*" "nvidia-npp==13.*" ninja open3d==0.20.0 pyrealsense2 opencv-python-headless \
    scipy plyfile pybind11
C=$CUDA_HOME && ln -sf lib $C/lib64 && ln -sf libcudart.so.13 $C/lib/libcudart.so
conda install -c conda-forge openusd=26.03   # USD export (PyPI usd-core has no aarch64 build)
```

`nvidia-nvvm` must stay at 13.0: newer versions emit PTX that ptxas does not accept (`Unsupported .version`).

### nvblox

Built from source with `patches/nvblox-thor.patch` (`main.py` uses the patched `add_color_frame`, so the release wheel does not work).
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
`NVBLOX_STREAM_PRIORITY` (`main.py` runs integration ahead of FFS on the GPU).

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
 cmake --build build -j8 --target ORB_SLAM3)

cmake -S recon/orbslam3_py -B recon/orbslam3_py/build -DPython3_EXECUTABLE=$(which python) \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build recon/orbslam3_py/build -j8   # -> recon/orbslam3_py/orbslam3_py.cpython-312-*.so
```

The patch: C++17 (Pangolin 0.9), `mnFullBAIdx` as `int`, no segfault printing settings of a rectified stereo rig,
`GetTracker()` / `GetAtlas()` accessors for the binding, stereo + IMU skips the IMU prediction for a frame without
preintegration instead of dereferencing a null pointer, and the "not enough motion" map reset only for monocular.
ORB-SLAM3, Fast-FoundationStereo and FastGS are expected in `~/`, as cloned here.

### Fast-FoundationStereo + TensorRT

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git ~/Fast-FoundationStereo
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/c-fast-foundationstereo', local_dir='$HOME/Fast-FoundationStereo/weights/c-ffs')"
pip install scikit-image omegaconf "tensorrt-cu13==11.3.0.99"
python -m recon.export_ffs_trt   # -> output/ffs/ffs_480x864_it8_v3_amp.engine (~5 min)
```

TensorRT is pip-installed into the env, leaving a system TensorRT alone. An engine is tied to its input size and TensorRT version.

### Person segmentation (TensorRT)

```bash
python -m recon.export_seg_trt   # -> output/seg/person_dlv3_r50_480x848_fp16.engine (~4 min)
```

DeepLabV3-ResNet50 (torchvision weights), fp16: ~8 ms a frame after FFS in the depth thread. Without the
engine people stay in the live map and `gs/offline.py` masks them with PyTorch.

### FastGS

```bash
git clone https://github.com/fastgs/FastGS.git ~/FastGS
git -C ~/FastGS apply $PWD/patches/fastgs-thor.patch
pip install websockets lpips
pip install --no-build-isolation ~/FastGS/submodules/simple-knn ~/FastGS/submodules/fused-ssim \
    ~/FastGS/submodules/diff-gaussian-rasterization_fastgs
```

The patch: CUDA 13 build fixes, a tile-list out-of-bounds write in the rasterizer
(`cudaErrorIllegalAddress` mid-training), and an off-center principal point in the projection matrix (the D455 color camera's is
off by several pixels; without it the Gaussians sit ~9 mm off the TSDF mesh), and per-image masks
(`<dataset>/masks/`, no loss where 0: the people found by `recon/person_mask.py`).

### Jetson

```bash
sudo jetson_clocks   # GPU and memory clocks at max (otherwise the GPU idles down between FFS calls)
# again after every reboot (or enabled as a service)
# The CPUs at max too, at every boot: JetPack 7.1's jetson_clocks leaves them to schedutil (972-2601 MHz),
# ORB-SLAM3 then takes 32 instead of 22 ms a frame (docs/profile)
sudo cp docs/install/cpu-performance.service /etc/systemd/system/
sudo systemctl enable --now cpu-performance.service
```
