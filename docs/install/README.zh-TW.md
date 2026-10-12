# 安裝

[English](README.md)

全部在同一個 conda 環境裡跑（下面用 `realsense-to-usd`，名稱不限）。所有指令都在 repo 根目錄執行。
`export` 那行只給這頁的編譯用（指到環境裡 pip 安裝的 CUDA 13 toolkit），整個安裝過程請用同一個終端機。
裝好之後，每次使用只要 `conda activate realsense-to-usd`。

```bash
conda create -n realsense-to-usd python=3.12 "numpy>=2" imageio matplotlib pip
conda activate realsense-to-usd
export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13 && export PATH=$CUDA_HOME/bin:$PATH
pip install torch==2.9.1+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130/
pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*" "nvidia-npp==13.*" ninja open3d==0.20.0 pyrealsense2 opencv-python-headless \
    scipy plyfile pybind11
C=$CUDA_HOME && ln -sf lib $C/lib64 && ln -sf libcudart.so.13 $C/lib/libcudart.so
conda install -c conda-forge openusd=26.03   # 匯出 USD（PyPI 的 usd-core 沒有 aarch64 版）
```

`nvidia-nvvm` 一定要固定在 13.0，新版產生的 PTX ptxas 不支援（`Unsupported .version`）。

### nvblox

套用 `patches/nvblox-thor.patch` 後從原始碼編譯（`live.py` 用到 patch 改過的 `add_color_frame`，官方 wheel 不能用）。
x86_64 上把 `CMAKE_CUDA_ARCHITECTURES` 改成自己的 GPU（例如 `89`），`CMAKE_CUDA_COMPILER` 指到 CUDA 13 的 `nvcc`。

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

`BUILD_RENDERER=0`：renderer 需要 Vulkan SDK，重建用不到。patch 內容：sm_110 的 CUDA 架構轉換、`numpy>=2` / `opencv-python-headless`
取代原本鎖定的版本、`NVBLOX_FREE_SPACE_VOX`（限制 free space 清除的距離）、`add_color_frame(..., depth_frame)`（用對齊的深度融合顏色）、
`NVBLOX_STREAM_PRIORITY`（`live.py` 讓融合在 GPU 上排在 FFS 前面）。

### ORB-SLAM3 與 Python binding

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

patch 內容：C++17（Pangolin 0.9 需要）、`mnFullBAIdx` 改成 `int`、Rectified 雙目印設定時不再 segfault、
給 binding 用的 `GetTracker()` / `GetAtlas()`、雙目 + IMU 遇到沒有預積分的幀時跳過 IMU 預測（原本解參考空指標）、
「Not enough motion」的地圖重置只用在單眼。
ORB-SLAM3、Fast-FoundationStereo、FastGS 都要照這裡的指令放在 `~/`。

### Fast-FoundationStereo + TensorRT

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git ~/Fast-FoundationStereo
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/c-fast-foundationstereo', local_dir='$HOME/Fast-FoundationStereo/weights/c-ffs')"
pip install scikit-image omegaconf "tensorrt-cu13==11.3.0.99"
python recon/export_ffs_trt.py   # -> output/ffs/ffs_480x864_it8_fp32_tf32.engine（約 5 分鐘）
```

TensorRT 用 pip 裝進環境，不動系統的 TensorRT。引擎綁定輸入大小與 TensorRT 版本。

### 人物分割（TensorRT）

```bash
python recon/export_seg_trt.py   # -> output/seg/person_dlv3_r50_480x848_fp16.engine（約 4 分鐘）
```

DeepLabV3-ResNet50（torchvision 權重）、fp16：在 depth thread 裡接在 FFS 後面，每幀約 8 ms。沒有這個引擎時
即時地圖裡會有人，`gs/offline.py` 再用 PyTorch 遮掉。

### FastGS

```bash
git clone https://github.com/fastgs/FastGS.git ~/FastGS
git -C ~/FastGS apply $PWD/patches/fastgs-thor.patch
pip install websockets lpips
pip install --no-build-isolation ~/FastGS/submodules/simple-knn ~/FastGS/submodules/fused-ssim \
    ~/FastGS/submodules/diff-gaussian-rasterization_fastgs
```

patch 內容：CUDA 13 編譯、光柵化的 tile 清單越界寫入（訓練中途 `cudaErrorIllegalAddress`）、
投影矩陣支援偏離中心的主點（D455 color 的主點偏好幾個 px，不修正時高斯和 TSDF mesh 差約 9 mm）、
每張影像的 mask（`<dataset>/masks/`，0 的地方不算 loss：`recon/person_mask.py` 找到的人）。

### Jetson

```bash
sudo jetson_clocks   # 鎖最高時脈（否則 GPU 在兩次 FFS 之間降頻）；每次開機後要重跑
```
