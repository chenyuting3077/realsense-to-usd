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

套用 `patches/nvblox-thor.patch` 後從原始碼編譯（`main.py` 用到 patch 改過的 `add_color_frame`，官方 wheel 不能用）。
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
`NVBLOX_STREAM_PRIORITY`（`main.py` 讓融合在 GPU 上排在 FFS 前面）。

### OpenVINS + RTAB-Map 與 Python binding

```bash
git clone --depth 1 --branch 2.2.0 https://github.com/ceres-solver/ceres-solver.git ~/ceres-2.2
git clone https://github.com/rpng/open_vins.git ~/open_vins && git -C ~/open_vins checkout 6948812
git clone --depth 1 --branch 0.23.13-jazzy https://github.com/introlab/rtabmap.git ~/rtabmap
recon/vio_py/build_deps.sh   # Ceres、OpenVINS（不用 ROS）、RTAB-Map 核心 -> ~/.local/{ceres,open_vins,rtabmap}
recon/vio_py/build.sh        # -> recon/vio_py/vio_py.cpython-312-*.so
```

全部用系統的 Eigen 3.4 和 OpenCV 4.6（conda 的 Ceres 是用 Eigen 5 編的）。OpenVINS 用 master：v2.7 用的是
Ceres 2.2 已移除的 `LocalParameterization`。腳本會隱藏已載入的 ROS 2。OpenVINS（它的 `config/rs_d455`）、
Fast-FoundationStereo、FastGS 都要照這裡的指令放在 `~/`。

### Fast-FoundationStereo + TensorRT

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git ~/Fast-FoundationStereo
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/c-fast-foundationstereo', local_dir='$HOME/Fast-FoundationStereo/weights/c-ffs')"
pip install scikit-image omegaconf "tensorrt-cu13==11.3.0.99"
python -m recon.export_ffs_trt   # -> output/ffs/ffs_480x864_it8_v3_amp.engine（約 5 分鐘）
```

TensorRT 用 pip 裝進環境，不動系統的 TensorRT。引擎綁定輸入大小與 TensorRT 版本。

### 人物分割（TensorRT）

```bash
python -m recon.export_seg_trt   # -> output/seg/person_dlv3_r50_480x848_fp16.engine（約 4 分鐘）
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
sudo jetson_clocks   # GPU 與記憶體鎖最高時脈（否則 GPU 在兩次 FFS 之間降頻）；每次開機後要重跑
```

`jetson_clocks` 不動 CPU（仍是 `schedutil`）；CPU 鎖最高時追蹤和建圖都比較快
（[docs/speed](../speed/README.zh-TW.md)）。每次開機自動設定：

```bash
sudo tee /etc/systemd/system/cpu-performance.service > /dev/null <<'EOF'
[Unit]
Description=CPU governor performance
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > "$g"; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now cpu-performance.service
```
