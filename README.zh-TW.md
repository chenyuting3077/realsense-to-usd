# realsense-to-usd

[English](README.md)

手持 Intel RealSense D455 → 即時 3D 重建 → 3D Gaussian Splatting → 給 NVIDIA Isaac Sim 的 USD。

- **即時**（`python main.py`）：ORB-SLAM3 雙目 + IMU 位姿、Fast-FoundationStereo 深度（TensorRT）、nvblox TSDF 預覽同時跑，
  視窗顯示地圖、追蹤中的影像、深度與狀態。按 Stop 後用存下的所有幀建距離加權的 TSDF mesh。
- **離線**（`python -m gs.offline`）：這段錄影的幀（遮掉人：即時或在這裡做）→ TSDF mesh → FastGS 3DGS → `scene.usdc`
  （Z-up、公尺、已水平；高斯負責顯示，減面後的 mesh 當靜態碰撞體）。

## 測試環境

| 項目 | 版本 |
|---|---|
| 硬體 | NVIDIA Jetson AGX Thor Developer Kit |
| 作業系統 | Ubuntu 24.04.5 LTS（kernel 6.8.12-tegra） |
| JetPack | 7.1（L4T R38.4.0） |
| CUDA | 13.0 |
| 相機 | Intel RealSense D455，韌體 5.17.3.10 |
| Python | 3.12（conda）、PyTorch 2.9.1+cu130、TensorRT 11.3、Open3D 0.20.0、pyrealsense2 2.58.4 |

Isaac Sim 算圖需要 RTX 的 ray tracing 硬體，USD 要在 x86_64 + RTX 的電腦上開。

## 範例：實驗室繞一圈 3 分鐘

手持 D455（848x480，投射器交替），187 秒，相機移動 19.4 m，Jetson AGX Thor。

| 步驟 | 結果 | 時間 |
|---|---|---|
| 追蹤（ORB-SLAM3 雙目 + IMU） | 4811 / 4829 組 IR；每組 35 ms（p90 43） | 即時 |
| 深度（FFS，TensorRT fp32 + TF32） | 934 幀，5.0 Hz；每幀 188 ms | 即時 |
| 拍到 → 進預覽地圖 | 263 ms（p90 460） | 即時 |
| Stop → 最終 mesh（4 m、距離加權 TSDF） | 331 萬個三角形 | 31 秒 |
| 3DGS -> USD：TSDF + 匯出 COLMAP + FastGS + USD | 36.4 萬個高斯；`scene.usdc` 169 MB | 約 11 分鐘 |

![最終 mesh](docs/lab_loop/mesh_views.png)

![實拍 vs 3DGS](docs/lab_loop/3dgs_test_views.png)

## 流程

```
D455 ──► 追蹤 process：投射器關的 IR ──► ORB-SLAM3 雙目（+ IMU），30 Hz
     │                                       │ 位姿（迴環 → 用修正後的軌跡重建地圖）
     └─► 深度 thread：最新一組投射器開的 IR ──► FFS（TensorRT）──► 對齊到 color 的深度
                         遮掉人（DeepLabV3，TensorRT）──┤
                     地圖：nvblox TSDF 預覽（深度 ≤ 2.5 m）──► 視窗
Stop ──► 最終軌跡 ──► 存下的所有幀做距離加權 TSDF（深度 ≤ 4 m）──► mesh_live.ply
     └─► frames/（color + FFS 深度 + 人物 mask + 位姿）、rec/（每組 frameset：color、IR、IMU）

3DGS -> USD（視窗）或 gs/offline.py：frames/ ──►（人物 mask，若即時沒做）──► TSDF + 點雲（不含人）
    ──► COLMAP text + mask ──► FastGS（人的像素不算 loss）──► scene.usdc
```

## 安裝

見 [docs/install](docs/install/README.zh-TW.md)：conda 環境、nvblox、ORB-SLAM3 與 Python binding、Fast-FoundationStereo + TensorRT、
人物分割、FastGS。TensorRT 加速前後的速度與精度：[docs/tensorrt](docs/tensorrt/README.zh-TW.md)；
更快的 FFS 設定：[docs/ffs_speed](docs/ffs_speed/README.zh-TW.md)；每個節點的時間與加速：
[docs/profile](docs/profile/README.zh-TW.md)；ORB-SLAM3 和 OpenVINS（+ RTAB-Map）、
VINS-Fusion、cuVSLAM 在錄影上的比較：[docs/vslam](docs/vslam/README.zh-TW.md)。

## 使用

```bash
conda activate realsense-to-usd
python main.py -o output/live_room       # 按視窗的 Stop 或 Ctrl+C 結束
python -m gs.offline output/live_room  # -> scene.usdc，或按視窗的 3DGS -> USD
```

開始時先拿穩相機 1 秒（用 IMU 轉正），再開始移動。D455 要接 USB 3。

## 參考文獻

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
- [nvblox](https://github.com/nvidia-isaac/nvblox)
- [FastGS](https://github.com/fastgs/FastGS)

## 授權

GPL-3.0。因為流程中含 Inria 3DGS 與 NVIDIA Fast-FoundationStereo，**僅限非商業研究使用**。
