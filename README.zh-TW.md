# realsense-to-usd

[English](README.md)

手持 Intel RealSense D455 → 即時 3D 重建 → 3D Gaussian Splatting → 給 NVIDIA Isaac Sim 的 USD。

- **即時**（`run_live.sh`）：ORB-SLAM3 雙目 + IMU 位姿、Fast-FoundationStereo 深度（TensorRT）、nvblox TSDF 預覽同時跑，
  視窗顯示地圖、追蹤中的影像、深度與狀態。按 Stop 後用存下的所有幀建距離加權的 TSDF mesh。
- **離線**（`run_offline.sh`）：這段錄影的幀 → TSDF mesh → FastGS 3DGS → `scene.usdc`
  （Z-up、公尺、已水平；高斯負責顯示，減面後的 mesh 當靜態碰撞體）。

在 Jetson AGX Thor（JetPack 7.1、CUDA 13）上開發與測試。
Isaac Sim 算圖需要 RTX 的 ray tracing 硬體，USD 要在 x86_64 + RTX 的電腦上開。

## 範例：實驗室繞一圈 3 分鐘

手持 D455（848x480，投射器交替），187 秒，相機移動 19.4 m，Jetson AGX Thor。

| 步驟 | 結果 | 時間 |
|---|---|---|
| 追蹤（ORB-SLAM3 雙目 + IMU） | 4811 / 4829 組 IR；每組 35 ms（p90 43） | 即時 |
| 深度（FFS，TensorRT fp32 + TF32） | 934 幀，5.0 Hz；每幀 188 ms | 即時 |
| 拍到 → 進預覽地圖 | 263 ms（p90 460） | 即時 |
| Stop → 最終 mesh（4 m、距離加權 TSDF） | 331 萬個三角形 | 31 秒 |
| `run_offline.sh`：TSDF + 匯出 COLMAP + FastGS（31k 步）+ USD | 測試視角 PSNR 25.45、SSIM 0.838、LPIPS 0.272；36.4 萬個高斯；`scene.usdc` 169 MB | 約 11 分鐘 |

![最終 mesh](docs/mesh_views.png)

![3DGS 測試視角](docs/3dgs_test_views.png)

## 流程

```
D455 ──► 追蹤 process：投射器關的 IR ──► ORB-SLAM3 雙目（+ IMU），30 Hz
     │                                       │ 位姿（迴環 → 用修正後的軌跡重建地圖）
     └─► 深度 thread：最新一組投射器開的 IR ──► FFS（TensorRT）──► 對齊到 color 的深度
                                             │
                     地圖：nvblox TSDF 預覽（深度 ≤ 2.5 m）──► 視窗
Stop ──► 最終軌跡 ──► 存下的所有幀做距離加權 TSDF（深度 ≤ 4 m）──► mesh_live.ply
     └─► frames/（color + FFS 深度 + 位姿）、rec/（每組 frameset：color、D455 深度、IR、IMU）

run_offline.sh：frames/ ──► TSDF + 點雲 ──► COLMAP text ──► FastGS ──► scene.usdc
```

## 安裝

見 [INSTALL.zh-TW.md](docs/INSTALL.zh-TW.md)：conda 環境、nvblox、ORB-SLAM3 與 Python binding、Fast-FoundationStereo + TensorRT、FastGS。

## 使用

```bash
./run_live.sh -o output/live_room   # 按視窗的 Stop 或 Ctrl+C 結束
./run_offline.sh output/live_room   # -> output/live_room/scene.usdc
```

開始時先拿穩相機 1 秒（用 IMU 轉正），再開始移動。D455 要接 USB 3。

## 授權

本 repo 為 GPL-3.0（即時追蹤透過 binding 連結 GPL-3.0 的 ORB-SLAM3）。第三方元件各有授權：
ORB-SLAM3 GPL-3.0；nvblox Apache-2.0；Fast-FoundationStereo 程式碼為 NVIDIA 自己的授權（見其 LICENSE.txt），C-FFS 權重為
NVIDIA Open Model Agreement；FastGS 為 MIT，但其中沿用的 Inria 3DGS 程式碼只能做非商業研究。
