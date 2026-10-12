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

![最終 mesh](docs/mesh_views.jpg)

*`mesh_live.ply` 從錄影的位姿渲染。遠處的玻璃與背景沒有表面（黑色）。*

![3DGS 測試視角](docs/3dgs_test_views.jpg)

*左：實拍，右：3DGS。錄影時內容在變的螢幕和走動的人模糊。*

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

見 [INSTALL.zh-TW.md](INSTALL.zh-TW.md)：conda 環境、nvblox、ORB-SLAM3 與 Python binding、Fast-FoundationStereo + TensorRT、FastGS。

## 使用

```bash
sudo jetson_clocks   # Jetson：鎖最高時脈（否則 GPU 在兩次 FFS 之間降頻）
./run_live.sh                        # 按視窗的 Stop 或 Ctrl+C 結束 -> output/live_<日期>_<時間>/
./run_live.sh --duration 120 -o output/live_room
./run_live.sh --no-imu --no-record   # 只用影像、不完整存檔
./run_offline.sh output/live_room    # -> output/live_room/scene.usdc
```

`--imu`（`run_live.sh` 的預設）時，開始先拿穩相機 1 秒（轉正），再移動一下，IMU 看到動作才開始追蹤。
D455 要接 USB 3；`rec/` 約 40 MB/s（每分鐘 2.4 GB）。

視窗顯示 nvblox 地圖、軌跡與目前相機（Free / Follow / Camera 視角）、追蹤中的 IR 影像與 ORB 特徵點、疊在 color 上的 FFS 深度、
以及追蹤 / 深度 / 地圖狀態。按鈕：**Stop**（結束並存檔；之後 **New recording** 開始下一段，輸出到 `<out>_2`…）、
**Discard & restart**、**Re-level (IMU)**。沒有螢幕時用 `--no-view`，數字寫在 `live.json`。`run_live.sh` 的 `DISPLAY` 預設 `:1`。

錄下的一段可以用同一套即時流程重播：`python recon/live.py output/live_room/rec -o output/replay --imu`。

### 輸出

```
output/live_<日期>_<時間>/
  mesh_live.ply     最終 mesh（世界座標：加速度計定的 Z 朝上，原點在第一幀相機）
  poses_live.npy    frames/ 每一幀的 Nx4x4 color 相機到世界
  live.json         時間：追蹤、深度、融合、延遲、重建
  frames/           融合過的幀：color/、depth_ffs/（uint16 mm）、meta.json、poses.npy   <- run_offline.sh 的輸入
  rec/              每組 frameset：color/、depth/（D455，對齊到 color）、ir1_on/ ir2_on/（投射器開）、
                    ir1/ ir2/（投射器關）、imu.csv、meta.json、poses.npy
  recon/ colmap/ fastgs/ scene.usdc     run_offline.sh 之後
```

離線腳本的輸入都是「幀資料夾」（`frames/` 或 `rec/`）：每個 stream 一個資料夾的編號 PNG，加上 `meta.json`
（color 內參、每幀時間戳；IR 的雙目內參、基線、`ir1_from_color` 與投射器模式）。
`rec/` 可用的離線工具：`recon/stereo_depth.py`（30 Hz 的 FFS 深度，即時版約 5 Hz）、`recon/track_orbslam3.py`。

### USD

```
/World                預設 prim
    DomeLight
    Scene             水平化（mesh 最大的水平面）
        Mesh          完整 mesh、頂點色，隱藏（--show-mesh 顯示）
        Collision     減面到 30 萬個三角形，隱藏，靜態碰撞體（UsdPhysics）
        Gaussians     UsdVol.ParticleField3DGaussianSplat，3 階 SH
/PhysicsScene         重力 -Z（放在預設 prim 外，以 reference 載入時不會帶進來）
```

`ParticleField3DGaussianSplat` 是 OpenUSD 26.03 的 3DGS schema，Isaac Sim 6.0 看得到；不支援的工具（例如 `usdview`）
要加 `--show-mesh` 才有畫面。

## 授權

本 repo 為 GPL-3.0（即時追蹤透過 binding 連結 GPL-3.0 的 ORB-SLAM3）。第三方元件各有授權：
ORB-SLAM3 GPL-3.0；nvblox Apache-2.0；Fast-FoundationStereo 程式碼為 NVIDIA 自己的授權（見其 LICENSE.txt），C-FFS 權重為
NVIDIA Open Model Agreement；FastGS 為 MIT，但其中沿用的 Inria 3DGS 程式碼只能做非商業研究。
