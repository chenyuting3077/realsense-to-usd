# Thor 上的速度

[English](README.md)

live pipeline 每個節點在 Jetson AGX Thor（JetPack 7.1、MAXN、TensorRT 11.3）上怎麼變快，以及對精度的影響。

## 現在

| | |
|---|---|
| FFS | TensorRT、848x480、max_disp 192、8 次迭代、FFS 自己的 fp16 autocast、`recon/ffs_volume.py` 的 cost volume：62 ms，ETH3D EPE 0.201 px |
| 人物分割 | DeepLabV3-ResNet50、TensorRT fp16：7.4 ms，與 FFS 同時跑 |
| ORB-SLAM3 | 每幀 22.1 / 25.8 ms（中位數 / p90），CPU governor `performance` |
| Live（重播 `output/live_20261012_171428/rec`） | depth 13.73 Hz，capture → map 122 / 192 ms |
| Stop → 存好 mesh 與 frames | 2.4 秒 |

## 腳本

```bash
bench/live/run_baseline.sh <rec> <out> 3                 # 重播 3 次 + 各階段表（recon/prof.py 的 span、tegrastats）
bench/live/run_replay.sh <rec> <out> [nsys|view]         # 重播一次，在 nsys 下或開著視窗
python -m bench.live.compare_outputs <run A> <run B> ...  # 重播之間的 ATE、深度、mesh Chamfer
bench/ffs/settings.sh <rec>,<rec>                        # build FFS engine；延遲、ETH3D、D455 深度
python -m bench.ffs.equal <engine A> <engine B> <rec>     # engine 時間、disparity 差
python -m bench.ffs.layers <engine>                       # 逐層時間（系統的 trtexec 讀不了 TensorRT 11.3 的 engine）
python -m bench.ffs.near_limit <engine 192> <engine 128> <rec>,<rec>
python -m bench.ffs.warp_equal <舊 ffs.py> <rec>
python -m bench.seg.compare <frames>
```

## Live pipeline 逐步

重播 `output/live_20261012_171428/rec` 並加 `--no-view`，Thor 上沒有其他工作；3 次的中位數（第一列 1 次）。

| | FFS 設定 | Depth | FFS | Capture → map | ORB-SLAM3 | Stop → 完成 |
|---|---|---:|---:|---:|---:|---:|
| 完整 IR，人物分割排在 FFS 後 | 848x480、TF32 | 5.9 Hz | | 215 / 383 ms | | |
| IR 縮成 0.75 倍（CPU 為 schedutil） | 636x360、TF32 | 9.34 Hz | 83.1 ms | 151 / 276 ms | 32.2 / 39.5 ms | 4.1 秒 |
| cost volume (1)、depth 執行緒 | 636x360、TF32 | 11.11 Hz | 71.3 ms | 149 / 260 ms | 32.0 / 39.7 ms | 4.6 秒 |
| CPU governor、Stop 之後 | 636x360、TF32 | 11.47 Hz | 71.0 ms | 141 / 218 ms | 22.1 / 25.5 ms | 2.2 秒 |
| 人物分割與 FFS 同時跑 | 636x360、TF32 | 12.26 Hz | 70.9 ms | 131 / 230 ms | 22.3 / 25.2 ms | 2.1 秒 |
| 848x480 + fp16 autocast | **848x480、amp** | 11.66 Hz | 74.8 ms | 135 / 240 ms | 22.0 / 25.3 ms | 2.1 秒 |
| cost volume (2) | **848x480、amp** | **13.73 Hz** | **62.0 ms** | **122 / 192 ms** | **22.1 / 25.8 ms** | **2.4 秒** |

開著視窗 FFS 多約 3 ms（在「人物分割」那一列為 11.72 Hz）。

### 精度

`compare_outputs` 對第 1 次；以同一份程式重跑的差異為準（每次選到的 frameset 不同）。

| | ATE（rmse） | 深度（中位數 / p90 / p99） | Chamfer（平均 / p90） |
|---|---:|---:|---:|
| 636x360 TF32，重跑 | 0.39–0.73 cm | 0 / 0 / 0 mm | 24.6–24.9 / 48 mm |
| 到人物分割為止的加速，對 636x360 TF32 | 0.29–0.57 cm | 0 / 1 / 4–5 mm | 20.5–28.4 / 40–64 mm |
| 848x480 amp，重跑 | 0.28–0.30 cm | 0 / 0 / 0 mm | 21.6–30.1 / 37–49 mm |
| cost volume (2)，對 848x480 amp | 0.24–0.40 cm | 0 / 2–3 / 16–18 mm | 17.8–33.6 / 31–68 mm |
| 848x480 amp 對 636x360 TF32 | 0.25–0.31 cm | **14 / 165** / 1300–2900 mm | 40–44 / 94–97 mm |

加速都在重跑的差異內；換 FFS 設定讓深度移動了 636x360 的誤差，軌跡不變。

## 時間花在哪

在 636x360 TF32 那一列（每次呼叫的中位數 / p90）：

| 階段 | 中位數 | p90 | 說明 |
|---|---:|---:|---|
| depth/ffs | 83.1 ms | 89.0 ms | engine 本身 81.2 ms |
| depth/seg | 8.6 ms | 9.4 ms | 在 FFS 之後 |
| depth/to color | 4.8 ms | 8.4 ms | disparity → color 相機的深度 |
| depth/wait frame | 6.8 ms | 8.6 ms | 從追蹤 process 收 frameset |
| map：地板、nvblox depth + color | 4 ms | 18 ms | |
| map/update mesh（每 1 秒） | 35 ms | 80 ms | |
| track/orbslam3 | 32.2 ms | 39.5 ms | 30 Hz 只有 33 ms |
| final/integrate | 1.0 秒 | | |
| final/mesh | 2.0 秒 | | `cluster_connected_triangles` 1.5 秒 |
| final/save frames | 1.1 秒 | | |

## FFS

### PyTorch → TensorRT

repo 外的實驗：640x480、8 次迭代、max_disp 192；ETH3D 用整張訓練集；room_1005 對 D455 晶片深度。

| | 設定 | 延遲 | ETH3D EPE | bad-1 | room 誤差中位數 | 對 A，p99 |
|---|---|---:|---:|---:|---:|---:|
| A | PyTorch fp16 autocast | 289.5 ms | 0.187 px | 1.68% | 16 mm | — |
| B | PyTorch fp32 | 189.1 ms | 0.188 px | 1.68% | 16 mm | 70 mm |
| C | TensorRT fp16 | 97.1 ms | 0.185 px | 1.70% | 16 mm | 65 mm |
| H | TensorRT fp32 + TF32 | 94.0 ms | 0.185 px | 1.70% | 16 mm | 64 mm |
| E | TensorRT fp16，4 次迭代 | 81.9 ms | 0.194 px | 1.84% | 16 mm | 578 mm |
| I | TensorRT fp16，max_disp 128 | 72.4 ms | 0.185 px | 1.75% | 17 mm | 819 mm |
| D | TensorRT FP8 | 68.0 ms | 0.195 px | 1.93% | 18 mm | 865 mm |
| G | TensorRT FP8，max_disp 128 | 51.8 ms | 0.195 px | 1.94% | 19 mm | 1113 mm |

TensorRT（C、H）保持精度；減少迭代、max_disp 128、FP8 會掉精度。

### Cost volume

`recon/ffs_volume.py` 用較少的工作算出 FFS 的 cost volume：(1) 平移用一次 Gather，每個特徵只正規化一次（原本每個視差一次）；
(2) group-wise correlation 改用 tensor core 上的矩陣乘法，不再建 BxGxCxDxHxW 的 6 維乘積。在 PyTorch 中 (1) 逐 bit 相同，
(2) 只差捨入。build 成 848x480 engine（`equal`，每一對在同一時段量）：

| Engine（848x480） | 單獨 | 與上一列的 disparity 差：中位數 / p99 / 最大 | ETH3D EPE |
|---|---:|---|---:|
| FFS 原本的 volume，TF32 | 127.2 ms | | 0.201 px |
| (1)，TF32 | 111.4 ms | 0.0033 / 1.26 / 132 px | 0.201 px |
| (1)，fp16 autocast | 74.4 ms | | 0.2011 px |
| **(1) + (2)，fp16 autocast** | **62.4 ms** | 0.028 / 3.95 / 198 px | **0.2011 px** |

差異是 TensorRT 的捨入：同一個 ONNX build 兩次就差 0.0029 / 1.74 / 138 px。

### 輸入尺寸、max disparity 與精度

amp 是 FFS 自己的 mixed precision（網路 fp16，disparity 與取樣座標 fp32），trace 進 ONNX（`--precision amp`）；
全部 fp16 不可行，它在 128–192 px 的間距（0.125 px）和模型誤差一樣大。`settings.sh` 用 (1) 的 volume；估計誤差 = z² / (fx · baseline) · EPE。

| | 延遲 | ETH3D EPE | bad-1 | bad-2 | 2.5 m / 4 m 的估計誤差 |
|---|---:|---:|---:|---:|---:|
| 848x480、max_disp 192、TF32 | 121.0 ms | 0.201 px | 1.63% | 0.66% | 31 / 79 mm |
| **848x480、max_disp 192、amp** | **68.7 ms** | **0.201 px** | **1.64%** | **0.66%** | **31 / 79 mm** |
| 848x480、max_disp 128、TF32 | 89.8 ms | 0.201 px | 1.61% | 0.64% | 31 / 79 mm |
| 848x480、max_disp 128、amp | 55.6 ms | 0.203 px | 1.61% | 0.64% | 31 / 79 mm |
| 636x360、max_disp 192、TF32 | 64.8 ms | 0.220 px | 2.02% | 0.83% | 34 / 86 mm |

D455 深度對 848x480 max_disp 192 TF32（覆蓋率 / 中位數 / p90）：

| | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|
| **848x480、max_disp 192、amp** | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480、max_disp 192、amp、(1) + (2) | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480、max_disp 128、TF32 | 100% / 73 / 145 mm | 100% / 0 / 30 mm | 100% / 0 / 8 mm | 100% / 2 / 45 mm | 100% / 8 / 100 mm | 100% / 10 / 229 mm |
| 848x480、max_disp 128、amp | 100% / 73 / 144 mm | 100% / 0 / 30 mm | 100% / 1 / 8 mm | 100% / 2 / 45 mm | 100% / 9 / 101 mm | 100% / 10 / 233 mm |
| 636x360、max_disp 192、TF32 | 99% / 5 / 47 mm | 100% / 1 / 39 mm | 100% / 5 / 65 mm | 100% / 23 / 349 mm | 100% / 52 / 376 mm | 100% / 83 / 838 mm |

amp 對同設定的 TF32（`accuracy`；D455 中位數 / p90，對 TF32 那個）：

| 848x480 | ETH3D EPE | bad-1 | bad-2 | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_disp 192、TF32 | 0.2005 px | 1.632% | 0.661% | | | | | | |
| max_disp 192、amp | 0.2011 px | 1.639% | 0.663% | 0 / 1 mm | 0 / 1 mm | 0 / 1 mm | 1 / 5 mm | 2 / 13 mm | 2 / 12 mm |
| max_disp 128、TF32 | 0.2014 px | 1.612% | 0.639% | | | | | | |
| max_disp 128、amp | 0.2031 px | 1.610% | 0.644% | 1 / 4 mm | 0 / 10 mm | 0 / 6 mm | 1 / 8 mm | 2 / 16 mm | 2 / 14 mm |

逐層（`layers`，848x480、max_disp 192）：

| | TF32、(1) | amp、(1) | amp、(1) + (2) |
|---|---:|---:|---:|
| engine 單獨 | 111.3 ms | 69.1 ms | 56.4 ms（另一個時段） |
| cost aggregation | 25.5 ms | 7.8 ms | 7.2 ms |
| update block（GRU） | 15.9 ms | 5.9 ms | 6.1 ms |
| corr_feature_att | 9.1 ms | `Corr` kernel，各 1.7 ms | 同左 |
| group-wise correlation | 13.7 ms | 14.1 ms（fp32） | 矩陣乘法 + 1.8 ms Gather |
| 最慢的 kernel | correlation | correlation | 4.4 ms，geometry encoding 取樣 |

- 636x360 在 2–5 m 差 2–8 cm；max_disp 128 失去近距離（見下）；amp 用 EPE +0.3–0.8%、0–2 mm 換 1.6–1.8 倍速度。
- 848x480、max_disp 192、amp（main.py）：848x480 TF32 的精度、636x360 TF32 的速度。

### max_disp 128 不適合近拍

它在 0.32 m 以內匹配不到，而測試錄影有 38% 的像素在這範圍；它在那裡輸出 112–145 px，和正確像素重疊，沒有門檻能分開（`near_limit`）：

| 切掉 disparity > | 去掉的近距離像素 | 去掉的正確像素（全部 / 100–128 px） |
|---:|---:|---:|
| 110 | 90.5% | 5.31% / 66.8% |
| 115 | 88.6% | 4.25% / 51.7% |
| 120 | 85.9% | 3.02% / 34.1% |
| 124 | 82.4% | 2.05% / 20.9% |
| 126 | 79.8% | 1.66% / 16.0% |

## 人物分割

`live_20261012_160515` 的 148 張 color，和 PyTorch fp32 比對（`seg.compare`）：

| | 延遲 | 和 fp32 的 IoU | 漏掉的人像素 | 有人的幀 |
|---|---:|---:|---:|---:|
| ResNet50 PyTorch fp32 | 46.1 ms | 1 | — | 131 / 148 |
| ResNet50 PyTorch fp16 | 17.6 ms | 0.9994 | 0.1% | 131 / 148 |
| **ResNet50 TensorRT fp16** | **7.4 ms** | **0.9992** | **0.0%** | **131 / 148** |
| MobileNetV3 TensorRT fp16 | 3.3 ms | 0.675 | 36% | 112 / 148 |

MobileNetV3 會漏掉只露出一部分的人。分割放在獨立執行緒時要排在 FFS 後等約 150 ms；在 FFS 之前排進自己的 stream 則幾乎不花時間
（FFS 70.9 vs 71.0 ms，等待 0.1 ms）。不分割的話 mesh 會留下人的殘影。

## Depth 執行緒

- `DisparityToColorDepth` 不用 boolean indexing（它會拿著 GIL 同步）：深度完全相同（`warp_equal`），4.8 → 2.6 ms。
- FFS 一結束就要下一組 frameset，約 7 ms 的傳輸跟 warp、分割重疊。

## CPU governor

`jetson_clocks` 鎖了 GPU 和記憶體，CPU 仍是 `schedutil`；改成 `performance`（`system/cpu-performance.service`）後
ORB-SLAM3 32.2 → 22.1 ms，幾乎不再跳過 IR。

## Stop 之後

- `clean_mesh`：用 scipy `connected_components` 找區塊，mesh 相同，1.5 → 0.24 秒。
- `FrameWriter` 在錄影時就寫 frames/：Stop 時 1.1 秒 → 2 ms。

## 沒做

- 3DGS 訓練（離線，佔 `gs.offline` 的 95%）：每步 14.8 ms 中 GPU 只忙 10.7 ms，其餘在等同步；這次只做 live。
- FP8 / FP4、減少迭代、MobileNet、GPU 上的 ORB：會改變結果。Thor 沒有 DLA。
- 還剩：FFS 約佔 depth 執行緒的 85%，最慢的 kernel 4.4 ms；視窗的 mesh 複製可以移出 mapper 迴圈。
