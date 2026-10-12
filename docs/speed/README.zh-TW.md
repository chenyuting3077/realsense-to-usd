# Thor 上的速度

[English](README.md)

在 Jetson AGX Thor（JetPack 7.1、MAXN、pip 裝的 TensorRT 11.3）上量 live pipeline 的每個節點，記錄每個節點怎麼變快、
對精度有什麼影響。目的是不改變結果的步驟，都落在 baseline 自身重跑的差異內；FFS 的設定則是用省下的時間換成更準的。

## 現在

| | |
|---|---|
| FFS | TensorRT、848x480、max_disp 192、8 次迭代、FFS 自己的 fp16 autocast、`recon/ffs_volume.py` 的 cost volume：62 ms（engine 單獨），ETH3D EPE 0.201 px |
| 人物分割 | DeepLabV3-ResNet50、TensorRT fp16：7.4 ms，在自己的 stream 上與 FFS 同時跑 |
| ORB-SLAM3 | 每幀 22.1 / 25.8 ms（中位數 / p90），CPU 為 `performance` governor |
| Live，重播 `output/live_20261012_171428/rec` | depth 13.73 Hz，capture → map 122 / 192 ms |
| Stop → 存好 mesh 與 frames | 2.4 秒 |

## 腳本

```bash
bench/live/run_baseline.sh output/live_20261012_171428/rec output/prof/base 3     # 重播 3 次 + 各階段表（同時記 tegrastats）
bench/live/run_replay.sh output/live_20261012_171428/rec output/prof/nsys nsys      # 在 nsys 下重播一次（PROFILE=1：NVTX range）
bench/live/run_replay.sh output/live_20261012_171428/rec output/prof/view view      # 開著桌面上的視窗
python -m bench.live.compare_outputs output/prof/base_1 output/prof/base_2 ...      # 一次重播的精度：ATE、深度、mesh Chamfer
bench/ffs/settings.sh <rec>,<rec>        # FFS engine：延遲、ETH3D、D455 深度（accuracy.py）
python -m bench.ffs.equal <engine A> <engine B> <rec>                         # 兩個 engine：engine 單獨的時間、|disparity 差|
python -m bench.ffs.layers <engine>                                             # engine 的逐層時間
python -m bench.ffs.near_limit <engine 192> <engine 128> <rec>,<rec>               # max_disp 128 在 0.32 m 以內
python -m bench.ffs.warp_equal <舊 ffs.py> <rec>                                    # DisparityToColorDepth 與另一個版本比較
python -m bench.seg.compare output/live_20261012_160515/frames                    # 人物分割的各種設定
```

各階段時間來自 `recon/prof.py`：`with span("depth/ffs"):` 區塊的中位數 / p90 寫進 `live.json`
（`stages`，以及追蹤 process 的 `tracker_stages`）。系統的 `trtexec` 是 TensorRT 10.13，讀不了 11.3 的 engine；
`bench/ffs/layers.py` 改從 Python 用 TensorRT 的逐層 profiler。

## Live pipeline 逐步

重播 `output/live_20261012_171428/rec`（36.5 秒、約 1000 組 IR）並加 `--no-view`，量測時 Thor 上沒有其他工作。
第一列是 1 次，其餘是 3 次的中位數。每一列都包含上一列。

| | FFS 設定 | Depth | FFS | Capture → map | ORB-SLAM3 | Stop → 完成 |
|---|---|---:|---:|---:|---:|---:|
| FFS 用完整 IR，人物分割排在 FFS 之後 | 848x480、TF32 | 5.9 Hz | | 215 / 383 ms | | |
| FFS 用 0.75 倍的 IR（CPU 為 schedutil） | 636x360、TF32 | 9.34 Hz | 83.1 ms | 151 / 276 ms | 32.2 / 39.5 ms | 4.1 秒 |
| cost volume 不重複計算 (1)、depth 執行緒 | 636x360、TF32 | 11.11 Hz | 71.3 ms | 149 / 260 ms | 32.0 / 39.7 ms | 4.6 秒 |
| CPU governor、Stop 之後 | 636x360、TF32 | 11.47 Hz | 71.0 ms | 141 / 218 ms | 22.1 / 25.5 ms | 2.2 秒 |
| 人物分割與 FFS 同時跑 | 636x360、TF32 | 12.26 Hz | 70.9 ms | 131 / 230 ms | 22.3 / 25.2 ms | 2.1 秒 |
| FFS 改為 848x480 + FFS 自己的 fp16 autocast | **848x480、amp** | 11.66 Hz | 74.8 ms | 135 / 240 ms | 22.0 / 25.3 ms | 2.1 秒 |
| correlation 改用矩陣乘法 (2) | **848x480、amp** | **13.73 Hz** | **62.0 ms** | **122 / 192 ms** | **22.1 / 25.8 ms** | **2.4 秒** |

FFS：`depth/ffs` 階段（engine 加前後處理）。視窗（Open3D，同一顆 GPU）讓 FFS 多約 3 ms、depth 少約 0.5 Hz（在「人物分割」那一列量：
11.72 Hz、FFS 73.7 ms）；給視窗的 mesh 更新（`map/update mesh` 35 / 68 ms、`map/show` 10 / 40 ms）每秒一次，在 mapper 迴圈裡。
depth 幀越多，房間裡被看到足夠次數、通過 mesh 最小權重的部分越多，最終 mesh 的三角形也越多。

### 精度

`bench.live.compare_outputs`：軌跡的 ATE、兩次都存下的幀的 |深度差|、mesh 的 Chamfer 距離（B 以軌跡對齊移到 A 的世界座標）。
每次重播選到的 projector-on frameset 不同（depth 執行緒空下來時取最新的一組），mesh 的覆蓋範圍也不同；以同一份程式重跑的差異為準。

| | ATE（rmse） | 深度（中位數 / p90 / p99） | Chamfer（平均 / p90） |
|---|---:|---:|---:|
| 636x360 TF32，重跑 | 0.39–0.73 cm | 0 / 0 / 0 mm | 24.6–24.9 / 48 mm |
| (1)、depth 執行緒、CPU、Stop 之後、人物分割：對 636x360 TF32（4 次） | 0.29–0.57 cm | 0 / 1 / 4–5 mm | 20.5–28.4 / 40–64 mm |
| 848x480 amp，重跑 | 0.28–0.30 cm | 0 / 0 / 0 mm | 21.6–30.1 / 37–49 mm |
| (2)：對 848x480 amp（3 次） | 0.24–0.40 cm | 0 / 2–3 / 16–18 mm | 17.8–33.6 / 31–68 mm |
| 848x480 amp 對 636x360 TF32（不同的 FFS 設定） | 0.25–0.31 cm | **14 / 165** / 1300–2900 mm | 40–44 / 94–97 mm |

加速的步驟都在重跑的差異內（深度差是 TensorRT 的捨入，見下）。最後一列是換 FFS 設定：深度的變化就是 636x360 的誤差
（見「輸入尺寸、max disparity 與精度」），軌跡不變。

## 時間花在哪

在 636x360 TF32 那一列，主 process（`stages`）與追蹤 process（`tracker_stages`），每次呼叫的中位數 / p90：

| 階段 | 中位數 | p90 | 說明 |
|---|---:|---:|---|
| depth/ffs | 83.1 ms | 89.0 ms | TensorRT engine 本身 81.2 ms |
| depth/seg | 8.6 ms | 9.4 ms | 在 FFS 之後、同一條 stream |
| depth/to color | 4.8 ms | 8.4 ms | disparity → color 相機的深度 |
| depth/wait frame | 6.8 ms | 8.6 ms | 要求 → 從追蹤 process 收到 frameset（pickle，約 2 MB） |
| map/floor update + apply、nvblox depth + color | 4 ms | 18 ms | mapper 不是瓶頸 |
| map/update mesh（每 1 秒） | 35 ms | 80 ms | |
| track/orbslam3 | 32.2 ms | 39.5 ms | 30 Hz 只有 33 ms；CPU 平均約 1.5 GHz |
| final/integrate | 1.0 秒 | | 341 幀的 Triton TSDF |
| final/mesh | 2.0 秒 | | 其中 `cluster_connected_triangles` 1.5 秒（單核） |
| final/save frames | 1.1 秒 | | Stop 之後寫 PNG |

depth 執行緒每幀約 97 ms，86% 是 FFS。GPU 記憶體複製可以忽略（統一記憶體，nsys：< 1% 的時間）。

## FFS

### PyTorch → TensorRT

TensorRT 11 只能建 strongly typed 網路（沒有 `BuilderFlag.FP16`）：精度在 ONNX 裡決定，fp32 的層在 builder 開 TF32。
engine 綁定輸入大小與 TensorRT 版本。

來自 repo 外的 FFS-on-Thor 實驗（輸入 640x480、C-FFS 權重、8 次迭代、max_disp 192）。延遲是模型呼叫本身（正規化、補邊、推論），
熱機 10 次後 100 次的平均。ETH3D two-view：27 組訓練集（整張）、non-occluded 像素、雷射掃描 ground truth。room_1005：D455 錄影中
122 組投射器開的 IR，和 D455 晶片深度在 0.3–2.5 m 比較（是參考值，不是 ground truth）。

| | 設定 | 延遲 | 加速 | ETH3D EPE | bad-1 | room 誤差中位數 | 誤差 < 2% | 對原本，p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | PyTorch fp16 autocast（原本） | 289.5 ms | 1.0× | 0.187 px | 1.68% | 16 mm | 67.8% | — |
| B | PyTorch fp32 | 189.1 ms | 1.5× | 0.188 px | 1.68% | 16 mm | 67.8% | 70 mm |
| C | TensorRT fp16 | 97.1 ms | 3.0× | 0.185 px | 1.70% | 16 mm | 67.8% | 65 mm |
| H | TensorRT fp32 + TF32 | 94.0 ms | 3.1× | 0.185 px | 1.70% | 16 mm | 67.8% | 64 mm |
| E | TensorRT fp16，4 次迭代 | 81.9 ms | 3.5× | 0.194 px | 1.84% | 16 mm | 67.9% | 578 mm |
| I | TensorRT fp16，max_disp 128 | 72.4 ms | 4.0× | 0.185 px | 1.75% | 17 mm | 66.1% | 819 mm |
| D | TensorRT FP8 | 68.0 ms | 4.3× | 0.195 px | 1.93% | 18 mm | 64.3% | 865 mm |
| G | TensorRT FP8，max_disp 128 | 51.8 ms | 5.6× | 0.195 px | 1.94% | 19 mm | 62.1% | 1113 mm |

- C 和 H 的深度和原本一樣：差異中位數 0 mm，p99 只是少數邊緣像素。
- 在 Thor 上 PyTorch fp16 autocast 比 fp32 慢（290 vs 189 ms）。
- 減少迭代、max_disp 128、FP8 都是用精度換速度；FP8 在 D455 的 IR 影像上損失比 ETH3D 大。
- fp16 的 engine C 不是照 FFS 自己的 autocast build 的（方式沒有記錄）；照那個 autocast trace 後，fp16 比 TF32 快 1.6–1.8 倍（見下）。

### Cost volume：同樣的值，少做很多工

`recon/ffs_volume.py` 取代 FFS `scripts/make_single_onnx.py` 的 ONNX 相容 cost volume。原本的做法用 48 次 pad + stack 建出每個視差
平移後的 target，每個視差都重新正規化一次特徵向量（48 次），group-wise correlation 則先建出 BxGxCxDxHxW 的乘積
（848x480 時是 224 通道 × 48 視差 × 120 × 216）再加總。改成：

1. 平移用在補過寬度上的一次 Gather，每個特徵向量在平移前只正規化一次（從 padding 平移進來的向量本來就是 0）；
2. correlation 用矩陣乘法：每個 group、每一列影像，所有左像素對所有右像素（216 × 216，走 tensor core），再用一次 Gather
   取出 48 條對角線；資料量是 6 維乘積的 1/6。

在 PyTorch 中兩者都等於 FFS 的 volume：(1) 逐 bit 相同，(2) 只差捨入（fp32 下 3e-7，fp16 下一個 fp16 間距 5e-4）。
build 成 848x480 的 engine（`bench.ffs.equal`：用 CUDA event 量 engine 單獨的時間、135 組 D455 的 |disparity 差|；每一對列都在同一時段量，
因為不同時段的絕對時間會漂移約 8%）：

| Engine（848x480） | 單獨 | 與上一列的 |disparity 差|：中位數 / p99 / 最大 | ETH3D EPE |
|---|---:|---|---:|
| FFS 原本的 volume，TF32 | 127.2 ms | | 0.201 px |
| (1)，TF32 | 111.4 ms | 0.0033 / 1.26 / 132 px | 0.201 px |
| (1)，fp16 autocast | 74.4 ms | （精度見下一節） | 0.2011 px |
| **(1) + (2)，fp16 autocast** | **62.4 ms** | 0.028 / 3.95 / 198 px | **0.2011 px** |

對照：同一個 ONNX build 兩次（TensorRT 每次選不同的 kernel；640x384、TF32）差 0.0029 / 1.74 / 138 px，ETH3D 0.2202 / 0.2201 px。
8 次 GRU 迭代會把捨入帶到少數邊緣像素。在 D455 上 (1) + (2) 的深度和 (1) 在每個距離都相同到毫米（下面 D455 表的兩列 amp 一樣）。
engine 檔名是 `_v3`，避免沿用較舊的 ONNX。

### 輸入尺寸、max disparity 與精度

FFS 有自己的 mixed precision（`args.mixed_precision`）：特徵、cost aggregation、GRU 網路在 fp16 autocast 下跑，disparity、
它的更新量和取樣座標維持 fp32。在 autocast 下 trace 匯出（`--precision amp`），轉型就寫在 ONNX 裡，TensorRT 照做。
整個 FFS 都用 fp16 則匯不出來（取樣座標是 fp32），也不該這樣做：fp16 在 128–192 px 之間的間距是 0.125 px，
和整個模型的誤差同一量級，8 次迭代還會累加。

`bench/ffs/settings.sh`（用 (1) 的 volume）。延遲：一組 848x480 IR 的模型呼叫（縮小、正規化、補邊、推論、disparity 放回 848x480），
熱機 10 次後 50 次的平均。ETH3D two-view：27 組訓練集，每組切成 848x480 的視窗（較小的整張），在全解析度以 non-occluded 的
雷射掃描 ground truth 評分；636x360 在縮小的影像上跑，disparity 再放大回來。估計誤差：把 ETH3D EPE 換成 D455 的深度誤差，
z² / (fx · baseline) · EPE，fx 430.6 px、baseline 95 mm（2.5 m 時每 px 0.15 m，4 m 時 0.39 m）；ETH3D 是可見光影像，
不是 D455 帶投射點的 IR。

| | 延遲 | fps | ETH3D EPE | bad-1 | bad-2 | 2.5 m / 4 m 的估計誤差 |
|---|---:|---:|---:|---:|---:|---:|
| 848x480、max_disp 192、TF32 | 121.0 ms | 8.3 | 0.201 px | 1.63% | 0.66% | 31 / 79 mm |
| **848x480、max_disp 192、amp** | **68.7 ms** | **14.5** | **0.201 px** | **1.64%** | **0.66%** | **31 / 79 mm** |
| 848x480、max_disp 128、TF32 | 89.8 ms | 11.1 | 0.201 px | 1.61% | 0.64% | 31 / 79 mm |
| 848x480、max_disp 128、amp | 55.6 ms | 18.0 | 0.203 px | 1.61% | 0.64% | 31 / 79 mm |
| 636x360、max_disp 192、TF32 | 64.8 ms | 15.4 | 0.220 px | 2.02% | 0.83% | 34 / 86 mm |

D455 沒有 ground truth：每個設定都對 848x480、max_disp 192、TF32 比（2 段錄影，每 4 組 projector-on 取 1）。
覆蓋率 / 中位數 / p90 |深度差|，其中也包含參考設定自己的雜訊：

| | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|
| **848x480、max_disp 192、amp** | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480、max_disp 192、amp、(1) + (2) | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480、max_disp 128、TF32 | 100% / 73 / 145 mm | 100% / 0 / 30 mm | 100% / 0 / 8 mm | 100% / 2 / 45 mm | 100% / 8 / 100 mm | 100% / 10 / 229 mm |
| 848x480、max_disp 128、amp | 100% / 73 / 144 mm | 100% / 0 / 30 mm | 100% / 1 / 8 mm | 100% / 2 / 45 mm | 100% / 9 / 101 mm | 100% / 10 / 233 mm |
| 636x360、max_disp 192、TF32 | 99% / 5 / 47 mm | 100% / 1 / 39 mm | 100% / 5 / 65 mm | 100% / 23 / 349 mm | 100% / 52 / 376 mm | 100% / 83 / 838 mm |

- **636x360** 距離越遠越不準：ETH3D EPE 多 10%；D455 上的差異中位數 2–3 m 是 2.3 cm、3–4 m 是 5 cm、4–5 m 是 8 cm
  （p90 35–84 cm：邊緣與遠處、紋理弱的地方）。近距離沒有損失。
- **max_disp 128** 保住遠處，但近距離不行（見「max_disp 128 不適合近拍」）。
- **amp 與同設定的 TF32 比**（`bench.ffs.accuracy <recs> <TF32 engine> <amp engine>`；D455 中位數 / p90，對 TF32 那個）：

| 848x480 | ETH3D EPE | bad-1 | bad-2 | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_disp 192、TF32 | 0.2005 px | 1.632% | 0.661% | | | | | | |
| max_disp 192、amp | 0.2011 px | 1.639% | 0.663% | 0 / 1 mm | 0 / 1 mm | 0 / 1 mm | 1 / 5 mm | 2 / 13 mm | 2 / 12 mm |
| max_disp 128、TF32 | 0.2014 px | 1.612% | 0.639% | | | | | | |
| max_disp 128、amp | 0.2031 px | 1.610% | 0.644% | 1 / 4 mm | 0 / 10 mm | 0 / 6 mm | 1 / 8 mm | 2 / 16 mm | 2 / 14 mm |

amp 快 1.6–1.8 倍，代價是 ETH3D EPE +0.3–0.8%、深度中位數 0–2 mm。這比 TensorRT 的捨入大（max_disp 128：|disparity 差|
0.085 / 6.6 / 156 px，重 build 一次是 0.003 / 1.7 / 138 px），但和輸入尺寸造成的差異相比很小。為什麼快（`bench.ffs.layers`），
848x480、max_disp 192：

| | TF32、(1) | amp、(1) | amp、(1) + (2) |
|---|---:|---:|---:|
| engine 單獨 | 111.3 ms | 69.1 ms | 56.4 ms（另一個時段） |
| cost aggregation | 25.5 ms | 7.8 ms | 7.2 ms |
| update block（GRU） | 15.9 ms | 5.9 ms | 6.1 ms |
| corr_feature_att | 9.1 ms | 融合進 `Corr` kernel（各 1.7 ms） | 同左 |
| group-wise correlation（一個融合 kernel） | 13.7 ms | 14.1 ms（FFS 用 fp32 正規化） | 矩陣乘法 + 1.8 ms 的 Gather |
| 最慢的 kernel | correlation | correlation | 4.4 ms，geometry encoding volume 的取樣 |

卷積在 fp16 下快 3–4 倍；卷積變快後，fp32 的 correlation 變成 engine 的 20%，(2) 把它拿掉。
848x480、max_disp 192 + amp 就是 main.py 的設定：848x480 TF32 的精度、636x360 TF32 的速度。

### max_disp 128 不適合近拍

它還能更快（(1) 時 55.6 ms），但比 fx · baseline / 128 = 0.32 m 近就匹配不到，而且仍會輸出 disparity。兩段測試錄影中，
38% 的像素（匹配點在右影像內）比這還近；128 engine 對它們給出 112–145 px（第 10 / 90 百分位；第 1 / 99：29 / 170），
和正確像素重疊，沒有門檻能分開（`bench.ffs.near_limit`）：

| 切掉 disparity > | 去掉的近距離像素 | 去掉的正確像素（全部 / 100–128 px 的） |
|---:|---:|---:|
| 110 | 90.5% | 5.31% / 66.8% |
| 115 | 88.6% | 4.25% / 51.7% |
| 118 | 87.1% | 3.57% / 42.0% |
| 120 | 85.9% | 3.02% / 34.1% |
| 122 | 84.3% | 2.49% / 26.8% |
| 124 | 82.4% | 2.05% / 20.9% |
| 126 | 79.8% | 1.66% / 16.0% |

## 人物分割

`live_20261012_160515`：148 張 848x480 color，131 張有人（`bench.seg.compare`）。沒有 ground truth：這裡的精度是
**和 PyTorch fp32 的 DeepLabV3-ResNet50 一不一致**，看的是加速設定會不會改變輸出，不是模型能不能找到每一個人。
延遲是一張（正規化 + 模型 + 取人的類別），熱機 10 次後 100 次的平均。

| | 延遲 | 和參考的 IoU | 不同的像素 | 參考的人像素漏掉 | 有人的幀 |
|---|---:|---:|---:|---:|---:|
| ResNet50 PyTorch fp32（參考） | 46.1 ms | 1 | — | — | 131 / 148 |
| ResNet50 PyTorch fp16 | 17.6 ms | 0.9994 | 0.002% | 0.1% | 131 / 148 |
| **ResNet50 TensorRT fp16** | **7.4 ms** | **0.9992** | **0.002%** | **0.0%** | **131 / 148** |
| MobileNetV3 PyTorch fp32 | 13.0 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 PyTorch fp16 | 11.6 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 TensorRT fp16 | 3.3 ms | 0.675 | 1.07% | 36% | 112 / 148 |

- TensorRT fp16 只改變 0.002% 的像素：沒有損失。
- MobileNetV3 的差距來自模型、不是精度（它的三種設定一致）：只露出一部分的人會漏掉（桌下的腿、半個身體在螢幕後），
  131 張有人的幀中有 19 張。
- `recon/person_mask.py`（離線）的 PyTorch mask 和 live depth 執行緒的 TensorRT mask 一致，IoU ≥ 0.998。

在 live pipeline 中（重播 `live_20261012_160515/rec`，778 組 IR，FFS 為 848x480 TF32），mask 讓人不進地圖：

| | Depth | 每個 depth 幀 | Capture → map | 最終 mesh |
|---|---:|---:|---:|---|
| 不分割 | 6.3 Hz | 149 ms | 202 ms | 50.7 萬個三角形，有兩個人的殘影 |
| 分割，在 FFS 之後、同一條 stream | 6.0 Hz | 158 ms（其中 mask 8.5 ms） | 213 ms | 45.1 萬個三角形，沒有人 |

放在獨立執行緒時，分割每張要排在 FFS 後面等約 150 ms，即使用高優先權的 CUDA stream 也一樣。在同一幀的 FFS 之前排進自己的
stream（`main.py` 的 `Depth`），它就能填 FFS 的空檔：有它 FFS 70.9 ms、沒有 71.0 ms，等 mask 的時間 8.4 → 0.1 ms。
只露出一部分的人可能還會留下痕跡（坐著的人的腿，模型漏掉一部分）：加大 `recon/person_mask.py` 的 `DILATE_PX` 或換更強的模型可以去掉。

## Depth 執行緒

`main.py`、`recon/ffs.py`：

- **warp 不用 boolean indexing**：`DisparityToColorDepth` 改成每個像素都算、無效的 scatter 到一個多出來的位置，不用 `disp[ok]`
  （它要等 GPU 算出個數，而且等的時候拿著 GIL）。108 幀中 108 幀深度完全相同（`bench.ffs.warp_equal`）；單獨 2.6 → 2.0 ms，pipeline 中 4.8 → 2.6 ms。
- **FFS 一跑完就要下一組 frameset**：從追蹤 process 傳過來（約 7 ms）跟 warp、分割重疊。再更早要的話，FFS 拿到時那幀就更舊。

## CPU governor

JetPack 7.1 的 `jetson_clocks`（這台以 `jetson-clocks.service` 開機自動執行）會鎖 GPU（1575 MHz）和記憶體（4266 MHz），
但 CPU 仍是 `schedutil`（972–2601 MHz，重播時平均約 1.5 GHz）。改成 `performance` 後 14 顆核都固定在 2601 MHz：
ORB-SLAM3 32.2 → 22.1 ms（p90 39.5 → 25.5），跳過的 IR 變少（1046 組追蹤到 1045 組，之前是 1008 組中約 1000 組），
capture → map p90 少 40 ms。`system/cpu-performance.service` 讓它每次開機自動設定（見 docs/install）。

## Stop 之後

- `clean_mesh`（`recon/reconstruct.py`）：用 numpy + scipy 的 `connected_components` 找 mesh 的區塊，取代 Open3D 單執行緒的
  `cluster_connected_triangles`：mesh 逐 byte 相同（頂點與三角形），1.5 → 0.24 秒。
- `FrameWriter`（`main.py`）：錄影時就由 4 個低優先權執行緒寫 frames/ 的 PNG；Stop 時只剩 meta.json 和 poses.npy（1.1 秒 → 2 ms）。

## 試過、沒採用

- **3DGS 訓練（離線）**：佔 `gs.offline` 的 95%（312 秒中的 296 秒）。nsys 量 3000 步：每步 14.8 ms 中 GPU 工作 10.7 ms，
  其餘是 CPU 在等同步（`loss.item()`、計時用的 `torch.cuda.synchronize()`、visibility filter 的 `nonzero()`、densification 統計用 mask 索引）。
  沒有繼續做：這次只要 live 的部分。
- 會改變結果：FP8 / FP4、FFS 少於 8 次迭代、MobileNet 分割（見上）、GPU 上的 ORB 特徵（特徵不同、軌跡就不同）。Thor 沒有 DLA。

## 還剩

- FFS 約佔 depth 執行緒的 85%。最慢的 kernel 是 geometry encoding volume 的取樣（4.4 ms），其餘分散在約 650 層。
- 給視窗的 mesh 複製（每秒一次，35 / 68 ms）可以移出 mapper 迴圈。
