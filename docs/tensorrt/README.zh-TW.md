# TensorRT 加速前後

[English](README.md)

即時 depth thread 裡的兩個網路都用 TensorRT 引擎跑在 Jetson AGX Thor 上（JetPack 7.1、pip 裝的 TensorRT 11.3）：

| | 引擎 | 之前 | 之後 | 精度 |
|---|---|---:|---:|---|
| 雙目深度：Fast-FoundationStereo | `recon/export_ffs_trt.py`，fp32 + TF32 | 289.5 ms（PyTorch fp16 autocast） | **94.0 ms**（3.1 倍） | 不變（ETH3D、D455 房間） |
| 人物分割：DeepLabV3-ResNet50 | `recon/export_seg_trt.py`，fp16 | 46.1 ms（PyTorch fp32）/ 17.6 ms（fp16） | **7.4 ms**（6.2 / 2.4 倍） | 不變（和 fp32 的 IoU 0.9992） |

TensorRT 11 只能建 strongly typed 網路（沒有 `BuilderFlag.FP16`），精度在 ONNX 裡決定：fp16 就用 fp16 的模型 trace，
fp32 則在 builder 開 TF32。引擎綁定輸入大小與 TensorRT 版本。

## Fast-FoundationStereo

建這個 repo 之前做的 FFS-on-Thor 實驗（輸入 640x480、C-FFS 權重、8 次迭代、max_disp 192）。
延遲是模型呼叫本身（正規化、補邊、推論），熱機 10 次後 100 次的平均。
ETH3D two-view：27 組訓練集、non-occluded 像素、雷射掃描 ground truth。room_1005：D455 錄影中 122 組投射器開的 IR，
和 D455 晶片深度在 0.3–2.5 m 比較（是參考值，不是 ground truth）。

| | 設定 | 延遲 | 加速 | ETH3D EPE | bad-1 | room 誤差中位數 | 誤差 < 2% | 和原本的差異 p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | PyTorch fp16 autocast（原本） | 289.5 ms | 1.0× | 0.187 px | 1.68% | 16 mm | 67.8% | — |
| B | PyTorch fp32 | 189.1 ms | 1.5× | 0.188 px | 1.68% | 16 mm | 67.8% | 70 mm |
| C | TensorRT fp16 | 97.1 ms | 3.0× | 0.185 px | 1.70% | 16 mm | 67.8% | 65 mm |
| **H** | **TensorRT fp32 + TF32（採用）** | **94.0 ms** | **3.1×** | **0.185 px** | **1.70%** | **16 mm** | **67.8%** | **64 mm** |
| E | TensorRT fp16，4 次迭代 | 81.9 ms | 3.5× | 0.194 px | 1.84% | 16 mm | 67.9% | 578 mm |
| I | TensorRT fp16，max_disp 128 | 72.4 ms | 4.0× | 0.185 px | 1.75% | 17 mm | 66.1% | 819 mm |
| D | TensorRT FP8 | 68.0 ms | 4.3× | 0.195 px | 1.93% | 18 mm | 64.3% | 865 mm |
| G | TensorRT FP8，max_disp 128 | 51.8 ms | 5.6× | 0.195 px | 1.94% | 19 mm | 62.1% | 1113 mm |

- C 和 H 的深度和原本一樣：差異中位數 0 mm，p99 只是少數邊緣像素。TF32 比 fp16 快 3 ms，精度相同。
- 在 Thor 上 PyTorch fp16 autocast 比 fp32 慢（290 vs 189 ms）。
- 減少迭代、max_disp 128（最近可量距離 0.19 m → 0.29 m）、FP8 都是用精度換速度；FP8 在 D455 的 IR 影像上損失比 ETH3D 大。

這個 repo 的引擎是 480x864（848x480 補到 32 的倍數）：在 depth thread 裡每幀 141–149 ms，含轉到 color 相機，
沒開人物分割（replay `output/live_20261012_160515/rec`）。這個尺寸沒有重新比較 PyTorch 與 TensorRT 的精度。

## 人物分割

`live_20261012_160515`：148 張 848x480 的 color 影像，131 張有人。沒有 ground truth：這裡的精度是
**和 PyTorch fp32 的 DeepLabV3-ResNet50 一致的程度**，看得出加速有沒有改變輸出，看不出模型有沒有抓到每個人。
延遲是一幀（正規化 + 模型 + 取 person 類別），熱機 10 次後 100 次的平均。
重現（[`bench_seg.py`](bench_seg.py)）：`python docs/tensorrt/bench_seg.py output/live_20261012_160515/frames`。

| | 延遲 | 和參考的 IoU | 不同的像素 | 參考的人像素漏掉 | 有人的幀 |
|---|---:|---:|---:|---:|---:|
| ResNet50 PyTorch fp32（參考） | 46.1 ms | 1 | — | — | 131 / 148 |
| ResNet50 PyTorch fp16 | 17.6 ms | 0.9994 | 0.002% | 0.1% | 131 / 148 |
| **ResNet50 TensorRT fp16（採用）** | **7.4 ms** | **0.9992** | **0.002%** | **0.0%** | **131 / 148** |
| MobileNetV3 PyTorch fp32 | 13.0 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 PyTorch fp16 | 11.6 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 TensorRT fp16 | 3.3 ms | 0.675 | 1.07% | 36% | 112 / 148 |

- TensorRT fp16 只改變 0.002% 的像素：沒有損失。
- MobileNetV3 的差距來自模型，不是精度（它的三種設定結果一樣）：只露出一部分的人（桌下的腿、半個身體在螢幕後）
  抓不到，131 張有人的幀裡有 19 張完全沒抓到。
- 離線 `recon/person_mask.py` 的 PyTorch mask 和即時 depth thread 的 TensorRT mask 一致，IoU ≥ 0.998。

### 在即時流程裡

分割接在同一幀的 FFS 後面跑，不另開 thread 同時跑：同時跑時它要排在 FFS 後面等 GPU
（每張約 150 ms，用高優先權 CUDA stream 也一樣）。replay 同一份 `rec/`（778 組 IR、29 秒）：

| | 深度 | 每個深度幀 | 拍到 → 進地圖 | 最終 mesh |
|---|---:|---:|---:|---|
| 不分割 | 6.3 Hz | 149 ms | 202 ms | 50.7 萬個三角形，有兩個人的重影 |
| 分割 | 6.0 Hz | 158 ms（mask 佔 8.5 ms） | 213 ms | 45.1 萬個三角形，沒有人 |

只露出一部分的人還是可能留下殘影（這裡是坐著的人的腿，模型部分沒抓到）：加大 `recon/person_mask.py` 的
`DILATE_PX` 或換更強的模型可以去掉。
