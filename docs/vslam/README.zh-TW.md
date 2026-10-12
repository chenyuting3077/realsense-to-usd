# D455 上的視覺 SLAM：ORB-SLAM3 vs OpenVINS（+ RTAB-Map）、VINS-Fusion、cuVSLAM

[English](README.md)

用同樣的輸入比較 ORB-SLAM3 和另外四套常用的視覺（慣性）SLAM，
**穩定性優先、精度其次**：5 段錄影的投射器關閉 IR 對（848x480、約 28 Hz）和 IMU（200 Hz），在 Jetson AGX Thor
上離線 replay。沒有 ground truth：穩定性用錄影自己的 IMU 判斷，精度則和 ORB-SLAM3 比，以及看相機回到同一處時
深度能不能對齊。

| | 類型 | 迴圈閉合 | 執行方式 |
|---|---|---|---|
| ORB-SLAM3 | 特徵點 SLAM，雙目 + IMU | 有（地點辨識、全域 BA、地圖合併） | 錄製當時即時跑的（`frames/poses.npy`） |
| OpenVINS | MSCKF 濾波 VIO，雙目 + IMU | 無 | docker 內 ROS 2 Jazzy，即時 |
| OpenVINS + RTAB-Map | RTAB-Map 的圖建在 OpenVINS 的里程計上，迴圈閉合用 IR 對（`stereo`）或彩色 + FFS 深度（`rgbd`） | 有 | docker 內 ROS 2 Jazzy，即時 |
| VINS-Fusion | 最佳化式 VIO，雙目 + IMU；`vins_loop`：加 loop_fusion（4 自由度位姿圖） | 有 | docker 內 ROS 1 Noetic，即時 |
| cuVSLAM 17 | GPU VIO + SLAM；`cuv_vio` 雙目 + IMU、`cuv_stereo` 純雙目 | 有 | Python（pip `cuvslam`），能跑多快就多快 |

重現（錄影在 `output/live_*`，結果在 `output/vslam/`）：

```bash
docker build -t ov-rtabmap:jazzy bench/vslam && docker build -t vins:noetic bench/vslam/vins
bench/vslam/run_all.sh              # OpenVINS、+ RTAB-Map stereo / rgbd
bench/vslam/run_all_cuvslam_vins.sh # cuVSLAM（conda 環境 realsense-to-usd）、VINS-Fusion
bench/vslam/score_all.sh            # 穩定性、和 ORB-SLAM3 比、重訪（ICP）、RTAB-Map 統計；NO_ICP=1 跳過 ICP
python -m bench.vslam.summary output output/vslam
```

## 結果

5 段錄影：四段 29–41 s（2–7 m，開頭相機放在桌上靜止約 9–14 s），一段 249 s（80 m，同一條走道來回好幾趟）。

| | 有位姿 | 第一個位姿 | 跳動（最大） | 靜止時：中位數 / 最大 | 和 ORB-SLAM3，4 段短的 | 和 ORB-SLAM3，249 s¹ |
|---|---:|---:|---:|---:|---:|---:|
| ORB-SLAM3 | 98% | 0.0 s | 2（0.46 m） | 0.03 / **124** cm/s | – | – |
| OpenVINS | 75% | 9.3 s | **0** | 0.08 / 1.0 cm/s | 0.6–3.6 cm | 7.9 cm |
| OpenVINS + RTAB-Map stereo | 75% | 9.3 s | **0** | 0.08 / 1.0 cm/s | 0.6–2.3 cm | **5.3 cm** |
| OpenVINS + RTAB-Map rgbd | 75% | 9.3 s | **0** | 0.09 / 1.0 cm/s | 1.1–4.6 cm | **5.3 cm** |
| VINS-Fusion | 98% | 0.8 s | **0** | 0.04 / **68** cm/s | 0.6–3.8 cm | 14.0 cm |
| VINS-Fusion + loop_fusion | 98% | 0.8 s | **0** | 0.04 / **68** cm/s | 0.4–5.6 cm | 9.5 cm |
| cuVSLAM 雙目 + IMU | 100% | 0.0 s | **2695**（發散） | 0.37 / 764 cm/s | 16–227 cm | 發散（6600 km） |
| cuVSLAM 純雙目 | 100% | 0.0 s | 44（0.50 m） | 0.22 / 5.7 cm/s | 3.5–39 cm | 9.2 cm |

¹ 只算到結束前 15 s：最後幾秒 ORB-SLAM3 自己出錯（見下）。

- **有位姿**：錄影中有位姿的時間比例。OpenVINS 要等相機第一次移動才初始化（靜態初始化），所以每段短錄影
  開頭靜止那 9–14 s 沒有位姿。
- **跳動**：比手持相機可能的速度還快的一步（相隔不超過 0.25 s 的兩個位姿之間 > 2 m/s 或 > 6 rad/s）。
- **靜止時**：IMU 判定相機靜止時（1 s 窗口，|陀螺儀| < 0.03 rad/s、加速度計標準差 < 0.05 m/s²），位姿移動的速度；
  實際上應為 0。各窗口的中位數 / 最差的窗口。
- **和 ORB-SLAM3**：剛體對齊（不縮放）後位置的 RMSE，在 `frames/` 的影格（約 5 Hz）上。這是和 ORB-SLAM3 的差，
  不是誤差：短錄影上 ORB-SLAM3 很可能才是比較準的那個。
- **重訪**（相機回到同一處時，用 FFS 深度做 ICP，`eval_revisits.py`）：這組設定還沒跑。

![249 s](traj_250s.png)

### 各自在哪裡出錯

- **cuVSLAM 雙目 + IMU** 在這些錄影上一直沒成功：靜止開頭後相機第一次移動時就發散（4 m 的路徑差 70 cm，
  長的那段到 6600 km）。IMU 資料是對的（陀螺儀的軸與時間和影像算出的轉動一致：各軸 ×1.0、差 4 ms；加速度計
  向上讀 +g；ORB-SLAM3、OpenVINS、VINS-Fusion 用同樣的資料和外參）。還沒解決。**cuVSLAM 純雙目** 能追蹤，但會跳
  （共 44 次，最大 0.5 m），大多發生在相機從桌上拿起或放下時。
- **ORB-SLAM3 和 VINS-Fusion 在 249 s 那段的最後**：最後約 3 s 相機靜止、靠近一個箱子，投射器關閉的 IR 影像上
  出現投射光點（5% 的像素，之前約 0.5%）。兩者都以 0.6–1.2 m/s 移動，ORB-SLAM3 還橫跳 3 m；OpenVINS（有零速度
  更新）和 cuVSLAM 純雙目都停在原地。光點是可能的原因，尚未證實。
- **OpenVINS** 需要兩個設定才穩：關掉線上相機–IMU 校正（開著時，第一次更新就把兩台相機的外參轉了 8°、時間差
  變成 43 ms，濾波器發散到 22 km），以及零速度更新（沒有它在桌上 5 s 沉了 35 cm：41 s 那段和 ORB-SLAM3 的差
  14.4 → 3.6 cm）。
- **RTAB-Map**（建在 OpenVINS 上）在 249 s 那段找到 101（stereo）/ 105（rgbd）次迴圈閉合和約 125 條鄰近連結，
  把 OpenVINS 和 ORB-SLAM3 的差從 7.9 cm 拉到 5.3 cm。短錄影上找到的很少。

### 成本

| | 每張 | 備註 |
|---|---:|---|
| ORB-SLAM3 | 35 ms（p90 42） | 即時；249 s 那段 6378 組 IR 中略過 760 組（忙不過來） |
| OpenVINS | 14 ms（p90 19） | Thor CPU |
| RTAB-Map | 每個節點 110–200 ms | 每秒 2 個節點，CPU |
| VINS-Fusion | – | 即時；它每兩組 IR 處理一組（多執行緒模式） |
| cuVSLAM 純雙目 | 0.5–0.7 ms（p90 0.8–1.5） | GPU |
| cuVSLAM 雙目 + IMU | 1.1–1.9 ms（p90 9–49） | GPU |

這些測試途中 CPU governor 改成 `performance`（所有核心固定 2.6 GHz），CPU 時間只是大概。

## 結論

- **最穩**：OpenVINS（+ RTAB-Map）：沒有跳動，靜止時最多 1 cm/s，連 ORB-SLAM3 和 VINS-Fusion 在長錄影最後出錯
  的地方也穩。弱點在開頭：相機動之前沒有位姿。
- **VINS-Fusion** 移動中一樣平順（沒有跳動），0.8 s 起就有位姿，但長錄影最後和 ORB-SLAM3 一樣漂；迴圈閉合在
  長錄影上有幫助（和 ORB-SLAM3 的差 14.0 → 9.5 cm）。
- **ORB-SLAM3** 精度好，但追蹤一變難就會跳。
- **cuVSLAM** 快得多，但在這份資料上最不穩（約 28 Hz、投射器交替的 IR）。

## main.py 的追蹤

main.py 用原生編譯的 OpenVINS + RTAB-Map 追蹤（`recon/vio.py`、`recon/vio_py`）：上面的設定，RTAB-Map 在背景執行緒
每 0.5 s 處理一個關鍵幀，OpenVINS 開始前的 IR（IMU 判定相機靜止）給它的第一個位姿。和 docker 版在同樣 5 段錄影上比
（`python -m bench.vslam.run_native`）：

| | 有位姿 | 跳動 | 靜止時最大 | 和 ORB-SLAM3，4 段短的 | 和 ORB-SLAM3，249 s（全段） |
|---|---:|---:|---:|---:|---:|
| docker OpenVINS + RTAB-Map stereo | 75% | 0 | 1.0 cm/s | 0.6–2.3 cm | 23.6 cm |
| recon/vio.py | **91%** | 0 | 0.5 cm/s | 1.5–3.7 cm | 23.3 cm |
