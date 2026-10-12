# Speed on Thor

[繁體中文](README.zh-TW.md)

How each node of the live pipeline got faster on the Jetson AGX Thor (JetPack 7.1, MAXN, TensorRT 11.3), and the effect on accuracy.

## Now

| | |
|---|---|
| FFS | TensorRT, 848x480, max_disp 192, 8 iterations, FFS's own fp16 autocast, the cost volumes of `recon/ffs_volume.py`: 62 ms, ETH3D EPE 0.201 px |
| Person segmentation | DeepLabV3-ResNet50, TensorRT fp16: 7.4 ms, beside FFS |
| ORB-SLAM3 | 22.1 / 25.8 ms a frame (median / p90), CPU governor `performance` |
| Live (replay of `output/live_20261012_171428/rec`) | depth 13.73 Hz, capture → map 122 / 192 ms |
| Stop → mesh and frames saved | 2.4 s |

## Scripts

```bash
bench/live/run_baseline.sh <rec> <out> 3                 # 3 replays + stage table (recon/prof.py spans, tegrastats)
bench/live/run_replay.sh <rec> <out> [nsys|view]         # one replay, under nsys or with the window
python -m bench.live.compare_outputs <run A> <run B> ...  # ATE, depth, mesh Chamfer between runs
bench/ffs/settings.sh <rec>,<rec>                        # build FFS engines; latency, ETH3D, D455 depth
python -m bench.ffs.equal <engine A> <engine B> <rec>     # engine time, disparity difference
python -m bench.ffs.layers <engine>                       # per-layer time (the system trtexec cannot load TensorRT 11.3 engines)
python -m bench.ffs.near_limit <engine 192> <engine 128> <rec>,<rec>
python -m bench.ffs.warp_equal <old ffs.py> <rec>
python -m bench.seg.compare <frames>
```

## The live pipeline, step by step

Replay of `output/live_20261012_171428/rec` with `--no-view`, nothing else on the Thor; median of 3 runs (first row: 1 run).

| | FFS setting | Depth | FFS | Capture → map | ORB-SLAM3 | Stop → done |
|---|---|---:|---:|---:|---:|---:|
| Full IR, segmentation after FFS | 848x480, TF32 | 5.9 Hz | | 215 / 383 ms | | |
| [IR at 0.75x](#input-size-max-disparity-and-precision) (CPU on schedutil) | 636x360, TF32 | 9.34 Hz | 83.1 ms | 151 / 276 ms | 32.2 / 39.5 ms | 4.1 s |
| [Cost volumes (1)](#cost-volumes), [depth thread](#depth-thread) | 636x360, TF32 | 11.11 Hz | 71.3 ms | 149 / 260 ms | 32.0 / 39.7 ms | 4.6 s |
| [CPU governor](#cpu-governor), [after Stop](#after-stop) | 636x360, TF32 | 11.47 Hz | 71.0 ms | 141 / 218 ms | 22.1 / 25.5 ms | 2.2 s |
| [Segmentation beside FFS](#person-segmentation) | 636x360, TF32 | 12.26 Hz | 70.9 ms | 131 / 230 ms | 22.3 / 25.2 ms | 2.1 s |
| [848x480 with fp16 autocast](#input-size-max-disparity-and-precision) | **848x480, amp** | 11.66 Hz | 74.8 ms | 135 / 240 ms | 22.0 / 25.3 ms | 2.1 s |
| [Cost volumes (2)](#cost-volumes) | **848x480, amp** | **13.73 Hz** | **62.0 ms** | **122 / 192 ms** | **22.1 / 25.8 ms** | **2.4 s** |

The window costs ~3 ms of FFS (11.72 Hz at the segmentation row).

### Accuracy

`compare_outputs` against the first run; re-runs of the same code are the yardstick (each picks other framesets).

| | ATE (rmse) | Depth (median / p90 / p99) | Chamfer (mean / p90) |
|---|---:|---:|---:|
| 636x360 TF32, re-runs | 0.39–0.73 cm | 0 / 0 / 0 mm | 24.6–24.9 / 48 mm |
| speed-ups up to segmentation, vs. 636x360 TF32 | 0.29–0.57 cm | 0 / 1 / 4–5 mm | 20.5–28.4 / 40–64 mm |
| 848x480 amp, re-runs | 0.28–0.30 cm | 0 / 0 / 0 mm | 21.6–30.1 / 37–49 mm |
| cost volumes (2), vs. 848x480 amp | 0.24–0.40 cm | 0 / 2–3 / 16–18 mm | 17.8–33.6 / 31–68 mm |
| 848x480 amp vs. 636x360 TF32 | 0.25–0.31 cm | **14 / 165** / 1300–2900 mm | 40–44 / 94–97 mm |

The speed-ups stay within the re-runs; the change of FFS setting moves the depth by the 636x360 error, not the trajectory.

## Where the time went

At the 636x360 TF32 row (median / p90 per call):

| Stage | median | p90 | Note |
|---|---:|---:|---|
| depth/ffs | 83.1 ms | 89.0 ms | engine alone 81.2 ms |
| depth/seg | 8.6 ms | 9.4 ms | after FFS |
| depth/to color | 4.8 ms | 8.4 ms | disparity → depth in the color camera |
| depth/wait frame | 6.8 ms | 8.6 ms | frameset from the tracker process |
| map: floor, nvblox depth + color | 4 ms | 18 ms | |
| map/update mesh (every 1 s) | 35 ms | 80 ms | |
| track/orbslam3 | 32.2 ms | 39.5 ms | 30 Hz leaves 33 ms |
| final/integrate | 1.0 s | | |
| final/mesh | 2.0 s | | `cluster_connected_triangles` 1.5 s |
| final/save frames | 1.1 s | | |

## FFS

### PyTorch → TensorRT

Experiments outside this repo: 640x480, 8 iterations, max_disp 192; ETH3D on the whole training pairs; room_1005 against the D455's own depth.

| | Setting | Latency | ETH3D EPE | bad-1 | room median error | vs. A, p99 |
|---|---|---:|---:|---:|---:|---:|
| A | PyTorch fp16 autocast | 289.5 ms | 0.187 px | 1.68% | 16 mm | — |
| B | PyTorch fp32 | 189.1 ms | 0.188 px | 1.68% | 16 mm | 70 mm |
| C | TensorRT fp16 | 97.1 ms | 0.185 px | 1.70% | 16 mm | 65 mm |
| H | TensorRT fp32 + TF32 | 94.0 ms | 0.185 px | 1.70% | 16 mm | 64 mm |
| E | TensorRT fp16, 4 iterations | 81.9 ms | 0.194 px | 1.84% | 16 mm | 578 mm |
| I | TensorRT fp16, max_disp 128 | 72.4 ms | 0.185 px | 1.75% | 17 mm | 819 mm |
| D | TensorRT FP8 | 68.0 ms | 0.195 px | 1.93% | 18 mm | 865 mm |
| G | TensorRT FP8, max_disp 128 | 51.8 ms | 0.195 px | 1.94% | 19 mm | 1113 mm |

TensorRT (C, H) keeps the accuracy; fewer iterations, max_disp 128 and FP8 lose it.

### Cost volumes

`recon/ffs_volume.py` computes FFS's cost volumes with less work: (1) the shift as one Gather, each feature normalized once
instead of once per disparity; (2) the group-wise correlation as matrix products on the tensor cores, without the 6-D
BxGxCxDxHxW product. In PyTorch (1) is bit-identical, (2) equal up to rounding. As 848x480 engines (`equal`, one session per pair):

| Engine (848x480) | Alone | Disparity difference to the row above: median / p99 / max | ETH3D EPE |
|---|---:|---|---:|
| FFS's volumes, TF32 | 127.2 ms | | 0.201 px |
| (1), TF32 | 111.4 ms | 0.0033 / 1.26 / 132 px | 0.201 px |
| (1), fp16 autocast | 74.4 ms | | 0.2011 px |
| **(1) + (2), fp16 autocast** | **62.4 ms** | 0.028 / 3.95 / 198 px | **0.2011 px** |

The differences are TensorRT's rounding: the same ONNX built twice differs by 0.0029 / 1.74 / 138 px.

### Input size, max disparity and precision

amp is FFS's own mixed precision (networks in fp16, disparity and sampling coordinates in fp32), traced into the ONNX
(`--precision amp`); all-fp16 cannot work, its step at 128–192 px (0.125 px) is the size of the model's error.
`settings.sh` with volumes (1); est. error = z² / (fx · baseline) · EPE.

| | Latency | ETH3D EPE | bad-1 | bad-2 | Est. error at 2.5 m / 4 m |
|---|---:|---:|---:|---:|---:|
| 848x480, max_disp 192, TF32 | 121.0 ms | 0.201 px | 1.63% | 0.66% | 31 / 79 mm |
| **848x480, max_disp 192, amp** | **68.7 ms** | **0.201 px** | **1.64%** | **0.66%** | **31 / 79 mm** |
| 848x480, max_disp 128, TF32 | 89.8 ms | 0.201 px | 1.61% | 0.64% | 31 / 79 mm |
| 848x480, max_disp 128, amp | 55.6 ms | 0.203 px | 1.61% | 0.64% | 31 / 79 mm |
| 636x360, max_disp 192, TF32 | 64.8 ms | 0.220 px | 2.02% | 0.83% | 34 / 86 mm |

D455 depth against 848x480 max_disp 192 TF32 (coverage / median / p90):

| | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|
| **848x480, max_disp 192, amp** | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480, max_disp 192, amp, (1) + (2) | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480, max_disp 128, TF32 | 100% / 73 / 145 mm | 100% / 0 / 30 mm | 100% / 0 / 8 mm | 100% / 2 / 45 mm | 100% / 8 / 100 mm | 100% / 10 / 229 mm |
| 848x480, max_disp 128, amp | 100% / 73 / 144 mm | 100% / 0 / 30 mm | 100% / 1 / 8 mm | 100% / 2 / 45 mm | 100% / 9 / 101 mm | 100% / 10 / 233 mm |
| 636x360, max_disp 192, TF32 | 99% / 5 / 47 mm | 100% / 1 / 39 mm | 100% / 5 / 65 mm | 100% / 23 / 349 mm | 100% / 52 / 376 mm | 100% / 83 / 838 mm |

amp against TF32 at the same setting (`accuracy`; D455 median / p90 against the TF32 one):

| 848x480 | ETH3D EPE | bad-1 | bad-2 | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_disp 192, TF32 | 0.2005 px | 1.632% | 0.661% | | | | | | |
| max_disp 192, amp | 0.2011 px | 1.639% | 0.663% | 0 / 1 mm | 0 / 1 mm | 0 / 1 mm | 1 / 5 mm | 2 / 13 mm | 2 / 12 mm |
| max_disp 128, TF32 | 0.2014 px | 1.612% | 0.639% | | | | | | |
| max_disp 128, amp | 0.2031 px | 1.610% | 0.644% | 1 / 4 mm | 0 / 10 mm | 0 / 6 mm | 1 / 8 mm | 2 / 16 mm | 2 / 14 mm |

Per layer (`layers`, 848x480, max_disp 192):

| | TF32, (1) | amp, (1) | amp, (1) + (2) |
|---|---:|---:|---:|
| engine alone | 111.3 ms | 69.1 ms | 56.4 ms (another session) |
| cost aggregation | 25.5 ms | 7.8 ms | 7.2 ms |
| update block (GRU) | 15.9 ms | 5.9 ms | 6.1 ms |
| corr_feature_att | 9.1 ms | `Corr` kernels, 1.7 ms each | the same |
| group-wise correlation | 13.7 ms | 14.1 ms (fp32) | matrix products + 1.8 ms Gather |
| slowest kernel | correlation | correlation | 4.4 ms, geometry encoding sampling |

- 636x360 loses 2–8 cm at 2–5 m; max_disp 128 loses the close range (below); amp costs +0.3–0.8% EPE and 0–2 mm for 1.6–1.8× speed.
- 848x480, max_disp 192, amp (main.py): the accuracy of 848x480 TF32 at the speed of 636x360 TF32.

### max_disp 128 does not suit close-ups

It cannot match closer than 0.32 m, and 38% of the test recordings' pixels are closer; there it outputs 112–145 px,
overlapping good pixels, so no threshold separates them (`near_limit`):

| Cut at disparity > | Near pixels removed | Good pixels removed (all / 100–128 px) |
|---:|---:|---:|
| 110 | 90.5% | 5.31% / 66.8% |
| 115 | 88.6% | 4.25% / 51.7% |
| 120 | 85.9% | 3.02% / 34.1% |
| 124 | 82.4% | 2.05% / 20.9% |
| 126 | 79.8% | 1.66% / 16.0% |

## Person segmentation

148 color frames of `live_20261012_160515`, agreement with PyTorch fp32 (`seg.compare`):

| | Latency | IoU vs. fp32 | Person pixels missed | Frames with people |
|---|---:|---:|---:|---:|
| ResNet50 PyTorch fp32 | 46.1 ms | 1 | — | 131 / 148 |
| ResNet50 PyTorch fp16 | 17.6 ms | 0.9994 | 0.1% | 131 / 148 |
| **ResNet50 TensorRT fp16** | **7.4 ms** | **0.9992** | **0.0%** | **131 / 148** |
| MobileNetV3 TensorRT fp16 | 3.3 ms | 0.675 | 36% | 112 / 148 |

MobileNetV3 misses partly visible people. In a thread of its own the segmentation waited ~150 ms behind FFS; queued on its own
stream right before FFS it costs nothing (FFS 70.9 vs 71.0 ms, 0.1 ms wait). Without it the mesh keeps ghosts of people.

## Depth thread

- `DisparityToColorDepth` without boolean indexing (which syncs holding the GIL): identical depth (`warp_equal`), 4.8 → 2.6 ms.
- The next frameset is requested when FFS ends, so its ~7 ms transfer overlaps the warp and segmentation.

## CPU governor

`jetson_clocks` locks GPU and memory but leaves the CPUs on `schedutil`; with `performance` (`system/cpu-performance.service`)
ORB-SLAM3 goes 32.2 → 22.1 ms and nearly no IR pair is skipped.

## After Stop

- `clean_mesh`: pieces via scipy `connected_components`, the same mesh, 1.5 → 0.24 s.
- `FrameWriter` writes frames/ during recording: 1.1 s → 2 ms at Stop.

## Not done

- 3DGS training (offline, 95% of `gs.offline`): the GPU is busy 10.7 of 14.8 ms a step, the rest waits on syncs; only live was asked for.
- FP8 / FP4, fewer iterations, MobileNet, ORB on the GPU: they change the result. Thor has no DLA.
- Left: FFS is ~85% of the depth thread, its slowest kernel 4.4 ms; the window's mesh copy could leave the mapper loop.
