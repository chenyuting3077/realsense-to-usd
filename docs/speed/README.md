# Speed on Thor

[繁體中文](README.zh-TW.md)

Every node of the live pipeline timed on the Jetson AGX Thor (JetPack 7.1, MAXN, TensorRT 11.3 from pip), how each got
faster, and what that did to the accuracy. Where a step was meant to keep the result, it stays within the spread of the
baseline's own re-runs; the FFS setting was changed for accuracy with the time saved.

## Now

| | |
|---|---|
| FFS | TensorRT, 848x480, max_disp 192, 8 iterations, FFS's own fp16 autocast, the cost volumes of `recon/ffs_volume.py`: 62 ms (engine alone), ETH3D EPE 0.201 px |
| Person segmentation | DeepLabV3-ResNet50, TensorRT fp16: 7.4 ms, on a stream of its own beside FFS |
| ORB-SLAM3 | 22.1 / 25.8 ms a frame (median / p90), CPUs on the `performance` governor |
| Live, replay of `output/live_20261012_171428/rec` | depth 13.73 Hz, capture → map 122 / 192 ms |
| Stop → mesh and frames saved | 2.4 s |

## Scripts

```bash
bench/run_baseline.sh output/live_20261012_171428/rec output/prof/base 3     # 3 replays + stage table (tegrastats beside)
bench/run_replay.sh output/live_20261012_171428/rec output/prof/nsys nsys      # one replay under nsys (PROFILE=1: NVTX ranges)
bench/run_replay.sh output/live_20261012_171428/rec output/prof/view view      # with the window on the desktop
python -m bench.compare_outputs output/prof/base_1 output/prof/base_2 ...      # accuracy of a run: ATE, depth, mesh Chamfer
bench/ffs_settings.sh <rec>,<rec>        # FFS engines: latency, ETH3D, D455 depth (ffs_accuracy.py)
python -m bench.ffs_equal <engine A> <engine B> <rec>                         # two engines: engine alone, |disparity difference|
python -m bench.ffs_layers <engine>                                             # per-layer time of an engine
python -m bench.near_limit <engine 192> <engine 128> <rec>,<rec>               # max_disp 128 below 0.32 m
python -m bench.warp_equal <old ffs.py> <rec>                                   # DisparityToColorDepth against another version
python -m bench.bench_seg output/live_20261012_160515/frames                    # person segmentation settings
```

The stages come from `recon/prof.py`: `with span("depth/ffs"):` blocks write their median / p90 to `live.json`
(`stages`, and `tracker_stages` from the tracker process). The system `trtexec` is TensorRT 10.13 and cannot load the
11.3 engines; `ffs_layers.py` uses TensorRT's per-layer profiler from Python instead.

## The live pipeline, step by step

Replay of `output/live_20261012_171428/rec` (36.5 s, ~1000 IR pairs) with `--no-view`, nothing else running on the Thor.
The first row is one run; the others are the median of 3. Each row adds to the one above.

| | FFS setting | Depth | FFS | Capture → map | ORB-SLAM3 | Stop → done |
|---|---|---:|---:|---:|---:|---:|
| FFS on the full IR, segmentation after FFS | 848x480, TF32 | 5.9 Hz | | 215 / 383 ms | | |
| [FFS on the IR at 0.75x](#input-size-max-disparity-and-precision) (CPU on schedutil) | 636x360, TF32 | 9.34 Hz | 83.1 ms | 151 / 276 ms | 32.2 / 39.5 ms | 4.1 s |
| [Cost volumes without repeated work](#cost-volumes-the-same-values-with-less-work) (1), [depth thread](#depth-thread) | 636x360, TF32 | 11.11 Hz | 71.3 ms | 149 / 260 ms | 32.0 / 39.7 ms | 4.6 s |
| [CPU governor](#cpu-governor), [after Stop](#after-stop) | 636x360, TF32 | 11.47 Hz | 71.0 ms | 141 / 218 ms | 22.1 / 25.5 ms | 2.2 s |
| [Segmentation beside FFS](#person-segmentation) | 636x360, TF32 | 12.26 Hz | 70.9 ms | 131 / 230 ms | 22.3 / 25.2 ms | 2.1 s |
| [FFS at 848x480 with its own fp16 autocast](#input-size-max-disparity-and-precision) | **848x480, amp** | 11.66 Hz | 74.8 ms | 135 / 240 ms | 22.0 / 25.3 ms | 2.1 s |
| [Correlation as matrix products](#cost-volumes-the-same-values-with-less-work) (2) | **848x480, amp** | **13.73 Hz** | **62.0 ms** | **122 / 192 ms** | **22.1 / 25.8 ms** | **2.4 s** |

FFS: the `depth/ffs` stage (engine with pre/post-processing). The window (Open3D, on the same GPU) costs ~3 ms of FFS and
~0.5 Hz of depth (at the segmentation row: 11.72 Hz, FFS 73.7 ms); its mesh updates (`map/update mesh` 35 / 68 ms,
`map/show` 10 / 40 ms) run once a second in the mapper loop. More depth frames give more of the room seen often enough to
pass the mesh's minimum weight: more triangles in the final mesh.

### Accuracy

`compare_outputs`: ATE of the trajectories, |depth difference| on the frames both runs stored, mesh Chamfer distance (B moved
into A's world by the trajectory alignment). Each run picks other projector-on framesets (the newest one when the depth thread
is free), so the meshes differ in coverage too; the spread of re-runs of the same code is the yardstick.

| | ATE (rmse) | Depth (median / p90 / p99) | Chamfer (mean / p90) |
|---|---:|---:|---:|
| 636x360 TF32, re-runs | 0.39–0.73 cm | 0 / 0 / 0 mm | 24.6–24.9 / 48 mm |
| (1), depth thread, CPU, after Stop, segmentation: against 636x360 TF32 (4 runs) | 0.29–0.57 cm | 0 / 1 / 4–5 mm | 20.5–28.4 / 40–64 mm |
| 848x480 amp, re-runs | 0.28–0.30 cm | 0 / 0 / 0 mm | 21.6–30.1 / 37–49 mm |
| (2): against 848x480 amp (3 runs) | 0.24–0.40 cm | 0 / 2–3 / 16–18 mm | 17.8–33.6 / 31–68 mm |
| 848x480 amp against 636x360 TF32 (another FFS setting) | 0.25–0.31 cm | **14 / 165** / 1300–2900 mm | 40–44 / 94–97 mm |

The speed-ups stay within the re-runs (their depth differences are TensorRT's rounding, below). The last row is the change of
FFS setting: the depth moves by the 636x360 error ([below](#input-size-max-disparity-and-precision)), the trajectory not.

## Where the time went

At the 636x360 TF32 row, main process (`stages`) and tracker process (`tracker_stages`), median / p90 per call:

| Stage | median | p90 | Note |
|---|---:|---:|---|
| depth/ffs | 83.1 ms | 89.0 ms | the TensorRT engine alone 81.2 ms |
| depth/seg | 8.6 ms | 9.4 ms | after FFS, on the same stream |
| depth/to color | 4.8 ms | 8.4 ms | disparity → depth in the color camera |
| depth/wait frame | 6.8 ms | 8.6 ms | request → frameset from the tracker process (pickled, ~2 MB) |
| map/floor update + apply, nvblox depth + color | 4 ms | 18 ms | the mapper is not the limit |
| map/update mesh (every 1 s) | 35 ms | 80 ms | |
| track/orbslam3 | 32.2 ms | 39.5 ms | 30 Hz leaves 33 ms; CPUs at ~1.5 GHz on average |
| final/integrate | 1.0 s | | Triton TSDF of 341 frames |
| final/mesh | 2.0 s | | of it `cluster_connected_triangles` 1.5 s (one core) |
| final/save frames | 1.1 s | | PNGs after Stop |

The depth thread took ~97 ms a frame, 86% of it FFS. GPU memory copies are negligible (unified memory, nsys: < 1% of the time).

## FFS

### PyTorch → TensorRT

TensorRT 11 builds strongly typed networks only (no `BuilderFlag.FP16`): the precision is set in the ONNX, and TF32 is enabled
in the builder for fp32 layers. An engine is tied to its input size and TensorRT version.

From FFS-on-Thor experiments outside this repo (640x480 input, C-FFS weights, 8 iterations, max_disp 192). Latency is the model
call (normalization, padding, inference), mean of 100 after 10 warm-up calls. ETH3D two-view: the 27 training pairs whole,
non-occluded pixels, laser-scan ground truth. room_1005: 122 projector-on IR pairs from a D455 recording, compared with the
D455's own depth at 0.3–2.5 m (a reference, not ground truth).

| | Setting | Latency | Speed-up | ETH3D EPE | bad-1 | room median error | error < 2% | vs. original, p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | PyTorch fp16 autocast (original) | 289.5 ms | 1.0× | 0.187 px | 1.68% | 16 mm | 67.8% | — |
| B | PyTorch fp32 | 189.1 ms | 1.5× | 0.188 px | 1.68% | 16 mm | 67.8% | 70 mm |
| C | TensorRT fp16 | 97.1 ms | 3.0× | 0.185 px | 1.70% | 16 mm | 67.8% | 65 mm |
| H | TensorRT fp32 + TF32 | 94.0 ms | 3.1× | 0.185 px | 1.70% | 16 mm | 67.8% | 64 mm |
| E | TensorRT fp16, 4 iterations | 81.9 ms | 3.5× | 0.194 px | 1.84% | 16 mm | 67.9% | 578 mm |
| I | TensorRT fp16, max_disp 128 | 72.4 ms | 4.0× | 0.185 px | 1.75% | 17 mm | 66.1% | 819 mm |
| D | TensorRT FP8 | 68.0 ms | 4.3× | 0.195 px | 1.93% | 18 mm | 64.3% | 865 mm |
| G | TensorRT FP8, max_disp 128 | 51.8 ms | 5.6× | 0.195 px | 1.94% | 19 mm | 62.1% | 1113 mm |

- C and H give the original depth: median difference 0 mm, the p99 is a few edge pixels.
- On Thor PyTorch fp16 autocast is slower than fp32 (290 vs 189 ms).
- Fewer iterations, max_disp 128 and FP8 trade accuracy for speed; FP8 loses more on the D455 IR images than on ETH3D.
- The fp16 engine C was built another way than FFS's own autocast (not recorded); traced under that autocast, fp16 is
  1.6–1.8× faster than TF32 ([below](#input-size-max-disparity-and-precision)).

### Cost volumes: the same values with less work

`recon/ffs_volume.py` replaces the ONNX-compatible cost volumes of FFS's `scripts/make_single_onnx.py`, which build the target
shifted by every disparity with 48 pads + a stack, normalize each feature vector once per disparity (48 times), and form the
group-wise correlation as a BxGxCxDxHxW product (224 channels × 48 disparities × 120 × 216 at 848x480) before summing it:

1. the shift as one Gather over a padded width, each feature vector normalized once before the shift (a vector shifted in from
   the padding is zero either way);
2. the correlation as matrix products: per group and image row, every left pixel with every right pixel (216 × 216 on the
   tensor cores), then the 48 diagonals picked by one Gather; 6× less data than the 6-D product.

In PyTorch both give FFS's volumes: (1) bit for bit, (2) up to rounding (3e-7 in fp32, one fp16 step of 5e-4 in fp16). As engines
at 848x480 (`ffs_equal`: the engine alone with CUDA events, and |disparity difference| on 135 D455 pairs; each pair of rows
measured in one session, as absolute times drift ~8% between sessions):

| Engine (848x480) | Alone | |disparity difference| to the row above: median / p99 / max | ETH3D EPE |
|---|---:|---|---:|
| FFS's volumes, TF32 | 127.2 ms | | 0.201 px |
| (1), TF32 | 111.4 ms | 0.0033 / 1.26 / 132 px | 0.201 px |
| (1), fp16 autocast | 74.4 ms | ([precision below](#input-size-max-disparity-and-precision)) | 0.2011 px |
| **(1) + (2), fp16 autocast** | **62.4 ms** | 0.028 / 3.95 / 198 px | **0.2011 px** |

For scale: the same ONNX built twice (TensorRT picks other kernels each build; 640x384, TF32) differs by 0.0029 / 1.74 / 138 px,
ETH3D 0.2202 / 0.2201 px. The 8 GRU iterations carry the rounding on to a few edge pixels. On the D455, (1) + (2) gives the depth
of (1) to the millimetre at every distance (the two amp rows of the D455 table below are the same). The engines are named `_v3`
so an older ONNX is not reused.

### Input size, max disparity and precision

FFS has its own mixed precision (`args.mixed_precision`): the feature, cost aggregation and GRU networks under fp16 autocast, the
disparity, its updates and the sampling coordinates in fp32. Exported traced under it (`--precision amp`), the casts are in the
ONNX and TensorRT keeps them. All of FFS in fp16 does not export (the sampling coordinates are fp32) and would not do: fp16 has
a step of 0.125 px between 128 and 192 px, the size of the model's whole error, added up over 8 iterations.

`ffs_settings.sh` (with the volumes of (1)). Latency: one 848x480 IR pair, the model call (downscaling, normalization, padding,
inference, disparity back to 848x480), mean of 50 after 10 warm-up calls. ETH3D two-view: the 27 training pairs, each cut to an
848x480 window (smaller ones whole), scored at full resolution on the non-occluded laser-scan ground truth; 636x360 runs on the
downscaled pair, its disparity scaled back up. Est. error: the ETH3D EPE as a D455 depth error, z² / (fx · baseline) · EPE
with fx 430.6 px and baseline 95 mm (0.15 m per px at 2.5 m, 0.39 m per px at 4 m); ETH3D is visible-light images, not the
D455's IR with the projector dots.

| | Latency | fps | ETH3D EPE | bad-1 | bad-2 | Est. error at 2.5 m / 4 m |
|---|---:|---:|---:|---:|---:|---:|
| 848x480, max_disp 192, TF32 | 121.0 ms | 8.3 | 0.201 px | 1.63% | 0.66% | 31 / 79 mm |
| **848x480, max_disp 192, amp** | **68.7 ms** | **14.5** | **0.201 px** | **1.64%** | **0.66%** | **31 / 79 mm** |
| 848x480, max_disp 128, TF32 | 89.8 ms | 11.1 | 0.201 px | 1.61% | 0.64% | 31 / 79 mm |
| 848x480, max_disp 128, amp | 55.6 ms | 18.0 | 0.203 px | 1.61% | 0.64% | 31 / 79 mm |
| 636x360, max_disp 192, TF32 | 64.8 ms | 15.4 | 0.220 px | 2.02% | 0.83% | 34 / 86 mm |

On the D455 there is no ground truth: each setting against 848x480, max_disp 192, TF32 (2 recordings, every 4th projector-on
pair). Coverage / median / p90 |depth difference|, which includes the noise of the reference itself:

| | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|
| **848x480, max_disp 192, amp** | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480, max_disp 192, amp, (1) + (2) | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480, max_disp 128, TF32 | 100% / 73 / 145 mm | 100% / 0 / 30 mm | 100% / 0 / 8 mm | 100% / 2 / 45 mm | 100% / 8 / 100 mm | 100% / 10 / 229 mm |
| 848x480, max_disp 128, amp | 100% / 73 / 144 mm | 100% / 0 / 30 mm | 100% / 1 / 8 mm | 100% / 2 / 45 mm | 100% / 9 / 101 mm | 100% / 10 / 233 mm |
| 636x360, max_disp 192, TF32 | 99% / 5 / 47 mm | 100% / 1 / 39 mm | 100% / 5 / 65 mm | 100% / 23 / 349 mm | 100% / 52 / 376 mm | 100% / 83 / 838 mm |

- **636x360** loses with distance: ETH3D EPE +10%, on the D455 a median difference of 2.3 cm at 2–3 m, 5 cm at 3–4 m and
  8 cm at 4–5 m (p90 35–84 cm: edges and far, weak texture). It keeps the close range.
- **max_disp 128** keeps the far range but not the close one ([below](#max_disp-128-does-not-suit-close-ups)).
- **amp against TF32 at the same setting** (`ffs_accuracy <recs> <TF32 engine> <amp engine>`; D455 median / p90 against the TF32 one):

| 848x480 | ETH3D EPE | bad-1 | bad-2 | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_disp 192, TF32 | 0.2005 px | 1.632% | 0.661% | | | | | | |
| max_disp 192, amp | 0.2011 px | 1.639% | 0.663% | 0 / 1 mm | 0 / 1 mm | 0 / 1 mm | 1 / 5 mm | 2 / 13 mm | 2 / 12 mm |
| max_disp 128, TF32 | 0.2014 px | 1.612% | 0.639% | | | | | | |
| max_disp 128, amp | 0.2031 px | 1.610% | 0.644% | 1 / 4 mm | 0 / 10 mm | 0 / 6 mm | 1 / 8 mm | 2 / 16 mm | 2 / 14 mm |

amp is 1.6–1.8× faster for +0.3–0.8% ETH3D EPE and 0–2 mm (median) of depth. That is more than TensorRT's rounding (max_disp
128: |disparity difference| 0.085 / 6.6 / 156 px, against 0.003 / 1.7 / 138 px for a rebuild), but a small change next to the
input size. Why it is faster (`ffs_layers`), 848x480, max_disp 192:

| | TF32, (1) | amp, (1) | amp, (1) + (2) |
|---|---:|---:|---:|
| engine alone | 111.3 ms | 69.1 ms | 56.4 ms (another session) |
| cost aggregation | 25.5 ms | 7.8 ms | 7.2 ms |
| update block (GRU) | 15.9 ms | 5.9 ms | 6.1 ms |
| corr_feature_att | 9.1 ms | fused into `Corr` kernels (1.7 ms each) | the same |
| group-wise correlation (one fused kernel) | 13.7 ms | 14.1 ms (FFS normalizes in fp32) | matrix products + a 1.8 ms Gather |
| slowest kernel | the correlation | the correlation | 4.4 ms, sampling the geometry encoding volume |

The convolutions are 3–4× faster in fp16; with them faster, the fp32 correlation became 20% of the engine, which (2) removes.
848x480, max_disp 192 with amp is the setting of main.py: the accuracy of 848x480 TF32 at the speed of 636x360 TF32.

### max_disp 128 does not suit close-ups

It would be faster still (55.6 ms with (1)), but it cannot match closer than fx · baseline / 128 = 0.32 m and still outputs a
disparity there. In the two test recordings 38% of the pixels (whose match is inside the right image) are nearer than that;
the 128 engine gives them 112–145 px (percentiles 10 / 90; 1 / 99: 29 / 170), overlapping the good pixels, so no threshold
separates them (`near_limit`):

| Cut at disparity > | Near pixels removed | Good pixels removed (all / those at 100–128 px) |
|---:|---:|---:|
| 110 | 90.5% | 5.31% / 66.8% |
| 115 | 88.6% | 4.25% / 51.7% |
| 118 | 87.1% | 3.57% / 42.0% |
| 120 | 85.9% | 3.02% / 34.1% |
| 122 | 84.3% | 2.49% / 26.8% |
| 124 | 82.4% | 2.05% / 20.9% |
| 126 | 79.8% | 1.66% / 16.0% |

## Person segmentation

`live_20261012_160515`: 148 color frames at 848x480, 131 with people (`bench_seg`). There is no ground truth: accuracy here is
**agreement with DeepLabV3-ResNet50 in PyTorch fp32**, which shows whether a faster setting changes the output, not whether the
model finds every person. Latency is one frame (normalization + model + person class), mean of 100 after 10 warm-up calls.

| | Latency | IoU vs. reference | Pixels differing | Reference person pixels missed | Frames with people |
|---|---:|---:|---:|---:|---:|
| ResNet50 PyTorch fp32 (reference) | 46.1 ms | 1 | — | — | 131 / 148 |
| ResNet50 PyTorch fp16 | 17.6 ms | 0.9994 | 0.002% | 0.1% | 131 / 148 |
| **ResNet50 TensorRT fp16** | **7.4 ms** | **0.9992** | **0.002%** | **0.0%** | **131 / 148** |
| MobileNetV3 PyTorch fp32 | 13.0 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 PyTorch fp16 | 11.6 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 TensorRT fp16 | 3.3 ms | 0.675 | 1.07% | 36% | 112 / 148 |

- TensorRT fp16 changes 0.002% of the pixels: no loss.
- MobileNetV3's gap is the model, not the precision (its three settings agree): it misses the people that show only in part
  (legs under a desk, half behind a monitor), 19 of the 131 frames with people.
- The PyTorch masks of `recon/person_mask.py` (offline) and the TensorRT ones of the live depth thread are the same, IoU ≥ 0.998.

In the live pipeline (replay of `live_20261012_160515/rec`, 778 IR pairs, 848x480 TF32 FFS), the masks keep people out of the map:

| | Depth | Per depth frame | Capture → map | Final mesh |
|---|---:|---:|---:|---|
| Without segmentation | 6.3 Hz | 149 ms | 202 ms | 507 k triangles, ghosts of two people |
| With segmentation, after FFS on its stream | 6.0 Hz | 158 ms (8.5 ms of it the mask) | 213 ms | 451 k triangles, no people |

In a thread of its own the segmentation waited ~150 ms a mask behind FFS, even on a high-priority CUDA stream. Queued on a
stream of its own right before FFS on the same frame (`main.py`'s `Depth`), it fills the gaps of FFS: FFS 70.9 ms with it,
71.0 without, and 8.4 → 0.1 ms waiting for the mask. People seen in part may still leave traces (the legs of a seated person,
partly missed by the model): a larger `DILATE_PX` in `recon/person_mask.py` or a stronger model would take them out.

## Depth thread

`main.py`, `recon/ffs.py`:

- **Warp without boolean indexing**: `DisparityToColorDepth` computes every pixel and scatters the invalid ones to a spare slot
  instead of `disp[ok]`, which waits for the GPU (holding the GIL) to learn the count. Identical depth on 108 of 108 frames
  (`warp_equal`); 2.6 → 2.0 ms alone, 4.8 → 2.6 ms in the pipeline.
- **The next frameset asked for when FFS is done**: its transfer from the tracker process (~7 ms) runs during the warp and the
  segmentation. Asked any earlier, the frame would be older when FFS gets it.

## CPU governor

JetPack 7.1's `jetson_clocks` (enabled as `jetson-clocks.service` here) locks the GPU (1575 MHz) and the memory (4266 MHz)
but leaves the CPUs on `schedutil` (972–2601 MHz, ~1.5 GHz on average in a replay). On `performance` all 14 cores stay at
2601 MHz: ORB-SLAM3 32.2 → 22.1 ms (p90 39.5 → 25.5), fewer IR pairs skipped (1045 of 1046 tracked instead of ~1000 of 1008),
capture → map p90 −40 ms. `system/cpu-performance.service` sets it at every boot (see docs/install).

## After Stop

- `clean_mesh` (`recon/reconstruct.py`): the pieces of the mesh found with numpy + scipy's `connected_components` instead of
  Open3D's single-threaded `cluster_connected_triangles`: the same mesh, byte for byte (vertices and triangles), 1.5 → 0.24 s.
- `FrameWriter` (`main.py`): frames/ PNGs written while recording by a low-priority pool of 4 threads; at Stop only meta.json
  and poses.npy are left (1.1 s → 2 ms).

## Tried, not used

- **3DGS training (offline)**: 95% of `gs.offline` (296 s of 312). nsys over 3000 steps: the GPU works 10.7 ms of the 14.8 ms of
  a step; the rest is the CPU waiting on syncs (`loss.item()`, `torch.cuda.synchronize()` for the timing, `nonzero()` of the
  visibility filter, indexing by masks in the densification stats). Not pursued: only the live part was asked for.
- They change the result: FP8 / FP4, fewer FFS iterations, MobileNet segmentation (above), ORB features on the GPU (other
  features, another trajectory). Thor has no DLA.

## Left

- FFS is ~85% of the depth thread. Its slowest kernel is the sampling of the geometry encoding volume (4.4 ms); the rest is
  spread over ~650 layers.
- The window's mesh copy (every second, 35 / 68 ms) could move out of the mapper loop.
