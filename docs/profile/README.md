# Profile of the live pipeline on Thor, and speed-ups

[繁體中文](README.zh-TW.md)

Every node of `main.py` timed on the Jetson AGX Thor (JetPack 7.1, MAXN), then made faster: first only where the output
stays the same (or within the spread of the baseline's own re-runs), then FFS moved to a more accurate setting with the time
saved. Replay of `output/live_20261012_171428/rec` (36.5 s, ~1000 IR pairs) with `--no-view`, 3 runs per row (median),
nothing else running on the Thor.

```bash
docs/profile/run_baseline.sh output/live_20261012_171428/rec output/prof/base 3     # 3 replays + stage table (tegrastats beside)
docs/profile/run_replay.sh output/live_20261012_171428/rec output/prof/nsys nsys      # one replay under nsys (PROFILE=1: NVTX ranges)
docs/profile/run_replay.sh output/live_20261012_171428/rec output/prof/view view      # with the window on the desktop
python -m docs.profile.compare_outputs output/prof/base_1 output/prof/base_2 ...      # accuracy: ATE, depth, mesh Chamfer
```

The stages come from `recon/prof.py`: `with span("depth/ffs"):` blocks write their median / p90 to `live.json`
(`stages`, and `tracker_stages` from the tracker process).

## Result

| | Depth | FFS (engine + pre/post) | Capture → map (median / p90) | ORB-SLAM3 per frame (median / p90) | Stop → mesh + frames saved |
|---|---:|---:|---:|---:|---:|
| Before | 9.34 Hz | 83.1 ms | 151 / 276 ms | 32.2 / 39.5 ms | 3.0 + 1.1 s |
| **Now** | **13.73 Hz** | **62.0 ms** | **122 / 192 ms** | **22.1 / 25.8 ms** | **2.4 + 0.0 s** |

Depth at 1.47× the rate on a more accurate FFS (ETH3D EPE 0.220 → 0.201 px; on the D455 2–8 cm closer to 848x480 TF32 at
2–5 m), ORB-SLAM3 1.45× faster (p90 under the 33 ms of 30 Hz), Stop → done 4.1 → 2.4 s (with 1.5× the depth frames).

Step by step (each row adds to the one above):

| | FFS setting | Depth | FFS | Capture → map | ORB-SLAM3 | Stop → done |
|---|---|---:|---:|---:|---:|---:|
| Before (CPU on schedutil) | 636x360, TF32 | 9.34 Hz | 83.1 ms | 151 / 276 ms | 32.2 / 39.5 ms | 4.1 s |
| [Cost volumes without repeated work](#cost-volumes-the-same-values-with-less-work) (1), [depth thread](#depth-thread) | 636x360, TF32 | 11.11 Hz | 71.3 ms | 149 / 260 ms | 32.0 / 39.7 ms | 4.6 s |
| [CPU governor](#cpu-governor-orb-slam3-32--22-ms), [after Stop](#after-stop) | 636x360, TF32 | 11.47 Hz | 71.0 ms | 141 / 218 ms | 22.1 / 25.5 ms | 2.2 s |
| [Segmentation beside FFS](#depth-thread) | 636x360, TF32 | 12.26 Hz | 70.9 ms | 131 / 230 ms | 22.3 / 25.2 ms | 2.1 s |
| [FFS at 848x480 with its own fp16 autocast](#848x480-with-ffss-own-fp16-autocast) | **848x480, amp** | 11.66 Hz | 74.8 ms | 135 / 240 ms | 22.0 / 25.3 ms | 2.1 s |
| [Correlation as matrix products](#cost-volumes-the-same-values-with-less-work) (2) (now) | **848x480, amp** | **13.73 Hz** | **62.0 ms** | **122 / 192 ms** | **22.1 / 25.8 ms** | **2.4 s** |

The window (Open3D, on the same GPU) costs ~3 ms of FFS and ~0.5 Hz of depth (measured at the segmentation row: 11.72 Hz,
FFS 73.7 ms). Its mesh updates (`map/update mesh` 35 / 68 ms, `map/show` 10 / 40 ms) run once a second in the mapper loop.

### Accuracy

`compare_outputs`: ATE of the trajectories, |depth difference| on the frames both runs stored, mesh Chamfer distance (B moved
into A's world by the trajectory alignment). Each run picks other projector-on framesets (the newest one when the depth thread
is free), so the meshes differ in coverage too; the spread of re-runs of the same code is the yardstick.

| | ATE (rmse) | Depth (median / p90 / p99) | Chamfer (mean / p90) |
|---|---:|---:|---:|
| before, re-runs | 0.39–0.73 cm | 0 / 0 / 0 mm | 24.6–24.9 / 48 mm |
| the speed-ups up to the segmentation row, against before (4 runs) | 0.29–0.57 cm | 0 / 1 / 4–5 mm | 20.5–28.4 / 40–64 mm |
| 848x480 amp, re-runs | 0.28–0.30 cm | 0 / 0 / 0 mm | 21.6–30.1 / 37–49 mm |
| correlation as matrix products, against 848x480 amp (3 runs) | 0.24–0.40 cm | 0 / 2–3 / 16–18 mm | 17.8–33.6 / 31–68 mm |
| 848x480 amp against 636x360 TF32 (another FFS setting) | 0.25–0.31 cm | **14 / 165** / 1300–2900 mm | 40–44 / 94–97 mm |

The speed-ups stay within the re-runs (their depth differences are TensorRT's rounding, below). The last row is the change of
FFS setting: the depth moves by the 636x360 error (see [FFS](#848x480-with-ffss-own-fp16-autocast)), the trajectory not.

## Where the time went (before)

Main process (`stages`) and tracker process (`tracker_stages`), median / p90 per call:

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

The depth thread was ~97 ms a frame, 86% of it FFS. GPU memory copies are negligible (unified memory, nsys: < 1% of the time).

## FFS

### Cost volumes: the same values with less work

`recon/ffs_volume.py` replaces the ONNX-compatible cost volumes of FFS's `scripts/make_single_onnx.py`, which build the target
shifted by every disparity with 48 pads + a stack, normalize each feature vector once per disparity (48 times), and form the
group-wise correlation as a BxGxCxDxHxW product (224 channels × 48 disparities × 120 × 216 at 848x480) before summing it:

1. the shift as one Gather over a padded width, each feature vector normalized once before the shift (a vector shifted in from
   the padding is zero either way);
2. the correlation as matrix products: per group and image row, every left pixel with every right pixel (216 × 216 on the
   tensor cores), then the 48 diagonals picked by one Gather; 6× less data than the 6-D product.

In PyTorch both give FFS's volumes: (1) bit for bit, (2) up to rounding (3e-7 in fp32, one fp16 step of 5e-4 in fp16). As engines
at 848x480 (`python -m docs.profile.ffs_equal <A> <B> <rec>`: the engine alone with CUDA events, and |disparity difference| on
135 D455 pairs; each pair measured in one session, as absolute times drift ~8% between sessions):

| Engine (848x480) | Alone | |disparity difference| to the row above: median / p99 / max | ETH3D EPE |
|---|---:|---|---:|
| FFS's volumes, TF32 | 127.2 ms | | 0.201 px |
| (1), TF32 | 111.4 ms | 0.0033 / 1.26 / 132 px | 0.201 px |
| (1), fp16 autocast | 74.4 ms | (precision, [below](#848x480-with-ffss-own-fp16-autocast)) | 0.2011 px |
| **(1) + (2), fp16 autocast (now)** | **62.4 ms** | 0.028 / 3.95 / 198 px | **0.2011 px** |

For scale: the same ONNX built twice (TensorRT picks other kernels each build; 640x384, TF32) differs by 0.0029 / 1.74 / 138 px,
ETH3D 0.2202 / 0.2201 px. The 8 GRU iterations carry the rounding on to a few edge pixels. On the D455, (1) + (2) gives the depth
of (1) to the millimetre at every distance (the two amp rows of the D455 table below are the same).

The engine is named `_v3` (`output/ffs/ffs_480x864_it8_v3_amp.engine`) so an old ONNX is not reused: `python -m recon.export_ffs_trt`
once (~5 min).

### 848x480 with FFS's own fp16 autocast

FFS has its own mixed precision (`args.mixed_precision`): the feature, cost aggregation and GRU networks under fp16 autocast, the
disparity, its updates and the sampling coordinates in fp32. Exported traced under it (`--precision amp`), the casts are in the
ONNX and TensorRT 11 (strongly typed) keeps them. All of FFS in fp16 does not export (the sampling coordinates are fp32) and would
not do: fp16 has a step of 0.125 px between 128 and 192 px, the size of the model's whole error, added up over 8 iterations.

`docs/profile/ffs_settings.sh <rec>,<rec>` (as docs/ffs_speed: latency of an 848x480
IR pair including pre/post-processing, ETH3D, D455 against 848x480 max_disp 192 TF32; measured with the volumes of (1)):

| | Latency | fps | ETH3D EPE | bad-1 | bad-2 |
|---|---:|---:|---:|---:|---:|
| 848x480, max_disp 192, TF32 | 121.0 ms | 8.3 | 0.201 px | 1.63% | 0.66% |
| **848x480, max_disp 192, amp** | **68.7 ms** | **14.5** | **0.201 px** | **1.64%** | **0.66%** |
| 848x480, max_disp 128, TF32 | 89.8 ms | 11.1 | 0.201 px | 1.61% | 0.64% |
| 848x480, max_disp 128, amp | 55.6 ms | 18.0 | 0.203 px | 1.61% | 0.64% |
| 636x360, max_disp 192, TF32 (before) | 64.8 ms | 15.4 | 0.220 px | 2.02% | 0.83% |

D455 (2 recordings, every 4th projector-on pair): coverage / median / p90 |depth difference| against 848x480, max_disp 192, TF32:

| | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|
| **848x480, max_disp 192, amp** | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480, max_disp 192, amp, (1) + (2) | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 0 / 1 mm | 100% / 1 / 5 mm | 100% / 2 / 13 mm | 100% / 2 / 12 mm |
| 848x480, max_disp 128, TF32 | 100% / 73 / 145 mm | 100% / 0 / 30 mm | 100% / 0 / 8 mm | 100% / 2 / 45 mm | 100% / 8 / 100 mm | 100% / 10 / 229 mm |
| 848x480, max_disp 128, amp | 100% / 73 / 144 mm | 100% / 0 / 30 mm | 100% / 1 / 8 mm | 100% / 2 / 45 mm | 100% / 9 / 101 mm | 100% / 10 / 233 mm |
| 636x360, max_disp 192, TF32 (before) | 99% / 5 / 47 mm | 100% / 1 / 39 mm | 100% / 5 / 65 mm | 100% / 23 / 349 mm | 100% / 52 / 376 mm | 100% / 83 / 838 mm |

**amp against TF32 at the same setting** (`python -m docs.profile.ffs_accuracy <recs> <TF32 engine> <amp engine>`; D455 median / p90
against the TF32 one):

| 848x480 | ETH3D EPE | bad-1 | bad-2 | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_disp 192, TF32 | 0.2005 px | 1.632% | 0.661% | | | | | | |
| max_disp 192, amp | 0.2011 px | 1.639% | 0.663% | 0 / 1 mm | 0 / 1 mm | 0 / 1 mm | 1 / 5 mm | 2 / 13 mm | 2 / 12 mm |
| max_disp 128, TF32 | 0.2014 px | 1.612% | 0.639% | | | | | | |
| max_disp 128, amp | 0.2031 px | 1.610% | 0.644% | 1 / 4 mm | 0 / 10 mm | 0 / 6 mm | 1 / 8 mm | 2 / 16 mm | 2 / 14 mm |

amp is 1.6–1.8× faster for +0.3–0.8% ETH3D EPE and 0–2 mm (median) of depth. That is more than TensorRT's rounding (max_disp
128: |disparity difference| 0.085 / 6.6 / 156 px, against 0.003 / 1.7 / 138 px for a rebuild), but a small change next to the
input size (the 636x360 rows). docs/tensorrt's fp16 engine (97 ms, slower than TF32, 640x480) came from the experiments before
this repo, built another way (not recorded).

Why it is faster (`python -m docs.profile.ffs_layers <engine>`: the engine alone with TensorRT's per-layer profiler; the system
`trtexec` is TensorRT 10.13 and cannot load 11.3 engines). 848x480, max_disp 192:

| | TF32, (1) | amp, (1) | amp, (1) + (2) |
|---|---:|---:|---:|
| engine alone | 111.3 ms | 69.1 ms | 56.4 ms (another session) |
| cost aggregation | 25.5 ms | 7.8 ms | 7.2 ms |
| update block (GRU) | 15.9 ms | 5.9 ms | 6.1 ms |
| corr_feature_att | 9.1 ms | fused into `Corr` kernels (1.7 ms each) | the same |
| group-wise correlation (one fused kernel) | 13.7 ms | 14.1 ms (FFS normalizes in fp32) | matrix products + a 1.8 ms Gather |
| slowest kernel | the correlation | the correlation | 4.4 ms, sampling the geometry encoding volume |

The convolutions are 3–4× faster in fp16; with them faster, the fp32 correlation became 20% of the engine, which (2) removes.

### max_disp 128 does not suit close-ups

It would be faster still (55.6 ms with (1)), but it cannot match closer than fx · baseline / 128 = 0.32 m. In the two test
recordings 38% of the pixels (whose match is inside the right image) are nearer than that; the 128 engine gives them
112–145 px (percentiles 10 / 90; 1 / 99: 29 / 170), overlapping the good pixels, so no threshold separates them
(`python -m docs.profile.near_limit <engine 192> <engine 128> <recs>`):

| Cut at disparity > | Near pixels removed | Good pixels removed (all / those at 100–128 px) |
|---:|---:|---:|
| 110 | 90.5% | 5.31% / 66.8% |
| 115 | 88.6% | 4.25% / 51.7% |
| 118 | 87.1% | 3.57% / 42.0% |
| 120 | 85.9% | 3.02% / 34.1% |
| 122 | 84.3% | 2.49% / 26.8% |
| 124 | 82.4% | 2.05% / 20.9% |
| 126 | 79.8% | 1.66% / 16.0% |

## Depth thread

`main.py`, `recon/ffs.py`:

- **Warp without boolean indexing**: `DisparityToColorDepth` computes every pixel and scatters the invalid ones to a spare slot
  instead of `disp[ok]`, which waits for the GPU (holding the GIL) to learn the count. Identical depth on 108 of 108 frames
  (`python -m docs.profile.warp_equal <old ffs.py> <rec>`); 2.6 → 2.0 ms alone, 4.8 → 2.6 ms in the pipeline.
- **The next frameset asked for when FFS is done**: its transfer from the tracker process (~7 ms) runs during the warp and the
  segmentation. Asked any earlier, the frame would be older when FFS gets it.
- **Segmentation beside FFS**: queued on a stream of its own right before FFS. It takes nothing from FFS (70.9 ms with, 71.0
  without) and is done when FFS is: 8.4 → 0.1 ms waiting for it. The masks are the same (same model, same image).
  docs/tensorrt found it waiting ~150 ms behind FFS in a thread of its own; queued first, on its own stream, it fills the gaps of FFS.

## CPU governor: ORB-SLAM3 32 → 22 ms

JetPack 7.1's `jetson_clocks` (enabled as `jetson-clocks.service` here) locks the GPU (1575 MHz) and the memory (4266 MHz)
but leaves the CPUs on `schedutil` (972–2601 MHz, ~1.5 GHz on average in a replay). On `performance` all 14 cores stay at
2601 MHz: ORB-SLAM3 32.2 → 22.1 ms (p90 39.5 → 25.5), fewer IR pairs skipped (1045 of 1046 tracked instead of ~1000 of 1008),
capture → map p90 −40 ms. `docs/install/cpu-performance.service` sets it at every boot (see docs/install).

## After Stop

- `clean_mesh` (`recon/reconstruct.py`): the pieces of the mesh found with numpy + scipy's `connected_components` instead of
  Open3D's single-threaded `cluster_connected_triangles`: the same mesh, byte for byte (vertices and triangles), 1.5 → 0.24 s.
- `FrameWriter` (`main.py`): frames/ PNGs written while recording by a low-priority pool of 4 threads; at Stop only meta.json
  and poses.npy are left (1.1 s → 2 ms).

## Tried, not used

- **3DGS training (offline)**: 95% of `gs.offline` (296 s of 312). nsys over 3000 steps: the GPU works 10.7 ms of the 14.8 ms of
  a step; the rest is the CPU waiting on syncs (`loss.item()`, `torch.cuda.synchronize()` for the timing, `nonzero()` of the
  visibility filter, indexing by masks in the densification stats). Not pursued: only the live part was asked for.
- Not tried because they change the result: FP8 / FP4, fewer FFS iterations, MobileNet segmentation (docs/tensorrt,
  docs/ffs_speed), ORB features on the GPU (other features, another trajectory). Thor has no DLA.

## Left

- FFS is ~85% of the depth thread. Its slowest kernel now is the sampling of the geometry encoding volume (4.4 ms); the rest is
  spread over ~650 layers.
- The window's mesh copy (every second, 35 / 68 ms) could move out of the mapper loop.
