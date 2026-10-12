# TensorRT: before and after

[繁體中文](README.zh-TW.md)

Both networks in the live depth thread run as TensorRT engines on the Jetson AGX Thor (JetPack 7.1, TensorRT 11.3 from pip):

| | Engine | Before | After | Accuracy |
|---|---|---:|---:|---|
| Stereo depth: Fast-FoundationStereo | `recon/export_ffs_trt.py`, fp32 + TF32 | 289.5 ms (PyTorch fp16 autocast) | **94.0 ms** (3.1×) | unchanged (ETH3D, D455 room) |
| Person segmentation: DeepLabV3-ResNet50 | `recon/export_seg_trt.py`, fp16 | 46.1 ms (PyTorch fp32) / 17.6 ms (fp16) | **7.4 ms** (6.2× / 2.4×) | unchanged (IoU 0.9992 vs fp32) |

TensorRT 11 builds strongly typed networks only (no `BuilderFlag.FP16`): the precision is set in the ONNX: an fp16
model traced for fp16, an fp32 one with TF32 enabled in the builder. An engine is tied to its input size and TensorRT version.

## Fast-FoundationStereo

From the FFS-on-Thor experiments run before this repo (640x480 input, C-FFS weights, 8 iterations, max_disp 192).
Latency is the model call (normalization, padding, inference), mean of 100 after 10 warm-up calls.
ETH3D two-view: the 27 training pairs, non-occluded pixels, laser-scan ground truth. room_1005: 122 projector-on IR
pairs from a D455 recording, compared with the D455's own depth at 0.3–2.5 m (a reference, not ground truth).

| | Setting | Latency | Speed-up | ETH3D EPE | bad-1 | room median error | error < 2% | vs. original, p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | PyTorch fp16 autocast (original) | 289.5 ms | 1.0× | 0.187 px | 1.68% | 16 mm | 67.8% | — |
| B | PyTorch fp32 | 189.1 ms | 1.5× | 0.188 px | 1.68% | 16 mm | 67.8% | 70 mm |
| C | TensorRT fp16 | 97.1 ms | 3.0× | 0.185 px | 1.70% | 16 mm | 67.8% | 65 mm |
| **H** | **TensorRT fp32 + TF32** | **94.0 ms** | **3.1×** | **0.185 px** | **1.70%** | **16 mm** | **67.8%** | **64 mm** |
| E | TensorRT fp16, 4 iterations | 81.9 ms | 3.5× | 0.194 px | 1.84% | 16 mm | 67.9% | 578 mm |
| I | TensorRT fp16, max_disp 128 | 72.4 ms | 4.0× | 0.185 px | 1.75% | 17 mm | 66.1% | 819 mm |
| D | TensorRT FP8 | 68.0 ms | 4.3× | 0.195 px | 1.93% | 18 mm | 64.3% | 865 mm |
| G | TensorRT FP8, max_disp 128 | 51.8 ms | 5.6× | 0.195 px | 1.94% | 19 mm | 62.1% | 1113 mm |

- C and H give the original depth: median difference 0 mm, the p99 is a few edge pixels. TF32 is 3 ms faster than fp16 at the same accuracy.
- On Thor PyTorch fp16 autocast is slower than fp32 (290 vs 189 ms).
- Fewer iterations, max_disp 128 (nearest depth 0.19 m → 0.29 m) and FP8 trade accuracy for speed;
  FP8 loses more on the D455 IR images than on ETH3D.

Here the 848x480 IR (padded to 864) took 141–149 ms a frame in the depth thread, including the warp into the color
camera, without the person segmentation running (replay of `output/live_20261012_160515/rec`). PyTorch vs. TensorRT
accuracy was not measured again at this size. main.py's FFS setting: [docs/profile](../profile/README.md).

## Person segmentation

`live_20261012_160515`: 148 color frames at 848x480, 131 with people. There is no ground truth: accuracy here is
**agreement with DeepLabV3-ResNet50 in PyTorch fp32**, which shows whether a faster setting changes the output, not
whether the model finds every person. Latency is one frame (normalization + model + person class), mean of 100 after 10 warm-up calls.
Reproduce with [`bench_seg.py`](bench_seg.py): `python -m docs.tensorrt.bench_seg output/live_20261012_160515/frames`.

| | Latency | IoU vs. reference | Pixels differing | Reference person pixels missed | Frames with people |
|---|---:|---:|---:|---:|---:|
| ResNet50 PyTorch fp32 (reference) | 46.1 ms | 1 | — | — | 131 / 148 |
| ResNet50 PyTorch fp16 | 17.6 ms | 0.9994 | 0.002% | 0.1% | 131 / 148 |
| **ResNet50 TensorRT fp16 (used)** | **7.4 ms** | **0.9992** | **0.002%** | **0.0%** | **131 / 148** |
| MobileNetV3 PyTorch fp32 | 13.0 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 PyTorch fp16 | 11.6 ms | 0.675 | 1.07% | 36% | 112 / 148 |
| MobileNetV3 TensorRT fp16 | 3.3 ms | 0.675 | 1.07% | 36% | 112 / 148 |

- TensorRT fp16 changes 0.002% of the pixels: no loss.
- MobileNetV3's gap is the model, not the precision (its three settings agree): it misses the people that show only in
  part (legs under a desk, half behind a monitor), 19 of the 131 frames with people.
- The PyTorch masks of `recon/person_mask.py` (offline) and the TensorRT ones of the live depth thread are the same, IoU ≥ 0.998.

### In the live pipeline

The segmentation runs after FFS on the same frame, not in a thread beside it: there it waits behind FFS for the GPU
(~150 ms a mask, even on a high-priority CUDA stream). Replay of the same `rec/` (778 IR pairs, 29 s):

| | Depth | Per depth frame | Capture → map | Final mesh |
|---|---:|---:|---:|---|
| Without segmentation | 6.3 Hz | 149 ms | 202 ms | 507 k triangles, ghosts of two people |
| With segmentation | 6.0 Hz | 158 ms (8.5 ms of it the mask) | 213 ms | 451 k triangles, no people |

People seen in part may still leave traces (here the legs of a seated person, partly missed by the model): a larger
`DILATE_PX` in `recon/person_mask.py` or a stronger model would take them out.
