# Faster Fast-FoundationStereo: smaller input or max_disp 128

[繁體中文](README.zh-TW.md)

FFS limits the live depth rate (the rest of the depth thread adds ~15 ms). Two ways to make it faster, against the setting
used (848x480 IR padded to 864, max disparity 192, TensorRT fp32 + TF32, 8 iterations), on the Jetson AGX Thor.
Reproduce with [`eval_ffs.py`](eval_ffs.py):

```bash
python -m recon.export_ffs_trt --height 384 --width 640   # 636x360 padded to 32
python -m recon.export_ffs_trt --max-disp 128
python -m docs.ffs_speed.eval_ffs output/live_20261012_160515/rec output/live_20261012_171428/rec
```

| | Latency | fps | ETH3D EPE | bad-1 | bad-2 | Est. error at 2.5 m / 4 m |
|---|---:|---:|---:|---:|---:|---:|
| 848x480, max_disp 192 (used) | 142.1 ms | 7.0 | 0.201 px | 1.63% | 0.66% | 31 / 79 mm |
| 636x360 (0.75x), max_disp 192 | **81.4 ms** | **12.3** | 0.220 px | 2.02% | 0.83% | 34 / 86 mm |
| 848x480, max_disp 128 | 98.9 ms | 10.1 | 0.201 px | 1.61% | 0.64% | 31 / 79 mm |

- Latency: one 848x480 IR pair, model call (downscaling, normalization, padding, inference, disparity back to 848x480),
  mean of 50 after 10 warm-up calls.
- ETH3D two-view: the 27 training pairs, each cut to an 848x480 window (smaller ones whole), scored at full resolution
  on the non-occluded laser-scan ground truth. 636x360 runs on the downscaled pair, its disparity scaled back up.
- Est. error: the ETH3D EPE as a D455 depth error, z² / (fx · baseline) · EPE with fx 430.6 px and baseline 95 mm
  (0.15 m per px at 2.5 m, 0.39 m per px at 4 m). ETH3D is visible-light images, not the D455's IR with the projector dots.

On D455 data there is no ground truth: the table compares each setting with the one used, by the depth the setting used
gives. 2 recordings, every 4th projector-on IR pair (480 pairs). Coverage / median / p90 of |depth difference|:

| | 0.2–0.35 m | 0.35–1 m | 1–2 m | 2–3 m | 3–4 m | 4–5 m |
|---|---:|---:|---:|---:|---:|---:|
| 636x360 (0.75x) | 99% / 5 / 47 mm | 100% / 1 / 39 mm | 100% / 5 / 65 mm | 100% / 23 / 349 mm | 100% / 52 / 376 mm | 100% / 83 / 837 mm |
| max_disp 128 | 100% / **73** / 145 mm | 100% / 0 / 30 mm | 100% / 0 / 8 mm | 100% / 2 / 45 mm | 100% / 8 / 100 mm | 100% / 10 / 229 mm |

The difference includes the noise of the setting used itself: it is not the error of the other setting alone.

## Conclusions

- **max_disp 128**: 1.44× faster (10.1 fps), the same depth from 0.35 m on (median difference 0–10 mm up to 5 m, ETH3D
  unchanged). Closer than fx · baseline / 128 = 0.32 m it cannot match and still outputs depth, ~7 cm too far: such
  depth would have to be dropped (disparity near 128). Close-ups of a desk at ~0.3 m are lost.
- **636x360**: 1.75× faster (12.3 fps). Less precise with distance: ETH3D EPE +10%, on the D455 a median difference of
  2.3 cm at 2–3 m, 5 cm at 3–4 m and 8 cm at 4–5 m (p90 35–84 cm: edges and far, weak texture). It keeps the close range.
- For the live map (depth ≤ 2.5 m) both are usable; for the final mesh up to 4 m, max_disp 128 keeps the far range.
