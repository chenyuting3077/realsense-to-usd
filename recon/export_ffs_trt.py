"""Export Fast-FoundationStereo to a single ONNX and build a TensorRT engine for recon/ffs.py.

Uses the ONNX-compatible cost volumes from Fast-FoundationStereo's scripts/make_single_onnx.py
(ImageNet normalisation stripped; recon/ffs.py applies it). TensorRT 11 only builds strongly
typed networks, so the precision is fixed in the ONNX:
  fp32   plain fp32 trace; the builder may use TF32 (default; --no-tf32 for strict fp32)
  fp16   traced under torch.autocast(float16), i.e. the same casts as the PyTorch fp16 path
  fp8    the fp16 ONNX quantised by NVIDIA ModelOpt: FP8 Q/DQ on convolutions and matmuls, every
         other layer keeps the fp16 / fp32 casts of the autocast trace (letting ModelOpt convert
         the fp32 ONNX to fp16 instead gives NaN disparities); calibrated on --calib-n IR pairs
         of another recording (--calib)

Usage (in the activated conda env; TensorRT from pip: tensorrt-cu13):
  python recon/export_ffs_trt.py -o output/ffs            # fp32 + TF32, 8 iterations (~4 min)
  -> output/ffs/ffs_480x640_it8_fp32.onnx, ffs_480x640_it8_fp32_tf32.engine
  python recon/export_ffs_trt.py -o output/ffs --width 864   # 848x480 (width padded to 32): what live.py uses
  python recon/export_ffs_trt.py -o output/ffs --precision fp8 --max-disp 128 --calib output/room/rec   # fastest (~25 min)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

from ffs import FFS_ROOT, MEAN, STD, load_model
from frames import load_meta


def export_onnx(path, height, width, valid_iters, max_disp, precision):
    model = load_model(valid_iters=valid_iters, max_disp=max_disp)
    model.args.mixed_precision = False
    sys.path.insert(0, os.path.join(FFS_ROOT, "scripts"))
    import core.foundation_stereo as fs
    import make_single_onnx as ms
    fs.normalize_image = lambda img: img
    fs.build_gwc_volume_optimized_pytorch1 = ms._build_gwc_volume_onnx
    fs.build_concat_volume_optimized_pytorch1 = ms._build_concat_volume_onnx
    wrapper = ms.FastFoundationStereoSingleOnnx(model).cuda().eval()
    x = [torch.randn(1, 3, height, width, device="cuda") for _ in range(2)]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
        torch.onnx.export(wrapper, tuple(x), path, opset_version=17, dynamo=False,
                          input_names=["left_image", "right_image"], output_names=["disparity"],
                          do_constant_folding=True)


def quantize_fp8(onnx_fp16, onnx_fp8, calib, n, height, width):
    """ModelOpt FP8 quantisation, calibrated on n projector-on IR pairs of the frames dir calib."""
    from modelopt.onnx.quantization import quantize
    meta = load_meta(calib)
    dirs = ("ir1_on", "ir2_on") if meta["stereo"].get("projector") == "alternate" else ("ir1", "ir2")
    ids = np.linspace(0, len(meta["frames"]) - 1, n).astype(int)
    data = {}
    for name, d in zip(("left_image", "right_image"), dirs):
        imgs = np.stack([cv2.imread(os.path.join(calib, d, f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) for i in ids])
        assert imgs.shape[1:] == (height, width), f"calibration images are {imgs.shape[1:]}, engine {height}x{width}"
        x = torch.from_numpy(imgs).float()[:, None].expand(-1, 3, -1, -1)
        data[name] = ((x - MEAN) / STD).numpy().astype(np.float32)
    quantize(onnx_fp16, quantize_mode="fp8", calibration_data=data, calibration_eps=["cpu"],
             high_precision_dtype="fp32", output_path=onnx_fp8)


def build_engine(onnx_path, engine_path, tf32, opt_level):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = opt_level
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(plan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="output/ffs")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--valid-iters", type=int, default=8)
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--precision", choices=["fp32", "fp16", "fp8"], default="fp32")
    p.add_argument("--calib", help="fp8: frames dir with IR to calibrate on (rec/ of a live session)")
    p.add_argument("--calib-n", type=int, default=16, help="fp8: number of calibration pairs")
    p.add_argument("--no-tf32", dest="tf32", action="store_false", help="fp32: no TF32 in TensorRT")
    p.add_argument("--opt-level", type=int, default=3, help="TensorRT builder optimisation level 0-5")
    p.add_argument("--name", help="base name (default ffs_<H>x<W>_it<N>[_md<max-disp>]_<precision>)")
    args = p.parse_args()
    assert args.height % 32 == 0 and args.width % 32 == 0, "height and width must be multiples of 32"
    if args.precision == "fp8" and not args.calib:
        p.error("--precision fp8 needs --calib")

    os.makedirs(args.out, exist_ok=True)
    md = f"_md{args.max_disp}" if args.max_disp != 192 else ""
    name = args.name or f"ffs_{args.height}x{args.width}_it{args.valid_iters}{md}_{args.precision}"
    onnx_path = os.path.join(args.out, f"{name}.onnx")
    tf32 = args.tf32 and args.precision == "fp32"
    engine_path = os.path.join(args.out, f"{name}{'_tf32' * tf32}.engine")
    t0 = time.time()
    if args.precision == "fp8":
        onnx_fp16 = onnx_path.replace("_fp8.onnx", "_fp16.onnx")
        if not os.path.exists(onnx_fp16):
            export_onnx(onnx_fp16, args.height, args.width, args.valid_iters, args.max_disp, "fp16")
        if not os.path.exists(onnx_path):
            quantize_fp8(onnx_fp16, onnx_path, args.calib, args.calib_n, args.height, args.width)
            print(f"ONNX + FP8 calibration {time.time() - t0:.0f}s -> {onnx_path}")
    elif not os.path.exists(onnx_path):
        export_onnx(onnx_path, args.height, args.width, args.valid_iters, args.max_disp, args.precision)
        print(f"ONNX {time.time() - t0:.0f}s -> {onnx_path}")
    t0 = time.time()
    build_engine(onnx_path, engine_path, tf32, args.opt_level)
    print(f"engine {time.time() - t0:.0f}s -> {engine_path}")


if __name__ == "__main__":
    main()
