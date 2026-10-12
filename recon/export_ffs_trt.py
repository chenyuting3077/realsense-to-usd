"""Build the Fast-FoundationStereo TensorRT engine of recon/ffs.py (8 iterations, 864x480 = the 848x480
IR padded to 32, max disparity 192, FFS's own fp16 autocast), with the ONNX-compatible cost volumes of
recon/ffs_volume.py (the same values as FFS's scripts/make_single_onnx.py ones, faster). Other sizes,
max disparities and TF32: recon.ffs.engine_path(), docs/profile.

Usage: python -m recon.export_ffs_trt [--height 384 --width 640] [--max-disp 128] [--precision tf32]   (~5 min)
"""
import argparse
import os
import sys

import torch

from recon.ffs import FFS_ROOT, PRECISIONS, engine_path, load_model
from recon.ffs_volume import build_concat_volume, build_gwc_volume


def export_onnx(path, height, width, max_disp, precision="tf32"):
    model = load_model(max_disp=max_disp)
    model.args.mixed_precision = precision == "amp"  # traced with FFS's autocast: its casts in the ONNX
    sys.path.insert(0, os.path.join(FFS_ROOT, "scripts"))
    import core.foundation_stereo as fs
    import make_single_onnx as ms
    fs.normalize_image = lambda img: img  # done in ffs.py
    fs.build_gwc_volume_optimized_pytorch1 = build_gwc_volume
    fs.build_concat_volume_optimized_pytorch1 = build_concat_volume
    wrapper = ms.FastFoundationStereoSingleOnnx(model).cuda().eval()
    x = [torch.randn(1, 3, height, width, device="cuda") for _ in range(2)]
    with torch.no_grad():
        torch.onnx.export(wrapper, tuple(x), path, opset_version=17, dynamo=False,
                          input_names=["left_image", "right_image"], output_names=["disparity"],
                          do_constant_folding=True)


def build_engine(onnx_path, engine_path, tf32=True):
    """A strongly typed engine (TensorRT 11: the precision is the ONNX's), TF32 for its fp32 layers."""
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
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(plan)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--precision", choices=list(PRECISIONS), default="amp")
    args = p.parse_args()
    engine = engine_path(args.height, args.width, args.max_disp, args.precision)
    os.makedirs(os.path.dirname(engine), exist_ok=True)
    onnx_path = engine.replace("_tf32.engine", ".engine").replace(".engine", ".onnx")
    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, args.height, args.width, args.max_disp, args.precision)
    build_engine(onnx_path, engine)
    print(f"-> {engine}")
