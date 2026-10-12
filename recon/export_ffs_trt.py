"""Build the Fast-FoundationStereo TensorRT engine of recon/ffs.py (fp32 with TF32, 8 iterations,
640x384 = the 0.75x 848x480 IR padded to 32, max disparity 192), through the ONNX-compatible cost
volumes of FFS's scripts/make_single_onnx.py. Other sizes / max disparities: recon.ffs.engine_path().

Usage: python -m recon.export_ffs_trt [--height 480 --width 864] [--max-disp 128]   (~5 min)
"""
import argparse
import os
import sys

import torch

from recon.ffs import FFS_ROOT, engine_path, load_model


def export_onnx(path, height, width, max_disp):
    model = load_model(max_disp=max_disp)
    model.args.mixed_precision = False
    sys.path.insert(0, os.path.join(FFS_ROOT, "scripts"))
    import core.foundation_stereo as fs
    import make_single_onnx as ms
    fs.normalize_image = lambda img: img  # done in ffs.py
    fs.build_gwc_volume_optimized_pytorch1 = ms._build_gwc_volume_onnx
    fs.build_concat_volume_optimized_pytorch1 = ms._build_concat_volume_onnx
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
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--max-disp", type=int, default=192)
    args = p.parse_args()
    engine = engine_path(args.height, args.width, args.max_disp)
    os.makedirs(os.path.dirname(engine), exist_ok=True)
    onnx_path = engine.replace("_tf32.engine", ".onnx")
    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, args.height, args.width, args.max_disp)
    build_engine(onnx_path, engine)
    print(f"-> {engine}")
