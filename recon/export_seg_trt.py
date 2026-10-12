"""Build the person segmentation TensorRT engine of person_mask.py's PersonSeg (DeepLabV3-ResNet50,
fp16, 848x480 color images): ~7 ms a frame on Thor, vs ~20 ms in PyTorch fp16.

Usage: python -m recon.export_seg_trt   (~4 min)
"""
import os

import torch

from recon.person_mask import ENGINE, load_model

HEIGHT, WIDTH = 480, 848


def export_onnx(path):
    with torch.no_grad():  # fp16 in the ONNX: TensorRT 11 has no fp16 flag, the network is strongly typed
        torch.onnx.export(load_model(), torch.randn(1, 3, HEIGHT, WIDTH, device="cuda").half(), path,
                          opset_version=17, dynamo=False, input_names=["image"], output_names=["person"])


def build_engine(onnx_path, engine_path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    if plan is None:
        raise SystemExit("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(plan)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(ENGINE), exist_ok=True)
    onnx_path = ENGINE.replace("_fp16.engine", ".onnx")
    if not os.path.exists(onnx_path):
        export_onnx(onnx_path)
    build_engine(onnx_path, ENGINE)
    print(f"-> {ENGINE}")
