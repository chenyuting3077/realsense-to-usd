"""Build the person segmentation TensorRT engine of person_mask.py's PersonSeg (DeepLabV3-ResNet50,
fp16, 848x480 color images): ~7 ms a frame on Thor, vs ~20 ms in PyTorch fp16.

Usage: python -m recon.export_seg_trt   (~4 min)
"""
import os

import torch

from recon.export_ffs_trt import build_engine
from recon.person_mask import ENGINE, load_model

HEIGHT, WIDTH = 480, 848


def export_onnx(path):
    with torch.no_grad():  # fp16 in the ONNX: TensorRT 11 has no fp16 flag, the network is strongly typed
        torch.onnx.export(load_model(), torch.randn(1, 3, HEIGHT, WIDTH, device="cuda").half(), path,
                          opset_version=17, dynamo=False, input_names=["image"], output_names=["person"])


if __name__ == "__main__":
    os.makedirs(os.path.dirname(ENGINE), exist_ok=True)
    onnx_path = ENGINE.replace("_fp16.engine", ".onnx")
    if not os.path.exists(onnx_path):
        export_onnx(onnx_path)
    build_engine(onnx_path, ENGINE, tf32=False)
    print(f"-> {ENGINE}")
