"""Person segmentation settings (docs/speed): latency of one 848x480 frame and
agreement with DeepLabV3-ResNet50 PyTorch fp32 on every color frame of a session. Builds the
MobileNetV3 comparison engine into output/seg/ the first time (~4 min).

Usage: python -m bench.seg.compare output/live_room/frames
"""
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

from torchvision.models import segmentation as S

from recon.export_seg_trt import HEIGHT, WIDTH, build_engine
from recon.person_mask import ENGINE, Person, normalize

R50 = (S.deeplabv3_resnet50, S.DeepLabV3_ResNet50_Weights.DEFAULT)
MBV3 = (S.deeplabv3_mobilenet_v3_large, S.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
MBV3_ENGINE = ENGINE.replace("dlv3_r50", "dlv3_mbv3")


def pytorch(model, half):
    ctor, weights = model
    m = Person(ctor(weights=weights, aux_loss=None)).cuda().eval()
    m = m.half() if half else m

    def run(color):
        x = normalize(color[None])
        with torch.inference_mode():
            return m(x if half else x.float())[0]
    return run


def tensorrt(engine):
    import tensorrt as trt
    if not os.path.exists(engine):
        onnx_path = engine.replace("_fp16.engine", ".onnx")
        with torch.no_grad():
            m = Person(MBV3[0](weights=MBV3[1], aux_loss=None)).cuda().eval().half()
            torch.onnx.export(m, torch.randn(1, 3, HEIGHT, WIDTH, device="cuda").half(), onnx_path,
                              opset_version=17, dynamo=False, input_names=["image"], output_names=["person"])
        build_engine(onnx_path, engine, tf32=False)
    e = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(engine, "rb").read())
    ctx = e.create_execution_context()
    out = torch.empty(tuple(e.get_tensor_shape("person")), dtype=torch.bool, device="cuda")
    ctx.set_tensor_address("person", out.data_ptr())
    stream = torch.cuda.Stream()

    def run(color):
        x = normalize(color[None]).contiguous()
        ctx.set_tensor_address("image", x.data_ptr())
        stream.wait_stream(torch.cuda.current_stream())
        ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(stream)
        return out[0].clone()
    return run


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "color", "*.png")))
    colors = [torch.from_numpy(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)).cuda() for f in files]
    configs = [("R50 PyTorch fp32 (reference)", lambda: pytorch(R50, False)),
               ("R50 PyTorch fp16", lambda: pytorch(R50, True)),
               ("R50 TensorRT fp16", lambda: tensorrt(ENGINE)),
               ("MBv3 PyTorch fp32", lambda: pytorch(MBV3, False)),
               ("MBv3 PyTorch fp16", lambda: pytorch(MBV3, True)),
               ("MBv3 TensorRT fp16", lambda: tensorrt(MBV3_ENGINE))]
    ref = None
    print("| | latency | IoU vs reference | pixels differing | reference person pixels missed | frames with people |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, make in configs:
        run = make()
        for _ in range(10):
            run(colors[0])
        ts = []
        for _ in range(100):  # model call incl. normalization, one frame
            torch.cuda.synchronize()
            t = time.perf_counter()
            run(colors[0])
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        masks = [run(c).cpu().numpy() for c in colors]
        ref = ref or masks
        iou = sum((a & b).sum() for a, b in zip(masks, ref)) / sum((a | b).sum() for a, b in zip(masks, ref))
        differ = np.mean([(a != b).mean() for a, b in zip(masks, ref)])
        missed = np.mean([(~a & b).sum() / b.sum() for a, b in zip(masks, ref) if b.any()])
        print(f"| {name} | {np.mean(ts):.1f} ms | {iou:.4f} | {differ:.3%} | {missed:.1%} | "
              f"{sum(a.any() for a in masks)} / {len(masks)} |", flush=True)
        del run
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
