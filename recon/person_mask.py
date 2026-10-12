"""People in the color images (DeepLabV3-ResNet50, COCO/VOC classes) -> keep masks: False on people,
dilated so the outline and the depth edge around a person go too. The TSDFs skip masked depth and
FastGS takes no loss there, so people (moving or not) stay out of the mesh and the Gaussians.
Live: PersonSeg, the TensorRT engine of export_seg_trt.py, in main.py's depth thread. Offline (this
script): PyTorch on a session's frames/color/ -> frames/mask/ (255 kept, 0 people), for sessions
recorded without the engine.

Usage: python -m recon.person_mask output/live_room/frames
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from recon.frames import iter_frames, load_meta

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ENGINE = os.path.join(ROOT, "output/seg/person_dlv3_r50_480x848_fp16.engine")  # export_seg_trt.py
PERSON = 15  # VOC class index
DILATE_PX = 8
BATCH = 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


class Person(torch.nn.Module):
    """A torchvision segmentation model: normalized Nx3xHxW -> NxHxW bool person."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m(x)["out"].argmax(1) == PERSON


def load_model():
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
    return Person(deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, aux_loss=None)).cuda().eval().half()


def normalize(colors):
    """NxHxWx3 uint8 RGB (CUDA) -> the model's fp16 input."""
    return ((colors.permute(0, 3, 1, 2).float() - MEAN.to(colors.device)) / STD.to(colors.device)).half()


def keep_mask(person):
    """NxHxW person -> NxHxW bool, False within DILATE_PX of a person."""
    k = 2 * DILATE_PX + 1
    return F.max_pool2d(person[:, None].half(), k, 1, DILATE_PX)[:, 0] == 0


class PersonSeg:
    """seg(color): HxWx3 uint8 RGB CUDA tensor -> HxW bool keep mask, on the current stream."""

    def __init__(self, engine=ENGINE):
        import tensorrt as trt
        with open(engine, "rb") as f:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"cannot load {engine} (built with another TensorRT version?)")
        self.context = self.engine.create_execution_context()
        self.shape = tuple(self.engine.get_tensor_shape("image"))[2:]
        self.out = torch.empty(tuple(self.engine.get_tensor_shape("person")), dtype=torch.bool, device="cuda")
        self.context.set_tensor_address("person", self.out.data_ptr())

    @torch.no_grad()
    def __call__(self, color):
        if tuple(color.shape[:2]) != self.shape:
            raise SystemExit(f"{ENGINE} is for {self.shape[1]}x{self.shape[0]} images, not "
                             f"{color.shape[1]}x{color.shape[0]}: rebuild it with python -m recon.export_seg_trt")
        x = normalize(color[None]).contiguous()
        self.context.set_tensor_address("image", x.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return keep_mask(self.out)[0]  # a new tensor: self.out is reused by the next call


def main():
    frames = sys.argv[1]
    meta = load_meta(frames)
    out = os.path.join(frames, "mask")
    os.makedirs(out, exist_ok=True)
    model = load_model()
    n, masked, batch = len(meta["frames"]), 0, []

    def flush():
        nonlocal masked
        with torch.inference_mode():
            keep = keep_mask(model(normalize(torch.stack([torch.from_numpy(c) for _, c in batch]).cuda())))
        for (i, _), k in zip(batch, keep.cpu().numpy()):
            masked += not k.all()
            cv2.imwrite(os.path.join(out, f"{i:06d}.png"), k.astype(np.uint8) * 255)
        batch.clear()
        print(f"frames {i + 1}/{n}", flush=True)

    for i, color, _ in iter_frames(frames, meta, "depth_ffs"):
        batch.append((i, color))
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    print(f"{masked} / {n} frames with people -> {out}")


if __name__ == "__main__":
    main()
