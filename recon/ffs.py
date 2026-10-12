"""Fast-FoundationStereo depth: TensorRT disparity of an IR pair -> filtered depth in the color camera."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from recon.frames import intrinsic_matrix

FFS_ROOT = os.path.expanduser("~/Fast-FoundationStereo")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EDGE_THRESH = 0.05  # flying pixels: 3x3 depth spread above this fraction of the depth
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


PRECISIONS = {"tf32": "fp32_tf32", "amp": "amp"}


def engine_path(height=480, width=864, max_disp=192, precision="amp"):
    """export_ffs_trt.py's engine for an input padded to height x width (8 iterations; v3: the cost
    volumes of recon/ffs_volume.py). precision: tf32 (fp32 with TF32) or amp (FFS's own fp16 autocast:
    the networks in fp16, the disparity and its sampling coordinates in fp32)."""
    md = "" if max_disp == 192 else f"_md{max_disp}"
    return os.path.join(ROOT, f"output/ffs/ffs_{height}x{width}_it8{md}_v3_{PRECISIONS[precision]}.engine")


# 848x480 IR at full size (padded to 864) with FFS's fp16 autocast: 62 ms, the accuracy of TF32 at 848x480
# (docs/profile)
ENGINE, SCALE = engine_path(), 1.0


def load_model(valid_iters=8, max_disp=192):
    """The PyTorch model (for the ONNX export)."""
    sys.path.insert(0, FFS_ROOT)
    from omegaconf import open_dict
    model = torch.load(os.path.join(FFS_ROOT, "weights/c-ffs/model_best_bp2_serialize.pth"), map_location="cpu",
                       weights_only=False)
    with open_dict(model.args):
        model.args.setdefault("normalize", True)  # missing from the C-FFS checkpoint
    model.args.valid_iters, model.args.max_disp = valid_iters, max_disp
    return model.cuda().eval()


class FFS:
    """ffs(left, right): HxW uint8 CUDA tensors -> HxW float32 disparity of the left image. With
    scale < 1 the pair is downscaled (area) for the engine and the disparity scaled back up."""

    def __init__(self, engine=ENGINE, scale=SCALE):
        import tensorrt as trt
        self.scale = scale
        if not os.path.exists(engine):
            raise SystemExit(f"{engine} not found: build it with python -m recon.export_ffs_trt")
        with open(engine, "rb") as f:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"cannot load {engine} (built with another TensorRT version?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.shape = tuple(self.engine.get_tensor_shape(self.inputs[0]))[2:]
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(output)), device="cuda")
        self.context.set_tensor_address(output, self.out.data_ptr())
        self.mean, self.std = MEAN.cuda(), STD.cuda()
        self.stream = torch.cuda.Stream()  # TensorRT syncs extra on the default stream

    @torch.no_grad()
    def __call__(self, left, right):
        h0, w0 = left.shape
        imgs = [torch.as_tensor(x, device="cuda").float()[None, None] for x in (left, right)]
        if self.scale != 1:
            imgs = [F.interpolate(x, scale_factor=self.scale, mode="area") for x in imgs]
        h, w = imgs[0].shape[2:]
        H, W = self.shape
        imgs = [F.pad(x.expand(1, 3, -1, -1), (0, W - w, 0, H - h), mode="replicate") for x in imgs]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            bufs = [((x - self.mean) / self.std).contiguous() for x in imgs]
            for name, b in zip(self.inputs, bufs):
                self.context.set_tensor_address(name, b.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
            disp = self.out[0, 0, :h, :w].clone()  # self.out is reused by the next call
            if self.scale != 1:
                disp = F.interpolate(disp[None, None], size=(h0, w0), mode="bilinear")[0, 0] * (w0 / w)
        torch.cuda.current_stream().wait_stream(self.stream)
        return disp


def drop_flying_pixels(d):
    """Zero depth next to a hole or on an edge (3x3 spread over EDGE_THRESH of it), where stereo
    smears foreground into background. max_pool2d pads with -inf: outside pixels are ignored."""
    x = d[None, None]
    hole = F.max_pool2d((x <= 0).float(), 3, 1, 1) > 0
    spread = F.max_pool2d(x, 3, 1, 1) + F.max_pool2d(-torch.where(x > 0, x, torch.inf), 3, 1, 1)
    return torch.where(hole | (spread > EDGE_THRESH * x), 0, x)[0, 0]


class DisparityToColorDepth:
    """Left-IR disparity -> filtered depth (m) in the color camera, 0 where none: each point splatted
    to its 2x2 pixel neighbourhood, nearest depth wins."""

    def __init__(self, meta, dev):
        st = meta["stereo"]
        self.H, self.W, self.dev = meta["height"], meta["width"], dev
        self.fx_baseline = st["fx"] * float(np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3]))
        self.K_ir = torch.tensor([[st["fx"], 0, st["cx"]], [0, st["fy"], st["cy"]], [0, 0, 1]], device=dev)
        self.K_c = torch.tensor(intrinsic_matrix(meta), device=dev, dtype=torch.float32)
        self.color_from_ir1 = torch.tensor(np.linalg.inv(np.array(st["ir1_from_color"])), device=dev,
                                           dtype=torch.float32)
        self.v, self.u = torch.meshgrid(torch.arange(st["height"], device=dev, dtype=torch.float32),
                                        torch.arange(st["width"], device=dev, dtype=torch.float32), indexing="ij")

    def __call__(self, disp):
        """Every pixel is computed and the invalid ones scattered to a spare slot: no boolean indexing,
        which would wait for the GPU (holding the GIL) to learn the count."""
        H, W, u, v, K_ir, K_c = self.H, self.W, self.u.ravel(), self.v.ravel(), self.K_ir, self.K_c
        disp = disp.ravel()
        ok = (disp > 0.5) & (u - disp >= 0)  # the match must fall inside the right image
        z = self.fx_baseline / torch.where(ok, disp, 1.0)
        X = torch.stack([(u - K_ir[0, 2]) * z / K_ir[0, 0], (v - K_ir[1, 2]) * z / K_ir[1, 1], z])
        Y = self.color_from_ir1[:3, :3] @ X + self.color_from_ir1[:3, 3:]
        pu, pv = Y[0] / Y[2] * K_c[0, 0] + K_c[0, 2], Y[1] / Y[2] * K_c[1, 1] + K_c[1, 2]
        depth = torch.full((H * W + 1,), torch.inf, device=self.dev)  # [H * W]: the invalid ones
        for du in (0, 1):
            for dv in (0, 1):
                x, y = torch.floor(pu).long() + du, torch.floor(pv).long() + dv
                m = ok & (x >= 0) & (x < W) & (y >= 0) & (y < H) & (Y[2] > 0)
                depth.scatter_reduce_(0, torch.where(m, y * W + x, H * W), Y[2], reduce="amin")
        return drop_flying_pixels(torch.where(torch.isinf(depth[:-1]), 0, depth[:-1]).reshape(H, W))
