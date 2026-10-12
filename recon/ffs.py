"""Fast-FoundationStereo depth: TensorRT disparity of an IR pair, warped into the color camera."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

FFS_ROOT = os.path.expanduser("~/Fast-FoundationStereo")
ENGINE = "output/ffs/ffs_480x864_it8_fp32_tf32.engine"  # recon/export_ffs_trt.py: 848x480 padded to 32
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


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
    """ffs(left, right): HxW uint8 CUDA tensors -> HxW float32 disparity of the left image."""

    def __init__(self, engine=ENGINE):
        import tensorrt as trt
        if not os.path.exists(engine):
            raise SystemExit(f"{engine} not found: build it with recon/export_ffs_trt.py")
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
        h, w = left.shape
        H, W = self.shape
        imgs = [F.pad(torch.as_tensor(x, device="cuda").float()[None, None].expand(1, 3, -1, -1),
                      (0, W - w, 0, H - h), mode="replicate") for x in (left, right)]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            bufs = [((x - self.mean) / self.std).contiguous() for x in imgs]
            for name, b in zip(self.inputs, bufs):
                self.context.set_tensor_address(name, b.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
            disp = self.out[0, 0, :h, :w].clone()  # self.out is reused by the next call
        torch.cuda.current_stream().wait_stream(self.stream)
        return disp


class DisparityToColorDepth:
    """Left-IR disparity -> depth (m) in the color camera, 0 where none: each point splatted to its
    2x2 pixel neighbourhood, nearest depth wins."""

    def __init__(self, meta, dev):
        st = meta["stereo"]
        self.H, self.W, self.dev = meta["height"], meta["width"], dev
        self.fx_baseline = st["fx"] * float(np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3]))
        self.K_ir = torch.tensor([[st["fx"], 0, st["cx"]], [0, st["fy"], st["cy"]], [0, 0, 1]], device=dev)
        self.K_c = torch.tensor([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]], device=dev)
        self.color_from_ir1 = torch.tensor(np.linalg.inv(np.array(st["ir1_from_color"])), device=dev,
                                           dtype=torch.float32)
        self.v, self.u = torch.meshgrid(torch.arange(st["height"], device=dev, dtype=torch.float32),
                                        torch.arange(st["width"], device=dev, dtype=torch.float32), indexing="ij")

    def __call__(self, disp):
        H, W, u, v, K_ir, K_c = self.H, self.W, self.u, self.v, self.K_ir, self.K_c
        ok = (disp > 0.5) & (u - disp >= 0)  # the match must fall inside the right image
        z = self.fx_baseline / disp[ok]
        X = torch.stack([(u[ok] - K_ir[0, 2]) * z / K_ir[0, 0], (v[ok] - K_ir[1, 2]) * z / K_ir[1, 1], z])
        Y = self.color_from_ir1[:3, :3] @ X + self.color_from_ir1[:3, 3:]
        pu, pv = Y[0] / Y[2] * K_c[0, 0] + K_c[0, 2], Y[1] / Y[2] * K_c[1, 1] + K_c[1, 2]
        depth = torch.full((H * W,), torch.inf, device=self.dev)
        for du in (0, 1):
            for dv in (0, 1):
                x, y = torch.floor(pu).long() + du, torch.floor(pv).long() + dv
                m = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (Y[2] > 0)
                depth.scatter_reduce_(0, y[m] * W + x[m], Y[2][m], reduce="amin")
        return torch.where(torch.isinf(depth), 0, depth).reshape(H, W)
