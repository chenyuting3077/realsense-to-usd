"""Fast-FoundationStereo disparity from a grey stereo pair, with a PyTorch or TensorRT backend.

    ffs = FFS(engine="output/ffs/ffs_480x640_it8_fp32_tf32.engine")  # TensorRT (fixed input size)
    ffs = FFS(valid_iters=8, precision="fp32")                # PyTorch (any size)
    disp = ffs(left, right)   # (H, W) uint8 / float CUDA tensors -> (H, W) float32 disparity, left image

The TensorRT engine comes from recon/export_ffs_trt.py (single ONNX, ImageNet normalisation
done here). The PyTorch model normalises inside forward().
"""
import os
import sys

import torch
import torch.nn.functional as F

FFS_ROOT = os.path.expanduser("~/Fast-FoundationStereo")
WEIGHTS = "weights/c-ffs/model_best_bp2_serialize.pth"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def load_model(root=FFS_ROOT, weights=WEIGHTS, valid_iters=8, max_disp=192):
    if root not in sys.path:
        sys.path.insert(0, root)
    from omegaconf import open_dict
    model = torch.load(os.path.join(root, weights), map_location="cpu", weights_only=False)
    with open_dict(model.args):
        model.args.setdefault("normalize", True)  # absent from the C-FFS checkpoint; code default
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    return model.cuda().eval()


class FFS:
    def __init__(self, engine=None, valid_iters=8, max_disp=192, precision="fp32", root=FFS_ROOT, weights=WEIGHTS):
        self.engine_path = engine
        if engine:
            import tensorrt as trt
            self.trt = trt
            with open(engine, "rb") as f:
                self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise SystemExit(f"cannot load {engine} (built with another TensorRT version?)")
            self.context = self.engine.create_execution_context()
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
            self.shape = tuple(self.engine.get_tensor_shape(self.inputs[0]))[2:]
            dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
            self.in_dtype = dtypes[self.engine.get_tensor_dtype(self.inputs[0])]
            self.out = torch.empty(tuple(self.engine.get_tensor_shape(self.output)), device="cuda",
                                   dtype=dtypes[self.engine.get_tensor_dtype(self.output)])
            self.context.set_tensor_address(self.output, self.out.data_ptr())
            self.mean, self.std = MEAN.cuda(), STD.cuda()
            self.stream = torch.cuda.Stream()  # TensorRT synchronises extra on the default stream
        else:
            self.model = load_model(root, weights, valid_iters, max_disp)
            # forward() has its own autocast blocks switched by the checkpoint's mixed_precision
            self.model.args.mixed_precision = precision == "fp16"
            self.valid_iters, self.precision = valid_iters, precision
            from core.utils.utils import InputPadder
            self.Padder = InputPadder

    @torch.no_grad()
    def __call__(self, left, right):
        h, w = left.shape
        imgs = [torch.as_tensor(x, device="cuda").float()[None, None].expand(1, 3, -1, -1) for x in (left, right)]
        if self.engine_path:
            H, W = self.shape
            if (h, w) != (H, W):  # pad bottom / right by replication, cropped off again below
                imgs = [F.pad(x, (0, W - w, 0, H - h), mode="replicate") for x in imgs]
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                bufs = [((x - self.mean) / self.std).to(self.in_dtype).contiguous() for x in imgs]
                for name, b in zip(self.inputs, bufs):
                    self.context.set_tensor_address(name, b.data_ptr())
                self.context.execute_async_v3(self.stream.cuda_stream)
                disp = self.out[0, 0, :h, :w].float().clone()  # self.out is reused by the next call
            torch.cuda.current_stream().wait_stream(self.stream)
            return disp
        pad = self.Padder(imgs[0].shape, divis_by=32, force_square=False)
        a, b = pad.pad(*imgs)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.precision == "fp16"):
            disp = self.model.forward(a, b, iters=self.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
        return pad.unpad(disp.float())[0, 0]
