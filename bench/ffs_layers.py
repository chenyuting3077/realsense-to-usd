"""Where the FFS engine spends its time: TensorRT's per-layer profiler (pip TensorRT; the system
trtexec is another version and cannot load the engine), layers grouped by the FFS module in their name.
Each profiled run is synchronous per layer, so the total is above the normal call; the shares count.

Usage: python -m bench.ffs_layers [engine]   (default: recon.ffs.ENGINE)
"""
import re
import sys
from collections import defaultdict

import numpy as np
import tensorrt as trt
import torch

from recon.ffs import ENGINE

RUNS, WARMUP = 20, 10


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms = defaultdict(list)

    def report_layer_time(self, name, ms):
        self.ms[name].append(ms)


def group(name):
    """The model module a layer comes from (ONNX node names keep the PyTorch module path)."""
    m = re.search(r"/((?:feature|cnet|stem|update_block|corr|cost_agg|classifier|context|gwc|concat|spx|"
                  r"refine|conv|sam|cam)[A-Za-z_0-9]*)", name)
    return m[1] if m else name.split("/")[1] if name.count("/") > 1 else "other"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ENGINE
    with open(path, "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    bufs = []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        t = torch.randn(tuple(engine.get_tensor_shape(n)), device="cuda")
        ctx.set_tensor_address(n, t.data_ptr())
        bufs.append(t)
    s = torch.cuda.Stream()
    for _ in range(WARMUP):
        ctx.execute_async_v3(s.cuda_stream)
    s.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    times = []
    for _ in range(RUNS):
        ev[0].record(s)
        ctx.execute_async_v3(s.cuda_stream)
        ev[1].record(s)
        s.synchronize()
        times.append(ev[0].elapsed_time(ev[1]))
    print(f"engine alone: {np.median(times):.1f} ms (median of {RUNS})")
    prof = Profiler()
    ctx.profiler = prof
    for _ in range(RUNS):
        ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
    layer = {k: float(np.median(v)) for k, v in prof.ms.items()}
    total = sum(layer.values())
    groups = defaultdict(float)
    for k, v in layer.items():
        groups[group(k)] += v
    print(f"profiled: {total:.1f} ms in {len(layer)} layers\n| part | ms | share |\n|---|---:|---:|")
    for k, v in sorted(groups.items(), key=lambda x: -x[1])[:15]:
        print(f"| {k} | {v:.1f} | {100 * v / total:.0f}% |")
    print("\nslowest layers:")
    for k, v in sorted(layer.items(), key=lambda x: -x[1])[:15]:
        print(f"  {v:6.2f} ms  {k[:150]}")


if __name__ == "__main__":
    main()
