"""Stage timings for docs/speed: `with span("ffs/trt"):` adds the block's wall time (ms) to times["ffs/trt"].
With PROFILE=1 every span is also an NVTX range (seen in nsys). GPU work counts only as far as the block
waits for it; nsys shows the rest. Each process has its own `times` (the tracker sends its own)."""
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

NVTX = os.environ.get("PROFILE") == "1"
times = defaultdict(list)

if NVTX:
    import nvtx  # noqa: F401  (pip install nvtx), no torch needed in the tracker process


@contextmanager
def span(name):
    if NVTX:
        import nvtx
        rng = nvtx.start_range(name)
    t = time.perf_counter()
    try:
        yield
    finally:
        times[name].append((time.perf_counter() - t) * 1000)
        if NVTX:
            nvtx.end_range(rng)


def summary(t=None):
    """{name: {n, median, p90, total_s}} of `times` (or t)."""
    return {k: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "total_s": float(np.sum(v)) / 1000} for k, v in sorted((times if t is None else t).items()) if v}
