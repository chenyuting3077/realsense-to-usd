"""A profiled run (run_replay.sh) -> its stage table: live.json's stages (recon/prof.py) of the main
process and the tracker, and the mean GPU load / clock, memory bandwidth and CPU load from tegrastats.

Usage: python -m bench.summarize <out dir> [<out dir> ...]   (several: median of the runs)
"""
import json
import os
import re
import sys

import numpy as np


def tegrastats(path):
    """Means over the log, or {}: CPU load and clock over the 14 cores, the busiest core, GPU power
    (Thor's tegrastats has no GPU load or memory bandwidth: nsys shows the GPU)."""
    if not os.path.exists(path):
        return {}
    cpu, mhz, top, gpu_w = [], [], [], []
    for line in open(path):
        m = re.search(r"CPU \[([^\]]*)\]", line)
        if m:
            cores = [(int(a), int(b)) for a, b in re.findall(r"(\d+)%@(\d+)", m[1])]
            if cores:
                cpu.append(np.mean([a for a, _ in cores]))
                top.append(max(a for a, _ in cores))
                mhz.append(np.mean([b for _, b in cores]))
        m = re.search(r"VDD_GPU (\d+)mW", line)
        if m:
            gpu_w.append(int(m[1]) / 1000)
    mean = lambda x: float(np.mean(x)) if x else None  # noqa: E731
    return {"cpu %": mean(cpu), "busiest core %": mean(top), "cpu MHz": mean(mhz), "gpu W": mean(gpu_w)}


def load(out):
    with open(os.path.join(out, "live.json")) as f:
        r = json.load(f)
    return r, {**r.get("stages", {}), **{f"[tracker] {k}": v for k, v in r.get("tracker_stages", {}).items()}}


def main():
    runs = [load(o) for o in sys.argv[1:]]
    med = lambda f: float(np.median([f(r) for r in runs]))  # noqa: E731
    r0 = runs[0][0]
    print(f"{len(runs)} run(s); run {med(lambda r: r[0]['run_s']):.1f} s, depth {med(lambda r: r[0]['depth']['hz']):.2f} Hz "
          f"({med(lambda r: r[0]['depth']['frames']):.0f} frames), tracked "
          f"{med(lambda r: r[0]['tracking']['tracked']):.0f}/{r0['tracking']['frames']}, capture -> map "
          f"{med(lambda r: (r[0]['latency_ms'] or {}).get('median', np.nan)):.0f} / "
          f"{med(lambda r: (r[0]['latency_ms'] or {}).get('p90', np.nan)):.0f} ms, "
          f"final mesh {med(lambda r: r[0]['final_mesh_s']):.1f} s, {med(lambda r: r[0]['mesh_triangles']):.0f} triangles")
    print(f"| stage | n | median ms | p90 ms | total s |\n|---|---:|---:|---:|---:|")
    for k in runs[0][1]:
        if all(k in s for _, s in runs):
            v = lambda f: med(lambda r: r[1][k][f])  # noqa: E731
            print(f"| {k} | {v('n'):.0f} | {v('median'):.1f} | {v('p90'):.1f} | {v('total_s'):.1f} |")
    for o in sys.argv[1:]:
        print(o, {k: round(v, 1) for k, v in tegrastats(os.path.join(o, "tegrastats.log")).items() if v is not None})


if __name__ == "__main__":
    main()
