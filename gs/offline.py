"""A live session -> <session>/scene.usdc: person masks (unless main.py made them), TSDF mesh + point
cloud (people left out), COLMAP dataset with the known poses, FastGS 3DGS (no loss on people), USD.
Prints "progress <0..1> <seconds left> <step>" lines (read by main.py's window); each run's step
times go to output/offline_times.jsonl, which the estimates come from.

Usage: python -m gs.offline output/live_room
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FASTGS = os.path.expanduser("~/FastGS")
HISTORY = os.path.join(ROOT, "output", "offline_times.jsonl")
STEPS = ["Person masks", "TSDF mesh", "COLMAP export", "3DGS training", "USD export"]
# s per frame, per frame, per frame, per training step, once; until there is a history
DEFAULT_RATES = {"mask": 0.03, "tsdf": 0.05, "colmap": 0.01, "train": 0.012, "usd": 20.0}


def session_size(frames):
    """(frames, training steps): each image seen ~38 times, at least 20k steps."""
    with open(os.path.join(frames, "meta.json")) as f:
        n = len(json.load(f)["frames"])
    return n, max(20000, (38 * n + 500) // 1000 * 1000)


def step_estimates(n, iters, masked, last=5):
    """Seconds per step from the mean rates of the last exports; no mask step when `masked`."""
    try:
        with open(HISTORY) as f:
            rows = [json.loads(line) for line in f][-last:]
    except FileNotFoundError:
        rows = []
    r = {}
    for k, v in DEFAULT_RATES.items():
        seen = [row[k] for row in rows if k in row]  # runs before a step existed lack its rate
        r[k] = sum(seen) / len(seen) if seen else v
    return [0 if masked else r["mask"] * n, r["tsdf"] * n, r["colmap"] * n, r["train"] * iters, r["usd"]]


def estimate(frames):
    """Estimated seconds of a whole run for a session's frames/."""
    return sum(step_estimates(*session_size(frames), os.path.isdir(os.path.join(frames, "mask"))))


class Progress:
    """Progress by estimated time; the time left is scaled by how far the run is off the estimate."""

    def __init__(self, est):
        self.est, self.t0 = est, time.perf_counter()

    def __call__(self, step, frac):
        done = sum(self.est[:step]) + self.est[step] * frac
        total = sum(self.est)
        elapsed = time.perf_counter() - self.t0
        left = (total - done) * (elapsed / done if done > 0.05 * total else 1)
        print(f"progress {done / total:.3f} {left:.0f} {STEPS[step]}", flush=True)


def run(progress, step, cmd, cwd=None, env=None):
    """cmd as step `step`; FastGS's tqdm "n/total" (or person_mask.py's "frames n/total") moves the
    progress within the step. Returns seconds."""
    t0 = time.perf_counter()
    progress(step, 0)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf, shown = b"", -1
    for chunk in iter(lambda: p.stdout.read1(4096), b""):
        *lines, buf = re.split(rb"[\r\n]", buf + chunk)
        for line in lines:
            m = re.search(rb"(?:Training progress:.*?|^frames )(\d+)/(\d+)", line)
            if m:
                pct = 100 * int(m[1]) // int(m[2])
                if pct != shown:
                    shown = pct
                    progress(step, pct / 100)
            elif line.strip():
                print(line.decode(errors="replace"), flush=True)
    if p.wait():
        raise SystemExit(f"{STEPS[step]} failed")
    return time.perf_counter() - t0


def main():
    s = os.path.abspath(sys.argv[1])
    frames, py = os.path.join(s, "frames"), sys.executable
    if not os.path.exists(os.path.join(frames, "poses.npy")):
        raise SystemExit(f"{frames}/poses.npy not found: not a live session?")
    n, iters = session_size(frames)
    masked = os.path.isdir(os.path.join(frames, "mask"))  # by main.py's depth thread
    progress = Progress(step_estimates(n, iters, masked))
    t = [0 if masked else run(progress, 0, [py, "-m", "recon.person_mask", frames], cwd=ROOT)]
    t.append(run(progress, 1, [py, "-m", "recon.reconstruct", frames, os.path.join(s, "recon")], cwd=ROOT))
    t.append(run(progress, 2, [py, "-m", "gs.export_colmap", frames,
                               os.path.join(s, "recon/pointcloud.ply"), os.path.join(s, "colmap")], cwd=ROOT))
    # Densify for the first half, at most to 15k: FastGS prunes Gaussians below 0.1 opacity every 3k
    # steps from 15k, and an opacity reset (every 3k while densifying) on such a step removes them all.
    t.append(run(progress, 3, [
        py, "train.py", "-s", os.path.join(s, "colmap"), "-i", "images", "-m", os.path.join(s, "fastgs"), "--quiet",
        "--optimizer_type", "default", "--densification_interval", "500", "--highfeature_lr", "0.02",
        "--grad_abs_thresh", "0.0008", "--iterations", str(iters), "--position_lr_max_steps", str(iters),
        "--densify_until_iter", str(min(iters // 2, 15000)), "--save_iterations", str(iters)],
        cwd=FASTGS, env=dict(os.environ, PYTHONPATH=FASTGS)))
    t.append(run(progress, 4, [py, "-m", "gs.export_usd", os.path.join(s, "recon/mesh.ply"),
                               os.path.join(s, f"fastgs/point_cloud/iteration_{iters}/point_cloud.ply"),
                               os.path.join(frames, "poses.npy"), os.path.join(s, "scene.usdc")], cwd=ROOT))
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a") as f:
        row = {"session": s, "frames": n, "iters": iters, "total_s": round(sum(t)), "tsdf": t[1] / n,
               "colmap": t[2] / n, "train": t[3] / iters, "usd": t[4]}
        f.write(json.dumps(row if masked else dict(row, mask=t[0] / n)) + "\n")
    print(f"3DGS -> USD done in {sum(t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
