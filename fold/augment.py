"""MimicGen-style augmentation for folding: random cloth pose, warp a random human stroke to it, replay,
keep only verified folds (grasp held + folded + released)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fold import replay as R            # noqa: E402
from fold.scene import CLOTH_XY         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/fold_augmented"
DXY, DYAW = np.array([0.02, 0.05]), np.radians(15)


def sample_pose(rng):
    return CLOTH_XY + rng.uniform(-1, 1, 2) * DXY, float(rng.uniform(-1, 1) * DYAW)


def _trial(i, seed=0):
    rng = np.random.default_rng(seed * 100003 + i)
    names = sorted(R._W["plans"])
    plan = R._W["plans"][names[rng.integers(len(names))]]
    xy, yaw = sample_pose(rng)
    ep = R.run_episode(R._W["s"], R._W["ik"], plan, xy, yaw)
    if ep["success"]:
        np.savez_compressed(OUT / f"aug_{i:05d}.npz", **ep)
    return dict(i=i, take=plan.name, success=bool(ep["success"]), corner_dist=round(float(ep["corner_dist"]), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-trials", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("aug_*.npz"):
        f.unlink()
    log, n_ok = [], 0
    with Pool(args.workers, initializer=R._init) as pool:
        for r in pool.imap_unordered(partial(_trial), range(args.max_trials), chunksize=2):
            log.append(r); n_ok += r["success"]
            if len(log) % 100 == 0:
                print(f"trials {len(log)} kept {n_ok} ({n_ok / len(log):.0%})", flush=True)
            if n_ok >= args.target:
                pool.terminate(); break
    summary = dict(trials=len(log), kept=n_ok, rate=n_ok / len(log))
    json.dump(summary, open(ROOT / "outputs/fold_augment_summary.json", "w"), indent=1)
    print(summary)
