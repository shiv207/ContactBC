"""MimicGen-style augmentation for folding: random cloth pose, warp a random human stroke to it, replay,
keep only verified folds (grasp held + folded + released)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker (else 20 workers oversubscribe)
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


def _trial(i, seed=0, noise=0.0, out=None):
    rng = np.random.default_rng(seed * 100003 + i)
    names = R._W.get("ok_names")
    if names is None:                  # MimicGen: only warp source demos whose direct replay succeeded
        ok = json.load(open(ROOT / "outputs/fold_replay_results.json"))
        names = R._W["ok_names"] = sorted(n for n in R._W["plans"] if ok.get(n, {}).get("success"))
    plan = R._W["plans"][names[rng.integers(len(names))]]
    xy, yaw = sample_pose(rng)
    ep = R.run_episode(R._W["s"], R._W["ik"], plan, xy, yaw, noise=noise, rng=rng)
    if ep["success"]:
        np.savez_compressed((out or OUT) / f"aug_s{seed}_{i:05d}.npz", **ep)
    return dict(i=i, take=plan.name, success=bool(ep["success"]), corner_dist=round(float(ep["corner_dist"]), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-trials", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/fold_augmented")
    ap.add_argument("--noise", type=float, default=0.0, help="DART noise std (rad) on executed arm targets")
    ap.add_argument("--resume", action="store_true", help="keep existing episodes, count them toward --target")
    args = ap.parse_args()
    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    have = sorted(OUT.glob("aug_*.npz"))
    if not args.resume:
        for f in have:
            f.unlink()
        have = []
    log, n_ok = [], len(have)
    with Pool(args.workers, initializer=R._init) as pool:
        for r in pool.imap_unordered(partial(_trial, seed=args.seed, noise=args.noise, out=OUT), range(args.max_trials), chunksize=2):
            log.append(r); n_ok += r["success"]
            if len(log) % 100 == 0:
                print(f"trials {len(log)} kept {n_ok} ({n_ok / len(log):.0%})", flush=True)
            if n_ok >= args.target:
                pool.terminate(); break
    summary = dict(trials_this_run=len(log), kept_total=n_ok, rate_this_run=(n_ok - len(have)) / max(1, len(log)))
    json.dump(summary, open(ROOT / f"outputs/fold_augment_summary_{OUT.name}.json", "w"), indent=1)
    print(summary)
