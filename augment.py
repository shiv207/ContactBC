"""Step 5 (MimicGen-style): randomize cup/coaster poses, warp each retargeted human take to the new poses,
replay in physics, keep only contact-confirmed successes.

Each source take is split into object-centric segments (approach | transport | retreat) by its grasp/release
keyframes; for a new scene the segments are re-anchored to the new cup / coaster poses (retarget.Plan.wrist_targets)
and executed with IK. Nothing is kept unless replay.label() says the grasp held AND the cup ended on the coaster.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/augmented"
# workspace (m, world frame) the G1 right arm reaches comfortably from its fixed base
X_RANGE, Y_RANGE, MIN_SEP = (0.27, 0.40), (-0.33, 0.0), 0.12

_W = {}


def sample_poses(rng):
    while True:
        cup = np.array([rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)])
        co = np.array([rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)])
        if np.linalg.norm(cup - co) >= MIN_SEP:
            return cup, co


def _init(wrist_source):
    from scene import Scene
    from ik import ArmIK
    from retarget import build_plans
    _W["s"] = Scene(); _W["ik"] = ArmIK(_W["s"]); _W["plans"] = build_plans(wrist_source)


def _trial(i, seed=0, gif=False):
    from replay import run_episode
    from scene import Recorder
    rng = np.random.default_rng(seed * 100003 + i)
    plan = _W["plans"][rng.integers(len(_W["plans"]))]
    cup, co = sample_poses(rng)
    rec = Recorder(_W["s"]) if gif else None
    ep = run_episode(_W["s"], _W["ik"], plan, cup, co, rec=rec)
    info = dict(i=i, take=plan.name, cup=cup.round(3).tolist(), coaster=co.round(3).tolist(),
                **{k: (bool(ep[k]) if isinstance(ep[k], (bool, np.bool_)) else round(float(ep[k]), 3))
                   for k in ["success", "grasp_held", "placed", "lift", "dist", "upright"]})
    if ep["success"]:
        np.savez_compressed(OUT / f"aug_{i:05d}.npz", **ep)
    if rec is not None:
        rec.save(ROOT / f"outputs/gifs/aug_{i:05d}_{'success' if ep['success'] else 'fail'}.gif")
        rec.r.close()
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="number of successful episodes to keep")
    ap.add_argument("--max-trials", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--wrist-source", default="egodex", choices=["egodex", "mediapipe"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("aug_*.npz"):
        f.unlink()
    log, n_ok = [], 0
    with Pool(args.workers, initializer=_init, initargs=(args.wrist_source,)) as pool:
        for info in pool.imap_unordered(partial(_trial, seed=args.seed), range(args.max_trials), chunksize=4):
            log.append(info); n_ok += info["success"]
            if len(log) % 50 == 0:
                print(f"trials {len(log)}  kept {n_ok}  ({n_ok / len(log):.0%})", flush=True)
            if n_ok >= args.target:
                pool.terminate(); break
    by_take = {}
    for r in log:
        t = by_take.setdefault(r["take"], [0, 0]); t[0] += r["success"]; t[1] += 1
    summary = dict(trials=len(log), kept=n_ok, rate=n_ok / max(1, len(log)),
                   per_take={k: f"{a}/{b}" for k, (a, b) in sorted(by_take.items())},
                   fail_grasp=sum(not r["grasp_held"] for r in log),
                   fail_place=sum(r["grasp_held"] and not r["placed"] for r in log))
    json.dump(dict(summary=summary, trials=log), open(OUT / "augment_log.json", "w"), indent=1)
    json.dump(summary, open(ROOT / "outputs/augment_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
