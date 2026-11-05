"""Recovery demos (failure-correction data, cf. "Failing Forward", arXiv 2605.08434): the planner deliberately
misses the grasp with one hand, opens it, moves to the true corner and grasps again while the other hand keeps
holding. Then the fold continues as usual. The missed hand is visible in the observation as "hand closed, grasp
flag 0", the state that learned policies get stuck in.

Timeline (base plan steps -> new steps):
  base[0:50]   reach + close; the missing hand aims MISS_UP above and MISS_OUT outside its corner (> GRASP_R away)
  R steps      missing hand opens, then moves to its true grasp point; the other hand holds closed
  base[40:]    close again (the holding hand stays closed), then transport / release / retreat as planned

  python fold/recovery.py --target 300
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fold import replay as R                         # noqa: E402
from fold.augment import sample_pose                 # noqa: E402
from fold.retarget import N_REACH, DWELL             # noqa: E402
from fold.scene import CLOSE, SIDES                  # noqa: E402
from retarget import smooth                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RT, OPEN_N, MOVE_N = 30, 8, 16            # inserted recovery steps: open over 8, move over 16, settle the rest
MISS_UP, MISS_OUT = 0.025, 0.025          # fixed offsets (--vary off)
VARY_UP, VARY_OUT, VARY_ANG = (0.01, 0.035), (0.02, 0.045), np.radians(60)   # --vary: randomized miss
C0 = N_REACH + DWELL                      # = base k_grasp (end of the first close)


class RecoveryPlan:
    def __init__(self, base, miss, out_dir, up=MISS_UP, out=MISS_OUT):
        """miss: side that misses; out_dir: (2,) unit xy vector pointing away from the cloth at that corner."""
        self.base, self.miss, self.name = base, miss, base.name
        self.shift = RT + DWELL
        self.T, self.k_grasp, self.k_release = base.T + self.shift, base.k_grasp + self.shift, base.k_release + self.shift
        self.off = np.r_[out * out_dir, up]
        open_ramp = 1 - smooth(np.clip(np.arange(RT) / OPEN_N, 0, 1))
        self.hand = {}
        for sd in SIDES:
            h = base.hand[sd]
            if sd == miss:
                mid = open_ramp[:, None] * CLOSE[sd]
                self.hand[sd] = np.concatenate([h[:C0], mid, h[N_REACH:]])
            else:
                hold = np.tile(CLOSE[sd], (RT + DWELL, 1))
                self.hand[sd] = np.concatenate([h[:C0], hold, h[C0:]])

    def wrist_targets(self, grasp, place):
        b = self.base.wrist_targets(grasp, place)
        out = {}
        for sd in SIDES:
            o = b[sd]
            if sd == self.miss:
                first = o[:C0].copy()
                ramp = smooth(np.clip((np.arange(C0) - N_REACH // 2) / (N_REACH // 2), 0, 1))[:, None]
                first += ramp * self.off              # drift onto the wrong point while descending
                wrong, right = first[-1], o[C0 - 1]
                k = np.arange(RT)
                a = smooth(np.clip((k - OPEN_N) / MOVE_N, 0, 1))[:, None]
                lift = 0.01 * np.sin(np.pi * a)            # small arc so the move doesn't drag the cloth
                mid = wrong + a * (right - wrong) + lift * [0, 0, 1]
                out[sd] = np.concatenate([first, mid, o[N_REACH:]])
            else:
                out[sd] = np.concatenate([o[:C0], np.tile(o[C0 - 1], (RT + DWELL, 1)), o[C0:]])
        return out


def _trial(i, seed=0, out=None, vary=False):
    rng = np.random.default_rng(seed * 100003 + i)
    names = R._W.get("ok_names")
    if names is None:
        ok = json.load(open(ROOT / "outputs/fold_replay_results.json"))
        names = R._W["ok_names"] = sorted(n for n in R._W["plans"] if ok.get(n, {}).get("success"))
    base = R._W["plans"][names[rng.integers(len(names))]]
    xy, yaw = sample_pose(rng)
    s = R._W["s"]
    s.reset(xy, yaw)
    c = s.corners()
    miss = SIDES[rng.integers(2)]
    k = SIDES.index(miss)
    d = (c[k] - c.mean(0))[:2]; d /= np.linalg.norm(d)
    if vary:              # policy misses vary in size and direction: rotate the outward vector, random magnitudes
        a = rng.uniform(-VARY_ANG, VARY_ANG); ca, sa = np.cos(a), np.sin(a)
        d = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
        plan = RecoveryPlan(base, miss, d, up=rng.uniform(*VARY_UP), out=rng.uniform(*VARY_OUT))
    else:
        plan = RecoveryPlan(base, miss, d)
    ep = R.run_episode(s, R._W["ik"], plan, xy, yaw)
    G = ep["grasp"]
    missed = G[C0 + 2, k] == 0 and G[C0 + 2, 1 - k] == 1       # the scripted miss really missed
    ok = bool(ep["success"]) and bool(missed)
    if ok:
        np.savez_compressed(out / f"rec_s{seed}_{i:05d}.npz", **ep)
    return dict(i=i, success=ok, missed=bool(missed), folded=bool(ep["success"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--max-trials", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/fold_recovery")
    ap.add_argument("--vary", action="store_true", help="randomize the miss offset (size and direction)")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.npz"):
        f.unlink()
    from functools import partial
    log, n = [], 0
    with Pool(a.workers, initializer=R._init) as pool:
        for r in pool.imap_unordered(partial(_trial, seed=a.seed, out=out, vary=a.vary), range(a.max_trials), chunksize=2):
            log.append(r); n += r["success"]
            if len(log) % 100 == 0:
                print(f"trials {len(log)} kept {n}  scripted miss took effect {np.mean([x['missed'] for x in log]):.0%}", flush=True)
            if n >= a.target:
                pool.terminate(); break
    s = dict(trials=len(log), kept=n, miss_rate=float(np.mean([x["missed"] for x in log])),
             fold_rate=float(np.mean([x["folded"] for x in log])))
    json.dump(s, open(ROOT / f"outputs/fold_recovery_summary_{out.name}.json", "w"), indent=1)
    print(s)
