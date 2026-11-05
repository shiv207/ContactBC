"""Self-improvement round (RECAP-style, as applied to laundry folding in pi0.6*): roll out a trained policy on
training cloth poses, keep only rollouts the simulator verifies as clean folds (folded, grasp held, released,
cloth never flung), and save them as extra BC data. Retraining on them moves the data toward the states the policy
actually visits. Simplified: plain success filtering, no advantage conditioning.

Training poses use RNG seeds disjoint from eval.py's held-out EVAL_SEED.

  python fold/self_improve.py --variant dart+rel --seeds 0 1 2 --out data/fold_self_r1 --round 1
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fold import eval as E   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H = E.H


def _rollout(args):
    i, rnd, out = args
    from fold.augment import sample_pose
    from fold.replay import obs_vec, home_q, label_state, calm_check
    s, ik = E._W["s"], E._W["ik"]
    seed = 7000 + 100 * rnd + E._W["seed"]
    assert seed != E.EVAL_SEED
    xy, yaw = sample_pose(np.random.default_rng(seed * 1000 + i))
    s.reset(xy, yaw, arm_q=home_q(s, ik))
    act = E.make_actor(E._W["pol"])
    log = {k: [] for k in ["obs", "action", "grasp", "cloth_top", "cloth_low", "cloth_speed"]}
    v_prev = s.verts()
    for t in range(H):
        o = obs_vec(s); a = s.clip_action(act(o, t))
        log["obs"].append(o); log["action"].append(a)
        s.step(a)
        log["grasp"].append(s.grasp_state())
        v = s.verts(); log["cloth_top"].append(v[:, 2].max()); log["cloth_low"].append(v[:, 2].min())
        log["cloth_speed"].append(np.linalg.norm(v - v_prev, axis=1).max() * 30); v_prev = v
    ep = {k: np.array(v) for k, v in log.items()}
    G = ep["grasp"]
    both = np.flatnonzero(G.min(1) > 0)
    if not len(both):
        return dict(i=i, success=False)
    k_grasp = int(both[0])
    after = np.flatnonzero(G[k_grasp:].max(1) == 0)
    k_release = k_grasp + int(after[0]) if len(after) else H - 1
    held_any = np.flatnonzero(G.max(1) > 0)
    r = label_state(s, G, slice(held_any[0], held_any[-1] + 1))
    r.update(calm_check(dict(success=r["success"], **{k: ep[k] for k in ["cloth_top", "cloth_low", "cloth_speed"]})))
    if r["success"]:
        np.savez_compressed(Path(out) / f"self_r{rnd}_s{E._W['seed']}_{i:05d}.npz", **ep, **r,
                            cloth_xy=xy, yaw=yaw, take="self", k_grasp=k_grasp, k_release=k_release)
    return dict(i=i, success=bool(r["success"]))


def _init(variant, seed):
    E._init(variant, seed); E._W["seed"] = seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dart+rel")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=250, help="rollouts per policy seed")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", default="data/fold_self_r1")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.npz"):
        f.unlink()
    summary = {}
    for sd in a.seeds:
        with Pool(a.workers, initializer=_init, initargs=(a.variant, sd)) as pool:
            res = pool.map(_rollout, [(i, a.round, str(out)) for i in range(a.trials)])
        summary[f"{a.variant}_s{sd}"] = sum(r["success"] for r in res) / len(res)
        print(f"{a.variant} seed {sd}: kept {sum(r['success'] for r in res)}/{len(res)}", flush=True)
    json.dump(summary, open(ROOT / f"outputs/fold_self_summary_r{a.round}.json", "w"), indent=1)
