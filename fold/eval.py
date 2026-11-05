"""Closed-loop evaluation of fold policies on held-out cloth poses (seed never used by augment.py).

  python fold/eval.py --variants plain aug aug+cg aug+rl --seeds 0 1 2
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
H, EVAL_SEED = 330, 4242
_W = {}


def _init(variant, seed):
    import torch
    torch.set_num_threads(1)
    from fold.scene import FoldScene, make_ik
    _W["s"] = FoldScene(); _W["ik"] = make_ik(_W["s"]); _W["variant"] = variant
    if "+rl" in variant:
        from fold.rl import load_rl
        _W["pol"] = load_rl(variant, seed)
    else:
        from fold.bc import load_policy
        _W["pol"] = load_policy(variant, seed)


def make_actor(pol):
    from fold.bc import Ensembler
    if hasattr(pol, "actor"):            # RL policy = BC ensembler + residual (deterministic at eval)
        return pol.actor(deterministic=True)
    ens = Ensembler(pol)
    return lambda obs, t: ens.act(obs, t)


def rollout(i, gif=False, log_feats=False):
    from fold.augment import sample_pose
    from fold.replay import obs_vec, home_q, label_state
    from scene import Recorder
    s, ik = _W["s"], _W["ik"]
    xy, yaw = sample_pose(np.random.default_rng(EVAL_SEED * 1000 + i))
    s.reset(xy, yaw, arm_q=home_q(s, ik))
    act = make_actor(_W["pol"])
    rec = Recorder(s) if gif else None
    G, F = [], []
    for t in range(H):
        o = obs_vec(s)
        a = s.clip_action(act(o, t))
        s.step(a)
        G.append(s.grasp_state())
        if log_feats:
            F.append(robot_features(s, a))
        if rec:
            rec.grab()
    G = np.array(G)
    # transport = steps where at least one hand held the cloth
    held_any = np.flatnonzero(G.max(1) > 0)
    tr = slice(held_any[0], held_any[-1] + 1) if len(held_any) else slice(0, 0)
    r = label_state(s, G, tr)
    r = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in r.items()}
    if not len(held_any):
        r["held"] = 0.0; r["success"] = False
    if rec:
        tag = _W["variant"].replace("+", "_")
        rec.save(ROOT / f"outputs/gifs/fold_eval_{tag}_{i:02d}_{'success' if r['success'] else 'fail'}.gif")
        rec.r.close()
    out = dict(i=i, **r)
    if log_feats:
        out["feats"] = np.array(F).tolist()
    return out


def robot_features(s, a):
    """Per-frame features shared with the human data for phases.py: per hand (wrist height above table,
    fingertip-pinch closure), plus wrist positions for speed / inter-hand distance."""
    from fold.scene import SIDES
    from scene import TABLE_Z
    d = s.data; f = []
    for sd in SIDES:
        w = d.xpos[s.model.body(f"{sd}_wrist_yaw_link").id]
        th = d.xpos[s.model.body(f"{sd}_hand_thumb_2_link").id]; ix = d.xpos[s.model.body(f"{sd}_hand_index_1_link").id]
        f += [*w, w[2] - TABLE_Z, np.linalg.norm(th - ix)]
    return f + list(s.grasp_state())


def evaluate(variant, seed, n=50, workers=20, gifs=False):
    with Pool(workers, initializer=_init, initargs=(variant, seed)) as pool:
        res = pool.map(rollout, range(n))
        if gifs:
            for want in (True, False):
                hit = next((r["i"] for r in res if r["success"] == want), None)
                if hit is not None:
                    pool.apply(rollout, (hit, True))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["plain", "aug", "aug+cg", "aug+rlv1", "aug+rl"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("-n", type=int, default=50)
    a = ap.parse_args()
    rows, allr = [], {}
    for v in a.variants:
        ps = []
        for sd in a.seeds:
            if not (ROOT / f"checkpoints/fold/{v}_s{sd}.pt").exists():
                continue                                   # e.g. RL was trained for seed 0 only
            res = evaluate(v, sd, a.n, gifs=(sd == a.seeds[0]))
            allr[f"{v}_s{sd}"] = res
            ps.append({k: float(np.mean([r[k] for r in res])) for k in ["success", "folded", "held", "corner_dist"]})
            print(f"{v:8s} seed {sd}: success {ps[-1]['success']:.0%} folded {ps[-1]['folded']:.0%} "
                  f"corner {100 * ps[-1]['corner_dist']:.1f} cm", flush=True)
        rows.append((v, ps))
    f = lambda xs: f"{100 * np.mean(xs):.0f} ± {100 * np.std(xs):.0f}%"
    lines = [f"Cloth folding: {a.n} held-out rollouts per training seed (mean ± std over the seeds listed).", "",
             "| variant | success | folded | grasp held (frac. of transport) | final corner error | per-seed |",
             "|---|---|---|---|---|---|"]
    for v, ps in rows:
        lines.append(f"| `{v}` | **{f([p['success'] for p in ps])}** | {f([p['folded'] for p in ps])} | "
                     f"{np.mean([p['held'] for p in ps]):.2f} | {100 * np.mean([p['corner_dist'] for p in ps]):.1f} cm | "
                     f"{', '.join(str(round(a.n * p['success'])) + '/' + str(a.n) for p in ps)} |")
    (ROOT / "outputs/fold_results.md").write_text("\n".join(lines) + "\n")
    json.dump(allr, open(ROOT / "outputs/fold_results.json", "w"))
    print("\n".join(lines))
