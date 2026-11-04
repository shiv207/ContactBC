"""Step 7: closed-loop evaluation, 50 rollouts per variant on held-out random cup/coaster poses.

Success (same criteria as data generation, but without plan keyframes):
  grasp held : >= 10 steps with the cup lifted >= 2 cm while >= 2 finger links touch it
  placed     : cup centre within 4 cm of the coaster centre, upright, resting, hand not touching it
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
H = 280                     # rollout horizon (control steps, 30 Hz) ~ longest demo + settle
EVAL_SEED = 999             # poses never used for data generation (augment uses seed 0)
_W = {}


def _init(variant, seed=0):
    import torch
    torch.set_num_threads(1)
    from scene import Scene
    from ik import ArmIK
    from train_bc import load_policy
    _W["s"] = Scene(); _W["ik"] = ArmIK(_W["s"]); _W["pol"] = load_policy(variant, seed)


def label_free(s, contact, cup):
    from replay import REST_Z
    from scene import TABLE_Z, CUP_HH
    lifted_held = ((contact[:, :3] > 0.05).sum(1) >= 2) & (cup[:, 2] - (TABLE_Z + CUP_HH) >= 0.02)
    d = float(np.linalg.norm(s.cup_pos()[:2] - s.coaster_xy()))
    up = float(s.cup_up()[2])
    resting = abs(s.cup_pos()[2] - REST_Z) < 0.01 and contact[-5:, 3].max() < 0.05
    grasp = int(lifted_held.sum()) >= 10
    placed = d < 0.04 and up > 0.95 and resting
    return dict(success=bool(grasp and placed), grasp_held=bool(grasp), placed=bool(placed), dist=d, upright=up)


def rollout(i, variant, gif=False):
    from augment import sample_poses
    from replay import obs_vec, home_q
    from scene import Recorder
    s, ik, pol = _W["s"], _W["ik"], _W["pol"]
    cup, co = sample_poses(np.random.default_rng(EVAL_SEED * 1000 + i))
    s.reset(cup, co, arm_q=home_q(s, ik))
    rec = Recorder(s) if gif else None
    from train_bc import Ensembler
    ens = Ensembler(pol)
    C, P = [], []
    for t in range(H):
        a = s.clip_action(ens.act(obs_vec(s), t))
        s.step(a)
        C.append(s.cup_contacts()); P.append(s.cup_pos())
        if rec:
            rec.grab()
    r = label_free(s, np.array(C), np.array(P))
    if rec:
        rec.save(ROOT / f"outputs/gifs/eval_{variant.replace('+', '_')}_{i:02d}_{'success' if r['success'] else 'fail'}.gif")
        rec.r.close()
    return dict(i=i, cup=cup.round(3).tolist(), coaster=co.round(3).tolist(), **r)


def evaluate(variant, seed=0, n=50, workers=10, gifs=True):
    with Pool(workers, initializer=_init, initargs=(variant, seed)) as pool:
        res = pool.map(partial(rollout, variant=variant), range(n))
    if gifs:                                   # GIFs for the first success and first failure
        with Pool(1, initializer=_init, initargs=(variant, seed)) as pool:
            for want in (True, False):
                hit = next((r["i"] for r in res if r["success"] == want), None)
                if hit is not None:
                    pool.apply(rollout, (hit, variant, True))
    return res


if __name__ == "__main__":
    from train_bc import VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()
    rows, all_res = [], {}
    for v in args.variants:
        per_seed = []
        for sd in args.seeds:
            res = evaluate(v, sd, args.n, gifs=(sd == args.seeds[0]))
            all_res[f"{v}_s{sd}"] = res
            per_seed.append({k: sum(r[k] for r in res) / len(res) for k in ["success", "grasp_held", "placed"]})
            print(f"{v:10s} seed {sd}: success {per_seed[-1]['success']:.0%}  grasp {per_seed[-1]['grasp_held']:.0%}"
                  f"  placed {per_seed[-1]['placed']:.0%}", flush=True)
        rows.append((v, per_seed))
    f = lambda xs: f"{100 * np.mean(xs):.0f} ± {100 * np.std(xs):.0f}%"
    lines = [f"Success over {args.n} held-out rollouts per seed, mean ± std over {len(args.seeds)} training seeds.", "",
             "| variant | training data | contact-gated loss | success | grasp held | placed | per-seed success |",
             "|---|---|---|---|---|---|---|"]
    for v, ps in rows:
        src, cg = VARIANTS[v]
        lines.append(f"| `{v}` | {'3 direct human replays' if src == 'demos' else '500 augmented episodes'} "
                     f"| {({'all': 'yes (all dims)', 'hand': 'yes (finger dims)'}).get(cg, 'no')} | **{f([p['success'] for p in ps])}** | {f([p['grasp_held'] for p in ps])} "
                     f"| {f([p['placed'] for p in ps])} | {', '.join(str(round(args.n * p['success'])) + '/' + str(args.n) for p in ps)} |")
    (ROOT / "outputs/results.md").write_text("\n".join(lines) + "\n")
    json.dump(all_res, open(ROOT / "outputs/results.json", "w"), indent=1)
    print("\n".join(lines))
