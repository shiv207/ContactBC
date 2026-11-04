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


def _init(variant):
    import torch
    torch.set_num_threads(1)
    from scene import Scene
    from ik import ArmIK
    from train_bc import load_policy
    _W["s"] = Scene(); _W["ik"] = ArmIK(_W["s"]); _W["pol"] = load_policy(variant)


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
    C, P = [], []
    for _ in range(H):
        a = s.clip_action(pol.act(obs_vec(s)))
        s.step(a)
        C.append(s.cup_contacts()); P.append(s.cup_pos())
        if rec:
            rec.grab()
    r = label_free(s, np.array(C), np.array(P))
    if rec:
        rec.save(ROOT / f"outputs/gifs/eval_{variant.replace('+', '_')}_{i:02d}_{'success' if r['success'] else 'fail'}.gif")
        rec.r.close()
    return dict(i=i, cup=cup.round(3).tolist(), coaster=co.round(3).tolist(), **r)


def evaluate(variant, n=50, workers=10):
    with Pool(workers, initializer=_init, initargs=(variant,)) as pool:
        res = pool.map(partial(rollout, variant=variant), range(n))
    # GIFs for the first success and first failure
    with Pool(1, initializer=_init, initargs=(variant,)) as pool:
        for want in (True, False):
            hit = next((r["i"] for r in res if r["success"] == want), None)
            if hit is not None:
                pool.apply(rollout, (hit, variant, True))
    return res


if __name__ == "__main__":
    from train_bc import VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()
    table, all_res = [], {}
    for v in args.variants:
        res = evaluate(v, args.n)
        all_res[v] = res
        k = lambda key: sum(r[key] for r in res)
        table.append((v, k("success"), k("grasp_held"), k("placed"), len(res)))
        print(f"{v:10s} success {k('success')}/{len(res)}  grasp {k('grasp_held')}  placed {k('placed')}", flush=True)
    lines = ["| variant | data | contact-gated loss | success | grasp held | placed |", "|---|---|---|---|---|---|"]
    for v, s_, g, p, n in table:
        src, cg = VARIANTS[v]
        lines.append(f"| `{v}` | {'3 human replays' if src == 'demos' else '500 augmented'} | {'yes' if cg else 'no'} "
                     f"| **{s_}/{n} ({s_ / n:.0%})** | {g}/{n} | {p}/{n} |")
    (ROOT / "outputs/results.md").write_text("\n".join(lines) + "\n")
    json.dump(all_res, open(ROOT / "outputs/results.json", "w"), indent=1)
    print("\n".join(lines))
