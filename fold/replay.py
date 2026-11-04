"""Replay retargeted human fold strokes on the G1 (both arms), label success, save GIFs.

Success = both near corners end within 5 cm (xy) of their far corners, fold error < 7 cm, both hands held
the cloth for >= 90% of transport, and both released at the end.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fold.scene import FoldScene, make_ik, SIDES, R_DOWN, CLOTH_XY   # noqa: E402
from fold.retarget import build_plans, HOME, HOVER                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data/fold_demos"
SETTLE = 20
_W = {}


def obs_vec(s):
    """28 robot joints + 4 cloth corners (12) + 2 grasp flags."""
    return np.concatenate([s.robot_q(), s.corners().ravel(), s.grasp_state()])


def home_q(s, ik):
    if "home" not in _W:
        _W["home"] = {sd: ik[sd].solve(HOME[sd], R_DOWN, q0=np.zeros(7), iters=300)[0] for sd in SIDES}
    return _W["home"]


def run_episode(s, ik, plan, cloth_xy=CLOTH_XY, yaw=0.0, rec=None):
    s.reset(cloth_xy, yaw, arm_q=home_q(s, ik))
    c = s.corners()
    grasp = {"left": c[0] + [0, 0, HOVER], "right": c[1] + [0, 0, HOVER]}
    place = {"left": c[2] + [0, 0, HOVER + 0.01], "right": c[3] + [0, 0, HOVER + 0.01]}
    tg = plan.wrist_targets(grasp, place)
    q = dict(home_q(s, ik))
    log = {k: [] for k in ["obs", "action", "grasp", "fold_err"]}
    for k in range(plan.T + SETTLE):
        kk = min(k, plan.T - 1)
        if plan.k_grasp <= kk < plan.k_release - 10:      # re-aim at the far corners' current positions
            c = s.corners()
            new = {"left": c[2] + [0, 0, HOVER + 0.01], "right": c[3] + [0, 0, HOVER + 0.01]}
            place = {sd: 0.8 * place[sd] + 0.2 * new[sd] for sd in SIDES}
            tg2 = plan.wrist_targets(grasp, place)
            for sd in SIDES:
                tg[sd][kk:] = tg2[sd][kk:]
        a = []
        for sd in SIDES:
            q[sd], _ = ik[sd].solve(tg[sd][kk], R_DOWN, q0=q[sd], iters=30)
            a += [q[sd], plan.hand[sd][kk]]
        a = s.clip_action(np.concatenate(a))
        log["obs"].append(obs_vec(s)); log["action"].append(a)
        s.step(a)
        log["grasp"].append(s.grasp_state()); log["fold_err"].append(s.fold_error())
        if rec is not None:
            rec.grab()
    ep = {k: np.array(v) for k, v in log.items()}
    ep.update(label(s, plan, ep))
    ep.update(cloth_xy=np.asarray(cloth_xy), yaw=float(yaw), take=plan.name,
              k_grasp=plan.k_grasp, k_release=plan.k_release)
    return ep


def label_state(s, grasp_log, transport):
    c = s.corners()
    corner = float(max(np.linalg.norm(c[0, :2] - c[2, :2]), np.linalg.norm(c[1, :2] - c[3, :2])))
    held = float(grasp_log[transport].min(1).mean()) if len(grasp_log[transport]) else 0.0
    released = bool(grasp_log[-1].sum() == 0)
    fe = s.fold_error()
    folded = corner < 0.05 and fe < 0.07
    return dict(success=bool(folded and held >= 0.9 and released), folded=bool(folded), held=held,
                released=released, corner_dist=corner, fold_error=fe)


def label(s, plan, ep):
    return label_state(s, ep["grasp"], slice(plan.k_grasp, plan.k_release - 10))


def _init():
    _W["s"] = FoldScene(); _W["ik"] = make_ik(_W["s"]); _W["plans"] = {p.name: p for p in build_plans()}


def _replay(name, gif=False):
    from scene import Recorder
    s, ik = _W["s"], _W["ik"]
    rec = Recorder(s) if gif else None
    ep = run_episode(s, ik, _W["plans"][name], rec=rec)
    np.savez_compressed(DEMOS / f"{name}.npz", **ep)
    if rec:
        rec.save(ROOT / f"outputs/gifs/fold_replay_{name}_{'success' if ep['success'] else 'fail'}.gif")
        rec.r.close()
    return name, {k: (bool(ep[k]) if isinstance(ep[k], (bool, np.bool_)) else round(float(ep[k]), 3))
                  for k in ["success", "folded", "held", "released", "corner_dist", "fold_error"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--gifs", type=int, default=3, help="GIFs for this many successes and failures")
    args = ap.parse_args()
    DEMOS.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in build_plans()]
    with Pool(args.workers, initializer=_init) as pool:
        res = dict(pool.map(_replay, names, chunksize=2))
        ok = [n for n in names if res[n]["success"]]; bad = [n for n in names if not res[n]["success"]]
        for n in ok[:args.gifs] + bad[:args.gifs]:
            pool.apply(_replay, (n, True))
    json.dump(res, open(ROOT / "outputs/fold_replay_results.json", "w"), indent=1)
    print(f"direct replay success {len(ok)}/{len(names)}  folded {sum(r['folded'] for r in res.values())}"
          f"  held {np.mean([r['held'] for r in res.values()]):.2f}")
