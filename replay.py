"""Step 4: replay retargeted human trajectories on the G1, log fingertip contacts, label success, save GIFs.

Success = cup ends upright on the coaster (centre within 4 cm), resting (not held), AND the grasp was held
during transport (cup lifted >= 2 cm while >= 2 finger links touched it for >= 80% of transport steps).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json
from pathlib import Path
import numpy as np
from scene import Scene, Recorder, TABLE_Z, CUP_HH, COASTER_HH, CUP_XY, COASTER_XY
from ik import ArmIK
from retarget import build_plans, HOME, DWELL

ROOT = Path(__file__).resolve().parent
DEMOS = ROOT / "data/demos"
R0 = np.eye(3)                     # side-grasp wrist orientation (G1 zero pose: fingers forward, palm sideways)
SETTLE = 15
GRASP_Z = TABLE_Z + CUP_HH
REST_Z = TABLE_Z + 2 * COASTER_HH + CUP_HH


def obs_vec(s):
    """Policy observation: 14 robot joints + cup pos (3) + cup up-axis (3) + coaster xy (2) + tip touch (3)."""
    return np.concatenate([s.robot_q(), s.cup_pos(), s.cup_up(), s.coaster_xy(), np.tanh(s.touch() / 5.0)])


_HOME_Q = {}


def home_q(s, ik):
    """IK for the home pose from the zero configuration (cached: keeps episodes independent)."""
    if id(s) not in _HOME_Q:
        _HOME_Q[id(s)] = ik.solve(HOME, R0, q0=np.zeros(7), iters=200)[0]
    return _HOME_Q[id(s)]


def place_point(coaster_xy, off):
    return np.r_[coaster_xy - off[:2], REST_Z - off[2] + 0.005]


def run_episode(s, ik, plan, cup_xy=CUP_XY, coaster_xy=COASTER_XY, rec=None):
    s.reset(cup_xy, coaster_xy, arm_q=home_q(s, ik))
    grasp_pt = np.r_[cup_xy, GRASP_Z]
    off = np.array([-0.016, -0.009, -0.006])                 # nominal in-hand offset; re-measured online
    targets = plan.wrist_targets(grasp_pt, place_point(coaster_xy, off), HOME)
    q = s.data.qpos[s.arm_q].copy()
    log = {k: [] for k in ["obs", "action", "contact", "touch", "cup"]}
    replanned = False
    for k in range(plan.T + SETTLE):
        kk = min(k, plan.T - 1)
        if plan.k_grasp + 4 <= kk < plan.k_release - DWELL:     # re-aim with the measured in-hand cup offset
            new = s.cup_pos() - s.data.site("grasp").xpos
            off = new if not replanned else 0.8 * off + 0.2 * new
            targets[kk:] = plan.wrist_targets(grasp_pt, place_point(coaster_xy, off), HOME)[kk:]
            replanned = True
        q, _ = ik.solve(targets[kk], R0, q0=q, iters=40)
        a = s.clip_action(np.r_[q, plan.hand[kk]])
        log["obs"].append(obs_vec(s)); log["action"].append(a)
        s.step(a)
        log["contact"].append(s.cup_contacts()); log["touch"].append(s.touch()); log["cup"].append(s.cup_pos())
        if rec is not None:
            rec.grab()
    ep = {k: np.array(v) for k, v in log.items()}
    ep.update(label(s, plan, ep))
    ep.update(cup_xy=np.asarray(cup_xy), coaster_xy=np.asarray(coaster_xy), take=plan.name,
              k_grasp=plan.k_grasp, k_release=plan.k_release)
    return ep


def label(s, plan, ep):
    c, cup = ep["contact"], ep["cup"]
    tr = slice(plan.k_grasp, plan.k_release - DWELL)
    n_tips = (c[tr, :3] > 0.05).sum(1)
    held_frac = float((n_tips >= 2).mean()) if len(n_tips) else 0.0
    lift = float(cup[tr, 2].max() - (TABLE_Z + CUP_HH)) if len(n_tips) else 0.0
    d = float(np.linalg.norm(s.cup_pos()[:2] - s.coaster_xy()))
    upright = float(s.cup_up()[2])
    resting = abs(s.cup_pos()[2] - REST_Z) < 0.01 and c[-5:, 3].max() < 0.05
    grasp_held = held_frac >= 0.8 and lift >= 0.02
    placed = d < 0.04 and upright > 0.95 and resting
    return dict(success=bool(grasp_held and placed), grasp_held=bool(grasp_held), placed=bool(placed),
                held_frac=held_frac, lift=lift, dist=d, upright=upright)


def contact_mask(ep, w=6):
    """Frames within w steps of a contact-state change (grasp onset / release) -> 1, else 0."""
    on = (ep["contact"][:, :3] > 0.05).sum(1) >= 2
    ch = np.flatnonzero(np.diff(on.astype(int)) != 0)
    m = np.zeros(len(on))
    for i in ch:
        m[max(0, i - w):i + w + 1] = 1
    return m


def save_episode(ep, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ep)


def summarize(ep):
    return {k: (round(float(ep[k]), 3) if not isinstance(ep[k], (bool, np.bool_)) else bool(ep[k]))
            for k in ["success", "grasp_held", "placed", "held_frac", "lift", "dist", "upright"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist-source", default="egodex", choices=["egodex", "mediapipe"])
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    s = Scene(); ik = ArmIK(s)
    results = {}
    rec = None if args.no_gif else Recorder(s)       # one renderer (EGL contexts don't like re-creation)
    for plan in build_plans(args.wrist_source):
        if rec:
            rec.frames = []
        ep = run_episode(s, ik, plan, rec=rec)
        save_episode(ep, DEMOS / f"{plan.name}.npz")
        results[plan.name] = summarize(ep)
        print(plan.name, results[plan.name])
        if rec:
            tag = "success" if ep["success"] else "fail"
            rec.save(ROOT / f"outputs/gifs/replay_{plan.name}_{tag}.gif")
    json.dump(results, open(ROOT / "outputs/replay_results.json", "w"), indent=1)
    print(f"success {sum(r['success'] for r in results.values())}/{len(results)}")
