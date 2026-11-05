"""Hand-relative observation features: each grasp site's position and its offset to its near corner (grasp target)
and far corner (place target), from forward kinematics on the observed joint angles.

The plain observation has only joint angles and cloth corners, so the policy has to learn the arm's kinematics to
know how far a hand is from its corner. These features give it that error directly (18 dims).
"""
import numpy as np
import mujoco

_M = {}
NQ = 28


def _model():
    if not _M:
        from fold.scene import FoldScene, SIDES
        s = FoldScene()
        _M.update(m=s.model, d=mujoco.MjData(s.model), sides=SIDES,
                  qadr=np.concatenate([np.r_[s.arm_q[sd], s.hand_q[sd]] for sd in SIDES]),
                  site=[s.site[sd] for sd in SIDES], qpos0=s.data.qpos.copy())
    return _M


def rel_feats(obs):
    """obs: (N, >=40) [28 joints, 4 corners x 3 (near-left, near-right, far-left, far-right), ...] -> (N, 18)."""
    M = _model(); m, d = M["m"], M["d"]
    obs = np.atleast_2d(obs)
    out = np.empty((len(obs), 18))
    d.qpos[:] = M["qpos0"]
    for n, o in enumerate(obs):
        d.qpos[M["qadr"]] = o[:NQ]
        mujoco.mj_kinematics(m, d)
        c = o[NQ:NQ + 12].reshape(4, 3)
        f = []
        for k, sid in enumerate(M["site"]):          # k = 0 left (corners 0, 2), k = 1 right (corners 1, 3)
            p = d.site_xpos[sid]
            f += [*p, *(c[k] - p), *(c[k + 2] - p)]
        out[n] = f
    return out
