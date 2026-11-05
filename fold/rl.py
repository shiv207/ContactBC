"""RL fine-tuning: PPO on a residual policy on top of the frozen BC policy (residual RL).

  action = BC_ensembled(obs, t) + delta,   delta ~ N(mu_theta(obs), sigma)  on the 14 arm joints only
  reward = +1 per hand on its first grasp
           - penalty for cloth vertex speed > 2.5 m/s or cloth > 30 cm above the table (no throwing)
           + 10 * (phi(s') - phi(s)) while BOTH hands hold the cloth (no reward for dragging it)
           - 1 if a hand lets go before the near corners are within 6 cm of the far corners
           - small action penalty, +3 at the end if the fold is verified
           phi = -mean xy distance between each near corner and its far corner

Grasp timing / finger closure stay with BC; RL corrects the arm motion. Rollouts run on CPU workers,
the PPO update runs on the GPU.

  python fold/rl.py --base aug --seed 0 --iters 30
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/fold"
H, ARM_DIMS, DELTA_SCALE = 330, np.r_[0:7, 14:21], 0.05
_W = {}


class ActorCritic(nn.Module):
    def __init__(self, od, ad=14, h=128):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(od, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, ad))
        self.v = nn.Sequential(nn.Linear(od, 2 * h), nn.Tanh(), nn.Linear(2 * h, 2 * h), nn.Tanh(), nn.Linear(2 * h, 1))
        self.log_std = nn.Parameter(torch.full((ad,), np.log(0.4)))     # in units of DELTA_SCALE
        nn.init.zeros_(self.mu[-1].weight); nn.init.zeros_(self.mu[-1].bias)  # start = pure BC
        for n in ["o_mu", "o_sd"]:
            self.register_buffer(n, torch.zeros(od))

    def norm(self, o):
        return (o - self.o_mu) / self.o_sd

    def dist(self, o):
        return torch.distributions.Normal(self.mu(self.norm(o)), self.log_std.exp())


def potential(s):
    c = s.corners()
    return -0.5 * (np.linalg.norm(c[0, :2] - c[2, :2]) + np.linalg.norm(c[1, :2] - c[3, :2]))


def _init(base, seed):
    torch.set_num_threads(1)
    from fold.scene import FoldScene, make_ik
    from fold.bc import load_policy
    _W["s"] = FoldScene(); _W["ik"] = make_ik(_W["s"]); _W["bc"] = load_policy(base, seed)


def _episode(args):
    """One stochastic rollout with the given actor weights. Returns transitions + episode stats."""
    sd, i, it, deterministic = args
    from fold.bc import Ensembler, with_phase
    from fold.augment import sample_pose
    from fold.replay import obs_vec, home_q, label_state, calm_check, MAX_LIFT, MAX_SPEED
    from scene import TABLE_Z
    s, ik = _W["s"], _W["ik"]
    ac = _W.get("ac")
    if ac is None or _W.get("ac_it") != it:
        ac = ActorCritic(sd["o_mu"].numel()); ac.load_state_dict(sd); ac.eval(); _W["ac"], _W["ac_it"] = ac, it
    rng = np.random.default_rng(10_000 * it + i)
    xy, yaw = sample_pose(rng)
    s.reset(xy, yaw, arm_q=home_q(s, ik))
    ens = Ensembler(_W["bc"])
    O, Aa, LP, R, G = [], [], [], [], []
    phi = potential(s)
    grasped_once = np.zeros(2, bool); prev_g = np.zeros(2)
    v_prev = s.verts(); top, low, spd = [], [], []
    for t in range(H):
        o = obs_vec(s); op = with_phase(o, t)
        base = ens.act(o, t)
        with torch.no_grad():
            d = ac.dist(torch.as_tensor(op, dtype=torch.float32)[None])
            u = d.mean if deterministic else d.sample()
            lp = d.log_prob(u).sum().item()
        u = u[0].numpy()
        a = base.copy(); a[ARM_DIMS] += DELTA_SCALE * u
        s.step(s.clip_action(a))
        phi2 = potential(s); g = s.grasp_state()
        r = -1e-3 * float((u ** 2).sum())
        if g.min() > 0:                                   # fold progress only counts while both hands hold
            r += 10 * (phi2 - phi)
        new_grasp = (g > 0) & ~grasped_once
        r += 1.0 * new_grasp.sum(); grasped_once |= g > 0
        if ((prev_g > 0) & (g == 0)).any() and phi2 < -0.06:
            r -= 1.0                                      # let go before the near corners reached the far ones
        v = s.verts(); sp = np.linalg.norm(v - v_prev, axis=1).max() * 30; v_prev = v
        top.append(v[:, 2].max()); low.append(v[:, 2].min()); spd.append(sp)
        r -= 0.5 * max(0.0, sp - MAX_SPEED) + 5.0 * max(0.0, v[:, 2].max() - TABLE_Z - MAX_LIFT)   # no throwing
        R.append(r); phi = phi2; prev_g = g
        O.append(op); Aa.append(u); LP.append(lp); G.append(s.grasp_state())
    G = np.array(G); held = np.flatnonzero(G.max(1) > 0)
    tr = slice(held[0], held[-1] + 1) if len(held) else slice(0, 0)
    lab = label_state(s, G, tr)
    ok = bool(calm_check(dict(success=lab["success"], cloth_top=np.array(top), cloth_low=np.array(low),
                              cloth_speed=np.array(spd)))["success"]) and len(held) > 0
    R[-1] += 3.0 * ok
    return dict(obs=np.array(O, np.float32), act=np.array(Aa, np.float32), logp=np.array(LP, np.float32),
                rew=np.array(R, np.float32), success=ok, corner=float(lab["corner_dist"]))


def gae(rew, val, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rew); last = 0.0
    for t in reversed(range(len(rew))):
        nv = val[t + 1] if t + 1 < len(val) else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last; adv[t] = last
    return adv, adv + val


def train(base="aug", seed=0, iters=30, eps_per_iter=40, workers=20, device="cuda"):
    torch.manual_seed(seed)
    from fold.bc import load as load_data, VARIANTS
    O, _, _ = load_data(VARIANTS[base][0])
    ac = ActorCritic(O.shape[1]).to(device)
    ac.o_mu.copy_(torch.tensor(O.mean(0))); ac.o_sd.copy_(torch.tensor(O.std(0) + 1e-3))
    opt = torch.optim.Adam(ac.parameters(), lr=1e-4)
    curve = []
    with Pool(workers, initializer=_init, initargs=(base, seed)) as pool:
        for it in range(iters):
            t0 = time.time()
            sd = {k: v.detach().cpu() for k, v in ac.state_dict().items()}
            eps = pool.map(_episode, [(sd, i, it, False) for i in range(eps_per_iter)])
            obs = torch.tensor(np.concatenate([e["obs"] for e in eps]), device=device)
            with torch.no_grad():
                val = ac.v(ac.norm(obs))[:, 0].cpu().numpy()
            advs, rets, k = [], [], 0
            for e in eps:
                n = len(e["rew"]); a_, r_ = gae(e["rew"], val[k:k + n]); advs.append(a_); rets.append(r_); k += n
            adv = torch.tensor(np.concatenate(advs), device=device); ret = torch.tensor(np.concatenate(rets), device=device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            act = torch.tensor(np.concatenate([e["act"] for e in eps]), device=device)
            lp0 = torch.tensor(np.concatenate([e["logp"] for e in eps]), device=device)
            for _ in range(5):
                perm = torch.randperm(len(obs), device=device)
                for j in range(0, len(obs), 2048):
                    b = perm[j:j + 2048]
                    d = ac.dist(obs[b]); lp = d.log_prob(act[b]).sum(1)
                    ratio = (lp - lp0[b]).exp()
                    pg = -torch.min(ratio * adv[b], ratio.clamp(0.8, 1.2) * adv[b]).mean()
                    vl = ((ac.v(ac.norm(obs[b]))[:, 0] - ret[b]) ** 2).mean()
                    loss = pg + 0.5 * vl - 1e-3 * d.entropy().sum(1).mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
            sr = float(np.mean([e["success"] for e in eps])); rr = float(np.mean([e["rew"].sum() for e in eps]))
            curve.append(dict(iter=it, success=sr, ret=rr, corner=float(np.mean([e["corner"] for e in eps])),
                              std=float(ac.log_std.exp().mean())))
            print(f"[rl {base} s{seed}] iter {it:2d} success {sr:.0%} return {rr:.2f} "
                  f"corner {100 * curve[-1]['corner']:.1f} cm  std {curve[-1]['std']:.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(ac=ac.cpu().state_dict(), base=base), CKPT / f"{base}+rl_s{seed}.pt")
    ac.to(device)
    json.dump(curve, open(ROOT / f"outputs/fold_rl_curve_s{seed}.json", "w"), indent=1)


class RLPolicy:
    def __init__(self, ac, bc):
        self.ac, self.bc = ac, bc

    def actor(self, deterministic=True):
        from fold.bc import Ensembler, with_phase
        ens = Ensembler(self.bc)

        def act(o, t):
            a = ens.act(o, t)
            with torch.no_grad():
                d = self.ac.dist(torch.as_tensor(with_phase(o, t), dtype=torch.float32)[None])
                u = (d.mean if deterministic else d.sample())[0].numpy()
            a = a.copy(); a[ARM_DIMS] += DELTA_SCALE * u
            return a
        return act


def load_rl(variant, seed=0):
    from fold.bc import load_policy
    ck = torch.load(CKPT / f"{variant}_s{seed}.pt", map_location="cpu")
    ac = ActorCritic(ck["ac"]["o_mu"].numel()); ac.load_state_dict(ck["ac"]); ac.eval()
    return RLPolicy(ac, load_policy(ck["base"], seed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="aug")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--eps", type=int, default=40)
    a = ap.parse_args()
    for s in a.seeds:
        train(a.base, s, a.iters, a.eps)
