"""Behaviour cloning for bimanual folding: obs (28 joints + 4 cloth corners + 2 grasp flags + phase) ->
chunk of K=10 future 28-D joint-target deltas, ACT-style temporal ensembling at test time.

Variants: plain = direct human replays only; aug = + MimicGen-style augmented episodes;
aug+cg = aug + contact-gated loss (frames within +-6 steps of a grasp/release event weighted 8x, finger dims).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/fold"
K, T_NORM, CG_W, CG_WIN, NQ = 10, 300.0, 8.0, 6, 28
VARIANTS = {"plain": ("demos", False), "aug": ("aug", False), "aug+cg": ("aug", True)}
HAND_DIMS = np.r_[7:14, 21:28]


def with_phase(obs, t):
    return np.concatenate([obs, [min(t / T_NORM, 1.2)]])


def event_mask(grasp, w=CG_WIN):
    m = np.zeros(len(grasp))
    for h in range(grasp.shape[1]):
        for i in np.flatnonzero(np.diff(grasp[:, h]) != 0):
            m[max(0, i - w):i + w + 1] = 1
    return m


def load(source):
    dirs = [ROOT / "data/fold_demos"] + ([ROOT / "data/fold_augmented"] if source == "aug" else [])
    O, A, M = [], [], []
    for d in dirs:
        for f in sorted(d.glob("*.npz")):
            e = np.load(f)
            if not bool(e["success"]):
                continue
            obs, act, T = e["obs"], e["action"], len(e["obs"])
            idx = np.minimum(np.arange(T)[:, None] + np.arange(K)[None], T - 1)
            O.append(np.stack([with_phase(o, t) for t, o in enumerate(obs)]))
            A.append((act[idx] - obs[:, None, :NQ]).reshape(T, -1))
            M.append(event_mask(e["grasp"]))
    print(f"{source}: {len(O)} episodes, {sum(len(o) for o in O)} frames")
    return np.concatenate(O), np.concatenate(A), np.concatenate(M)


class Policy(nn.Module):
    def __init__(self, od, ad, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(od, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
        for n in ["o_mu", "o_sd", "a_mu", "a_sd"]:
            self.register_buffer(n, torch.zeros(od if n[0] == "o" else ad))

    def forward(self, o):
        return self.net((o - self.o_mu) / self.o_sd)

    @torch.no_grad()
    def chunk(self, o):
        o = torch.as_tensor(o, dtype=torch.float32)[None]
        d = (self(o)[0] * self.a_sd + self.a_mu).numpy().reshape(K, -1)
        return o[0, :NQ].numpy() + d


class Ensembler:
    def __init__(self, pol, m=0.1):
        self.pol, self.m, self.buf = pol, m, []

    def act(self, obs, t):
        self.buf.append((t, self.pol.chunk(with_phase(obs, t))))
        self.buf = [(t0, c) for t0, c in self.buf if t - t0 < K]
        p = np.stack([c[t - t0] for t0, c in self.buf]); w = np.exp(-self.m * np.arange(len(p)))
        return (w[:, None] * p).sum(0) / w.sum()


def train(variant, seed=0, steps=30000, bs=1024, lr=1e-3, device="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    src, cg = VARIANTS[variant]
    O, A, M = load(src)
    pol = Policy(O.shape[1], A.shape[1]).to(device)
    pol.o_mu.copy_(torch.tensor(O.mean(0))); pol.o_sd.copy_(torch.tensor(O.std(0) + 1e-3))
    pol.a_mu.copy_(torch.tensor(A.mean(0))); pol.a_sd.copy_(torch.tensor(A.std(0) + 1e-3))
    W = 1 + (CG_W - 1) * M if cg else np.ones(len(M))
    O, A, W = (torch.tensor(x, dtype=torch.float32, device=device) for x in (O, A, W))
    An = (A - pol.a_mu) / pol.a_sd
    hand = torch.zeros(NQ, dtype=torch.bool, device=device); hand[HAND_DIMS] = True
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        i = torch.randint(0, len(O), (bs,), device=device)
        o = O[i] + 0.01 * torch.randn_like(O[i]) * pol.o_sd
        err = ((pol(o) - An[i]) ** 2).view(bs, K, NQ)
        if cg:   # gate the finger dims near grasp/release events; arm dims uniform
            loss = 0.5 * err[..., ~hand].mean() + 0.5 * (W[i, None, None] * err[..., hand]).mean() / W[i].mean()
        else:
            loss = err.mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 10000 == 0 or it == steps - 1:
            print(f"  [{variant} s{seed}] step {it} loss {loss.item():.4f}", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(pol.cpu().state_dict(), CKPT / f"{variant}_s{seed}.pt")


def load_policy(variant, seed=0):
    sd = torch.load(CKPT / f"{variant}_s{seed}.pt", map_location="cpu")
    p = Policy(sd["o_mu"].numel(), sd["a_mu"].numel()); p.load_state_dict(sd); p.eval()
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=30000)
    a = ap.parse_args()
    for s in a.seeds:
        for v in a.variants:
            train(v, s, a.steps)
