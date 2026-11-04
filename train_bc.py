"""Step 6: MLP behaviour cloning on (robot state, cup/coaster pose, touch) -> 14-D joint targets.

Variants (ablation):
  plain       : the successful *direct* human replays only (data/demos), uniform loss
  plain+cg    : same data, contact-gated loss
  aug         : + MimicGen-style augmented episodes (data/augmented), uniform loss
  aug+cg      : ContactBC = augmented data + contact-gated loss (frames near grasp/release weighted x CG_W)
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
CG_W, CG_WIN = 8.0, 6          # contact-gated loss: weight and half-window (control steps)
VARIANTS = {"plain": ("demos", False), "plain+cg": ("demos", True), "aug": ("aug", False), "aug+cg": ("aug", True)}


def contact_mask(contact, w=CG_WIN):
    """1 on frames within w steps of a change in finger-contact state (grasp onset / release)."""
    on = (contact[:, :3] > 0.05).sum(1) >= 2
    m = np.zeros(len(on))
    for i in np.flatnonzero(np.diff(on.astype(int)) != 0):
        m[max(0, i - w):i + w + 1] = 1
    return m


def load(source):
    files = sorted((ROOT / "data/demos").glob("ep*.npz")) if source == "demos" else \
        sorted((ROOT / "data/augmented").glob("aug_*.npz"))
    O, A, M = [], [], []
    for f in files:
        e = np.load(f)
        if not bool(e["success"]):
            continue
        O.append(e["obs"]); A.append(e["action"] - e["obs"][:, :14]); M.append(contact_mask(e["contact"]))
    print(f"{source}: {len(O)} successful episodes, {sum(len(o) for o in O)} frames")
    return np.concatenate(O), np.concatenate(A), np.concatenate(M)


class Policy(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(od, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
        for n in ["o_mu", "o_sd", "a_mu", "a_sd"]:
            self.register_buffer(n, torch.zeros(od if n[0] == "o" else ad))

    def forward(self, o):                      # normalised in, normalised out
        return self.net((o - self.o_mu) / self.o_sd)

    @torch.no_grad()
    def act(self, o):
        o = torch.as_tensor(o, dtype=torch.float32, device=self.o_mu.device)[None]
        d = self(o)[0] * self.a_sd + self.a_mu
        return o[0, :14].cpu().numpy() + d.cpu().numpy()


def train(variant, steps=20000, bs=1024, lr=1e-3, seed=0, device="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    source, cg = VARIANTS[variant]
    O, A, M = load(source)
    W = 1 + (CG_W - 1) * M if cg else np.ones(len(M))
    pol = Policy(O.shape[1], A.shape[1]).to(device)
    pol.o_mu.copy_(torch.tensor(O.mean(0))); pol.o_sd.copy_(torch.tensor(O.std(0) + 1e-3))
    pol.a_mu.copy_(torch.tensor(A.mean(0))); pol.a_sd.copy_(torch.tensor(A.std(0) + 1e-3))
    O, A, W = (torch.tensor(x, dtype=torch.float32, device=device) for x in (O, A, W))
    An = (A - pol.a_mu) / pol.a_sd
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        i = torch.randint(0, len(O), (bs,), device=device)
        o = O[i] + 0.01 * torch.randn_like(O[i]) * pol.o_sd           # small state noise for robustness
        loss = (W[i] * ((pol(o) - An[i]) ** 2).mean(1)).sum() / W[i].sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 5000 == 0 or it == steps - 1:
            print(f"  [{variant}] step {it} loss {loss.item():.4f}")
    CKPT.mkdir(exist_ok=True)
    torch.save(pol.state_dict(), CKPT / f"{variant}.pt")
    return pol


def load_policy(variant, device="cpu"):
    sd = torch.load(CKPT / f"{variant}.pt", map_location=device)
    pol = Policy(sd["o_mu"].numel(), sd["a_mu"].numel()).to(device)
    pol.load_state_dict(sd); pol.eval()
    return pol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()
    for v in args.variants:
        train(v, steps=args.steps)
