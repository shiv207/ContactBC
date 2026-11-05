"""Behaviour cloning for bimanual folding: obs (28 joints + 4 cloth corners + 2 grasp flags + phase) ->
chunk of K=10 future 28-D joint-target deltas, ACT-style temporal ensembling at test time.

Variants: plain = direct human replays only; aug = + MimicGen-style augmented episodes;
aug+cg = aug + contact-gated loss (frames within +-6 steps of a grasp/release event weighted 8x, finger dims);
dart = aug + DART noise-injected episodes.
dart+sub* = FineART-style hierarchical policy (arXiv 2609.36416, fold/subtask.py): a System-2 head predicts the current
subtask (reach / descend / grasp / lift / carry / lower / release / retreat) from the observation, and a System-1
action head is conditioned on it (ground-truth subtask in training, i.e. teacher forcing; its own prediction at test
time). Knowledge insulation (KI): System 1 reads System 2's features through a stop-gradient, so the action loss
cannot reshape the subtask representation. dart+sub-ki = no KI (shared trunk). dart+sub-t = no clock input: the
subtask is the policy's only notion of task progress.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CKPT = ROOT / "checkpoints/fold"
K, T_NORM, CG_W, CG_WIN, NQ = 10, 300.0, 8.0, 6, 28
LAM_ACT, LAM_SUB = 10.0, 1.0           # FineART loss weights (lambda_flow = 10, lambda_text = 1)
VARIANTS = {"plain": ("demos", False), "aug": ("aug", False), "aug+cg": ("aug", True), "dart": ("dart", False),
            "dart+sub": ("dart", False), "dart+sub-ki": ("dart", False), "dart+sub-t": ("dart", False)}
HIER = {"dart+sub": dict(ki=True, use_time=True), "dart+sub-ki": dict(ki=False, use_time=True),
        "dart+sub-t": dict(ki=True, use_time=False)}
# dart+rel: + hand-relative features (fold/features.py). dart+rel+self: + the policy's own verified successes
# (fold/self_improve.py, RECAP-style self-improvement), and so on for later rounds.
# +q: quality filter, drop training episodes whose fold ended more than MAX_CORNER from the target.
VARIANTS.update({v: ("dart", False) for v in ["dart+rel", "dart+rel+self", "dart+rel+self2", "dart+rel+self2+q",
                                                  "dart+rel+self2+rec"]})
REL = {"dart+rel", "dart+rel+self", "dart+rel+self2", "dart+rel+self2+q", "dart+rel+self2+rec"}
EXTRA = {"dart+rel+self": ["data/fold_self_r1"], "dart+rel+self2": ["data/fold_self_r1", "data/fold_self_r2"],
         "dart+rel+self2+q": ["data/fold_self_r1", "data/fold_self_r2"],
         "dart+rel+self2+rec": ["data/fold_self_r1", "data/fold_self_r2", "data/fold_recovery"]}   # fold/recovery.py
MAX_CORNER = {"dart+rel+self2+q": 0.035}
HAND_DIMS = np.r_[7:14, 21:28]


def with_phase(obs, t):
    return np.concatenate([obs, [min(t / T_NORM, 1.2)]])


def add_rel(op):
    """[obs, phase] -> [obs, 18 hand-relative features, phase] (single frame or batch)."""
    from fold.features import rel_feats
    op = np.atleast_2d(op)
    return np.concatenate([op[:, :-1], rel_feats(op[:, :-1]), op[:, -1:]], 1)


def event_mask(grasp, w=CG_WIN):
    m = np.zeros(len(grasp))
    for h in range(grasp.shape[1]):
        for i in np.flatnonzero(np.diff(grasp[:, h]) != 0):
            m[max(0, i - w):i + w + 1] = 1
    return m


def load(source, subtasks=False, rel=False, extra=(), max_corner=None):
    dirs = [ROOT / "data/fold_demos"] + ([ROOT / "data/fold_augmented"] if source in ("aug", "dart") else []) \
        + ([ROOT / "data/fold_dart"] if source == "dart" else []) + [ROOT / x for x in extra]
    from fold.subtask import labels
    O, A, M, Y = [], [], [], []
    for d in dirs:
        for f in sorted(d.glob("*.npz")):
            e = np.load(f)
            if not bool(e["success"]) or (max_corner and float(e["corner_dist"]) > max_corner):
                continue
            obs, act, T = e["obs"], e["action"], len(e["obs"])
            idx = np.minimum(np.arange(T)[:, None] + np.arange(K)[None], T - 1)
            op = np.stack([with_phase(o, t) for t, o in enumerate(obs)])
            O.append(add_rel(op) if rel else op)
            A.append((act[idx] - obs[:, None, :NQ]).reshape(T, -1))
            M.append(event_mask(e["grasp"]))
            Y.append(labels(T, int(e["k_grasp"]), int(e["k_release"])))
    print(f"{source}{'+' + '+'.join(extra) if extra else ''}: {len(O)} episodes, {sum(len(o) for o in O)} frames")
    out = np.concatenate(O), np.concatenate(A), np.concatenate(M)
    return out + (np.concatenate(Y),) if subtasks else out


class Policy(nn.Module):
    def __init__(self, od, ad, h=512, rel=False):
        super().__init__()
        self.rel = rel
        self.net = nn.Sequential(nn.Linear(od, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
        for n in ["o_mu", "o_sd", "a_mu", "a_sd"]:
            self.register_buffer(n, torch.zeros(od if n[0] == "o" else ad))

    def forward(self, o):
        return self.net((o - self.o_mu) / self.o_sd)

    @torch.no_grad()
    def chunk(self, o):
        o = torch.as_tensor(add_rel(o)[0] if self.rel else o, dtype=torch.float32)[None]
        d = (self(o)[0] * self.a_sd + self.a_mu).numpy().reshape(K, -1)
        return o[0, :NQ].numpy() + d


class HierPolicy(nn.Module):
    """System 2: obs -> trunk features -> subtask logits. System 1: (obs, features, subtask embedding) -> action chunk."""

    def __init__(self, od, ad, ki=True, use_time=True, h=512, e=32):
        super().__init__()
        from fold.subtask import NSUB
        self.s2 = nn.Sequential(nn.Linear(od, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.s2_head = nn.Linear(h, NSUB)
        self.emb = nn.Embedding(NSUB, e)
        self.s1 = nn.Sequential(nn.Linear(od + h + e, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
        for n in ["o_mu", "o_sd", "a_mu", "a_sd"]:
            self.register_buffer(n, torch.zeros(od if n[0] == "o" else ad))
        self.register_buffer("cfg", torch.tensor([float(ki), float(use_time)]))
        self.last_sub = -1

    def forward(self, o, sub=None):
        on = (o - self.o_mu) / self.o_sd
        if not self.cfg[1]:
            on = torch.cat([on[..., :-1], torch.zeros_like(on[..., -1:])], -1)    # hide the clock
        f = self.s2(on)
        logits = self.s2_head(f)
        if sub is None:                      # hierarchical inference: condition on System 2's own prediction
            sub = logits.argmax(-1)
        fa = f.detach() if self.cfg[0] else f
        return self.s1(torch.cat([on, fa, self.emb(sub)], -1)), logits

    @torch.no_grad()
    def chunk(self, o):
        o = torch.as_tensor(o, dtype=torch.float32)[None]
        a, logits = self(o)
        self.last_sub = int(logits.argmax(-1))
        d = (a[0] * self.a_sd + self.a_mu).numpy().reshape(K, -1)
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
    rel = variant in REL
    O, A, M, Y = load(src, subtasks=True, rel=rel, extra=EXTRA.get(variant, ()), max_corner=MAX_CORNER.get(variant))
    hier = variant in HIER
    pol = (HierPolicy(O.shape[1], A.shape[1], **HIER[variant]) if hier
           else Policy(O.shape[1], A.shape[1], rel=rel)).to(device)
    pol.o_mu.copy_(torch.tensor(O.mean(0))); pol.o_sd.copy_(torch.tensor(O.std(0) + 1e-3))
    pol.a_mu.copy_(torch.tensor(A.mean(0))); pol.a_sd.copy_(torch.tensor(A.std(0) + 1e-3))
    W = 1 + (CG_W - 1) * M if cg else np.ones(len(M))
    O, A, W = (torch.tensor(x, dtype=torch.float32, device=device) for x in (O, A, W))
    Y = torch.tensor(Y, dtype=torch.long, device=device)
    An = (A - pol.a_mu) / pol.a_sd
    hand = torch.zeros(NQ, dtype=torch.bool, device=device); hand[HAND_DIMS] = True
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        i = torch.randint(0, len(O), (bs,), device=device)
        o = O[i] + 0.01 * torch.randn_like(O[i]) * pol.o_sd
        if hier:                 # teacher forcing: System 1 sees the annotated subtask during training
            pred, logits = pol(o, Y[i])
            err = ((pred - An[i]) ** 2).view(bs, K, NQ)
            ce = nn.functional.cross_entropy(logits, Y[i])
            loss = LAM_ACT * err.mean() + LAM_SUB * ce
        else:
            err = ((pol(o) - An[i]) ** 2).view(bs, K, NQ)
            if cg:   # gate the finger dims near grasp/release events; arm dims uniform
                loss = 0.5 * err[..., ~hand].mean() + 0.5 * (W[i, None, None] * err[..., hand]).mean() / W[i].mean()
            else:
                loss = err.mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 10000 == 0 or it == steps - 1:
            extra = f" subtask acc {(logits.argmax(-1) == Y[i]).float().mean().item():.3f}" if hier else ""
            print(f"  [{variant} s{seed}] step {it} loss {loss.item():.4f}{extra}", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(pol.cpu().state_dict(), CKPT / f"{variant}_s{seed}.pt")


class SeedEnsemble:
    """Average the action chunks of several training seeds of one variant (test-time variance reduction)."""

    def __init__(self, pols):
        self.pols = pols

    def chunk(self, o):
        return np.mean([p.chunk(o) for p in self.pols], 0)


def load_policy(variant, seed=0):
    if variant.startswith("ens:"):           # ens:<variant> = seeds 0, 1, 2 of <variant>, averaged
        return SeedEnsemble([load_policy(variant[4:], sd) for sd in (0, 1, 2)])
    sd = torch.load(CKPT / f"{variant}_s{seed}.pt", map_location="cpu")
    if "cfg" in sd:
        ki, use_time = (bool(x) for x in sd["cfg"])
        p = HierPolicy(sd["o_mu"].numel(), sd["a_mu"].numel(), ki=ki, use_time=use_time)
    else:
        p = Policy(sd["o_mu"].numel(), sd["a_mu"].numel(), rel=variant in REL)
    p.load_state_dict(sd); p.eval()
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
