"""Unsupervised phase discovery (no labels) to check the pipeline.

1. Fit a Gaussian HMM on HUMAN fold strokes (+-1 s context) using per-frame features averaged over both hands:
   wrist height above the table, wrist speed, pinch aperture (MediaPipe). K chosen by BIC.
2. Name the discovered states from their mean features (lifted / low+closed / low+open ...).
3. Checks:
   a. keyframe agreement: do the HMM's own boundaries into/out of the "lifted" state coincide with the
      hand-designed grasp/release keyframes used for retargeting?
   b. the canonical human phase order (most frequent collapsed state sequence).
   c. robot rollouts (BC / RL policies) decoded with the SAME human HMM: do they follow the human phase order,
      and does following it separate successful from failed rollouts?

  python fold/phases.py --variants aug aug+rl
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fold.scene import SIDES                      # noqa: E402
from fold.retarget import detect_strokes, STRETCH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed_fold"
CTX = 30


def feats(wrist, ap, time_scale=1.0):
    """wrist: {side:(T,3)}, ap: {side:(T,)} -> (T,3) [height, speed, aperture] averaged over hands."""
    h = np.mean([gaussian_filter1d(wrist[s][:, 2] - np.percentile(wrist[s][:, 2], 5), 1.5) for s in SIDES], 0)
    v = np.mean([gaussian_filter1d(np.linalg.norm(np.gradient(wrist[s], axis=0), axis=1), 1.5) * 30 * time_scale
                 for s in SIDES], 0)
    a = np.mean([gaussian_filter1d(ap[s], 1.5) for s in SIDES], 0)
    return np.stack([h, v, a], 1)


def human_sequences():
    seqs, keys = [], []
    for f in sorted(PROC.glob("*.npz")):
        d = np.load(f)
        w = {s: d[f"{s}_wrist"] for s in SIDES}
        F = feats(w, {s: d[f"{s}_aperture"] for s in SIDES})
        for a, b in detect_strokes(w):
            lo, hi = max(0, a - CTX), min(len(F), b + CTX + 1)
            seqs.append(F[lo:hi]); keys.append((a - lo, b - lo))
    return seqs, keys


def collapse(z):
    return [int(z[0])] + [int(x) for i, x in enumerate(z[1:], 1) if x != z[i - 1]]


def is_subseq(sub, seq):
    it = iter(seq)
    return all(x in it for x in sub)


def robot_sequences(variant, seed=0, n=50):
    from fold import eval as E
    with Pool(20, initializer=E._init, initargs=(variant, seed)) as pool:
        res = pool.starmap(E.rollout, [(i, False, True) for i in range(n)])
    out = []
    for r in res:
        F = np.array(r.pop("feats"))              # per hand: wrist xyz, height, thumb-index distance; grasp flags
        w = {s: F[:, 5 * k:5 * k + 3] for k, s in enumerate(SIDES)}
        ap = {s: F[:, 5 * k + 4] for k, s in enumerate(SIDES)}
        G = F[:, 10:12].max(1)
        Fr = feats(w, ap, time_scale=STRETCH)
        held = np.flatnonzero(G > 0)
        if len(held):                             # same window as the human strokes: +-1 human-second
            c = int(CTX * STRETCH)                # around the (first grasp, last release)
            Fr = Fr[max(0, held[0] - c):held[-1] + c + 1]
        out.append((Fr[::int(STRETCH)], r["success"], bool(len(held))))   # back to human frame rate
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["aug", "aug+rl"])
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    seqs, keys = human_sequences()
    X = np.concatenate(seqs); L = [len(s) for s in seqs]
    mu_h, sd_h = X.mean(0), X.std(0) + 1e-6
    Z = lambda F, mu=mu_h, sd=sd_h: (F - mu) / sd
    best = None
    for k in range(3, 7):
        m = GaussianHMM(k, covariance_type="full", n_iter=60, tol=1e-2, random_state=0).fit(Z(X), L)
        n_par = k * k + 2 * k * 3 + k * 6
        bic = -2 * m.score(Z(X), L) + n_par * np.log(len(X))
        if best is None or bic < best[0]:
            best = (bic, k, m)
    _, K, hmm = best
    means = hmm.means_ * sd_h + mu_h                    # physical units
    lifted = int(np.argmax(means[:, 0]))
    order_h = np.argsort(means[:, 0])
    names = {}
    for s in range(K):
        hgt, spd, apt = means[s]
        if s == lifted:
            names[s] = "lifted/carry"
        elif hgt < np.median(means[:, 0]) and apt <= np.median(means[:, 2]):
            names[s] = "low, pinched (grasp/set-down)"
        elif hgt < np.median(means[:, 0]):
            names[s] = "low, open (reach/release)"
        else:
            names[s] = "mid-height, moving (lift/lower)"
    print(f"HMM: K={K} states chosen by BIC on {len(seqs)} human strokes ({len(X)} frames)")
    for s in order_h:
        print(f"  state {s}: {names[s]:32s} height {100 * means[s, 0]:5.1f} cm  speed {100 * means[s, 1]:5.1f} cm/s  "
              f"aperture {means[s, 2]:.2f}")

    # (a) keyframe agreement vs chance: distance from each heuristic keyframe (grasp a, release b) to the
    #     nearest HMM boundary between a LOW state and a non-LOW state; chance = same number of boundaries
    #     placed uniformly at random in the same sequence.
    low = {s for s in range(K) if names[s].startswith("low")}
    is_low = lambda z: np.array([x in low for x in z])
    rng = np.random.default_rng(0)
    dk, dch, zs = [], [], []
    for F, (a, b) in zip(seqs, keys):
        z = hmm.predict(Z(F)); zs.append(z)
        bd = np.flatnonzero(np.diff(is_low(z).astype(int)) != 0)
        if not len(bd):
            continue
        rb = rng.integers(0, len(z), size=(200, len(bd)))
        for k in (a, b):
            dk.append(np.abs(bd - k).min()); dch.append(np.median(np.abs(rb - k).min(1)))
    print(f"(a) heuristic keyframe -> nearest HMM low/non-low boundary: median {np.median(dk):.0f} frames "
          f"({np.median(dk) / 30:.2f} s); chance {np.median(dch):.0f} frames")

    # (b) phase-order checks that use the aperture signal (never used by the stroke detector):
    #     P1 pinch-before-lift: a 'low, pinched' frame within 1 s before the first non-low frame
    #     P2 open-after-set-down: a 'low, open' frame after the last non-low frame
    pinched = {s for s in range(K) if "pinched" in names[s]}; opened = {s for s in range(K) if "open" in names[s]}

    def order_checks(z):
        nl = np.flatnonzero(~is_low(z))
        if not len(nl):
            return False, False
        p1 = any(x in pinched for x in z[max(0, nl[0] - 30):nl[0]])
        p2 = any(x in opened for x in z[nl[-1] + 1:])
        return bool(p1), bool(p2)
    hc = np.array([order_checks(z) for z in zs])
    print(f"(b) human strokes: pinch-before-lift {hc[:, 0].mean():.0%}, open-after-set-down {hc[:, 1].mean():.0%}, "
          f"both {hc.all(1).mean():.0%}")
    report = dict(K=K, states={int(s): dict(name=names[s], height_cm=float(100 * means[s, 0]),
                                             speed_cms=float(100 * means[s, 1]), aperture=float(means[s, 2]))
                               for s in range(K)},
                  keyframe_to_boundary_frames=float(np.median(dk)), keyframe_chance_frames=float(np.median(dch)),
                  human_pinch_before_lift=float(hc[:, 0].mean()), human_open_after_setdown=float(hc[:, 1].mean()),
                  human_both=float(hc.all(1).mean()), robot={})
    # (c) robot rollouts, decoded with the same human HMM
    examples = {}
    for v in args.variants:
        rs = robot_sequences(v, 0, args.n)
        Fr = np.concatenate([F for F, _, _ in rs])
        mu_r, sd_r = Fr.mean(0), Fr.std(0) + 1e-6
        rows = []
        for F, ok, grasped in rs:
            Fz = Z(F).copy()
            Fz[:, 2] = (F[:, 2] - mu_r[2]) / sd_r[2]   # aperture units differ (Dex3 vs human): z-score per source
            z = hmm.predict(Fz)
            rows.append((ok, grasped and all(order_checks(z)), np.bincount(z, minlength=K) / len(z), len(collapse(z))))
            examples.setdefault(v, []).append((z, ok))
        okm = np.array([r[0] for r in rows]); fol = np.array([r[1] for r in rows])
        frac = np.mean([r[2] for r in rows], 0)
        out = dict(n=len(rows), success=float(okm.mean()), follow_all=float(fol.mean()),
                   follow_given_success=float(fol[okm].mean()) if okm.any() else None,
                   follow_given_fail=float(fol[~okm].mean()) if (~okm).any() else None,
                   success_given_follow=float(okm[fol].mean()) if fol.any() else None,
                   success_given_not_follow=float(okm[~fol].mean()) if (~fol).any() else None,
                   n_segments=float(np.mean([r[3] for r in rows])),
                   lifted_frac_success=float(np.mean([r[2][lifted] for r in rows if r[0]])) if okm.any() else None,
                   lifted_frac_fail=float(np.mean([r[2][lifted] for r in rows if not r[0]])) if (~okm).any() else None,
                   low_frac_success=float(np.mean([sum(r[2][x] for x in low) for r in rows if r[0]])) if okm.any() else None,
                   low_frac_fail=float(np.mean([sum(r[2][x] for x in low) for r in rows if not r[0]])) if (~okm).any() else None,
                   time_fraction={names[s]: float(frac[s]) for s in range(K)})
        report["robot"][v] = out
        print(f"    {v}: time in 'lifted' state: successes {out['lifted_frac_success']:.2f} vs failures "
              f"{out['lifted_frac_fail']:.2f}; time in low states: {out['low_frac_success']:.2f} vs {out['low_frac_fail']:.2f}")
        print(f"(c) {v}: success {out['success']:.0%}; follows human order (pinch-before-lift & open-after) {out['follow_all']:.0%} "
              f"(successes {out['follow_given_success']}, failures {out['follow_given_fail']}); "
              f"P(success|follows) {out['success_given_follow']}, P(success|not) {out['success_given_not_follow']}")
    hum_frac = np.bincount(np.concatenate(zs), minlength=K) / sum(len(z) for z in zs)
    report["human_time_fraction"] = {names[s]: float(hum_frac[s]) for s in range(K)}
    json.dump(report, open(ROOT / "outputs/fold_phases.json", "w"), indent=1)

    # figure: phase timelines (human strokes vs robot rollouts)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    rows = [("human", z, True) for z in zs[:6]]
    for v in args.variants:
        ex = examples.get(v, [])
        rows += [(f"{v} ok", z, ok) for z, ok in ex if ok][:3] + [(f"{v} fail", z, ok) for z, ok in ex if not ok][:3]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(rows) + 1.2))
    for r, (lab, z, _) in enumerate(rows):
        t = np.linspace(0, 1, len(z))
        ax.scatter(t, np.full(len(z), r), c=[cmap(x) for x in z], s=12, marker="s")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("normalised time"); ax.invert_yaxis()
    for s in range(K):
        ax.scatter([], [], c=[cmap(s)], marker="s", label=names[s])
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=K)
    ax.set_title("Unsupervised HMM phases (fit on human strokes only)")
    plt.tight_layout(); plt.savefig(ROOT / "assets/fold_phases.png", dpi=100)
    print("wrote outputs/fold_phases.json, assets/fold_phases.png")
