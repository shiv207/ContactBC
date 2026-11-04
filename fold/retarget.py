"""Human bimanual fold strokes -> G1 dual-arm + Dex3 joint-target plans.

1. Stroke detection (per take, ARKit wrists, z-up): a fold stroke = both wrists lifted > 3 cm above their
   resting height together for >= 0.3 s; it starts/ends when the first hand is back at the cloth. Kept if
   peak lift is 3-35 cm and the hands' midpoint moves >= 5 cm.
2. Canonical frame: the stroke is rotated so the midpoint displacement points +x (away from the robot,
   = the fold direction); if the human's left hand ends up on the -y side, the stroke is mirrored in y.
3. Object-centric warp (as in v1 / MimicGen): each hand's path is anchored at its near cloth corner at grasp and
   at the matching far corner at release, keeping the human's own lift profile and path shape (plus a minimum
   clearance so the cloth flips over). Canonical reach before, finger-axis-free retreat after.
4. Fingers: MediaPipe finger angles -> Dex3 (v1 map, mirrored for the left hand), locked to the pinch posture
   while grasping.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget import human_finger_map, smooth   # noqa: E402
from fold.scene import CLOSE, SIDES             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed_fold"
STRETCH, DWELL, N_REACH, N_RETREAT = 2.0, 10, 40, 30
HOVER = 0.03                    # grasp-site height above the cloth when grasping / releasing
CLEAR = 0.10                    # min mid-transport lift (the near half must flip over the far half)
HOME = {"left": np.array([0.22, 0.30, 1.0]), "right": np.array([0.22, -0.30, 1.0])}


def detect_strokes(w, up=0.03, rest=0.012):
    """w: {side: (T,3)} wrist paths. Returns [(t_start, t_end)]. Start/end are extended out to the moment
    the first hand is back near the cloth (both hands are on the cloth at grasp and at release)."""
    h = {s: gaussian_filter1d(w[s][:, 2] - np.percentile(w[s][:, 2], 5), 2) for s in SIDES}
    both = (h["left"] > up) & (h["right"] > up)
    lab, n = label(both)
    out = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(lab == k)
        if len(idx) < 9:
            continue
        a, b = idx[0], idx[-1]
        while a > 0 and h["left"][a] > rest and h["right"][a] > rest:
            a -= 1
        while b < len(both) - 1 and h["left"][b] > rest and h["right"][b] > rest:
            b += 1
        mid = 0.5 * (w["left"] + w["right"])
        peak = min(h["left"][a:b + 1].max(), h["right"][a:b + 1].max())
        disp = np.linalg.norm(mid[b, :2] - mid[a, :2])
        if 0.03 <= peak <= 0.35 and disp >= 0.05 and 15 <= b - a <= 200 and (not out or a > out[-1][1]):
            out.append((int(a), int(b)))
    return out


def canonical(w, a, b):
    """Rotate (about z) so the midpoint displacement a->b is +x; mirror y if needed so left hand is at +y."""
    mid = 0.5 * (w["left"] + w["right"])
    d = mid[b, :2] - mid[a, :2]
    yaw = np.arctan2(d[1], d[0])
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    out = {s: (w[s] - mid[a]) @ R.T for s in SIDES}
    mirrored = out["left"][a, 1] < out["right"][a, 1]
    if mirrored:
        out = {"left": out["right"] * [1, -1, 1], "right": out["left"] * [1, -1, 1]}
    return out, mirrored


def resample(x, n):
    t0, t1 = np.linspace(0, 1, len(x)), np.linspace(0, 1, n)
    return np.stack([np.interp(t1, t0, x[:, k]) for k in range(x.shape[1])], 1)


class FoldPlan:
    def __init__(self, take, a, b, d):
        self.name = f"{take}_{a:04d}"
        w = {s: gaussian_filter1d(d[f"{s}_wrist"], 1.0, axis=0) for s in SIDES}
        cw, self.mirrored = canonical(w, a, b)
        pre, post = 0, min(len(w["left"]) - 1, b + 15)
        n = lambda k: int(round(k * STRETCH))
        seg = slice(a, post + 1)
        self.h = {s: resample(cw[s][seg], n(post - a + 1)) for s in SIDES}
        ang = {s: d[f"{s}_finger_angles"] for s in SIDES}
        if self.mirrored:
            ang = {"left": ang["right"], "right": ang["left"]}
        self.tr = n(b - a)                                     # release (relative to stroke start)
        Th = len(self.h["left"])
        # control-step -> human-frame index: reach + grasp dwell at frame 0, transport, release dwell, retreat
        self.idx = np.r_[[0] * (N_REACH + DWELL), np.arange(0, self.tr), [self.tr] * DWELL,
                         np.arange(self.tr, Th)]
        self.T = len(self.idx)
        self.k_grasp = N_REACH + DWELL
        self.k_release = self.k_grasp + self.tr + DWELL
        closed = np.zeros(self.T); ramp = np.linspace(0, 1, DWELL)
        closed[N_REACH:N_REACH + DWELL] = ramp
        closed[self.k_grasp:self.k_release - DWELL] = 1
        closed[self.k_release - DWELL:self.k_release] = 1 - ramp
        self.closed = closed
        self.hand = {}
        for s in SIDES:
            hm = resample(human_finger_map(ang[s]), Th)[self.idx]
            hm = hm if s == "right" else -hm                   # left Dex3 is the mirror image
            hm = np.clip(hm * np.sign(CLOSE[s] + 1e-9), 0, None) * np.sign(CLOSE[s] + 1e-9)  # no anti-closing
            lock = CLOSE[s]
            hand = (1 - closed[:, None]) * np.zeros(7) + closed[:, None] * lock
            hand[:N_REACH] = 0                                 # open while reaching
            self.hand[s] = hand
        self.lift = float(min(self.h[s][:self.tr + 1, 2].max() - self.h[s][0, 2] for s in SIDES))
        self.disp = float(np.linalg.norm(0.5 * (self.h["left"][self.tr] + self.h["right"][self.tr])[:2]))

    def wrist_targets(self, grasp, place):
        """grasp/place: {side: xyz}. Returns {side: (T,3)} grasp-site targets."""
        out = {}
        for s in SIDES:
            h, tr = self.h[s], self.tr
            p = np.zeros_like(h)
            al = (np.arange(tr + 1) / max(1, tr))[:, None]
            p[:tr + 1] = grasp[s] + (h[:tr + 1] - h[0]) + al * ((place[s] - grasp[s]) - (h[tr] - h[0]))
            line = grasp[s][2] + al[:, 0] * (place[s][2] - grasp[s][2])
            clear = CLEAR * np.minimum(1, np.minimum(4 * al[:, 0], 4 * (1 - al[:, 0])))
            p[:tr + 1, 2] = np.maximum(p[:tr + 1, 2], line + clear)
            n = len(h) - tr                                    # retreat: straight up 8 cm, then human
            r1 = smooth(np.clip(np.arange(n) / 10, 0, 1))[:, None]
            r2 = smooth(np.clip(np.arange(n) / 10 - 1, 0, 1))[:, None]
            p[tr:] = place[s] + r1 * [0, 0, 0.08] + r2 * (h[tr:] - h[tr]) * [1, 1, 0]
            o = p[self.idx]
            pre = grasp[s] + [0, 0, 0.08]
            n1 = N_REACH // 2
            o[:n1] = HOME[s] + smooth(np.linspace(0, 1, n1))[:, None] * (pre - HOME[s])
            o[n1:N_REACH] = pre + smooth(np.linspace(0, 1, N_REACH - n1))[:, None] * (grasp[s] - pre)
            o[N_REACH:N_REACH + DWELL] = grasp[s]
            out[s] = o
        return out


def build_plans(max_takes=None):
    plans = []
    for f in sorted(PROC.glob("*.npz"))[:max_takes]:
        d = np.load(f)
        w = {s: d[f"{s}_wrist"] for s in SIDES}
        for a, b in detect_strokes(w):
            plans.append(FoldPlan(f.stem, a, b, d))
    return plans


if __name__ == "__main__":
    plans = build_plans()
    takes = {p.name.split("_")[0] for p in plans}
    print(f"{len(plans)} fold strokes from {len(takes)} takes")
    L = np.array([p.lift for p in plans]); D = np.array([p.disp for p in plans]); T = np.array([p.tr for p in plans])
    print(f"lift cm: median {100 * np.median(L):.0f} (range {100 * L.min():.0f}-{100 * L.max():.0f}); "
          f"displacement cm: median {100 * np.median(D):.0f}; transport steps: median {np.median(T):.0f}; "
          f"mirrored {np.mean([p.mirrored for p in plans]):.0%}")
