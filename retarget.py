"""Step 2: human wrist trajectory + finger angles -> G1 right arm + Dex3 hand joint targets.

Human -> robot mapping
  * wrist translation: the human wrist path (yaw-aligned to the wearer's heading at t=0, time-stretched
    to robot speed) is split at the human grasp/release keyframes into approach / transport / retreat
    segments. Each segment is re-expressed relative to the object it interacts with (MimicGen-style):
    the grasp keyframe lands on the sim cup, the release keyframe lands above the coaster.
  * wrist orientation: fixed side-grasp orientation (Dex3 palm facing the cup). See README.
  * fingers: human flexion angles (MediaPipe) -> Dex3 joints with per-joint linear maps, clipped to
    joint limits. Between grasp and release the hand is locked to at least the closure needed to hold
    the cup (a 3-finger Dex3 cannot pinch a 5 cm cup the way the human pinches a teacup).
  * arm joints: damped-least-squares IK on the 'grasp' site, warm-started each frame.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data/processed"

STRETCH = 2.0            # robot executes the human motion 2x slower (Dex3 needs time to close)
DWELL = 12               # control steps to dwell at grasp / release while fingers close / open
OPEN = np.zeros(7)
CLOSE = np.array([0.0, -0.8, -1.2, 1.2, 1.2, 1.2, 1.2])       # thumb_0..2, index_0..1, middle_0..1
RELEASE = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])        # open + thumb swung away from the cup
HOME = np.array([0.22, -0.32, 1.0])                            # grasp-site home position
N_REACH = 40             # control steps for the canonical reach
BACKOFF_FRAMES = 16      # human frames (post-stretch) spent backing off after release
GRIP_MAX = 1.0           # cap on closure beyond CLOSE (over-squeezing flings the cup on release)
CLEAR = 0.04             # min transport clearance (m) at mid-transport


def keyframes(wrist):
    """EgoDex clips start at hand-object contact: grasp = motion onset (speed > 25% of max);
    release = the speed valley in the 50-92% window (the hand slows to set the object down)."""
    T = len(wrist)
    speed = gaussian_filter1d(np.linalg.norm(np.gradient(wrist, axis=0), axis=1), 1.5)
    t_g = int(np.argmax(speed > 0.25 * speed.max()))
    a, b = max(t_g + 5, int(0.5 * T)), int(0.92 * T)
    t_r = a + int(np.argmin(speed[a:b]))
    return t_g, t_r


def human_finger_map(ang):
    """(T,15) human flexion angles -> (T,7) Dex3 targets. Baseline = most-open frame of the take."""
    a = ang - np.percentile(ang, 5, axis=0)
    q = np.zeros((len(a), 7))
    q[:, 0] = 0.8 * a[:, 2]                    # thumb_0  <- thumb opposition
    q[:, 1] = -1.2 * a[:, 0]                   # thumb_1  <- thumb MCP flex (closes negative)
    q[:, 2] = -1.5 * a[:, 1]                   # thumb_2  <- thumb IP flex
    q[:, 3] = 1.5 * a[:, 3]                    # index_0  <- index MCP
    q[:, 4] = 1.0 * (a[:, 4] + a[:, 5])        # index_1  <- index PIP+DIP
    q[:, 5] = 1.5 * a[:, 6]                    # middle_0 <- middle MCP
    q[:, 6] = 1.0 * (a[:, 7] + a[:, 8])        # middle_1 <- middle PIP+DIP
    return q


def lock_closed(q, closed):
    """Where closed==1, push each joint at least to CLOSE in its closing direction."""
    sign = np.sign(CLOSE)
    lock = sign * np.clip(sign * q, np.abs(CLOSE), np.abs(CLOSE) * GRIP_MAX)
    return np.where(closed[:, None] > 0, (1 - closed[:, None]) * q + closed[:, None] * lock, q)


def load_take(name, wrist_source="egodex"):
    d = np.load(PROC / f"{name}.npz")
    w = d["wrist_pos_gt"] if wrist_source == "egodex" else d["wrist_pos"]
    fwd = d["head_fwd"][0, :2]; yaw = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    w = gaussian_filter1d(w @ Rz.T, 1.0, axis=0)          # heading -> +x
    return dict(name=name, wrist=w, ang=d["finger_angles"], ang_gt=d["finger_angles_gt"])


def resample(x, n):
    t0 = np.linspace(0, 1, len(x)); t1 = np.linspace(0, 1, n)
    return np.stack([np.interp(t1, t0, x[:, k]) for k in range(x.shape[1])], 1)


class Plan:
    """Time-indexed task-space plan for one human take, re-targetable to any cup/coaster pose."""

    def __init__(self, take):
        w, ang = take["wrist"], take["ang"]
        t_g, t_r = keyframes(w)
        n = lambda k: int(round(k * STRETCH))
        Tn = n(len(w))
        self.h = resample(w, Tn)                                     # human wrist (time-stretched)
        self.hand_h = resample(human_finger_map(ang), Tn)
        self.tg, self.tr = n(t_g), n(t_r)
        # insert dwell at grasp and release (fingers close / open while wrist holds still)
        # reach: the clips have none, so prepend a canonical approach (home -> above cup -> descend)
        idx = np.r_[[self.tg] * N_REACH, [self.tg] * DWELL, np.arange(self.tg, self.tr), [self.tr] * DWELL,
                    np.arange(self.tr, Tn)]
        self.idx = idx                                               # control step -> human frame
        self.n_reach = N_REACH
        self.k_grasp = N_REACH + DWELL                               # step at which grasp is complete
        self.k_release = N_REACH + DWELL + (self.tr - self.tg) + DWELL                         # step at which release is complete
        self.T = len(idx)
        closed = np.zeros(self.T)
        ramp = np.linspace(0, 1, DWELL)
        g, r = N_REACH, self.k_release - DWELL                       # grasp / release start steps
        closed[g:g + DWELL] = ramp
        closed[g + DWELL:r] = 1
        closed[r:r + DWELL] = 1 - ramp
        self.closed = closed
        hand = lock_closed(self.hand_h[idx], closed)           # clipped to joint limits at execution
        # the clips start at contact, so the human "preshape" is already a grasp: keep Dex3 open while
        # reaching and while backing off after release
        hand[:N_REACH] = OPEN
        hand[r:r + DWELL] = (1 - ramp)[:, None] * lock_closed(self.hand_h[idx], np.ones(self.T))[r:r + DWELL] + ramp[:, None] * RELEASE
        hand[r + DWELL:] = RELEASE     # after release the human hand relaxes; Dex3 stays open (no re-curl near cup)
        self.hand = hand
        self.name = take["name"]

    def wrist_targets(self, grasp_pt, place_pt, start_pt=HOME):
        """Warp the human wrist path so grasp keyframe -> grasp_pt and release keyframe -> place_pt."""
        h, tg, tr = self.h, self.tg, self.tr
        p = np.zeros_like(h)
        # transport: human path relative to the cup, linearly warped to end at the coaster
        al = ((np.arange(tg, tr + 1) - tg) / max(1, tr - tg))[:, None]
        p[tg:tr + 1] = grasp_pt + (h[tg:tr + 1] - h[tg]) + al * ((place_pt - grasp_pt) - (h[tr] - h[tg]))
        # clearance above the straight line: some takes barely lift, and the cup must come down onto the coaster
        line_z = grasp_pt[2] + al[:, 0] * (place_pt[2] - grasp_pt[2])
        clear = CLEAR * np.minimum(1, np.minimum(4 * al[:, 0], 4 * (1 - al[:, 0])))   # lift, carry, descend
        p[tg:tr + 1, 2] = np.maximum(p[tg:tr + 1, 2], line_z + clear)
        # retreat: human path relative to the coaster; never below release height
        # Dex3's thumb hooks the cup if the hand leaves sideways, so first back off along the fingers (-x)
        n = len(h) - tr
        ramp = smooth(np.clip(np.arange(n) / BACKOFF_FRAMES, 0, 1))[:, None]
        ramp2 = smooth(np.clip(np.arange(n) / BACKOFF_FRAMES - 1, 0, 1))[:, None]
        p[tr:] = place_pt + ramp * np.array([-0.08, 0, 0.02]) + ramp2 * (h[tr:] - h[tr])
        p[tr:, 2] = np.maximum(p[tr:, 2], place_pt[2] + 0.02 * ramp[:, 0])
        out = p[self.idx]
        # canonical reach: home -> 10 cm above the cup -> straight descent
        pre = grasp_pt + [0, 0, 0.10]
        n1 = N_REACH // 2; s1 = smooth(np.linspace(0, 1, n1))[:, None]; s2 = smooth(np.linspace(0, 1, N_REACH - n1))[:, None]
        out[:n1] = start_pt + s1 * (pre - start_pt)
        out[n1:N_REACH] = pre + s2 * (grasp_pt - pre)
        out[N_REACH:N_REACH + DWELL] = grasp_pt
        return out


def smooth(s):
    return 3 * s ** 2 - 2 * s ** 3


def build_plans(wrist_source="egodex"):
    takes = sorted(p.stem for p in PROC.glob("ep*.npz"))
    return [Plan(load_take(t, wrist_source)) for t in takes]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist-source", default="egodex", choices=["egodex", "mediapipe"])
    args = ap.parse_args()
    for pl in build_plans(args.wrist_source):
        print(f"{pl.name}: {pl.T} control steps, grasp@{pl.tg} release@{pl.tr} "
              f"transport {np.linalg.norm(pl.h[pl.tr] - pl.h[pl.tg]):.2f} m, "
              f"peak lift {pl.h[pl.tg:pl.tr + 1, 2].max() - pl.h[pl.tg, 2]:.2f} m")
