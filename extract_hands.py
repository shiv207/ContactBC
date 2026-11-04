"""Step 1: egocentric video -> MediaPipe 21-landmark hands -> smoothed wrist pose + finger joint angles.

For every take in data/raw/takes/*.mp4:
  * MediaPipe Hands on each frame -> 21 metric 3D "world" landmarks + 2D image landmarks (right hand)
  * wrist pose in the camera frame from solvePnP(3D world landmarks, 2D landmarks, intrinsics),
    lifted to a gravity-aligned z-up world frame with the head/camera pose
  * Savitzky-Golay smoothing, finger flexion angles + thumb-index aperture
  * the same quantities from the EgoDex ARKit skeleton (ground truth) for validation

Output: data/processed/<take>.npz
"""
import json
from pathlib import Path
import numpy as np
import cv2
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
import mediapipe as mp

ROOT = Path(__file__).resolve().parent
TAKES = ROOT / "data/raw/takes"
PARQ = ROOT / "data/raw/egodex/test/basic_pick_place/data/chunk-000"
OUT = ROOT / "data/processed"

# ARKit world is y-up; convert to z-up (x, y, z) -> (x, -z, y)
Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
FINGERS = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
# EgoDex joint name for each MediaPipe landmark index
EGODEX_21 = ["Hand",
             "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"] + \
            [f"{f}{j}" for f in FINGERS[1:] for j in ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]]


def finger_angles(L):
    """L: (T,21,3). Returns (T,15) flexion angles [thumb mcp, ip, opposition, then mcp,pip,dip x4] in rad."""
    def ang(a, b):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return np.arccos(np.clip((a * b).sum(-1), -1, 1))
    out = []
    t = L[:, [0, 1, 2, 3, 4]]
    out += [ang(t[:, 2] - t[:, 1], t[:, 3] - t[:, 2]), ang(t[:, 3] - t[:, 2], t[:, 4] - t[:, 3]),
            ang(t[:, 2] - t[:, 1], L[:, 5] - L[:, 0])]            # thumb metacarpal vs index metacarpal
    for f in range(1, 5):
        c = L[:, [0, 4 * f + 1, 4 * f + 2, 4 * f + 3, 4 * f + 4]]
        out += [ang(c[:, i + 1] - c[:, i], c[:, i + 2] - c[:, i + 1]) for i in range(3)]
    return np.stack(out, 1)


def aperture(L):
    """Thumb-tip to index/middle-tip distance normalised by palm length (scale-free)."""
    palm = np.linalg.norm(L[:, 9] - L[:, 0], axis=-1)
    return 0.5 * (np.linalg.norm(L[:, 4] - L[:, 8], axis=-1) + np.linalg.norm(L[:, 4] - L[:, 12], axis=-1)) / palm


def fill_and_smooth(x, valid, win=9, poly=3):
    """Linear-interpolate missing frames, then Savitzky-Golay along time."""
    x = x.copy(); T = len(x); idx = np.arange(T)
    flat = x.reshape(T, -1)
    for k in range(flat.shape[1]):
        flat[:, k] = np.interp(idx, idx[valid], flat[valid, k])
    win = min(win, T - (1 - T % 2))
    return savgol_filter(flat, win, poly, axis=0).reshape(x.shape)


def load_egodex(ep):
    cols = ["episode_index", "frame_index", "camera_intrinsics", "observation.state.camera"] + \
           [f"observation.state.right{j}" for j in EGODEX_21]
    for f in sorted(PARQ.glob("*.parquet")):
        t = pq.read_table(f, columns=cols, filters=[("episode_index", "=", ep)]).to_pandas()
        if len(t):
            break
    t = t.sort_values("frame_index")
    M = lambda c: np.stack([np.stack(list(v)) for v in t[c]]).astype(float)   # (T,4,4)
    K = np.stack(list(t["camera_intrinsics"].iloc[0])).astype(float)
    joints = np.stack([M(f"observation.state.right{j}")[:, :3, 3] for j in EGODEX_21], 1)  # (T,21,3) ARKit world
    return dict(K=K, cam=M("observation.state.camera"), wrist=M("observation.state.rightHand"), joints=joints)


def run_mediapipe(video):
    cap = cv2.VideoCapture(str(video)); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    T = len(frames); H, W = frames[0].shape[:2]
    world = np.zeros((T, 21, 3)); img = np.zeros((T, 21, 2)); valid = np.zeros(T, bool)
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
        for i, f in enumerate(frames):
            r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if not r.multi_hand_landmarks:
                continue
            # MediaPipe labels assume a mirrored selfie image; in un-mirrored egocentric video the
            # wearer's right hand is labelled "Left".
            for lm, wl, hd in zip(r.multi_hand_landmarks, r.multi_hand_world_landmarks, r.multi_handedness):
                if hd.classification[0].label == "Left":
                    img[i] = [[p.x * W, p.y * H] for p in lm.landmark]
                    world[i] = [[p.x, p.y, p.z] for p in wl.landmark]
                    valid[i] = True
    return world, img, valid, (W, H), frames


def wrist_from_pnp(world, img, valid, K):
    """Wrist pose in the (OpenCV) camera frame from 21 metric landmarks + their 2D projections."""
    T = len(world); pos = np.full((T, 3), np.nan); R = np.tile(np.eye(3), (T, 1, 1))
    for i in np.where(valid)[0]:
        ok, rv, tv = cv2.solvePnP(world[i].astype(np.float64), img[i].astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            R[i] = cv2.Rodrigues(rv)[0]; pos[i] = (R[i] @ (-world[i, 0]) + tv.ravel())  # wrist landmark
    return pos, R


def process(video, ep):
    gt = load_egodex(ep)
    world, img, valid, (W, H), frames = run_mediapipe(video)
    T = min(len(world), len(gt["cam"]))
    world, img, valid = world[:T], img[:T], valid[:T]
    K = gt["K"].copy(); K[:2] *= W / 1920.0                       # intrinsics for the resized take
    print(f"{video.stem}: {T} frames, MediaPipe right hand in {valid.mean():.0%}")

    world_s = fill_and_smooth(world, valid)
    img_s = fill_and_smooth(img, valid)
    p_cam, R_cam = wrist_from_pnp(world_s, img_s, np.ones(T, bool), K)
    # camera -> ARKit world -> z-up world
    Tc = gt["cam"][:T]                    # EgoDex camera pose is camera->world, OpenCV axes (x right, y down, z fwd)
    p_w = np.einsum("tij,tj->ti", Tc[:, :3, :3], p_cam) + Tc[:, :3, 3]
    wrist_mp = p_w @ Y2Z.T
    wrist_mp = fill_and_smooth(wrist_mp, np.isfinite(wrist_mp).all(1))
    R_mp = np.einsum("ij,tjk,tkl->til", Y2Z, Tc[:, :3, :3], R_cam)

    wrist_gt = gt["wrist"][:T, :3, 3] @ Y2Z.T
    R_gt = np.einsum("ij,tjk->tik", Y2Z, gt["wrist"][:T, :3, :3])
    joints_gt = gt["joints"][:T] @ Y2Z.T
    head = gt["cam"][:T, :3, 3] @ Y2Z.T
    head_fwd = gt["cam"][:T, :3, 2] @ Y2Z.T

    ang_mp, ang_gt = finger_angles(world_s), finger_angles(joints_gt)
    ap_mp, ap_gt = aperture(world_s), aperture(joints_gt)

    # project GT wrist into the image to check handedness/PnP consistency (pixels)
    Tw2c = np.linalg.inv(Tc)
    pc = np.einsum("tij,tj->ti", Tw2c[:, :3, :3], gt["wrist"][:T, :3, 3]) + Tw2c[:, :3, 3]
    uv = (pc @ K.T); uv = uv[:, :2] / uv[:, 2:]
    stats = dict(take=video.stem, episode=int(ep), frames=int(T), detect_rate=float(valid.mean()),
                 wrist_px_err=float(np.median(np.linalg.norm(uv[valid] - img[valid, 0], axis=1))) if valid.any() else None,
                 wrist_3d_err_m=float(np.median(np.linalg.norm(wrist_mp - wrist_gt, axis=1))),
                 wrist_rel_err_m=float(np.median(np.linalg.norm((wrist_mp - wrist_mp[0]) - (wrist_gt - wrist_gt[0]), axis=1))),
                 finger_angle_mae_deg=float(np.degrees(np.abs(ang_mp - ang_gt)[:, 3:9].mean())),
                 aperture_corr=float(np.corrcoef(ap_mp, ap_gt)[0, 1]))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{video.stem}.npz", fps=30, landmarks=world_s, landmarks_raw=world, landmarks_2d=img_s,
             valid=valid, wrist_pos=wrist_mp, wrist_rot=R_mp, wrist_pos_gt=wrist_gt, wrist_rot_gt=R_gt,
             finger_angles=ang_mp, finger_angles_gt=ang_gt, aperture=ap_mp, aperture_gt=ap_gt,
             head_pos=head, head_fwd=head_fwd, joints_gt=joints_gt)
    overlay(frames[:T], img, valid, uv, OUT / f"{video.stem}_overlay.mp4")
    return stats


def overlay(frames, img, valid, uv_gt, path):
    import imageio
    out = []
    for f, p, v, g in zip(frames, img, valid, uv_gt):
        f = f.copy()
        if v:
            for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                cv2.line(f, tuple(p[a].astype(int)), tuple(p[b].astype(int)), (0, 255, 0), 2)
        cv2.circle(f, tuple(np.nan_to_num(g).astype(int)), 6, (0, 0, 255), -1)   # GT wrist (red)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, fps=30)


if __name__ == "__main__":
    stats = []
    for v in sorted(TAKES.glob("ep*.mp4")):
        stats.append(process(v, int(v.stem[2:])))
        print("  ", {k: round(x, 3) if isinstance(x, float) else x for k, x in stats[-1].items()})
    json.dump(stats, open(OUT / "extract_stats.json", "w"), indent=1)
