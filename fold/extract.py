"""Bimanual version of extract_hands.py for the folding takes.

Per take, for BOTH hands: MediaPipe 21 landmarks (smoothed) -> finger angles + thumb-index aperture, and the
EgoDex ARKit wrist trajectory (z-up world) + ARKit finger joints for validation. Output: data/processed_fold/*.npz
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import cv2
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_hands import finger_angles, aperture, fill_and_smooth, EGODEX_21, Y2Z   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAKES = ROOT / "data/raw/fold_takes"
OUT = ROOT / "data/processed_fold"
PARQ = {"t": ROOT / "data/raw/egodex/test/basic_fold/data/chunk-000",
        "r": ROOT / "data/raw/egodex/train/basic_fold-part1/data/chunk-000"}
SIDES = ["left", "right"]


def load_gt(tag, ep):
    cols = ["episode_index", "frame_index", "observation.state.camera"] + \
           [f"observation.state.{s}{j}" for s in SIDES for j in EGODEX_21]
    for f in sorted(PARQ[tag].glob("*.parquet")):
        t = pq.read_table(f, columns=cols, filters=[("episode_index", "=", ep)]).to_pandas()
        if len(t):
            break
    t = t.sort_values("frame_index")
    M = lambda c: np.stack([np.stack(list(v)) for v in t[c]]).astype(float)
    out = {"cam": M("observation.state.camera")}
    for s in SIDES:
        out[f"{s}_wrist"] = M(f"observation.state.{s}Hand")
        out[f"{s}_joints"] = np.stack([M(f"observation.state.{s}{j}")[:, :3, 3] for j in EGODEX_21], 1)
    return out


def mediapipe_both(video):
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video)); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    T = len(frames)
    res = {s: (np.zeros((T, 21, 3)), np.zeros(T, bool)) for s in SIDES}
    # un-mirrored egocentric video: MediaPipe "Left" = wearer's right hand
    label2side = {"Left": "right", "Right": "left"}
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
        for i, f in enumerate(frames):
            r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if not r.multi_hand_landmarks:
                continue
            for wl, hd in zip(r.multi_hand_world_landmarks, r.multi_handedness):
                s = label2side[hd.classification[0].label]
                res[s][0][i] = [[p.x, p.y, p.z] for p in wl.landmark]; res[s][1][i] = True
    return res, T


def process(video):
    tag, ep = video.stem[0], int(video.stem[1:])
    gt = load_gt(tag, ep)
    mpres, T = mediapipe_both(video)
    T = min(T, len(gt["cam"]))
    out = dict(fps=30, head_fwd=(gt["cam"][:T, :3, 2] @ Y2Z.T), head_pos=gt["cam"][:T, :3, 3] @ Y2Z.T)
    stats = dict(take=video.stem, frames=int(T))
    for s in SIDES:
        L, valid = mpres[s][0][:T], mpres[s][1][:T]
        jg = gt[f"{s}_joints"][:T] @ Y2Z.T
        out[f"{s}_wrist"] = gt[f"{s}_wrist"][:T, :3, 3] @ Y2Z.T
        out[f"{s}_wrist_rot"] = np.einsum("ij,tjk->tik", Y2Z, gt[f"{s}_wrist"][:T, :3, :3])
        out[f"{s}_aperture_gt"] = aperture(jg)
        out[f"{s}_finger_angles_gt"] = finger_angles(jg)
        out[f"{s}_valid"] = valid
        if valid.sum() >= 5:
            Ls = fill_and_smooth(L, valid)
            out[f"{s}_landmarks"] = Ls
            out[f"{s}_finger_angles"] = finger_angles(Ls)
            out[f"{s}_aperture"] = aperture(Ls)
            stats[f"{s}_detect"] = float(valid.mean())
            stats[f"{s}_aperture_corr"] = float(np.corrcoef(out[f"{s}_aperture"], out[f"{s}_aperture_gt"])[0, 1])
        else:                                   # hand never seen: fall back to ARKit fingers
            out[f"{s}_finger_angles"] = out[f"{s}_finger_angles_gt"]
            out[f"{s}_aperture"] = out[f"{s}_aperture_gt"]
            stats[f"{s}_detect"] = float(valid.mean())
    np.savez_compressed(OUT / f"{video.stem}.npz", **out)
    return stats


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    vids = sorted(TAKES.glob("*.mp4"))
    with Pool(12) as pool:
        stats = pool.map(process, vids, chunksize=1)
    json.dump(stats, open(OUT / "extract_stats.json", "w"), indent=1)
    for k in ["left_detect", "right_detect", "left_aperture_corr", "right_aperture_corr"]:
        v = [s[k] for s in stats if k in s and np.isfinite(s[k])]
        print(f"{k}: median {np.median(v):.2f}  (n={len(v)})")
