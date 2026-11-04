"""Step 0: download the egocentric human takes.

Source: Apple EgoDex (Apple Vision Pro, egocentric 1080p @30 fps + ARKit hand tracking), via the public LeRobot
conversion griffinlabs/EgoDex-LeRobot-v3.0 (CC-BY-NC-ND 4.0). We take every right-hand "pick up the <cup> from
the table and place it on/in <X>" episode from the basic_pick_place test split and cut each into its own clip.

If you have your own phone recordings, skip this and drop them into data/raw/takes/ as epNNN.mp4 (see README).
"""
import subprocess
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
REPO, TASK = "griffinlabs/EgoDex-LeRobot-v3.0", "test/basic_pick_place"
RAW = ROOT / "data/raw/egodex"
TAKES = ROOT / "data/raw/takes"
CUP_TASK = r"Pick up (?:the |a )?[\w\s-]*(?:tea ?cup|cup) from the table"


def get(f):
    return Path(hf_hub_download(REPO, f"{TASK}/{f}", repo_type="dataset", local_dir=RAW))


if __name__ == "__main__":
    ep = pd.read_parquet(get("meta/episodes/chunk-000/file-000.parquet"))
    ep["task"] = ep["tasks"].apply(lambda t: t[0])
    cups = ep[ep.task.str.contains(CUP_TASK, case=False, regex=True)]
    print(f"{len(cups)} cup pick-and-place episodes")
    vk = "videos/observation.images.camera"
    TAKES.mkdir(parents=True, exist_ok=True)
    for _, r in cups.iterrows():
        get(f"data/chunk-000/file-{int(r['data/file_index']):03d}.parquet")
        video = get(f"{vk}/chunk-000/file-{int(r[f'{vk}/file_index']):03d}.mp4")
        out = TAKES / f"ep{int(r.episode_index):03d}.mp4"
        if not out.exists():          # AV1 -> H.264 960x540 (OpenCV's bundled ffmpeg cannot decode AV1)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(r[f"{vk}/from_timestamp"]),
                            "-to", str(r[f"{vk}/to_timestamp"]), "-i", str(video), "-vf", "scale=960:-2",
                            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
        print(f"  {out.name}: {r.length} frames  {r.task.split(': ', 1)[1]}")
