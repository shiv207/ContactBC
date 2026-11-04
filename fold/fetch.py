"""Download EgoDex `basic_fold` (egocentric bimanual T-shirt folding) and cut every episode into its own clip.

  python fold/fetch.py                 # test split: 77 takes, ~0.6 GB
  python fold/fetch.py --train-files 2 # + 2 video files of train/basic_fold-part1 (~80 more takes)
"""
import argparse, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "griffinlabs/EgoDex-LeRobot-v3.0"
RAW = ROOT / "data/raw/egodex"
TAKES = ROOT / "data/raw/fold_takes"
VK = "videos/observation.images.camera"


def get(f):
    return Path(hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW))


def cut(args):
    video, t0, t1, out = args
    if not out.exists():                      # AV1 -> H.264 960x540 (OpenCV cannot decode AV1)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-to", str(t1), "-i", str(video),
                        "-vf", "scale=960:-2", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-an",
                        str(out)], check=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-files", type=int, default=0)
    args = ap.parse_args()
    TAKES.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, task, nfiles in [("test", "basic_fold", None), ("train", "basic_fold-part1", args.train_files)]:
        if nfiles == 0:
            continue
        ep = pd.read_parquet(get(f"{split}/{task}/meta/episodes/chunk-000/file-000.parquet"))
        if nfiles is not None:
            ep = ep[ep[f"{VK}/file_index"] < nfiles]
        for _, r in ep.iterrows():
            get(f"{split}/{task}/data/chunk-000/file-{int(r['data/file_index']):03d}.parquet")
            video = get(f"{split}/{task}/{VK}/chunk-000/file-{int(r[f'{VK}/file_index']):03d}.mp4")
            tag = "t" if split == "test" else "r"
            jobs.append((video, r[f"{VK}/from_timestamp"], r[f"{VK}/to_timestamp"],
                         TAKES / f"{tag}{int(r.episode_index):04d}.mp4"))
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(cut, jobs))
    print(f"{len(outs)} takes in {TAKES}")
