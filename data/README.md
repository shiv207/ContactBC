# Data

## How the human data was collected

- **Source:** Apple **EgoDex** ([paper](https://arxiv.org/abs/2505.11709)), recorded with an **Apple Vision Pro**.
  It is head-mounted and egocentric (first person), 1920×1080 at 30 fps, with ARKit hand and head tracking.
- **Access:** through the public LeRobot conversion
  [griffinlabs/EgoDex-LeRobot-v3.0](https://huggingface.co/datasets/griffinlabs/EgoDex-LeRobot-v3.0),
  `test/basic_pick_place`.
- **Selection:** every right-hand episode whose instruction matches "pick up the (tea)cup from the table and place
  it on/in X". That gives **N = 7 takes**.
- **Preprocessing:** `fetch_data.py` cuts each take from the AV1 source to a 960×540 H.264 clip in
  `data/raw/takes/`.
- **Why not phone video:** no phone recordings of my own existed when this was built. To use your own, see the main
  README. Any `*.mp4` in `data/raw/takes/` not named `epNNN` is treated as your own video.

| take | frames | duration | instruction | MediaPipe right hand detected |
|---|---|---|---|---|
| ep016 | 55 | 1.8 s | Pick up the white and pink tea cup from the table with the right hand and place it on the green bowl. | 98% |
| ep062 | 97 | 3.2 s | Pick up a blue cup from the table and place it in a speaker case. | 53% |
| ep128 | 52 | 1.7 s | Pick up the pink and white tea cup from the table with the right hand and place it into the green bowl. | 79% |
| ep202 | 59 | 2.0 s | Pick up a yellow cup from the table and place it on a baking tray. | 63% |
| ep217 | 100 | 3.3 s | Pick up the teacup from the table with the right hand and place it on the yellow bowl. | 100% |
| ep237 | 57 | 1.9 s | Pick up a yellow cup from the table and place it on a baking tray. | 60% |
| ep270 | 69 | 2.3 s | Pick up the pink and white tea cup from the table with the right hand and place it into the green bowl. | 38% |

## License

EgoDex is **CC-BY-NC-ND 4.0**, so the footage is not redistributed here; run `fetch_data.py`.
`data/sample/` holds only outputs derived in this repo:

| file | contents |
|---|---|
| `ep217_mediapipe.npz` | MediaPipe landmarks (raw and smoothed, 3D and 2D), finger angles, aperture, PnP wrist pose |
| `extract_stats.json` | per-take MediaPipe vs EgoDex-ARKit validation numbers |
| `replay_ep217.npz` | one retargeted replay in sim: obs, actions, fingertip contacts, label |
| `aug_*.npz` | one MimicGen-style augmented episode |
