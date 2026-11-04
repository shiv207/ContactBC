#!/usr/bin/env bash
# One command: env -> data -> hands -> retarget -> replay -> augment -> train -> eval.   ~15 min on RTX 5060 + 24 cores
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv            # MediaPipe has no wheels for Python >= 3.13
  uv pip install --python .venv/bin/python -r <(grep -v '^torch' requirements.txt)
  UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
fi
PY=.venv/bin/python

if [ ! -d third_party/mujoco_menagerie/unitree_g1 ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1
fi

$PY verify_env.py
$PY fetch_data.py                     # skip if you put your own takes in data/raw/takes/
$PY extract_hands.py                  # MediaPipe -> data/processed/*.npz
$PY retarget.py                       # prints per-take keyframes / transport
$PY replay.py                         # direct human replays -> data/demos, outputs/gifs/replay_*
$PY augment.py --target 500           # MimicGen-style -> data/augmented
$PY train_bc.py                       # 5 variants x 3 seeds -> checkpoints/
$PY eval.py                           # 50 rollouts each -> outputs/results.md, outputs/gifs/eval_*
cat outputs/results.md
