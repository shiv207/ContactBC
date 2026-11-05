#!/usr/bin/env bash
# Folding pipeline, one command (after ./run.sh has set up .venv and Menagerie). ~2.5 h on RTX 5060 + 24 cores.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-egl} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
PY=.venv/bin/python
$PY fold/fetch.py --train-files 2      # 157 egocentric T-shirt folding takes (EgoDex basic_fold)
$PY fold/extract.py                    # MediaPipe both hands + ARKit wrists -> data/processed_fold
$PY fold/retarget.py                   # bimanual fold strokes (298 from 120 takes)
$PY fold/replay.py                     # direct replay on the G1 -> data/fold_demos (148 successes)
$PY fold/augment.py --target 500       # MimicGen-style cloth-pose augmentation -> data/fold_augmented
$PY fold/bc.py                         # BC: plain / aug / aug+cg, 3 seeds
$PY fold/rl.py --base aug --seeds 0 1 2 --iters 25 --eps 80   # residual PPO on top of BC
$PY fold/eval.py --variants plain aug aug+cg aug+rl            # 50 held-out rollouts per seed
$PY fold/phases.py --variants aug aug+rl                       # unsupervised HMM phase check
cat outputs/fold_results.md
