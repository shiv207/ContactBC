"""Watch fold policies live in the interactive MuJoCo viewer (real time, loops over held-out cloth poses).

  python fold/watch.py                    # aug+rl, seed 0, eval rollouts 0,1,2,...
  python fold/watch.py --variant aug --start 21
Mouse: left-drag rotate, right-drag pan, scroll zoom. Close the window to quit.
"""
import os
os.environ["MUJOCO_GL"] = "glfw"           # on-screen window, not headless EGL
import argparse, sys, time
from pathlib import Path
import numpy as np
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fold.scene import FoldScene, make_ik          # noqa: E402
from fold.augment import sample_pose               # noqa: E402
from fold.replay import obs_vec, home_q, label_state  # noqa: E402
from fold.eval import make_actor, H, EVAL_SEED     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="aug+rl")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start", type=int, default=0, help="first held-out rollout index")
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()

if "+rl" in a.variant:
    from fold.rl import load_rl
    pol = load_rl(a.variant, a.seed)
else:
    from fold.bc import load_policy
    pol = load_policy(a.variant, a.seed)

s = FoldScene(); ik = make_ik(s)
with mujoco.viewer.launch_passive(s.model, s.data) as v:
    v.cam.lookat[:] = [0.35, 0, 0.8]; v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 180, -35
    i = a.start
    while v.is_running():
        xy, yaw = sample_pose(np.random.default_rng(EVAL_SEED * 1000 + i))
        s.reset(xy, yaw, arm_q=home_q(s, ik))
        act = make_actor(pol); G = []
        print(f"rollout {i}: cloth at ({xy[0]:.2f}, {xy[1]:.2f}) yaw {np.degrees(yaw):+.0f} deg ...", end=" ", flush=True)
        for t in range(H):
            if not v.is_running():
                break
            t0 = time.time()
            s.step(s.clip_action(act(obs_vec(s), t))); G.append(s.grasp_state())
            v.sync()
            time.sleep(max(0.0, (1 / 30) / a.speed - (time.time() - t0)))
        G = np.array(G); held = np.flatnonzero(G.max(1) > 0) if len(G) else []
        tr = slice(held[0], held[-1] + 1) if len(held) else slice(0, 0)
        ok = len(held) > 0 and bool(label_state(s, G, tr)["success"])
        print("FOLDED" if ok else "failed", flush=True)
        t1 = time.time()
        while v.is_running() and time.time() - t1 < 1.5:   # pause on the final state
            v.sync(); time.sleep(0.03)
        i += 1
