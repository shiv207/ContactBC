"""Sanity check: MuJoCo (EGL offscreen) + torch CUDA + MediaPipe."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, torch, mediapipe as mp

m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); r.update_scene(d); img = r.render()
print("mujoco", mujoco.__version__, "egl render ok", img.shape)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
x = torch.randn(512, 512, device="cuda"); print("cuda matmul ok", float((x @ x).mean()))
print("mediapipe", mp.__version__)
