"""Bimanual cloth-folding scene: fixed-base G1 + two Dex3 hands + an 11x9 flex cloth on a table.

Grasping: Dex3 fingertips cannot pinch a 2 mm cloth off a table reliably in MuJoCo, so a hand "grasps" by
activating a point constraint (MuJoCo `connect` equality) between its fingertip grasp site and the nearest cloth
vertex, but only when the hand is commanded closed AND the fingertip is within GRASP_R of the cloth.
Opening the hand releases it. This is the one physics shortcut in the folding pipeline (see README).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene import build_spec, G1_XML, TABLE_Z, Recorder   # noqa: E402

NX, NY, SP = 9, 9, 0.025                  # cloth grid (20 x 20 cm)
CLOTH_XY = np.array([0.36, 0.0])          # nominal cloth centre
GRASP_R = 0.04                            # max fingertip-vertex distance for a grasp to engage
SIDES = ["left", "right"]
ARM = {s: [f"{s}_{j}_joint" for j in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                         "wrist_roll", "wrist_pitch", "wrist_yaw"]] for s in SIDES}
HAND = {s: [f"{s}_hand_{j}_joint" for j in ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1",
                                              "middle_0", "middle_1"]] for s in SIDES}
CLOSE_R = np.array([0.0, -0.6, -0.9, 0.9, 0.9, 0.9, 0.9])     # pinch posture, right hand
CLOSE = {"right": CLOSE_R, "left": -CLOSE_R}                    # left hand is the mirror (all signs flip)
TIP = [0.205, 0.0, 0.0]                                         # grasp site: between index/middle tips


def make_ik(scene):
    """Arm IK per side; shoulder roll limited so elbows stay out (collision-unaware IK otherwise tucks the
    elbows into the torso)."""
    from ik import ArmIK
    iks = {}
    for i, sd in enumerate(SIDES):
        k = ArmIK(scene, joints=ARM[sd], site=f"grasp_{sd}")
        if sd == "right":
            k.hi[1] = min(k.hi[1], -0.05)
        else:
            k.lo[1] = max(k.lo[1], 0.05)
        iks[sd] = k
    return iks


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


R_DOWN = _ry(np.radians(45))              # fingers pitched 45 deg down at the cloth (reachable x 0.20-0.44)


def build_xml():
    spec = build_spec(cup=False)
    spec.meshdir = str(G1_XML.parent / "assets")
    spec.option.noslip_iterations = 0
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    for g in spec.geoms:                  # cloth (collision bit 2) touches only the table: hand-cloth contact
        if g.parent.name == "table":      # squeezed the flex into the table and blew it up; grasps are welds
            g.contype, g.conaffinity = 3, 3
    for s in SIDES:
        spec.body(f"{s}_wrist_yaw_link").add_site(name=f"grasp_{s}", pos=TIP, size=[0.006] * 3, rgba=[1, 1, 0, 0])
    xml = spec.to_xml()
    cloth = f'''
    <flexcomp name="cloth" type="grid" count="{NX} {NY} 1" spacing="{SP} {SP} {SP}"
              pos="{CLOTH_XY[0]} {CLOTH_XY[1]} {TABLE_Z + 0.006}" radius="0.005" dim="2" mass="0.06"
              rgba="0.15 0.35 0.85 1">
      <contact contype="2" conaffinity="2" condim="3" solref="0.003 1" selfcollide="none" friction="0.8"/>
      <edge equality="true" damping="0.002"/>
    </flexcomp>'''
    eqs = "\n".join(f'<connect name="g_{s}_{k}" body1="{s}_wrist_yaw_link" body2="cloth_{k}" '
                    f'anchor="{TIP[0]} {TIP[1]} {TIP[2]}" active="false" solref="0.01 1"/>'
                    for s in SIDES for k in range(NX * NY))
    xml = xml.replace("</worldbody>", cloth + "\n</worldbody>", 1)
    xml = xml.replace("</mujoco>", f"<equality>\n{eqs}\n</equality>\n</mujoco>")
    return xml


class FoldScene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_q = {s: np.array([m.joint(j).qposadr[0] for j in ARM[s]]) for s in SIDES}
        self.hand_q = {s: np.array([m.joint(j).qposadr[0] for j in HAND[s]]) for s in SIDES}
        self.u = np.array([m.actuator(j).id for s in SIDES for j in ARM[s] + HAND[s]])  # 28 actuators
        self.vb = np.array([m.body(f"cloth_{k}").id for k in range(NX * NY)])
        self.rest = m.body_pos[self.vb].copy()
        self.eq = {s: np.array([m.equality(f"g_{s}_{k}").id for k in range(NX * NY)]) for s in SIDES}
        for s in SIDES:   # connect's body2 anchor is computed from qpos0 at compile time: pin it to the vertex
            m.eq_data[self.eq[s], 3:6] = 0
        self.site = {s: m.site(f"grasp_{s}").id for s in SIDES}
        # grid indices (vertex k = i*NY + j, i along x, j along y) -- checked in __main__
        g = np.arange(NX * NY).reshape(NX, NY)
        self.grid = g
        self.n_sub = int(round((1 / 30) / m.opt.timestep))
        self.reset()

    def reset(self, cloth_xy=CLOTH_XY, yaw=0.0, arm_q=None):
        m, d = self.model, self.data
        c, s_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        ctr = np.r_[CLOTH_XY, 0]
        m.body_pos[self.vb] = (self.rest - ctr) @ R.T + np.r_[cloth_xy, 0]
        mujoco.mj_resetData(m, d)
        for sd in SIDES:                  # only the grasp welds (the cloth's own edge constraint stays on)
            d.eq_active[self.eq[sd]] = 0
        if arm_q is not None:
            for s in SIDES:
                d.qpos[self.arm_q[s]] = arm_q[s]
        for i in range(m.nu):
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        mujoco.mj_forward(m, d)
        self.grasped = {s: -1 for s in SIDES}

    # --- observations -------------------------------------------------------
    def verts(self):
        return self.data.xpos[self.vb].copy()

    def corners(self):
        """near-left, near-right, far-left, far-right (near = small x, left = +y)."""
        v, g = self.verts(), self.grid
        return np.stack([v[g[0, -1]], v[g[0, 0]], v[g[-1, -1]], v[g[-1, 0]]])

    def fold_error(self):
        """Mean distance between each near-half vertex and its mirror on the far half (0 = perfect fold)."""
        v, g = self.verts(), self.grid
        h = NX // 2
        return float(np.linalg.norm(v[g[:h]] - v[g[::-1][:h]], axis=-1).mean())

    def robot_q(self):
        return np.concatenate([np.r_[self.data.qpos[self.arm_q[s]], self.data.qpos[self.hand_q[s]]] for s in SIDES])

    def grasp_state(self):
        return np.array([float(self.grasped[s] >= 0) for s in SIDES])

    # --- control ------------------------------------------------------------
    def closure(self, s, hand_target):
        return float(np.clip(np.dot(hand_target, CLOSE[s]) / np.dot(CLOSE[s], CLOSE[s]), 0, 1.5))

    def step(self, action):
        """action: 28 joint targets [left arm 7, left hand 7, right arm 7, right hand 7]."""
        d = self.data
        d.ctrl[self.u] = action
        for i, s in enumerate(SIDES):
            c = self.closure(s, action[14 * i + 7:14 * i + 14])
            if c > 0.6 and self.grasped[s] < 0:
                dist = np.linalg.norm(self.verts() - d.site_xpos[self.site[s]], axis=1)
                k = int(np.argmin(dist))
                other = self.grasped[SIDES[1 - i]]
                if dist[k] < GRASP_R and k != other:
                    d.eq_active[self.eq[s][k]] = 1; self.grasped[s] = k
            elif c < 0.4 and self.grasped[s] >= 0:
                d.eq_active[self.eq[s][self.grasped[s]]] = 0; self.grasped[s] = -1
        mujoco.mj_step(self.model, d, nstep=self.n_sub)

    def clip_action(self, a):
        r = self.model.actuator_ctrlrange[self.u]
        return np.clip(a, r[:, 0], r[:, 1])


if __name__ == "__main__":
    import imageio, time
    s = FoldScene()
    v = s.verts()
    print("nq", s.model.nq, "neq", s.model.neq, "vertex 0", v[0].round(3), "vertex 1", v[1].round(3),
          f"vertex {NY}", v[NY].round(3))
    print("corners", s.corners().round(3), "fold error (flat)", round(s.fold_error(), 3))
    t = time.time()
    for _ in range(30):
        s.step(s.robot_q())
    print(f"1 s sim in {time.time() - t:.2f}s wall")
    r = mujoco.Renderer(s.model, 360, 480)
    for cam in ["front", "ego"]:
        r.update_scene(s.data, camera=cam); imageio.imwrite(f"outputs/fold_scene_{cam}.png", r.render())
    r.close()
