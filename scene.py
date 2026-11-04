"""MuJoCo scene: fixed-base Unitree G1 (Dex3-1 hands) + table + cup + coaster + fingertip touch sensors.

The G1 pelvis is welded in place (no locomotion/balance): this project is about
the hand/arm manipulation, not whole-body control.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
G1_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml"

TABLE_Z = 0.78                      # table top height (m)
CUP_R, CUP_HH = 0.025, 0.045        # cup radius, half height
COASTER_R, COASTER_HH = 0.05, 0.002
CUP_XY = np.array([0.33, -0.24])    # nominal poses (randomized by augment.py)
COASTER_XY = np.array([0.33, -0.06])

ARM = [f"right_{j}_joint" for j in
       ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
HAND = [f"right_hand_{j}_joint" for j in
        ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]]
TIPS = {"thumb": "right_hand_thumb_2_link", "index": "right_hand_index_1_link", "middle": "right_hand_middle_1_link"}


def build_spec(cup=True):
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.delete(spec.joint("floating_base_joint"))      # weld pelvis to world
    for k in list(spec.keys):
        spec.delete(k)
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3                    # helps grasps hold
    w = spec.worldbody

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[.2, .3, .4], rgb2=[.1, .2, .3], width=300, height=300)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[5, 5], reflectance=0.1)
    w.add_light(pos=[0.5, 0, 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], material="grid")

    # table
    tx, ty = (0.45, -0.1) if cup else (0.45, 0.0)      # cloth scene: wider table (front edge x = 0.16)
    t = w.add_body(name="table", pos=[tx, ty, TABLE_Z / 2])
    t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.29 if not cup else 0.25, 0.4, TABLE_Z / 2], rgba=[.55, .4, .3, 1],
               friction=[1, .005, .0001])

    if not cup:                                          # cloth scene (fold/) builds its own objects
        _add_sensors_cameras(spec, w)
        return spec

    # coaster (static, its own body so augment.py can move it)
    c = w.add_body(name="coaster", pos=[*COASTER_XY, TABLE_Z + COASTER_HH])
    c.add_geom(name="coaster", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[COASTER_R, COASTER_HH, 0], rgba=[.1, .6, .3, 1])
    c.add_site(name="coaster_site", size=[0.005] * 3, rgba=[0, 0, 0, 0])

    # cup: light, high friction solid cylinder
    cup = w.add_body(name="cup", pos=[*CUP_XY, TABLE_Z + CUP_HH + 0.001])
    cup.add_freejoint(name="cup_free")
    cup.add_geom(name="cup", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_R, CUP_HH, 0], mass=0.08,
                 rgba=[.9, .9, .95, 1], friction=[1.5, .05, .002], condim=6)   # torsional friction: no pivoting in the grasp
    cup.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_R * 1.02, 0.008, 0], pos=[0, 0, CUP_HH * 0.5],
                 rgba=[.8, .2, .2, 1], contype=0, conaffinity=0, mass=0)   # red band (visual)

    _add_sensors_cameras(spec, w)
    return spec


def _add_sensors_cameras(spec, w):
    # fingertip touch sensors: a sphere site covering each distal link
    for name, body in TIPS.items():
        b = spec.body(body)
        b.add_site(name=f"tip_{name}", pos=[0, 0, 0], size=[0.03, 0, 0], rgba=[1, 0, 0, 0])
        spec.add_sensor(name=f"touch_{name}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"tip_{name}")
    # grasp site: centre of the hand's grasp aperture, in the wrist frame
    spec.body("right_wrist_yaw_link").add_site(name="grasp", pos=[0.107, 0.048, 0.0], size=[0.008] * 3,
                                                rgba=[1, 1, 0, 0])

    # cameras: third-person and egocentric (head-mounted, like the human videos)
    w.add_camera(name="front", pos=[1.25, -0.55, 1.35], xyaxes=[0.45, 0.89, 0, -0.42, 0.21, 0.88])
    spec.body("torso_link").add_camera(name="ego", pos=[0.08, 0, 0.42], xyaxes=[0, -1, 0, 0.75, 0, 0.66])


class Scene:
    def __init__(self):
        self.model = build_spec().compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_q = np.array([m.joint(j).qposadr[0] for j in ARM])
        self.hand_q = np.array([m.joint(j).qposadr[0] for j in HAND])
        self.arm_dof = np.array([m.joint(j).dofadr[0] for j in ARM])
        self.arm_u = np.array([m.actuator(j).id for j in ARM])
        self.hand_u = np.array([m.actuator(j).id for j in HAND])
        self.cup_q = m.joint("cup_free").qposadr[0]
        self.cup_geom = m.geom("cup").id
        self.coaster_body = m.body("coaster").id
        self.tip_bodies = {k: m.body(v).id for k, v in TIPS.items()}
        self.hand_bodies = {m.body(b).id for b in range(m.nbody) if "right_hand" in m.body(b).name
                            or m.body(b).name == "right_wrist_yaw_link"}
        self.n_sub = int(round((1 / 30) / m.opt.timestep))   # control at 30 Hz (EgoDex frame rate)
        self.reset()

    # --- state -------------------------------------------------------------
    def reset(self, cup_xy=CUP_XY, coaster_xy=COASTER_XY, arm_q=None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.body_pos[self.coaster_body][:2] = coaster_xy
        d.qpos[self.cup_q:self.cup_q + 3] = [*cup_xy, TABLE_Z + CUP_HH + 0.001]
        d.qpos[self.cup_q + 3:self.cup_q + 7] = [1, 0, 0, 0]
        if arm_q is not None:
            d.qpos[self.arm_q] = arm_q
        d.ctrl[:] = d.qpos[m.actuator_trnid[:, 0].clip(0)] * 0      # zero targets for all...
        for i in range(m.nu):                                        # ...then hold current pose
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        mujoco.mj_forward(m, d)

    def robot_q(self):
        return np.concatenate([self.data.qpos[self.arm_q], self.data.qpos[self.hand_q]])

    def cup_pos(self):
        return self.data.qpos[self.cup_q:self.cup_q + 3].copy()

    def cup_up(self):
        return self.data.xmat[self.model.body("cup").id].reshape(3, 3)[:, 2].copy()

    def coaster_xy(self):
        return self.model.body_pos[self.coaster_body][:2].copy()

    def touch(self):
        return np.array([self.data.sensor(f"touch_{k}").data[0] for k in TIPS])

    def cup_contacts(self):
        """Normal force between the cup and each fingertip link (thumb, index, middle) + whole hand."""
        m, d = self.model, self.data
        f = np.zeros(4); f6 = np.zeros(6)
        tip_ids = list(self.tip_bodies.values())
        for i in range(d.ncon):
            c = d.contact[i]
            if self.cup_geom not in (c.geom1, c.geom2):
                continue
            other = m.geom_bodyid[c.geom2 if c.geom1 == self.cup_geom else c.geom1]
            if other not in self.hand_bodies:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if other in tip_ids:
                f[tip_ids.index(other)] += abs(f6[0])
            f[3] += abs(f6[0])
        return f

    # --- control -----------------------------------------------------------
    def step(self, action):
        """action = 14 joint position targets (7 arm + 7 hand)."""
        self.data.ctrl[self.arm_u] = action[:7]
        self.data.ctrl[self.hand_u] = action[7:]
        mujoco.mj_step(self.model, self.data, nstep=self.n_sub)

    def clip_action(self, a):
        lo = self.model.actuator_ctrlrange[np.r_[self.arm_u, self.hand_u], 0]
        hi = self.model.actuator_ctrlrange[np.r_[self.arm_u, self.hand_u], 1]
        return np.clip(a, lo, hi)


class Recorder:
    def __init__(self, scene, cam="front", w=320, h=240):
        self.s, self.cam = scene, cam
        self.r = mujoco.Renderer(scene.model, h, w)
        self.frames = []

    def grab(self):
        self.r.update_scene(self.s.data, camera=self.cam)
        self.frames.append(self.r.render())

    def save(self, path, fps=12):
        """mp4 via imageio, or a compact GIF (shared 64-colour palette via ffmpeg)."""
        import imageio, subprocess, tempfile
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".gif":
            return imageio.mimsave(path, self.frames, fps=30)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "v.mp4"
            imageio.mimsave(tmp, self.frames, fps=30, macro_block_size=1)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-vf",
                            f"fps={fps},split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
                            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(path)], check=True)


if __name__ == "__main__":
    import imageio
    s = Scene()
    print("nq", s.model.nq, "nu", s.model.nu, "n_sub", s.n_sub)
    r = mujoco.Renderer(s.model, 360, 480)
    for cam in ["front", "ego"]:
        r.update_scene(s.data, camera=cam)
        imageio.imwrite(ROOT / f"outputs/scene_{cam}.png", r.render())
    r.close()
    print("grasp site", s.data.site("grasp").xpos, "cup", s.cup_pos())
