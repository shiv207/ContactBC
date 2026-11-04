"""Damped-least-squares IK for the G1 right arm (7 DoF), targeting the 'grasp' site pose."""
import numpy as np
import mujoco


def quat_err(R_target, R):
    """Rotation-vector error that rotates R onto R_target."""
    q, qt = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten()); mujoco.mju_mat2Quat(qt, R_target.flatten())
    qc = np.zeros(4); mujoco.mju_negQuat(qc, q)
    qe = np.zeros(4); mujoco.mju_mulQuat(qe, qt, qc)
    v = np.zeros(3); mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


class ArmIK:
    def __init__(self, scene, rot_weight=1.0, damping=1e-2, joints=None, site="grasp"):
        self.s = scene
        self.m = scene.model
        self.d = mujoco.MjData(self.m)          # scratch data: never disturbs the sim
        self.site = self.m.site(site).id
        self.rw, self.lam = rot_weight, damping
        joints = joints or scene_arm_joints(scene)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in joints])
        self.dadr = np.array([self.m.joint(j).dofadr[0] for j in joints])
        self.lo = self.m.jnt_range[[self.m.joint(j).id for j in joints], 0]
        self.hi = self.m.jnt_range[[self.m.joint(j).id for j in joints], 1]

    SEEDS = [np.zeros(7), np.array([-0.5, -0.3, 0, 1.0, 0, 0, 0]), np.array([-1.0, 0, 0, 0.5, 0, 0.5, 0]),
             np.array([-0.5, 0.3, 0, 1.0, 0, 0, 0]), np.array([-1.0, 0, 0, 0.5, 0, -0.5, 0])]

    def solve(self, pos, R=None, q0=None, iters=60, tol=1e-3, retry=0.01):
        """Warm-started DLS; if the result misses by > retry (m), also try SEEDS and keep the best."""
        q, e = self._solve(pos, R, q0, iters, tol)
        if retry is not None and e > retry:
            for sd in self.SEEDS:
                q2, e2 = self._solve(pos, R, sd, 150, tol)
                if e2 < e:
                    q, e = q2, e2
        return q, e

    def _solve(self, pos, R=None, q0=None, iters=60, tol=1e-3):
        s, m, d = self.s, self.m, self.d
        d.qpos[:] = s.data.qpos
        if q0 is not None:
            d.qpos[self.qadr] = q0
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            e = pos - d.site_xpos[self.site]
            J = np.zeros((0, 7)); err = e
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = jp[:, self.dadr]
            if R is not None:
                er = quat_err(R, d.site_xmat[self.site].reshape(3, 3))
                err = np.r_[e, self.rw * er]
                J = np.r_[J, self.rw * jr[:, self.dadr]]
            if np.linalg.norm(e) < tol and (R is None or np.linalg.norm(err[3:]) < 5 * tol):
                break
            dq = J.T @ np.linalg.solve(J @ J.T + self.lam * np.eye(len(err)), err)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        return d.qpos[self.qadr].copy(), float(np.linalg.norm(pos - d.site_xpos[self.site]))


def scene_arm_joints(scene):
    from scene import ARM
    return ARM
