import numpy as np
import mujoco


def quat_apply_wxyz(quat, vec):
    quat = np.asarray(quat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    qw, qx, qy, qz = quat
    q_vec = np.array([qx, qy, qz], dtype=np.float64)
    t = 2.0 * np.cross(q_vec, vec)
    return vec + qw * t + np.cross(q_vec, t)


def quat_rotate_inverse_wxyz(quat, vec):
    quat = np.asarray(quat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    qw, qx, qy, qz = quat
    q_vec = np.array([qx, qy, qz], dtype=np.float64)
    a = vec * (2.0 * qw * qw - 1.0)
    b = np.cross(q_vec, vec) * qw * 2.0
    c = q_vec * np.dot(q_vec, vec) * 2.0
    return a - b + c


class MujocoRenderUtils:
    """Draw command (blue) and actual velocity (green) arrows above the robot base."""

    def __init__(
        self,
        model,
        base_body_names=("base", "base_link"),
        arrow_height=0.45,
        vel_arrow_height=0.58,
        arrow_width=0.015,
        arrow_scale=0.6,
        min_arrow_len=0.08,
        cmd_arrow_rgba=(0.15, 0.45, 1.0, 0.95),
        vel_arrow_rgba=(0.15, 0.85, 0.25, 0.95),
    ):
        self.model = model
        self.data = None
        self.cmd = np.zeros(3, dtype=np.float64)
        self.actual_vel = np.zeros(3, dtype=np.float64)
        self.base_body_id = self._find_base_body(base_body_names)
        self.arrow_height = arrow_height
        self.vel_arrow_height = vel_arrow_height
        self.arrow_width = arrow_width
        self.arrow_scale = arrow_scale
        self.min_arrow_len = min_arrow_len
        self.cmd_arrow_rgba = np.array(cmd_arrow_rgba, dtype=np.float32)
        self.vel_arrow_rgba = np.array(vel_arrow_rgba, dtype=np.float32)

    def _find_base_body(self, names):
        for name in names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        return 1

    def update(self, cmd, actual_vel, data):
        self.cmd = np.asarray(cmd, dtype=np.float64)
        self.actual_vel = np.asarray(actual_vel, dtype=np.float64)
        self.data = data

    def update_external_rendering(self, viewer_or_renderer, ctype="viewer"):
        if ctype != "viewer" or self.data is None:
            return

        scn = viewer_or_renderer.user_scn
        scn.ngeom = 0
        self._add_velocity_arrow(
            scn,
            self.cmd[:2],
            self.cmd_arrow_rgba,
            self.arrow_height,
        )
        self._add_velocity_arrow(
            scn,
            self.actual_vel[:2],
            self.vel_arrow_rgba,
            self.vel_arrow_height,
        )

    def _add_velocity_arrow(self, scn, vel_xy, rgba, height):
        if scn.ngeom >= scn.maxgeom:
            return

        vel_norm = np.linalg.norm(vel_xy)
        if vel_norm < 1e-4:
            return

        base_pos = self.data.xpos[self.base_body_id].copy()
        base_quat = self.data.xquat[self.base_body_id].copy()
        start = base_pos + np.array([0.0, 0.0, height], dtype=np.float64)

        direction = quat_apply_wxyz(
            base_quat, np.array([vel_xy[0], vel_xy[1], 0.0], dtype=np.float64)
        )
        direction_norm = np.linalg.norm(direction[:2])
        if direction_norm < 1e-6:
            return
        direction = direction / direction_norm

        arrow_len = max(vel_norm * self.arrow_scale, self.min_arrow_len)
        end = start + direction * arrow_len

        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3, dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            width=self.arrow_width,
            from_=start,
            to=end,
        )
        scn.ngeom += 1
