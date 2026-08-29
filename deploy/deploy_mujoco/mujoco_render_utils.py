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
    """Draw command (blue) and actual velocity (green) arrows above the robot base,
    and per-foot contact force arrows (orange) at each wheel body."""

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
        foot_force_arrow_scale=0.0004,  # [m/N] 箭头长度 = |F| * scale, 0 关闭; viewer 自带接触力箭头不可调, 用自绘替代
        foot_force_arrow_width=0.006,
        foot_force_min_arrow_len=0.02,
        foot_force_max_arrow_len=0.12,  # 长度封顶, 冲击瞬间也不会满屏
        foot_force_rgba=(1.0, 0.55, 0.1, 0.9),
        foot_body_keywords=("wheel", "foot"),
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
        self.foot_force_arrow_scale = foot_force_arrow_scale
        self.foot_force_arrow_width = foot_force_arrow_width
        self.foot_force_min_arrow_len = foot_force_min_arrow_len
        self.foot_force_max_arrow_len = foot_force_max_arrow_len
        self.foot_force_rgba = np.array(foot_force_rgba, dtype=np.float32)
        self.foot_body_ids = self._find_foot_bodies(foot_body_keywords)

    def _find_base_body(self, names):
        for name in names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        return 1

    def _find_foot_bodies(self, keywords):
        """按名字关键词(如 wheel/foot)自动定位足/轮 body。"""
        ids = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and any(kw in name.lower() for kw in keywords):
                ids.append(i)
        return ids

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
        self._add_foot_force_arrows(scn)

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

    def _add_foot_force_arrows(self, scn):
        """在每个足/轮 body 处画接触力箭头(长度 = |cfrc| * 可配置 scale, 足底太小可调小 scale)。"""
        if self.foot_force_arrow_scale <= 0 or not self.foot_body_ids or self.data is None:
            return
        for body_id in self.foot_body_ids:
            if scn.ngeom >= scn.maxgeom:
                return
            force = self.data.cfrc_int[body_id, :3]  # 世界系接触力 (叶子 body 即自身受力)
            force_norm = np.linalg.norm(force)
            if force_norm < 1.0:  # <1N 视为腾空, 不画
                continue
            start = self.data.xpos[body_id].copy()
            arrow_len = force_norm * self.foot_force_arrow_scale
            arrow_len = min(max(arrow_len, self.foot_force_min_arrow_len), self.foot_force_max_arrow_len)
            end = start + force / force_norm * arrow_len

            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.zeros(3, dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=self.foot_force_rgba,
            )
            mujoco.mjv_connector(
                geom,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                width=self.foot_force_arrow_width,
                from_=start,
                to=end,
            )
            scn.ngeom += 1
