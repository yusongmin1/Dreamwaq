import os
import time
from collections import deque

import matplotlib
matplotlib.use('TkAgg')  # 本机 Qt 的 xcb 插件不可用，强制使用 Tk 后端
import matplotlib.pyplot as plt
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import onnxruntime as ort
import pygame
from pynput import keyboard

from mujoco_render_utils import MujocoRenderUtils, quat_rotate_inverse_wxyz

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def resolve_config_path(config_file):
    if os.path.isabs(config_file) and os.path.isfile(config_file):
        return config_file
    candidates = [
        config_file,
        os.path.join(CONFIG_DIR, config_file),
        os.path.join(REPO_ROOT, config_file),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Config file not found: {config_file}")


GEAR_SPEEDS = {1: 0.5, 2: 1.0, 3: 1.5}


def get_xbox_command(joystick, max_cmd, dead_zone=0.1):
    pygame.event.pump()
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    if abs(lx) < dead_zone:
        lx = 0.0
    if abs(ly) < dead_zone:
        ly = 0.0
    if abs(rx) < dead_zone:
        rx = 0.0
    cmd_x = -ly * max_cmd[0]
    cmd_y = -lx * max_cmd[1]
    cmd_yaw = -rx * max_cmd[2]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)


class JointVelocityPlotter:
    """实时滚动绘制关节速度曲线（含轮子/foot 关节），按四条腿分为 2x2 四张子图。

    关节顺序为 FL, FR, RL, RR，每条腿 [hip, thigh, calf, foot]，
    四个关节全部绘制（foot 即轮子），子图顺序：左上 FL、右上 FR、左下 RL、右下 RR。
    """

    LEGS = ['FL', 'FR', 'RL', 'RR']
    JOINTS = ['hip', 'thigh', 'calf', 'foot']

    def __init__(self, dt=0.005, window_s=10.0, refresh_every=10):
        """
        Args:
            dt: 仿真步长 [s]
            window_s: 曲线滚动窗口长度 [s]
            refresh_every: 每多少个仿真步刷新一次绘图（10 步 = 20Hz）
        """
        self.refresh_every = refresh_every
        # 数据每个仿真步都会存入缓冲区，按 dt 计算点数才能覆盖 window_s
        self.max_points = max(2, int(window_s / dt))
        self.t_buf = deque(maxlen=self.max_points)
        self.dq_buf = deque(maxlen=self.max_points)
        self._count = 0

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6.5))
        try:
            self.fig.canvas.manager.set_window_title('Joint Velocities (real-time)')
        except Exception:
            pass
        self.lines = {}
        for k, leg in enumerate(self.LEGS):
            ax = self.axes[k // 2][k % 2]
            self.lines[leg] = []
            for joint in self.JOINTS:
                line, = ax.plot([], [], label=f'{leg} {joint}', linewidth=1.2)
                self.lines[leg].append(line)
            ax.set_xlabel('time [s]')
            ax.set_ylabel('joint velocity [rad/s]')
            ax.set_title(leg)
            ax.grid(True, alpha=0.4)
            ax.legend(loc='upper right', ncol=4, fontsize=8)
        self.fig.suptitle('Joint velocities (rolling window)')
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, t, dq):
        """每个仿真步调用一次：缓存数据，按 refresh_every 频率重绘。

        Args:
            t: 当前仿真时间 [s]
            dq: 16 维关节速度（策略顺序 FL/FR/RL/RR × [hip, thigh, calf, foot]），
                四个关节全部绘制（foot 即轮子速度）
        """
        self.t_buf.append(float(t))
        self.dq_buf.append(np.asarray(dq, dtype=np.float64).reshape(4, 4).copy())
        self._count += 1
        if self._count % self.refresh_every != 0:
            return
        ts = np.asarray(self.t_buf)
        dqs = np.asarray(self.dq_buf)  # (N, 4 legs, 4 joints)
        for k, leg in enumerate(self.LEGS):
            for line, col in zip(self.lines[leg], dqs[:, k, :].T):
                line.set_data(ts, col)
            ax = self.axes[k // 2][k % 2]
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)


class MujocoDeploy:
    def __init__(self, default_config="m20.yaml"):
        import argparse

        self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd = 0.05, 0.0, 0.0
        listener = keyboard.Listener(on_press=self._on_key_press)
        listener.start()

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c",
            "--config_file",
            type=str,
            default=default_config,
            help="config file name in the config folder",
        )
        args = parser.parse_args()
        self.config_file = resolve_config_path(args.config_file)
        with open(self.config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.config = config
            self.policy_path = resolve_repo_path(config["policy_path"])
            self.xml_path = resolve_repo_path(config["xml_path"])
            print(f"config: {self.config_file}")
            print(f"xml: {self.xml_path}")
            print(f"policy: {self.policy_path}")

            self.simulation_duration = config["simulation_duration"]
            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.torque_limits = np.array(config["torque_limits"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.vel_scale = config["vel_scale"]
            self.base_max_cmd = np.array(
                config.get("max_cmd", [1.5, 1.0, 3.0]), dtype=np.float32
            )
            self.gear_speeds = config.get("gear_speeds", GEAR_SPEEDS)
            self.gear = config.get("default_gear", 1)
            self.max_cmd = self.base_max_cmd.copy()
            self.max_cmd[0] = self.gear_speeds[self.gear]
            self._prev_lb = False
            self._prev_rb = False

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.num_obs_hist = config.get("num_obs_hist", 5)
            self.init_base_height = config.get("init_base_height", None)

            self.wheel_indices = config["wheel_sim_indices"]
            self.cmd = np.array(config["cmd_init"], dtype=np.float32)
            self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd = self.cmd.tolist()
            self.mujoco_to_policy_idx = None
            self.policy_to_mujoco_idx = None

        self._init_joystick(config.get("use_joystick", True))

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.obs_hist_len = self.num_obs * (self.num_obs_hist + 1)
        self.obs_hist_buf = np.zeros(self.obs_hist_len, dtype=np.float32)

        self.dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_vel = np.zeros(self.num_actions, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.line_vel = np.zeros(3, dtype=np.float32)

        self.counter = 0

        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt

        self.policy = ort.InferenceSession(
            self.policy_path, providers=["CPUExecutionProvider"]
        )
        self.render_utils = MujocoRenderUtils(
            self.m,
            base_body_names=tuple(config.get("base_body_names", ["base", "base_link"])),
            arrow_height=config.get("arrow_height", 0.45),
            vel_arrow_height=config.get("vel_arrow_height", 0.58),
            arrow_width=config.get("arrow_width", 0.015),
            arrow_scale=config.get("arrow_scale", 0.6),
            cmd_arrow_rgba=tuple(config.get("cmd_arrow_rgba", [0.15, 0.45, 1.0, 0.95])),
            vel_arrow_rgba=tuple(config.get("vel_arrow_rgba", [0.15, 0.85, 0.25, 0.95])),
        )
        self._setup_dof_mapping(config)
        self._print_joint_order()
        self._reset_robot_state()
        print(f"Gear {self.gear}: max vx = {self.max_cmd[0]:.1f} m/s")

    def _init_joystick(self, use_joystick):
        self.use_joystick = False
        self.joystick = None
        if not use_joystick:
            print("Joystick disabled in config. Using keyboard commands.")
            return

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.use_joystick = True
            print(f"Detected joystick: {self.joystick.get_name()}")
        else:
            print("No joystick detected. Using keyboard commands.")

    def _set_gear(self, gear):
        gear = int(np.clip(gear, 1, 3))
        if gear == self.gear:
            return
        self.gear = gear
        self.max_cmd[0] = self.gear_speeds[self.gear]
        self._clamp_cmd()
        print(f"Gear {self.gear}: max vx = {self.max_cmd[0]:.1f} m/s")

    def _clamp_cmd(self):
        max_vx = self.max_cmd[0]
        self.x_vel_cmd = float(np.clip(self.x_vel_cmd, -max_vx, max_vx))
        self.cmd[:] = [self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd]

    def _on_key_press(self, key):
        try:
            if key.char == "6":
                self.x_vel_cmd = min(self.x_vel_cmd + 0.1, self.max_cmd[0])
            elif key.char == "7":
                self.x_vel_cmd = max(self.x_vel_cmd - 0.1, -self.max_cmd[0])
            elif key.char == "8":
                self.y_vel_cmd += 0.3
            elif key.char == "9":
                self.y_vel_cmd -= 0.3
            elif key.char == "-":
                self.yaw_vel_cmd += 0.5
            elif key.char == "=":
                self.yaw_vel_cmd -= 0.5
            elif key.char == "1":
                self.x_vel_cmd = 0.0
                self.y_vel_cmd = 0.0
                self.yaw_vel_cmd = 0.0
            elif key.char == "2":
                self._set_gear(1)
                return
            elif key.char == "3":
                self._set_gear(2)
                return
            elif key.char == "4":
                self._set_gear(3)
                return
            else:
                return
            self.cmd[:] = [self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd]
            print(
                f"Updated velocities: vx={self.x_vel_cmd:.2f}, "
                f"vy={self.y_vel_cmd:.2f}, dyaw={self.yaw_vel_cmd:.2f}, "
                f"gear={self.gear}"
            )
        except AttributeError:
            pass

    def _update_command(self):
        if self.use_joystick and self.joystick is not None:
            lb = self.joystick.get_button(4)
            rb = self.joystick.get_button(5)
            if rb and not self._prev_rb:
                self._set_gear(self.gear + 1)
            elif lb and not self._prev_lb:
                self._set_gear(self.gear - 1)
            self._prev_lb = lb
            self._prev_rb = rb

            cmd = get_xbox_command(self.joystick, self.max_cmd)
            self.x_vel_cmd = float(np.clip(cmd[0], -self.max_cmd[0], self.max_cmd[0]))
            self.y_vel_cmd, self.yaw_vel_cmd = cmd[1], cmd[2]
        self.cmd[:] = [self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd]

    def _setup_dof_mapping(self, config):
        if "isaac_dof_names" in config:
            mj_names = self._get_joint_names_in_qpos_order()
            isaac_names = config["isaac_dof_names"]
            if len(isaac_names) != self.num_actions:
                raise ValueError("isaac_dof_names length must match num_actions")
            self.policy_to_mujoco_idx = np.array(
                [mj_names.index(name) for name in isaac_names], dtype=np.int32
            )
        elif "policy_to_mujoco_idx" in config:
            self.policy_to_mujoco_idx = np.array(
                config["policy_to_mujoco_idx"], dtype=np.int32
            )
        elif "mujoco_to_policy_idx" in config:
            mujoco_to_policy = np.array(config["mujoco_to_policy_idx"], dtype=np.int32)
            self.policy_to_mujoco_idx = np.zeros(self.num_actions, dtype=np.int32)
            for mj_i, pol_i in enumerate(mujoco_to_policy):
                self.policy_to_mujoco_idx[pol_i] = mj_i
        else:
            return

        self.mujoco_to_policy_idx = np.zeros(self.num_actions, dtype=np.int32)
        for pol_i, mj_i in enumerate(self.policy_to_mujoco_idx):
            self.mujoco_to_policy_idx[mj_i] = pol_i
        print(
            "Isaac Gym -> MuJoCo dof map (policy_to_mujoco_idx):",
            self.policy_to_mujoco_idx.tolist(),
        )

    def _get_joint_names_in_qpos_order(self):
        names = []
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            names.append(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j))
        return names

    def _print_joint_order(self):
        qpos_joints = self._get_joint_names_in_qpos_order()
        print("MuJoCo joint order (qpos[7:]):")
        for i, name in enumerate(qpos_joints):
            print(f"  [{i}] {name}")
        print("MuJoCo actuator order (ctrl):")
        for i in range(self.m.nu):
            print(f"  [{i}] {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
        if len(qpos_joints) == self.m.nu:
            act_joint_names = []
            for i in range(self.m.nu):
                trn_id = self.m.actuator_trnid[i, 0]
                act_joint_names.append(
                    mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, trn_id)
                )
            mismatch = [
                i
                for i, (a, b) in enumerate(zip(qpos_joints, act_joint_names))
                if a != b
            ]
            if mismatch:
                print(f"WARNING: qpos/actuator order mismatch at indices {mismatch}")

    def _mujoco_to_policy(self, mj_vals):
        if self.mujoco_to_policy_idx is None:
            return mj_vals.copy()
        return mj_vals[self.mujoco_to_policy_idx]

    def _policy_to_mujoco(self, policy_vals):
        if self.policy_to_mujoco_idx is None:
            return policy_vals.copy()
        mj_vals = np.zeros_like(policy_vals)
        for pol_i, mj_i in enumerate(self.policy_to_mujoco_idx):
            mj_vals[mj_i] = policy_vals[pol_i]
        return mj_vals

    def _reset_robot_state(self):
        if self.init_base_height is not None:
            self.d.qpos[2] = self.init_base_height
        self.d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.d.qpos[7 : 7 + self.num_actions] = self._policy_to_mujoco(self.default_angles)
        self.d.qvel[:] = 0.0
        self.d.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.get_robot_state()

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def get_robot_state(self):
        self.dof_pos = self._mujoco_to_policy(self.d.qpos[7 : 7 + self.num_actions])
        self.dof_vel = self._mujoco_to_policy(self.d.qvel[6 : 6 + self.num_actions])
        self.quat = self.d.qpos[3:7]
        self.ang_vel = self.d.qvel[3:6]
        self.base_lin_vel = quat_rotate_inverse_wxyz(self.quat, self.d.qvel[:3])
        self.line_vel = self.base_lin_vel

    def compute_torques(self, actions):
        dof_err = self.default_angles - self.dof_pos
        dof_err[self.wheel_indices] = 0

        actions_scaled = actions * self.action_scale
        actions_scaled[self.wheel_indices] = 0
        vel_ref = np.zeros_like(actions_scaled)
        vel_tmp = actions * self.vel_scale
        vel_ref[self.wheel_indices] = vel_tmp[self.wheel_indices]

        torques = self.kps * (actions_scaled + dof_err) + self.kds * (
            vel_ref - self.dof_vel
        )
        return np.clip(torques, -self.torque_limits, self.torque_limits)

    def apply_torques(self, torques):
        self.d.ctrl[:] = self._policy_to_mujoco(torques)

    def compute_observation(self):
        dof_error = (self.dof_pos - self.default_angles) * self.dof_pos_scale
        dof_error[self.wheel_indices] = 0

        dof_vel = self.dof_vel * self.dof_vel_scale
        gravity_orientation = self.get_gravity_orientation(self.quat)
        omega = self.ang_vel * self.ang_vel_scale
        self.obs[0] = self.x_vel_cmd * self.cmd_scale[0]
        self.obs[1] = self.y_vel_cmd * self.cmd_scale[1]
        self.obs[2] = self.yaw_vel_cmd * self.cmd_scale[2]
        self.obs[3:6] = omega
        self.obs[6:9] = gravity_orientation
        self.obs[9 : 9 + self.num_actions] = dof_error
        self.obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dof_vel
        self.obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action

        self.obs_hist_buf = self.obs_hist_buf[self.num_obs :]
        self.obs_hist_buf = np.concatenate((self.obs_hist_buf, self.obs), axis=-1)

    def _setup_viewer_camera(self, viewer):
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = self.render_utils.base_body_id
        viewer.cam.distance = self.config.get("camera_distance", 2.5)
        viewer.cam.elevation = self.config.get("camera_elevation", -20.0)
        viewer.cam.azimuth = self.config.get("camera_azimuth", 60.0)

    def run_sim(self):
        # 实时 16 关节速度曲线窗口（按四条腿分四张子图）
        dq_plotter = JointVelocityPlotter(
            dt=self.simulation_dt, window_s=10.0, refresh_every=10
        )
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            self._setup_viewer_camera(viewer)
            start = time.time()

            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()

                if self.counter % self.control_decimation == 0:
                    self._update_command()

                tau = self.compute_torques(self.action)
                self.apply_torques(tau)
                mujoco.mj_step(self.m, self.d)
                self.get_robot_state()

                actual_vel = np.array(
                    [self.base_lin_vel[0], self.base_lin_vel[1], self.ang_vel[2]],
                    dtype=np.float64,
                )
                self.render_utils.update(self.cmd, actual_vel, self.d)
                self.render_utils.update_external_rendering(viewer, ctype="viewer")

                self.counter += 1
                if self.counter % self.control_decimation == 0:
                    self.compute_observation()

                    out_name = self.policy.get_outputs()[0].name
                    actions = self.policy.run(
                        [out_name], {"obs": self.obs_hist_buf.reshape(1, -1)}
                    )
                    self.action = actions[0][0]

                viewer.sync()

                # 实时更新关节速度曲线
                dq_plotter.update(self.d.time, self.dof_vel)

                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        dq_plotter.close()
