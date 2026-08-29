# M20 CTS (Concurrent Teacher-Student) 任务环境
# 以 m20 任务为基准(地形课程/高台命令/拖拽辅助/奖励全部继承),
# 只重写观测: student 用 57 维本体观测 + 5 帧历史; teacher encoder 用
# 264 维特权观测(域随机化参数70 + 4轮接触 + 187高度 + 3线速度)。
# 轮足处理: 轮子 dof_pos 在观测中置零(继承 m20), 轮速(×0.05)保留在观测。
from legged_gym.envs.M20.m20 import M20_Robot
from legged_gym.envs.M20_CTS.m20_cts_config import M20_Cts_Cfg_Yu
import torch


class M20_Cts_Robot(M20_Robot):

    def __init__(self, cfg: M20_Cts_Cfg_Yu, sim_params, physics_engine, sim_device, headless):
        # 记录每个 env 的基座质量/质心偏移(m20 中只存最后一个 env 的值, 这里按 env 记录)
        self._mass_list = []
        self._com_list = []
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # CTS 附加接口
        self.num_critic_obs = self.cfg.env.num_critic_obs            # 321 = 57 + 264
        self.num_history_obs = self.cfg.env.num_history_obs          # 285
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs,
                                          device=self.device, dtype=torch.float)

        # per-env 基座质量/质心(特权观测用)
        if len(self._mass_list) == self.num_envs:
            self.priv_added_base_masses = torch.tensor(
                self._mass_list, device=self.device, dtype=torch.float).unsqueeze(1)
            self.priv_added_base_com = torch.tensor(
                self._com_list, device=self.device, dtype=torch.float)
        else:
            self.priv_added_base_masses = torch.zeros(self.num_envs, 1, device=self.device)
            self.priv_added_base_com = torch.zeros(self.num_envs, 3, device=self.device)

    def _process_rigid_shape_props(self, env_id, props):
        props = super()._process_rigid_shape_props(env_id, props)
        # super() 每次会覆盖 added_base_masses/added_base_com 为当前 env 的采样值, 记录下来
        if hasattr(self, 'added_base_masses'):
            self._mass_list.append(float(self.added_base_masses.flatten()[-1]))
        else:
            self._mass_list.append(0.0)
        if hasattr(self, 'added_base_com'):
            self._com_list.append(self.added_base_com.flatten().tolist())
        else:
            self._com_list.append([0.0, 0.0, 0.0])
        return props

    def compute_observations(self):
        # ---- student 本体观测(与 m20 完全一致) ----
        self.obs_hist_buf = self.obs_hist_buf[:, self.num_obs:]
        self.obs_hist_buf = torch.cat((self.obs_hist_buf, self.obs_buf), dim=-1)
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:, self.wheel_indices] = 0
        self.dof_pos[:, self.wheel_indices] = 0
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                             -1, 1.) * self.obs_scales.height_measurements

        obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.dof_err * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        # ---- teacher 特权观测: 域随机化参数(70) + 4轮接触 + 187高度 + 3线速度 = 264 ----
        contact_mask = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.).float()
        domain_rand_info = torch.cat((
            self.friction_coeffs,            # 1
            self.restitution_coeffs,         # 1
            self.priv_added_base_masses,     # 1
            self.priv_added_base_com,        # 3
            self.p_gains_multiplier,         # 16
            self.d_gains_multiplier,         # 16
            self.torques_multiplier,         # 16
            self.motor_zero_offsets,         # 16
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            domain_rand_info,                                   # 70
            contact_mask,                                       # 4
            heights,                                            # 187
            self.base_lin_vel * self.obs_scales.lin_vel,        # 3
        ), dim=-1)

        # ---- critic 观测: 干净本体观测 + 特权 (57+264=321) ----
        self.critic_obs_buf = torch.cat((obs, self.privileged_obs_buf), dim=-1)

        self.obs_buf = obs
        self.vel_buf = self.base_lin_vel
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def step(self, actions):
        # 返回 (obs, privileged, obs_hist, critic_obs, rew, dones, infos)
        obs, _, obs_hist, _, rews, dones, infos = super().step(actions)
        clip_obs = self.cfg.normalization.clip_observations
        critic_obs_buf = torch.clip(self.critic_obs_buf, -clip_obs, clip_obs)
        return obs, self.privileged_obs_buf, obs_hist, critic_obs_buf, rews, dones, infos

    def get_observations(self):
        # CTS runner 需要 4 元组
        return self.obs_buf, self.privileged_obs_buf, self.obs_hist_buf, self.critic_obs_buf
