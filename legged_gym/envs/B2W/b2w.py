import torch

from legged_gym.envs.M20.m20 import M20_Robot
from legged_gym.envs.B2W.b2w_config import B2W_Cfg

class B2W_Robot(M20_Robot):
    """参考 m20 任务的 B2W 轮足机器人环境 (16 DOF, 复用 M20_Robot 逻辑)。

    奖励风格与 GO2W 任务一致(参考 Wheel_Legged_Gym-main 仓库):
    实现相同的奖励函数在 M20_Robot 中已存在(lin_vel_z/ang_vel_xy/orientation/base_height/
    torques/dof_acc/action_rate/collision/termination/tracking_lin_vel/stand_still/hip_default),
    此处只覆盖实现不同或 M20 缺失的部分, 系数见 B2W_Cfg.rewards.scales。
    """
    def __init__(self, cfg: B2W_Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _reward_dof_vel(self):
        # M20 的 clone 写法: 不污染 self.dof_vel, 观测保留真实轮速(policy 可利用轮速信息);
        # 奖罚语义仍以腿关节为主, 轮速只按 0.1 权重计入
        vel = self.dof_vel.clone()
        vel[:, self.wheel_indices] *= 0.1
        return torch.sum(torch.square(vel), dim=1)

    def _reward_tracking_ang_vel(self):
        # 与原仓库一致: 无高台屏蔽
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # 与原仓库一致: 鼓励长迈步, 仅在腾空后首次触地时结算, 零指令不给奖励
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_stumble(self):
        # 与原仓库一致: 惩罚脚(轮)撞竖直面(水平接触力 > 5倍垂直接触力)
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
            5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
