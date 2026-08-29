from legged_gym.envs.M20.m20 import M20_Robot
from legged_gym.envs.B2W.b2w_config import B2W_Cfg

class B2W_Robot(M20_Robot):
    """参考 m20 任务的 B2W 轮足机器人环境 (16 DOF, 结构与 M20 一致, 复用 M20_Robot 逻辑)"""
    def __init__(self, cfg: B2W_Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
