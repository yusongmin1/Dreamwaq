from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.M20.m20_config import M20_Cfg_Yu, M20_PPO_Yu

class B2W_Cfg(M20_Cfg_Yu):
    """参考 m20 任务的 B2W 轮足机器人配置, kp/kd 取自原 b2w 目录配置"""
    class init_state(M20_Cfg_Yu.init_state):
        pos = [0.0, 0.0, 0.7] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'FL_thigh_joint': 0.8,   # [rad]
            'FL_calf_joint': -1.5,  # [rad]
            'FL_foot_joint': 0.0,   # [rad]

            'FR_hip_joint': 0.0,     # [rad]
            'FR_thigh_joint': 0.8,   # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'FR_foot_joint': 0.0,   # [rad]

            'RL_hip_joint': 0.0,   # [rad]
            'RL_thigh_joint': 0.8,    # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'RL_foot_joint': 0.0,    # [rad]

            'RR_hip_joint': 0.0,   # [rad]
            'RR_thigh_joint': 0.8,    # [rad]
            'RR_calf_joint': -1.5,  # [rad]
            'RR_foot_joint': 0.0,    # [rad]
        }

    class control(M20_Cfg_Yu.control):
        # PD Drive parameters: kp/kd 来自原 legged_gym/envs/b2w/b2w_config.py
        control_type = 'P'
        stiffness = {'hip_joint': 400., 'thigh_joint': 400., 'calf_joint': 400., 'foot_joint': 0.}  # [N*m/rad]
        damping   = {'hip_joint': 20., 'thigh_joint': 20., 'calf_joint': 20., 'foot_joint': 5.}     # [N*m*s/rad]
        action_scale = 0.25
        vel_scale = 10.0
        decimation = 4

    class asset(M20_Cfg_Yu.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b2w_description/urdf/b2w_description_fix.urdf'
        name = "b2w"
        foot_name = "foot"
        wheel_name = ["foot"]
        penalize_contacts_on = ["hip", "thigh", "calf", "base"]
        terminate_after_contacts_on = []

    class rewards(M20_Cfg_Yu.rewards):
        base_height_target = 0.5

class B2W_PPO(M20_PPO_Yu):
    class runner(M20_PPO_Yu.runner):
        experiment_name = 'B2W'
