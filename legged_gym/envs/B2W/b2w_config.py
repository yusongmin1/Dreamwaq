from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.M20.m20_config import M20_Cfg_Yu, M20_PPO_Yu

class B2W_Cfg(M20_Cfg_Yu):
    """参考 m20 任务的 B2W 轮足机器人配置, kp/kd 取自原 b2w 目录配置"""
    class init_state(M20_Cfg_Yu.init_state):
        pos = [0.0, 0.0, 0.70] # x,y,z [m] 默认站姿站立高度约0.624, 略高避免初始穿模
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
        stiffness = {'hip_joint': 160., 'thigh_joint': 160., 'calf_joint': 160., 'foot_joint': 0.}  # [N*m/rad]
        damping   = {'hip_joint': 5., 'thigh_joint': 5., 'calf_joint': 5., 'foot_joint': 1.}     # [N*m*s/rad]
        action_scale = 0.25
        vel_scale = 5.0
        decimation = 4

    class asset(M20_Cfg_Yu.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b2w_description/urdf/b2w_description_fix.urdf'
        name = "b2w"
        foot_name = "foot"
        wheel_name = ["foot"]
        penalize_contacts_on = ["hip", "thigh", "calf", "base"]
        terminate_after_contacts_on = []
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
    class rewards(M20_Cfg_Yu.rewards):
        # 奖励系数与 GO2W 任务一致(参考 Wheel_Legged_Gym-main 仓库, 整体替换 scales,
        # M20 特有的 highplatform_*/wheel_dof_vel/stand_still_wheel_*/run_still 等项不再激活)
        class scales:
            termination = -0.8
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -1
            ang_vel_xy = -0.05
            orientation = -0.5
            torques = -0.000005
            dof_vel = -1e-7
            dof_acc = -1e-7
            base_height = -10
            feet_air_time = 0.5
            collision = -1
            feet_stumble = -0.1
            action_rate = -0.01
            stand_still = -0.5
            hip_default = -0.5
        only_positive_rewards = True
        tracking_sigma = 0.25
        base_height_target = 0.65 # 期望高度 [m] (默认站姿站立约0.624, 负载下轻微压缩)
        max_contact_force = 100.
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
class B2W_PPO(M20_PPO_Yu):
    class runner(M20_PPO_Yu.runner):
        experiment_name = 'B2W'
