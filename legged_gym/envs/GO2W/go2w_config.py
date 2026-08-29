from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.M20.m20_config import M20_Cfg_Yu, M20_PPO_Yu

class GO2W_Cfg(M20_Cfg_Yu):
    """参考 m20 任务的 GO2W 轮足机器人配置, kp/kd 取自原 go2w 目录配置"""
    class init_state(M20_Cfg_Yu.init_state):
        pos = [0.0, 0.0, 0.42] # x,y,z [m] 默认站姿(thigh 0.8/calf -1.5)站立高度约0.408, 略高避免初始穿模
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'FL_foot_joint': 0.0,

            'FR_hip_joint': 0.0,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,
            'FR_foot_joint': 0.0,

            'RL_hip_joint': 0.0,
            'RL_thigh_joint': 0.8,
            'RL_calf_joint': -1.5,
            'RL_foot_joint': 0.0,

            'RR_hip_joint': 0.0,
            'RR_thigh_joint': 0.8,
            'RR_calf_joint': -1.5,
            'RR_foot_joint': 0.0,
        }

    class terrain(M20_Cfg_Yu.terrain):
        terrain_proportions = [0.15, 0.15, 0.3, 0.3, 0.1]
        num_rows= 10 # number of terrain rows (levels)
    class control(M20_Cfg_Yu.control):
        # PD Drive parameters: kp/kd 来自原 legged_gym/envs/go2w/go2w_config.py
        control_type = 'P'
        stiffness = {'hip_joint': 40., 'thigh_joint': 40., 'calf_joint': 40., 'foot_joint': 0.}  # [N*m/rad]
        damping   = {'hip_joint': 1., 'thigh_joint': 1., 'calf_joint': 1., 'foot_joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25
        vel_scale = 5.0
        decimation = 4

    class asset(M20_Cfg_Yu.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w_fix.urdf'
        name = "go2w"
        foot_name = "foot"
        wheel_name = ["foot"]
        penalize_contacts_on = ["hip", "thigh", "calf", "base"]
        terminate_after_contacts_on = []
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
    class rewards:
        # 奖励系数与 Wheel_Legged_Gym-main 仓库 go2w 任务完全一致(整体替换 scales,
        # M20 特有的 highplatform_*/wheel_dof_vel/stand_still_wheel_*/run_still 等项不再激活)
        class scales:
            termination = -0.8
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -1
            ang_vel_xy = -0.05
            orientation = -0.5
            torques = -0.0003
            dof_vel = -1e-7
            dof_acc = -1e-7
            base_height = -10
            # feet_air_time = 0.5
            collision = -1
            feet_stumble = -0.1
            action_rate = -0.01
            stand_still = -0.5
            hip_default = -0.5
        only_positive_rewards = True
        tracking_sigma = 0.25
        base_height_target = 0.4
        max_contact_force = 100.
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.5

class GO2W_PPO(M20_PPO_Yu):
    class runner(M20_PPO_Yu.runner):
        experiment_name = 'GO2W'
