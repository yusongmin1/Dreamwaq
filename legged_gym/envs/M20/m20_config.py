from legged_gym.envs.base.base_config import BaseConfig

class M20_Cfg_Yu(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 57
        num_obs_hist = 5
        num_privileged_obs = 57+3+187 
        num_latent_dims=16
        num_explicit_dims=3
        num_history_obs=num_obs_hist*num_observations
        num_actions = 16
        episode_length_s = 20 # episode length in seconds
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts=True
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 40 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, highplatform]
        terrain_proportions = [0.1, 0.15, 0.25, 0.25, 0.1,0.15]
        # terrain_proportions =   [0., 0., 0., 0., 0.,1.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
    class commands:
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0,1.2] # min max [m/s]
            lin_vel_y = [-0.6,0.6]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
        class highplatform:
            lin_vel_x_neg = [-1.0, -0.35]  # 反向 min max [m/s]
            lin_vel_x_pos = [0.35, 1.0]    # 正向 min max [m/s]
            lin_vel_y = [0.0, 0.0]   # forward only
            heading = [-0.0, 0.0]  # -10 deg to 10 deg [rad], resampled to commands[:, 3]
    class init_state:
        pos = [0.0, 0.0, 0.60] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'fl_hipx_joint': 0.0,   # [rad]
            'fl_hipy_joint': -0.6,   # [rad]
            'fl_knee_joint': 1.0,  # [rad]
            'fl_wheel_joint': -0.0,   # [rad]

            'fr_hipx_joint': 0.0,     # [rad]
            'fr_hipy_joint': -0.6,   # [rad]
            'fr_knee_joint': 1.0,     # [rad]
            'fr_wheel_joint': 0.0,   # [rad]

            'hl_hipx_joint': 0.0,   # [rad]
            'hl_hipy_joint': 0.6,    # [rad]
            'hl_knee_joint': -1.0,  # [rad]
            'hl_wheel_joint': 0.0,    # [rad]

            'hr_hipx_joint': 0.0,   # [rad]
            'hr_hipy_joint': 0.6,    # [rad]
            'hr_knee_joint': -1.0,  # [rad]
            'hr_wheel_joint': 0.0,    # [rad]
        }

    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hipx_joint': 80.,'hipy_joint': 80.,'knee_joint': 80.,'wheel_joint': 0.}  # [N*m/rad]
        damping   = {'hipx_joint': 2.0,'hipy_joint': 2.0,'knee_joint': 2.0,'wheel_joint': 0.6}     # [N*m*s/rad]
        action_scale = 0.25
        vel_scale = 5.0
        decimation = 4
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/m20/urdf/m20.urdf'
        name = "m20"
        foot_name = "wheel"
        wheel_name =["wheel"] 
        penalize_contacts_on = ["hipx", "hipy","knee","base"]
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2,1.25]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.6
        upward_drag = True
        upward_drag_cmd_threshold = 0.5      # command planar speed [m/s]
        upward_drag_vel_threshold = 0.2      # actual base planar speed [m/s]
        upward_drag_z_force = 10000.0         # upward force [N]
        upward_drag_z_vel = 1.0              # upward velocity impulse [m/s]
        upward_drag_forward_offset = 0.1     # force application point, 10cm in front of base [m]
        upward_drag_max_count = 4           # max drag attempts per episode
        upward_drag_cooldown_steps = 100     # steps between drag attempts
        randomize_base_mass = True
        added_base_mass_range = [-1,5]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.85, 1.15]  
        damping_multiplier_range = [0.85, 1.15]     
        torque_multiplier_range=[0.85, 1.15]  

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        delay = True

    class rewards:
        class scales:
            tracking_lin_vel = 2.0 # 惩罚当前机器人在X、Y方向速度与命令不一致
            tracking_ang_vel = 1.0 # 惩罚当前机器人在角度转向速度与命令不一致
            lin_vel_z = -2 # 惩罚机器人在Z轴上的速度 对应现象为机器人上下起伏很大
            ang_vel_xy = -0.05 # 惩罚机器人在X轴和Y轴上的角速度 对应现象为遏制机器人左右晃动和前后晃动
            orientation = -0.2 # 强烈鼓励机器人与初始姿态的基座方向一致
            base_height=-10.0
            torques = -0.00005#
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            collision = -1.
            # stumble = -0.1
            action_rate = -0.01
            stand_still=-0.5
            dof_pos_limits = -5.0
            dof_vel_limits=-1.0
            hip_default = -0.5
            run_still=-0.05
            highplatform_yaw = -2.0
            highplatform_world_vel = -10.0 # 高台世界系线速度>0.75时大惩罚
            highplatform_leg_dof_vel = -1.0 # 高台上腿关节(不含轮子)速度超限的大惩罚
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        highplatform_world_speed_limit = 0.75 # [m/s] world-frame linear speed limit on highplatform
        highplatform_leg_dof_vel_limit = 16.0 # [rad/s] 高台上腿关节(不含轮子)速度限值
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.9
        base_height_target = 0.52
        max_contact_force = 200. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class M20_PPO_Yu( BaseConfig ):
    seed = 1
    runner_class_name = 'DreamWaQRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_obs=57
    class runner:
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        num_steps_per_env = 24 # per iteration
        run_name = ''
        experiment_name = 'M20'
        save_interval = 100 
        max_iterations = 300000
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt