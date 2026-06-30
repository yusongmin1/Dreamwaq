from legged_gym.envs.base.base_config import BaseConfig

class GO2W_Cfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 57
        num_obs_hist = 5
        num_privileged_obs = 57 + 3 + 187
        num_latent_dims = 16
        num_explicit_dims = 3
        num_history_obs = num_obs_hist * num_observations
        num_actions = 16
        episode_length_s = 20
        env_spacing = 3.
        send_timeouts = True

    class terrain:
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, highplatform]
        terrain_proportions = [0.1, 0.1, 0.3, 0.25, 0.1, 0.15]
        slope_treshold = 0.75

    class commands:
        curriculum = False
        max_curriculum = 1.2
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        class ranges:
            lin_vel_x = [-1.0, 1.2]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]
        class highplatform:
            lin_vel_x = [0.5, 1.1]
            lin_vel_y = [0.0, 0.0]
            heading = [-0.174533, 0.174533]  # -10 deg to 10 deg [rad]

    class init_state:
        pos = [0.0, 0.0, 0.4]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
            'FL_foot_joint': 0.0,
            'RL_foot_joint': 0.0,
            'FR_foot_joint': 0.0,
            'RR_foot_joint': 0.0,
        }

    class control:
        control_type = 'P'
        stiffness = {'hip_joint': 40., 'thigh_joint': 40., 'calf_joint': 40., 'foot_joint': 0.}
        damping = {'hip_joint': 1., 'thigh_joint': 1., 'calf_joint': 1., 'foot_joint': 0.5}
        action_scale = 0.25
        vel_scale = 10.0
        decimation = 4

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf'
        name = "go2w"
        foot_name = "foot"
        wheel_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.6
        upward_drag = False
        upward_drag_cmd_threshold = 0.5
        upward_drag_vel_threshold = 0.2
        upward_drag_z_force = 10000.0
        upward_drag_z_vel = 1.0
        upward_drag_forward_offset = 0.1
        upward_drag_max_count = 4
        upward_drag_cooldown_steps = 100
        randomize_base_mass = True
        added_base_mass_range = [-1, 5]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.85, 1.15]
        damping_multiplier_range = [0.85, 1.15]
        torque_multiplier_range = [0.85, 1.15]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 3]

    class rewards:
        class scales:
            termination = -0.8
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2
            ang_vel_xy = -0.05
            orientation = -0.2
            base_height = -5.0
            torques = -0.000005
            dof_vel = -1e-6
            dof_acc = -2.5e-7
            collision = -1.
            action_rate = -0.01
            stand_still = -0.5
            dof_pos_limits = -5.0
            hip_default = -0.5
            run_still = -0.05
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.5
        max_contact_force = 200.

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
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class GO2W_PPO_Cfg(BaseConfig):
    seed = 1
    runner_class_name = 'DreamWaQRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_obs = 57

    class runner:
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        num_steps_per_env = 24
        run_name = ''
        experiment_name = 'rough_go2w'
        save_interval = 100
        max_iterations = 300000
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
