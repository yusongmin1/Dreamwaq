# M20 CTS 任务配置
# 环境继承 M20_Cfg_Yu(地形课程/高台命令/奖励/域随机化全部沿用),
# 只覆盖特权观测维度; 训练配置为 CTS(OnPolicyRunnerCTS)专属。
from legged_gym.envs.M20.m20_config import M20_Cfg_Yu
from legged_gym.envs.base.base_config import BaseConfig


class M20_Cts_Cfg_Yu(M20_Cfg_Yu):
    class env(M20_Cfg_Yu.env):
        # student 本体观测 57 维不变; 特权观测换为 CTS 版本:
        # 域随机化参数70 + 4轮接触 + 187高度 + 3线速度 = 264
        num_privileged_obs = 264
        # critic 观测 = 本体57 + 特权264
        num_critic_obs = 321


class M20_Cts_PPO_Yu(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunnerCTS'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        teacher_encoder_hidden_dims = [512, 256]
        student_encoder_hidden_dims = [512, 256]
        latent_dim = 32
        norm_type = 'l2norm'
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        student_encoder_learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        teacher_env_ratio = 0.75  # 75% env 为 teacher, 每 4 个 env 1 个 student

    class runner:
        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'CTS'
        num_steps_per_env = 24
        run_name = ''
        experiment_name = 'M20_CTS'
        save_interval = 100
        max_iterations = 300000
        resume = False
        load_run = -1  # -1 means last run
        checkpoint = -1
        resume_path = None
