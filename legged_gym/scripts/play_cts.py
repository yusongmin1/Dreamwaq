# CTS 任务测试脚本: 加载训练好的 m20_cts 模型, 用 student 路径推理并导出 jit 策略
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from _bootstrap import setup_repo_paths

setup_repo_paths()

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_cts, task_registry
import torch


def play(args, *, export_policy=True, use_teacher=False):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override some parameters for testing.
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 60)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, privileged_buf, obs_hist, critic_obs_buf = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
    )

    if use_teacher:
        # teacher 路径: 用特权观测(域随机化+接触+高度+线速度)过 teacher_encoder
        model = ppo_runner.alg.model

        def policy(obs, privileged_obs):
            latent = model.teacher_encoder(privileged_obs)
            return model.actor(torch.cat([latent, obs], dim=-1))

        print('>>> 使用 TEACHER 策略(特权观测)进行测试 <<<')
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        print('>>> 使用 STUDENT 策略(历史观测)进行测试 <<<')

    if export_policy:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name,
            'exported', 'policies',
        )
        export_policy_as_cts(ppo_runner.alg.model, path, env.num_obs)
        print('Exported CTS policy (student) to:', path)

    for i in range(10 * int(env.max_episode_length)):
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        if use_teacher:
            actions = policy(obs.detach(), privileged_buf.detach())
        else:
            actions = policy(obs.detach(), obs_hist.detach())
        obs, privileged_buf, obs_hist, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    # --teacher 标志先取出, 避免 get_args 的 argparse 报未知参数
    use_teacher = '--teacher' in sys.argv
    if use_teacher:
        sys.argv.remove('--teacher')
    play(get_args(), use_teacher=use_teacher)
