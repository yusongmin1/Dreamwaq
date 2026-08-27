# 单机器人: 每回合随机重置在高台/上楼梯/下楼梯的不同难度(level)上,
# matplotlib 实时绘制两条前腿末端(前轮中心 fl_wheel/fr_wheel)相对基座的机体系 x 偏移
import os
import sys
import random

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from _bootstrap import setup_repo_paths

setup_repo_paths()

import numpy as np
import isaacgym
from isaacgym.torch_utils import quat_rotate_inverse, quat_apply
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

import matplotlib
import matplotlib.pyplot as plt

NUM_ROWS = 5   # level 0~4 -> difficulty 0~0.8, 台阶/高台高度不同
HISTORY = 300  # 绘图窗口保留的最近步数


def type_cols(proportions, num_cols, tname):
    """按 proportions 累积区间, 返回该地形类型占据的列号列表"""
    cum = np.cumsum(proportions)
    cols = []
    for j in range(num_cols):
        c = j / num_cols + 0.001
        if c < cum[0]:
            name = "smooth_slope"
        elif c < cum[1]:
            name = "rough_slope"
        elif c < cum[3]:
            name = "stairs_down" if c < cum[2] else "stairs_up"
        elif c < cum[4]:
            name = "obstacles"
        else:
            name = "highplatform"
        if name == tname:
            cols.append(j)
    return cols


def play(args):
    # 保持 viewer 画面(不加 --headless 时可以看到机器人), matplotlib 另开窗口画曲线
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = NUM_ROWS
    # 用课程地形: 行=难度等级(高度不同), 列=地形类型
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = NUM_ROWS - 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 地形已按课程方式生成; 之后手动控制重置位置, 关闭自动课程更新
    env.cfg.terrain.curriculum = False

    obs, obs_hist, _ = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # 只在高台上随机(level 0~4 -> 不同高度)
    num_cols = env_cfg.terrain.num_cols
    choices = {
        'highplatform': type_cols(env_cfg.terrain.terrain_proportions, num_cols, 'highplatform'),
    }
    print("可选列:", choices)
    terrain_name = 'highplatform'

    def random_reset():
        """随机选地形类型和难度等级, 把机器人重置过去"""
        nonlocal terrain_name
        terrain_name = random.choice(list(choices.keys()))
        col = random.choice(choices[terrain_name])
        level = random.randint(0, NUM_ROWS - 1)
        env.terrain_levels[0] = level
        env.terrain_types[0] = col
        env.env_origins[0] = env.terrain_origins[level, col]
        env.reset_idx(torch.tensor([0], device=env.device))
        zh = env.env_origins[0, 2].item()
        print(f"[reset] {terrain_name}  col={col}  level={level}  平台高={zh:.2f} m")
        ax.set_title(f"{terrain_name}  level={level}  h={zh:.2f} m")

    # ---- matplotlib 实时绘图 ----
    if matplotlib.get_backend().lower() == 'agg':
        print("警告: matplotlib 后端是 Agg(无显示), 曲线窗口将无法显示")
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel('step')
    ax.set_ylabel('前腿末端相对基座 x 偏移 [m]')
    ax.grid(True)
    line_fl, = ax.plot([], [], label='FL front-left', color='tab:blue')
    line_fr, = ax.plot([], [], label='FR front-right', color='tab:orange')
    ax.legend(loc='upper right')

    fl_hist, fr_hist = [], []
    # 前腿末端 = 前两个轮子(URDF 顺序 fl_wheel, fr_wheel)
    front_feet = env.feet_indices[:2]

    random_reset()

    for i in range(100 * int(env.max_episode_length)):
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        actions = policy(obs.detach(), obs_hist.detach())
        obs, _, obs_hist, _, rews, dones, infos = env.step(actions.detach())

        # 前腿末端位置 -> 基座系, 取 x 偏移
        foot_pos = env.rigid_body_states[0, front_feet, 0:3]          # (2,3)
        base_pos = env.root_states[0, 0:3]
        quat2 = env.base_quat[0].unsqueeze(0).expand(2, -1)           # (2,4)
        p_rel = quat_rotate_inverse(quat2, foot_pos - base_pos)       # (2,3)
        fl_hist.append(p_rel[0, 0].item())
        fr_hist.append(p_rel[1, 0].item())

        # 相机跟随机器人: 始终位于机体正后上方, 看着基座(转向也跟着转)
        fwd = quat_apply(env.base_quat[0].unsqueeze(0),
                         torch.tensor([1., 0., 0.], device=env.device)).cpu().numpy()
        bp = base_pos.cpu().numpy()
        cam_pos = bp - 3.5 * fwd + np.array([0.0, 0.0, 1.5])
        env.set_camera(cam_pos, bp)

        # 回合结束 -> 随机换地形和高度
        if env.reset_buf[0]:
            random_reset()
            fl_hist.clear()
            fr_hist.clear()

        if i % 2 == 0:  # 每2步刷新一次, 降低绘图开销
            n = min(len(fl_hist), HISTORY)
            xs = np.arange(n)
            line_fl.set_data(xs, fl_hist[-n:])
            line_fr.set_data(xs, fr_hist[-n:])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)  # 处理 GUI 事件, 否则窗口白屏不刷新

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    play(get_args())
