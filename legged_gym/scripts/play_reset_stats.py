# 5x5 地形(难度全部 0.7)统计: 各地形类型 + 各重置原因的重置次数
import os
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from _bootstrap import setup_repo_paths

setup_repo_paths()

import numpy as np
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from legged_gym.utils.terrain import Terrain

NUM_ROWS = 5
NUM_COLS = 5
NUM_ENVS = 100  # 多个 env 会分布在同一批子地形上

# 与 terrain.py make_terrain 中分支对应的地形名
TERRAIN_NAMES = [
    "smooth_slope",   # proportions[0]
    "rough_slope",    # proportions[1]
    "stairs_down",    # [1], [2): step_height *= -1
    "stairs_up",      # [2], [3)
    "obstacles",      # [3], [4)
    "highplatform",   # [4], [5)
]


def terrain_choice_to_name(choice, proportions):
    cum = np.cumsum(proportions)
    if choice < cum[0]:
        return "smooth_slope"
    elif choice < cum[1]:
        return "rough_slope"
    elif choice < cum[3]:
        return "stairs_down" if choice < cum[2] else "stairs_up"
    elif choice < cum[4]:
        return "obstacles"
    else:
        return "highplatform"


# ---- monkey-patch Terrain.randomized_terrain: 难度固定0.7, 并记录每个子地形的choice ----
subterrain_choice = {}  # (row, col) -> choice

def randomized_terrain_recorded(self):
    for k in range(self.cfg.num_sub_terrains):
        (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
        choice = np.random.uniform(0, 1)
        difficulty = np.random.choice([0.7])
        subterrain_choice[(i, j)] = choice
        terrain = self.make_terrain(choice, difficulty)
        self.add_terrain_to_map(terrain, i, j)

Terrain.randomized_terrain = randomized_terrain_recorded


def play(args):
    args.headless = True  # 无头模式
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.terrain.num_rows = NUM_ROWS
    env_cfg.terrain.num_cols = NUM_COLS
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_hist, _ = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ---- monkey-patch check_termination: 记录重置原因 ----
    reset_cause = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # 0=无
    # 原因编码: 1=接触碰撞, 2=翻倒, 3=关节超速, 4=超时
    orig_check_termination = env.check_termination

    def check_termination_recorded():
        orig_check_termination()
        cause = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        contact = torch.any(torch.norm(
            env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)
        tipped = env.projected_gravity[:, 2] > 0.
        over_speed = torch.abs(env.dof_vel) > env.dof_vel_limits * env.cfg.rewards.soft_dof_vel_limit
        over_speed[:, env.wheel_indices] = False
        over_speed = torch.any(over_speed, dim=1)
        timeout = env.time_out_buf

        cause[contact] = 1
        cause[(~contact) & tipped] = 2
        cause[(~contact) & (~tipped) & over_speed] = 3
        cause[(~contact) & (~tipped) & (~over_speed) & timeout] = 4
        reset_cause[:] = torch.where(env.reset_buf, cause, torch.zeros_like(cause))

    env.check_termination = check_termination_recorded

    # ---- 每个 env 所在子地形 -> 地形类型名 ----
    proportions = env_cfg.terrain.terrain_proportions
    env_terrain_name = []
    for e in range(env.num_envs):
        lvl = env.terrain_levels[e].item()
        typ = env.terrain_types[e].item()
        ch = subterrain_choice.get((lvl, typ), 0.5)
        env_terrain_name.append(terrain_choice_to_name(ch, proportions))
    print("env -> terrain type:")
    for e, name in enumerate(env_terrain_name):
        print(f"  env{e:2d} (level={env.terrain_levels[e].item()}, col={env.terrain_types[e].item()}): {name}")

    # 统计: stats[terrain_name][cause] += 1
    stats = defaultdict(lambda: defaultdict(int))
    cause_names = {1: "contact碰撞", 2: "tipped翻倒", 3: "overspeed关节超速", 4: "timeout超时"}

    num_episodes = 0
    for i in range(10 * int(env.max_episode_length)):
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        actions = policy(obs.detach(), obs_hist.detach())
        obs, _, obs_hist, _, rews, dones, infos = env.step(actions.detach())

        done_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        for e in done_ids.cpu().numpy():
            stats[env_terrain_name[e]][cause_names.get(reset_cause[e].item(), "other")] += 1
        num_episodes += len(done_ids)

    # ---- 汇总输出 ----
    print("\n========== 重置统计 (5x5, 难度全部0.7) ==========")
    all_causes = list(cause_names.values())
    header = f"{'terrain':<14}" + "".join(f"{c:<16}" for c in all_causes) + "total"
    print(header)
    terrain_totals = {}
    for name in TERRAIN_NAMES + [k for k in stats if k not in TERRAIN_NAMES]:
        row = stats.get(name, {})
        total = sum(row.values())
        terrain_totals[name] = total
        print(f"{name:<14}" + "".join(f"{row.get(c, 0):<16}" for c in all_causes) + str(total))

    print(f"\n总重置次数: {num_episodes}")
    ranked = sorted(terrain_totals.items(), key=lambda kv: -kv[1])
    print("\n按地形重置次数排序:")
    for name, total in ranked:
        if total > 0:
            top_cause = max(stats[name].items(), key=lambda kv: kv[1])
            print(f"  {name:<14} {total:4d} 次 (主要原因: {top_cause[0]} {top_cause[1]} 次)")


if __name__ == '__main__':
    play(get_args())
