
# Wheel-Legged Robot Reinforcement Learning Library Dreamwaq


## 🧩 Introduction

This repository provides a **Reinforcement Learning (RL) framework** for wheel-legged robots.  
It supports **simulation-based policy training**, **Sim2Sim validation**, and **Sim2Real deployment** on real robots such as **Unitree Go2** and **custom wheel-legged platforms**.

### 🚀 Features
- ✅ Supports vision-proprioception fusion reinforcement learning  
- ✅ End-to-end pipeline for training, validation, and deployment  
- ✅ Extensible to multiple robot platforms (Go2w, m20, B2W etc.)  
- ✅ Supports custom reward functions and curriculum learning

---

## 🛠️ 使用指南
### 0. 安装依赖
python环境：3.8
#### Isaacgym 安装

```bash
pip install -e ~/isaacgym/python
```
#### rsl 
```bash
pip install -e ./rsl_rl
```
#### legged_gym
```bash
pip install -e .
```
#### 山猫m20 已经部署实物
![m20](m20_deploy.gif)
## Use

### 1. train
```bash
python legged_gym/scripts/train.py --task=m20 --headless

# b2w / go2w 任务(参考 m20 任务添加, kp/kd 取自各自原目录配置)
python legged_gym/scripts/train.py --task=b2w --headless
python legged_gym/scripts/train.py --task=go2w --headless

# CTS(Concurrent Teacher-Student, arXiv 2405.10830)版本任务
python legged_gym/scripts/train.py --task=m20_cts --headless
```

任务说明:

| 任务 | 机器人 | 初始高度 | kp/kd (hip/thigh/calf) | 轮 kp/kd | URDF |
|------|--------|----------|------------------------|----------|------|
| `m20` | 山猫 m20 | 0.60 | 80 / 2 | 0 / 0.6 | `resources/robots/m20` |
| `b2w` | Unitree B2-W | 0.70 | 400 / 20 | 0 / 5 | `resources/robots/b2w_description` |
| `go2w` | Unitree GO2-W | 0.42 | 40 / 1 | 0 / 0.5 | `resources/robots/go2w` |

b2w / go2w 均为 16 DOF(4 腿 x hip/thigh/calf/wheel), 复用 M20_Robot 全部逻辑(地形课程/高台命令/奖励/域随机化), 仅机器人参数不同。

m20_cts 以 m20 任务为基准(地形课程/高台命令/拖拽/奖励全部沿用),75% env 走 teacher(264维特权观测:域随机化参数70+4轮接触+187高度+3线速度),25% env 走 student(285维5帧历史),共享 actor/critic 并行训练,latent 蒸馏。测试与导出:

```bash
python legged_gym/scripts/play_cts.py --task=m20_cts
```

### 2. play policy

```bash
python legged_gym/scripts/play.py --task=m20 --num_envs=50
```
```bash
python legged_gym/scripts/play.py --task=go2w --num_envs=50
```
### 2.1 专项测试脚本

```bash
# 5x5 全高台地形测试(开启高台拖拽辅助)
python legged_gym/scripts/play_highplatform.py --task=m20

# 重置原因统计: 100个机器人, 5x5地形(难度全部0.7), 按地形类型/重置原因(碰撞/翻倒/关节超速/超时)汇总
python legged_gym/scripts/play_reset_stats.py --task=m20

# 单机器人: 每回合随机重置在高台不同难度(level 0~4), matplotlib 实时绘制两条前腿末端相对基座的x偏移, 相机自动跟随
python legged_gym/scripts/play_foot_offset_plot.py --task=m20
```

训练时 TensorBoard 与终端会打印每种地形的课程水平(`terrain_level_smooth_slope / rough_slope / stairs_down / stairs_up / obstacles / highplatform`)以及总水平,用于观察高台等短板地形练到了几级。

### 3. Sim2Sim (MuJoCo)

在 Isaac Gym 中训练完成后，可将导出的 ONNX 策略在 MuJoCo 中验证。需先安装依赖：

```bash
pip install mujoco onnxruntime pynput pygame pyyaml
```

#### M20

```bash
# 默认赛道场景
python deploy/deploy_mujoco/deploy_mujoco_m20.py

# 地形场景
python deploy/deploy_mujoco/deploy_mujoco_m20.py -c m20_terrain.yaml
```

#### M20 CTS

```bash
# 先用 play_cts.py 导出 policy_cts.onnx, 然后:
python deploy/deploy_mujoco/deploy_mujoco_m20_cts.py
```

#### B2W / GO2W

mjcf 来自 rl_sar (`src/robots/*_description/mjcf`), 场景为 `race_track.xml`(include 原始机器人 xml + M20 全套赛道地形: 台阶/坡道/高台/hfield 等)。训练并导出 onnx (`logs/B2W/exported/policies/policy_dwaq.onnx`, go2w 同理)后:

```bash
python deploy/deploy_mujoco/deploy_mujoco_b2w.py
python deploy/deploy_mujoco/deploy_mujoco_go2w.py
```

注意: rl_sar 原始 mjcf 的电机顺序(FR/FL/RR/RL, 轮子单列)与 qpos/策略顺序(FL/FR/RL/RR 交错)不一致, `deploy_mujoco/{B2W,GO2W}/mjcf/` 下的副本已重排为与 qpos 同序; 轮子在 mjcf 中名为 `*_wheel_joint`(训练时为 `*_foot_joint`), 由 yaml 中的 `isaac_dof_names` 按名字映射。

运行时会弹出关节速度实时曲线窗口(2x2 按腿分图,含轮子 foot 关节速度)。

可通过 `-c` 指定配置文件(位于 `deploy/deploy_mujoco/configs/`,可选 `m20.yaml`、`m20_cts.yaml`、`b2w.yaml`、`go2w.yaml`),例如：

```bash
python deploy/deploy_mujoco/deploy_mujoco_m20.py -c m20_terrain.yaml
```

#### 操控说明

| 按键 | 功能 |
|------|------|
| `6` / `7` | 增加 / 减少前进速度 |
| `8` / `9` | 增加 / 减少横向速度 |
| `-` / `=` | 增加 / 减少偏航角速度 |
| `1` | 停止（速度归零） |
| `2` / `3` / `4` | 切换 1 / 2 / 3 档 |

**档位设置**（前进最大速度）：

| 档位 | 最大 vx (m/s) |
|------|---------------|
| 1 档 | 0.5 |
| 2 档 | 1.0 |
| 3 档 | 1.5 |

连接 Xbox 手柄时，左摇杆控制移动、右摇杆控制转向；**LB / RB** 可降档 / 升档。

### problem
如果你发现自己的urdf训练出来轮子不转，而是抬腿，把urdf中的轮子的continuous改为revolute,上下限改一下-99999 99999，即可训练出轮子转的模型

如果训练开始时(还没学到东西)轮子就有很大的速度, 多半是 `init_state.pos` 初始高度远大于默认站姿的站立高度, 机器人落地冲击把近似无驱动的轮子(kp=0)踢飞。按 URDF 正运动学把初始高度设为略高于站立高度即可(如 go2w 站立约 0.408, 初始设 0.42)。

### 为什么可以训练出轮子转的模型

注意到

action_scale = 0.25

vel_scale = 5.0

故轮子的action更能表现出来故更容易训练出轮子转的模型

### 当前问题

- 上高台的动作过于剧烈，不如原始项目柔和
- 下高台时后腿容易被卡住(下高台没有经过训练)
- 零速度下的偏移问题

---

## 📋 优化方向，待实验

- [x] 添加高台的yaw误差惩罚（yaw偏移改善，上高台的动作还是不对，在课程难度低的1200轮，看起来还可以，再难度升高后动作不对劲了）
- [x] 上高台速度有一个大的冲击，限制世界系下的高速z轴速度
- [x] 将关节速度超限加入到终止条件中，上楼梯以及高台速度关节速度明显下降 2026/8/26
- [x] 高台速度方向按 env 编号奇偶固定分配(奇数正向/偶数负向,严格各半)
- [x] 高台卡住时向上拖拽辅助(外力持续10个控制步0.2s,每回合最多4次,冷却100步),防止高台卡死躺平
- [x] Sim2Sim 曲线窗口加入轮子(foot)关节速度
- [ ] 急加减速轮子会腾空
- [ ] 零速度下的轮子力矩惩罚
- [ ] 原地站立时机器人的姿态不规范
- [ ] 原地转圈依旧没有一个完整的解决方案，虽然现在的可以使用
- [ ] 上楼梯的步态收敛随机，全是在后面的可以，云深处那样大跨步也可以，全伸在前面那种不可以，一前一后的也不可以
- [ ] 下楼梯步态姿态过低，像趴在楼梯上一样
- [ ] 下高台时后腿容易被卡住
- [ ] 上高台一条腿会被卡住
- [x] go2w 以及b2w任务(2026/8/29: 训练任务 `--task=b2w/go2w` + MuJoCo sim2sim `deploy_mujoco_b2w.py/deploy_mujoco_go2w.py`, race_track 全地形场景)

---

# 致谢
https://github.com/XinLang2019/Wheel_Legged_Gym