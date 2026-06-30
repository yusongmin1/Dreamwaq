
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
```

```bash
python legged_gym/scripts/train.py --task=go2w --headless
```

```bash
python legged_gym/scripts/play.py --task=b2w --headless
```

### 2. play policy

```bash
python legged_gym/scripts/play.py --task=m20 --num_envs=50
```

```bash
python legged_gym/scripts/play.py --task=go2w --num_envs=50
```

```bash
python legged_gym/scripts/play.py --task=b2w --num_envs=50
```

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

#### M20 Big

```bash
python deploy/deploy_mujoco/deploy_mujoco.py
```

#### Go2

```bash
python deploy/deploy_mujoco/deploy_mujoco_go2.py
```

可通过 `-c` 指定配置文件，例如：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py -c m20_big.yaml
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

- [x] 上高台的速度降低 
- [x] 添加高台的yaw误差惩罚（yaw偏移改善，上高台的动作还是不对，在课程难度低的1200轮，看起来还可以，再难度升高后动作不对劲了）
- [ ] 上高台速度有一个大的冲击，限制世界系下的高速z轴速度
- [ ] 高台的高速度关节惩罚
- [ ] 零速度下的轮子力矩惩罚
- [ ] 全部地形改为粗糙能够让机器人学会优雅转圈吗
 

---

# 致谢
https://github.com/XinLang2019/Wheel_Legged_Gym