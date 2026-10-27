
# 轮足机器人强化学习库 DreamWaQ


## 🧩 项目介绍

本仓库提供一套面向**轮足(wheel-legged)机器人**的**强化学习(RL)训练框架**, 基于 DreamWaQ 算法(隐式世界模型 + VAE 速度估计蒸馏)。

完整支持三个阶段的全流程:

- **仿真训练** (Isaac Gym): 大规模并行环境 + 地形课程学习
- **Sim2Sim 验证** (MuJoCo): ONNX 策略在赛道/楼梯等场景中离线验证
- **Sim2Real 实机部署** (rl_sar): 已在山猫 m20 实机上完成部署

### 🚀 功能特性
- ✅ 基于 DreamWaQ 的隐式世界模型训练, 观测历史经 VAE 蒸馏速度估计
- ✅ 训练 → Sim2Sim 验证 → 实机部署 的端到端流水线
- ✅ 多机器人平台可扩展(山猫 m20 / Unitree GO2-W / Unitree B2-W, 16 DOF 统一结构)
- ✅ 自定义奖励函数与地形/命令双课程学习
- ✅ 高台/楼梯/坡道/hfield 等全套地形课程

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
山猫 m20 已完成实机部署:

![m20](m20_deploy.gif)
### 1. 训练 train
```bash
python legged_gym/scripts/train.py --task=m20 --headless

# b2w / go2w 任务(参考 m20 任务添加, kp/kd 取自各自原目录配置)
python legged_gym/scripts/train.py --task=b2w --headless
python legged_gym/scripts/train.py --task=go2w --headless

```

任务说明:

| 任务 | 机器人 | 初始高度 | kp/kd (hip/thigh/calf) | 轮 kp/kd | URDF |
|------|--------|----------|------------------------|----------|------|
| `m20` | 山猫 m20 | 0.60 | 80 / 2 | 0 / 0.6 | `resources/robots/m20` |
| `b2w` | Unitree B2-W | 0.65 | 160 / 5 | 0 / 1 | `resources/robots/b2w_description` |
| `go2w` | Unitree GO2-W | 0.42 | 40 / 1 | 0 / 0.5 | `resources/robots/go2w` |

b2w / go2w 均为 16 DOF(4 腿 x hip/thigh/calf/wheel), 复用 M20_Robot 全部逻辑(地形课程/高台命令/奖励/域随机化), 仅机器人参数不同。

m20_cts 以 m20 任务为基准(地形课程/高台命令/奖励全部沿用),75% env 走 teacher(264维特权观测:域随机化参数70+4轮接触+187高度+3线速度),25% env 走 student(285维5帧历史),共享 actor/critic 并行训练,latent 蒸馏。测试与导出:


### 2. 策略回放 play

```bash
python legged_gym/scripts/play.py --task=m20 --num_envs=50
```
```bash
python legged_gym/scripts/play.py --task=b2w --num_envs=50
```
```bash
python legged_gym/scripts/play.py --task=go2w --num_envs=50
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
```

#### B2W / GO2W

mjcf 来自 rl_sar (`src/robots/*_description/mjcf`), 场景为 `race_track.xml`(include 原始机器人 xml + M20 全套赛道地形: 台阶/坡道/高台/hfield 等)。训练并导出 onnx (`logs/B2W/exported/policies/policy_dwaq.onnx`, go2w 同理)后:

```bash
python deploy/deploy_mujoco/deploy_mujoco_b2w.py
```

```bash
python deploy/deploy_mujoco/deploy_mujoco_go2w.py
```

#### 关节速度曲线（可选，默认关闭）

实时绘制 16 个关节速度（按 FL/FR/RL/RR 分四张子图）。默认不开启；需要时任选一种方式：

```bash
# 命令行开启
python deploy/deploy_mujoco/deploy_mujoco_b2w.py --plot_joint_vel
python deploy/deploy_mujoco/deploy_mujoco_m20.py --plot_joint_vel
python deploy/deploy_mujoco/deploy_mujoco_go2w.py --plot_joint_vel
```

或在对应 yaml（如 `deploy/deploy_mujoco/configs/b2w.yaml`）中设置：

```yaml
plot_joint_vel: true
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

### 4. 常见问题
如果你发现自己的urdf训练出来轮子不转，而是抬腿，把urdf中的轮子的continuous改为revolute,上下限改一下-99999 99999，即可训练出轮子转的模型

如果训练开始时(还没学到东西)轮子就有很大的速度, 多半是 `init_state.pos` 初始高度远大于默认站姿的站立高度, 机器人落地冲击把近似无驱动的轮子(kp=0)踢飞。按 URDF 正运动学把初始高度设为略高于站立高度即可(如 go2w 站立约 0.408, 初始设 0.42)。

#### 为什么可以训练出轮子转的模型

注意到

action_scale = 0.25

vel_scale = 5.0

故轮子的action更能表现出来故更容易训练出轮子转的模型


## 📋 TODO

- [x] 上高台偏移，添加上高台的偏移惩罚以及偏移一段数值终止，好很多
- [x] 较大的速度超限惩罚，上楼梯的速度明显下降 2026/9/2
- [x] go2w 以及b2w任务(2026/8/29: 训练任务 `--task=b2w/go2w` + MuJoCo sim2sim `deploy_mujoco_b2w.py/deploy_mujoco_go2w.py`, race_track 全地形场景)，解决之前的bug，勉强能可以使用了
- [ ] 急加减速轮子会腾空，会扑出去，鲁棒性不足
- [ ] 原地站立以及高速运动时机器人的姿态不规范
- [ ] 原地转圈依旧没有一个完整的解决方案，
- [ ] 下高台时后腿容易被卡住
- [ ] 上高台超过90cm的时候前腿叉开上的高台，而不是滑上高台做借力

