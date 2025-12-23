# 多智能体强化学习 (MARL) 框架设计文档

## 概述

本文档描述了在RLinf框架基础上设计的多智能体强化学习(MARL)训练框架。该框架支持多种MARL算法和训练范式，可以灵活地处理多智能体环境下的强化学习训练。

## 架构设计

### 核心组件

1. **MultiAgentEnvWorker** (`rlinf/workers/env/multi_agent_env_worker.py`)
   - 多智能体环境Worker
   - 处理多智能体格式的观察、动作、奖励
   - 支持单智能体环境的自动适配
   - 管理多智能体环境的交互流程

2. **MultiAgentActorGroup** (`rlinf/runners/marl_runner.py`)
   - 多智能体Actor组管理类
   - 支持独立策略、共享策略、部分共享策略
   - 协调多个智能体的策略网络训练
   - 处理权重同步和优势计算

3. **MARLRunner** (`rlinf/runners/marl_runner.py`)
   - MARL训练运行器
   - 协调环境、rollout和actor的交互
   - 实现完整的MARL训练循环
   - 支持多种MARL算法

4. **MARL算法模块** (`rlinf/algorithms/marl_algorithms.py`)
   - MAPPO优势计算
   - IPPO优势计算
   - MADDPG优势计算
   - QMIX值分解（基础实现）
   - 多智能体策略损失和价值损失计算

### 数据流

```
环境 (MultiAgentEnvWorker)
    ↓ (多智能体观察)
Rollout Worker
    ↓ (多智能体动作)
环境
    ↓ (多智能体奖励、done)
Actor Group (MultiAgentActorGroup)
    ↓ (计算优势、回报)
训练更新
```

## 支持的算法

### 1. MAPPO (Multi-Agent PPO)
- **特点**: 集中训练分散执行(CTDE)
- **优势计算**: 使用集中式价值函数，但每个智能体独立计算优势
- **适用场景**: 需要利用全局信息但执行时分散的场景

### 2. IPPO (Independent PPO)
- **特点**: 完全独立学习
- **优势计算**: 每个智能体完全独立，不共享任何信息
- **适用场景**: 智能体间交互较少或异构智能体

### 3. MADDPG
- **特点**: 多智能体DDPG，适用于连续动作空间
- **优势计算**: 基于Q-learning的优势计算
- **适用场景**: 连续动作空间的多智能体任务

### 4. QMIX
- **特点**: 值分解方法
- **优势计算**: 通过混合网络学习全局Q值
- **适用场景**: 需要值分解的协作任务

## 设计特点

### 1. 灵活的策略共享

支持三种策略共享模式：
- **独立策略 (Independent)**: 每个智能体有独立的策略网络
- **共享策略 (Shared)**: 所有智能体共享同一个策略网络
- **部分共享 (Partial)**: 部分智能体共享策略网络（需要自定义实现）

### 2. 环境适配机制

- 自动检测环境返回格式（单智能体 vs 多智能体）
- 单智能体环境自动适配为多智能体格式
- 支持自定义奖励分配策略

### 3. 集中训练分散执行 (CTDE)

- 支持使用全局状态进行训练
- 支持集中式critic
- 执行时每个智能体独立决策

### 4. 可扩展的算法接口

- 通过注册机制添加新算法
- 统一的优势计算接口
- 灵活的策略损失和价值损失计算

## 使用方式

### 基本配置

```yaml
marl:
  num_agents: 2
  agent_ids: ["agent_0", "agent_1"]
  algorithm: "mappo"
  policy_sharing: "independent"
  use_global_state: true
  centralized_critic: true
```

### 代码示例

```python
from rlinf.runners import MARLRunner, MultiAgentActorGroup
from rlinf.workers.env.multi_agent_env_worker import MultiAgentEnvWorker

# 创建Actor组
actor_group = MultiAgentActorGroup(cfg, actor_workers)

# 创建环境Worker
env_group = MultiAgentEnvWorker.create_group(cfg).launch(...)

# 创建Runner
runner = MARLRunner(cfg, actor_group, rollout, env_group)
runner.init_workers()
runner.run()
```

## 文件结构

```
rlinf/
├── workers/
│   └── env/
│       └── multi_agent_env_worker.py    # 多智能体环境Worker
├── runners/
│   └── marl_runner.py                  # MARL Runner和ActorGroup
├── algorithms/
│   └── marl_algorithms.py              # MARL算法实现
└── examples/
    └── marl/
        ├── README.md                    # 使用文档
        ├── main_marl.py                # 示例主程序
        └── config/
            └── mappo_example.yaml      # 配置示例
```

## 扩展指南

### 添加新的MARL算法

1. 在`marl_algorithms.py`中实现优势计算函数：

```python
@register_advantage("your_algorithm")
def compute_your_algorithm_advantages(...):
    # 实现算法逻辑
    return advantages, returns
```

2. 在配置文件中使用：

```yaml
algorithm:
  advantage_type: "your_algorithm"
```

### 自定义通信机制

1. 在Actor Worker中添加通信逻辑
2. 在环境交互时传递消息
3. 在配置中启用通信：

```yaml
marl:
  communication:
    enabled: true
    method: "your_method"
```

## 注意事项

1. **环境接口**: 多智能体环境需要返回字典格式的观察、奖励和done标志
2. **策略共享**: 共享策略可以显著减少内存和计算开销，但需要智能体是同构的
3. **性能优化**: 使用CTDE可以利用全局信息提升训练效率
4. **通信机制**: 当前版本支持基础通信，复杂通信需要自定义实现

## 未来改进方向

1. **更完善的通信机制**: 支持更复杂的智能体间通信
2. **更多算法支持**: 添加更多MARL算法（如COMA, VDN等）
3. **异构智能体支持**: 更好地支持异构智能体场景
4. **性能优化**: 进一步优化多智能体训练的性能
5. **可视化工具**: 添加多智能体训练的可视化工具

## 参考资源

- [MARL算法综述](https://arxiv.org/abs/2006.07869)
- [MAPPO论文](https://arxiv.org/abs/2103.01955)
- [MADDPG论文](https://arxiv.org/abs/1706.02275)
- [QMIX论文](https://arxiv.org/abs/1803.11485)

