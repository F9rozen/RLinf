# 多智能体强化学习 (MARL) 训练框架

本目录包含多智能体强化学习训练框架的使用示例和配置。

## 概述

MARL框架支持多种多智能体强化学习算法和训练范式：

### 支持的算法

1. **IPPO (Independent PPO)**: 独立学习，每个智能体独立训练自己的策略
2. **MAPPO (Multi-Agent PPO)**: 集中训练分散执行，使用集中式价值函数
3. **MADDPG**: 多智能体DDPG，适用于连续动作空间
4. **QMIX**: 值分解方法，通过混合网络学习全局Q值

### 支持的训练范式

- **独立学习 (Independent Learning)**: 每个智能体完全独立，不共享信息
- **集中训练分散执行 (CTDE)**: 训练时可以使用全局信息，执行时每个智能体独立决策
- **完全集中式 (Fully Centralized)**: 训练和执行都使用集中式策略

## 目录结构

```
examples/marl/
├── README.md                    # 本文档
├── config/                      # 配置文件目录
│   ├── mappo_example.yaml      # MAPPO配置示例
│   └── ippo_example.yaml       # IPPO配置示例
├── main_marl.py                # MARL训练主程序
└── run_marl.sh                 # 运行脚本
```

## 快速开始

### 1. 配置文件

MARL配置需要在原有配置基础上添加`marl`部分：

```yaml
marl:
  num_agents: 2                    # 智能体数量
  agent_ids: ["agent_0", "agent_1"]  # 智能体ID列表（可选）
  algorithm: "mappo"               # 算法类型: mappo, ippo, maddpg, qmix
  policy_sharing: "independent"    # 策略共享方式: independent, shared, partial
  use_global_state: true           # 是否使用全局状态（CTDE）
  communication:                   # 智能体间通信配置（可选）
    enabled: false
    method: "message_passing"      # 通信方法
```

### 2. 环境适配

多智能体环境需要返回字典格式的观察、奖励和done标志：

```python
# 环境返回格式
obs = {
    "agent_0": obs_tensor_0,  # shape: [batch_size, obs_dim]
    "agent_1": obs_tensor_1,
    # ...
}

rewards = {
    "agent_0": reward_tensor_0,  # shape: [batch_size]
    "agent_1": reward_tensor_1,
    # ...
}

dones = {
    "agent_0": done_tensor_0,  # shape: [batch_size]
    "agent_1": done_tensor_1,
    # ...
}
```

如果使用单智能体环境，框架会自动适配为多智能体格式。

### 3. 运行训练

```bash
# 使用MAPPO算法
python main_marl.py --config config/mappo_example.yaml

# 或使用提供的脚本
bash run_marl.sh
```

## 核心组件

### 1. MultiAgentEnvWorker

多智能体环境Worker，负责：
- 管理多智能体环境的交互
- 处理多智能体格式的观察、动作、奖励
- 支持单智能体环境的自动适配

### 2. MultiAgentActorGroup

多智能体Actor组，管理：
- 每个智能体的策略网络
- 策略权重同步
- 优势函数和回报计算
- 训练步骤执行

### 3. MARLRunner

MARL训练运行器，协调：
- 环境交互
- Rollout生成
- 优势计算
- 策略更新

## 配置说明

### MARL配置项

```yaml
marl:
  # 必需配置
  num_agents: 2                    # 智能体数量
  
  # 可选配置
  agent_ids: ["agent_0", "agent_1"]  # 智能体ID，默认自动生成
  algorithm: "mappo"               # 算法: mappo, ippo, maddpg, qmix
  policy_sharing: "independent"    # 策略共享: independent, shared, partial
  
  # CTDE配置
  use_global_state: true           # 是否使用全局状态
  centralized_critic: true        # 是否使用集中式critic
  
  # 通信配置
  communication:
    enabled: false                 # 是否启用通信
    method: "message_passing"      # 通信方法
    message_dim: 64                # 消息维度
```

### 算法特定配置

#### MAPPO
```yaml
algorithm:
  advantage_type: "mappo"          # 使用MAPPO优势计算
  normalize_advantages: true
  use_global_done: true            # 使用全局done标志
```

#### IPPO
```yaml
algorithm:
  advantage_type: "ippo"           # 使用IPPO优势计算
  normalize_advantages: true
  use_global_done: false           # 不使用全局done
```

#### MADDPG
```yaml
algorithm:
  advantage_type: "maddpg"         # 使用MADDPG优势计算
  use_target_network: true
  soft_update_tau: 0.01
```

## 示例代码

### 基本使用

```python
from rlinf.runners import MARLRunner, MultiAgentActorGroup
from rlinf.workers.env.multi_agent_env_worker import MultiAgentEnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.scheduler.cluster import Cluster
from rlinf.utils.placement import HybridComponentPlacement

# 初始化集群和placement
cluster = Cluster(num_nodes)
component_placement = HybridComponentPlacement(cfg, cluster)

# 创建多智能体Actor组
actor_workers = {}
for agent_id in cfg.marl.agent_ids:
    actor_placement = component_placement.get_strategy(f"actor_{agent_id}")
    actor_workers[agent_id] = create_actor_worker(cfg, actor_placement)

actor_group = MultiAgentActorGroup(cfg, actor_workers)

# 创建环境Worker
env_placement = component_placement.get_strategy("env")
env_group = MultiAgentEnvWorker.create_group(cfg).launch(
    cluster, placement_strategy=env_placement
)

# 创建Rollout Worker
rollout_placement = component_placement.get_strategy("rollout")
rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
    cluster, placement_strategy=rollout_placement
)

# 创建Runner并运行
runner = MARLRunner(
    cfg=cfg,
    actor_group=actor_group,
    rollout=rollout_group,
    env=env_group,
)
runner.init_workers()
runner.run()
```

## 注意事项

1. **环境适配**: 如果使用单智能体环境，框架会自动适配，但可能需要根据具体场景调整奖励分配策略

2. **策略共享**: 
   - `independent`: 每个智能体独立策略，适合异构智能体
   - `shared`: 所有智能体共享策略，适合同构智能体
   - `partial`: 部分智能体共享策略，需要额外配置

3. **通信机制**: 当前版本支持基础的通信机制，复杂通信需要自定义实现

4. **性能优化**: 
   - 对于同构智能体，使用共享策略可以显著减少内存和计算开销
   - 使用CTDE范式可以利用全局信息提升训练效率

## 扩展开发

### 添加新的MARL算法

1. 在`rlinf/algorithms/marl_algorithms.py`中添加优势计算函数：

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

实现自定义的智能体间通信需要在Actor Worker中添加通信逻辑，并在环境交互时传递消息。

## 参考资源

- [MARL算法综述](https://arxiv.org/abs/2006.07869)
- [MAPPO论文](https://arxiv.org/abs/2103.01955)
- [MADDPG论文](https://arxiv.org/abs/1706.02275)
- [QMIX论文](https://arxiv.org/abs/1803.11485)

