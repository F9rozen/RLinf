# SimpleMARLEnv - 简单多智能体环境

## 概述

`SimpleMARLEnv` 是一个简单的多智能体环境实现示例，用于测试和演示MARL训练框架。

## 环境描述

- **类型**: 2D网格世界
- **任务**: 多个智能体需要到达各自的目标位置
- **观察**: 每个智能体只能看到自己的位置和目标位置
- **动作**: 连续动作空间，控制位置增量
- **奖励**: 基于到目标的距离，到达目标有额外奖励

## 使用方法

### 1. 在配置文件中指定

```yaml
env:
  train:
    simulator_type: "simple_marl"
    total_num_envs: 64
    max_steps_per_rollout_epoch: 100
    seed: 42
    grid_size: 10        # 网格大小（可选，默认10）
    max_steps: 100       # 最大步数（可选，默认100）
```

### 2. 确保MARL配置正确

```yaml
marl:
  num_agents: 2
  agent_ids: ["agent_0", "agent_1"]
```

### 3. 确保Actor模型配置匹配

```yaml
actor:
  model:
    obs_dim: 24    # 观察维度（至少7，会自动填充）
    action_dim: 8  # 动作维度（至少2，使用前2维作为位置增量）
```

## 环境接口

### reset()

返回多智能体观察字典：

```python
obs_dict, infos = env.reset()
# obs_dict = {
#     "agent_0": tensor([num_envs, obs_dim]),
#     "agent_1": tensor([num_envs, obs_dim]),
# }
```

### step(actions)

执行动作并返回结果：

```python
actions = {
    "agent_0": tensor([num_envs, action_dim]),
    "agent_1": tensor([num_envs, action_dim]),
}
obs, rewards, dones, truncations, infos = env.step(actions)
```

### chunk_step(chunk_actions)

执行chunk动作（用于多步rollout）：

```python
chunk_actions = {
    "agent_0": tensor([num_envs, num_chunks, action_dim]),
    "agent_1": tensor([num_envs, num_chunks, action_dim]),
}
obs, rewards, terminations, truncations, infos = env.chunk_step(chunk_actions)
```

## 观察空间

每个智能体的观察包含：
- 自己的位置 (2维)
- 目标位置 (2维)
- 相对位置 (2维)
- 距离 (1维)
- 填充到 `obs_dim` 维度

## 动作空间

- 动作维度: `action_dim`
- 使用前2维作为位置增量
- 动作值被clamp到[-1, 1]范围

## 奖励函数

- 基础奖励: `-distance` (距离越近奖励越高)
- 到达奖励: 到达目标时额外+10.0

## 终止条件

- **Done**: 智能体到达目标（距离 < 0.5）
- **Truncation**: 达到最大步数

## 测试

运行测试脚本验证环境：

```bash
# 方式1: 使用模块测试
python -m rlinf.envs.simple_marl.test_simple_marl_env

# 方式2: 使用独立测试脚本
python examples/marl/test_env.py
```

## 注意事项

1. 这是一个简单的示例环境，主要用于测试框架功能
2. 实际使用时，建议实现更复杂的环境
3. 观察维度至少需要7，会自动填充到配置的`obs_dim`
4. 动作维度至少需要2，使用前2维作为位置增量

