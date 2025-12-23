# 环境准备快速参考

## 快速检查清单

- [ ] 环境类继承自 `gym.Env`
- [ ] 实现 `reset()` 方法，返回 `(obs_dict, infos)`
- [ ] 实现 `step()` 方法，返回 `(obs_dict, rewards_dict, dones_dict, truncations_dict, infos)`
- [ ] 实现 `chunk_step()` 方法（用于多步rollout）
- [ ] 观察、奖励、done都是字典格式：`{agent_id: tensor}`
- [ ] 在 `rlinf/envs/__init__.py` 中注册环境
- [ ] 配置文件中指定 `simulator_type`

## 必需的方法签名

```python
def reset(self, options={}):
    """返回: (obs_dict, infos)"""
    return obs_dict, infos

def step(self, actions):
    """actions: {agent_id: action_tensor}
    返回: (obs_dict, rewards_dict, dones_dict, truncations_dict, infos)
    """
    return obs_dict, rewards_dict, dones_dict, truncations_dict, infos

def chunk_step(self, chunk_actions):
    """chunk_actions: {agent_id: action_tensor} shape [num_envs, num_chunks, action_dim]
    返回: (obs_dict, rewards_dict, terminations_dict, truncations_dict, infos)
    """
    return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos
```

## 数据格式

### 观察字典
```python
obs_dict = {
    "agent_0": torch.Tensor,  # [num_envs, obs_dim]
    "agent_1": torch.Tensor,
}
```

### 奖励字典
```python
rewards_dict = {
    "agent_0": torch.Tensor,  # [num_envs] 或 [num_envs, num_chunks]
    "agent_1": torch.Tensor,
}
```

### Done字典
```python
dones_dict = {
    "agent_0": torch.Tensor,  # [num_envs] 或 [num_envs, num_chunks], dtype=bool
    "agent_1": torch.Tensor,
}
```

## 注册环境

在 `rlinf/envs/__init__.py` 的 `get_env_cls` 函数中添加：

```python
elif simulator_type == "your_env_name":
    from rlinf.envs.your_env.your_env import YourEnv
    return YourEnv
```

## 配置文件

```yaml
env:
  train:
    simulator_type: "your_env_name"  # 必须与注册的名称一致
    total_num_envs: 64
    max_steps_per_rollout_epoch: 100
    seed: 42
```

## 测试环境

```python
from omegaconf import OmegaConf
from rlinf.envs import get_env_cls

cfg = OmegaConf.create({
    "marl": {"num_agents": 2, "agent_ids": ["agent_0", "agent_1"]},
    "actor": {"model": {"obs_dim": 24, "action_dim": 8}},
    "seed": 42
})

env_cls = get_env_cls("your_env_name", cfg)
env = env_cls(cfg, rank=0, num_envs=4)

# 测试
obs_dict, _ = env.reset()
actions = {agent_id: torch.randn(4, 8) for agent_id in ["agent_0", "agent_1"]}
obs, rewards, dones, truncations, infos = env.step(actions)
```

## 常见错误

1. **返回格式错误**: 必须返回字典，不是tensor
2. **键名不匹配**: `agent_ids` 必须与配置中的一致
3. **形状错误**: 确保所有tensor的batch维度是 `num_envs`
4. **未注册环境**: 必须在 `__init__.py` 中注册

## 参考

- 详细指南: [ENV_SETUP.md](ENV_SETUP.md)
- 示例代码: [env_example.py](env_example.py)
- 现有环境: `rlinf/envs/maniskill/maniskill_env.py`

