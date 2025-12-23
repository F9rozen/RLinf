# 多智能体环境准备指南

本文档详细说明如何为MARL训练准备多智能体环境。

## 目录

1. [环境接口要求](#环境接口要求)
2. [实现多智能体环境](#实现多智能体环境)
3. [注册环境](#注册环境)
4. [单智能体环境适配](#单智能体环境适配)
5. [完整示例](#完整示例)

## 环境接口要求

### 多智能体环境必须实现的接口

多智能体环境需要继承自 `gym.Env` 并实现以下方法：

```python
class MultiAgentEnv(gym.Env):
    def __init__(self, cfg, rank, num_envs, ret_device="cpu"):
        """初始化环境"""
        pass
    
    def reset(self, options={}):
        """重置环境，返回多智能体观察"""
        # 必须返回: obs_dict, infos
        # obs_dict: {agent_id: obs_tensor}
        return obs_dict, infos
    
    def step(self, actions):
        """执行动作，返回多智能体结果"""
        # actions: {agent_id: action_tensor} 或 torch.Tensor
        # 必须返回: obs_dict, rewards_dict, dones_dict, truncations_dict, infos
        return obs_dict, rewards_dict, dones_dict, truncations_dict, infos
    
    def chunk_step(self, chunk_actions):
        """执行chunk动作（用于多步rollout）"""
        # 类似step，但处理chunked actions
        return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos
```

### 数据格式要求

#### 观察格式 (obs_dict)

```python
obs_dict = {
    "agent_0": obs_tensor_0,  # shape: [num_envs, obs_dim]
    "agent_1": obs_tensor_1,
    # ... 每个智能体一个键
}
```

#### 奖励格式 (rewards_dict)

```python
rewards_dict = {
    "agent_0": reward_tensor_0,  # shape: [num_envs, num_chunks] 或 [num_envs]
    "agent_1": reward_tensor_1,
    # ...
}
```

#### Done标志格式 (dones_dict)

```python
dones_dict = {
    "agent_0": done_tensor_0,  # shape: [num_envs, num_chunks] 或 [num_envs]
    "agent_1": done_tensor_1,
    # ...
}
```

## 实现多智能体环境

### 步骤1: 创建环境类

在 `rlinf/envs/` 下创建你的环境目录和文件：

```python
# rlinf/envs/your_marl_env/your_marl_env.py

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

class YourMARLEnv(gym.Env):
    """你的多智能体环境实现"""
    
    def __init__(self, cfg: DictConfig, rank: int, num_envs: int, ret_device: str = "cpu"):
        self.cfg = cfg
        self.rank = rank
        self.num_envs = num_envs
        self.ret_device = ret_device
        self.seed = cfg.seed + rank
        
        # 多智能体配置
        self.num_agents = cfg.marl.num_agents
        self.agent_ids = cfg.marl.get("agent_ids", [f"agent_{i}" for i in range(self.num_agents)])
        
        # 初始化环境
        self._init_environment()
        self._init_reset_state_ids()
    
    def _init_environment(self):
        """初始化具体的环境实例"""
        # 例如：初始化物理仿真器、场景等
        # self.simulator = YourSimulator(...)
        pass
    
    def _init_reset_state_ids(self):
        """初始化重置状态ID和随机数发生器"""
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        # 设置重置状态逻辑
        self.reset_state_ids = list(range(self.num_envs))
    
    def reset(self, options={}):
        """重置环境，返回多智能体观察"""
        # 重置所有环境实例
        # self.simulator.reset()
        
        # 获取每个智能体的观察
        obs_dict = {}
        for agent_id in self.agent_ids:
            # 获取该智能体的观察
            obs = self._get_agent_observation(agent_id)  # shape: [num_envs, obs_dim]
            obs_dict[agent_id] = obs.to(self.ret_device)
        
        infos = {}
        return obs_dict, infos
    
    def step(self, actions):
        """执行动作"""
        # actions可能是字典或tensor
        if isinstance(actions, dict):
            # 多智能体动作字典
            agent_actions = actions
        else:
            # 单tensor，需要分割给各个智能体
            # 假设actions shape: [num_envs, num_agents * action_dim]
            agent_actions = self._split_actions(actions)
        
        # 执行动作
        # self.simulator.step(agent_actions)
        
        # 获取结果
        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        truncations_dict = {}
        
        for agent_id in self.agent_ids:
            obs_dict[agent_id] = self._get_agent_observation(agent_id).to(self.ret_device)
            rewards_dict[agent_id] = self._get_agent_reward(agent_id).to(self.ret_device)
            dones_dict[agent_id] = self._get_agent_done(agent_id).to(self.ret_device)
            truncations_dict[agent_id] = self._get_agent_truncation(agent_id).to(self.ret_device)
        
        infos = self._get_info()
        
        return obs_dict, rewards_dict, dones_dict, truncations_dict, infos
    
    def chunk_step(self, chunk_actions):
        """执行chunk动作（用于多步rollout）"""
        # chunk_actions: {agent_id: action_tensor} 
        # action_tensor shape: [num_envs, num_chunks, action_dim]
        
        obs_dict = {}
        rewards_dict = {}
        terminations_dict = {}
        truncations_dict = {}
        
        # 对每个chunk执行step
        num_chunks = chunk_actions[self.agent_ids[0]].shape[1]
        
        for chunk_idx in range(num_chunks):
            # 提取当前chunk的动作
            current_chunk_actions = {
                agent_id: actions[:, chunk_idx, :] 
                for agent_id, actions in chunk_actions.items()
            }
            
            # 执行step
            chunk_obs, chunk_rewards, chunk_dones, chunk_truncations, chunk_infos = self.step(current_chunk_actions)
            
            # 累积结果
            if chunk_idx == 0:
                # 初始化
                for agent_id in self.agent_ids:
                    obs_dict[agent_id] = chunk_obs[agent_id].unsqueeze(1)  # [num_envs, 1, obs_dim]
                    rewards_dict[agent_id] = chunk_rewards[agent_id].unsqueeze(1)  # [num_envs, 1]
                    terminations_dict[agent_id] = chunk_dones[agent_id].unsqueeze(1)  # [num_envs, 1]
                    truncations_dict[agent_id] = chunk_truncations[agent_id].unsqueeze(1)  # [num_envs, 1]
            else:
                # 拼接
                for agent_id in self.agent_ids:
                    obs_dict[agent_id] = torch.cat([obs_dict[agent_id], chunk_obs[agent_id].unsqueeze(1)], dim=1)
                    rewards_dict[agent_id] = torch.cat([rewards_dict[agent_id], chunk_rewards[agent_id].unsqueeze(1)], dim=1)
                    terminations_dict[agent_id] = torch.cat([terminations_dict[agent_id], chunk_dones[agent_id].unsqueeze(1)], dim=1)
                    truncations_dict[agent_id] = torch.cat([truncations_dict[agent_id], chunk_truncations[agent_id].unsqueeze(1)], dim=1)
        
        infos = chunk_infos  # 使用最后一个chunk的infos
        
        return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos
    
    def _get_agent_observation(self, agent_id: str) -> torch.Tensor:
        """获取指定智能体的观察"""
        # 实现你的观察获取逻辑
        # 返回: [num_envs, obs_dim]
        pass
    
    def _get_agent_reward(self, agent_id: str) -> torch.Tensor:
        """获取指定智能体的奖励"""
        # 实现你的奖励计算逻辑
        # 返回: [num_envs]
        pass
    
    def _get_agent_done(self, agent_id: str) -> torch.Tensor:
        """获取指定智能体的done标志"""
        # 返回: [num_envs]
        pass
    
    def _get_agent_truncation(self, agent_id: str) -> torch.Tensor:
        """获取指定智能体的truncation标志"""
        # 返回: [num_envs]
        pass
    
    def _get_info(self) -> dict:
        """获取环境信息"""
        return {}
    
    def _split_actions(self, actions: torch.Tensor) -> dict:
        """将单tensor动作分割为多智能体动作字典"""
        # 假设actions shape: [num_envs, num_agents * action_dim]
        action_dim = self.cfg.actor.model.action_dim
        agent_actions = {}
        for i, agent_id in enumerate(self.agent_ids):
            start_idx = i * action_dim
            end_idx = (i + 1) * action_dim
            agent_actions[agent_id] = actions[:, start_idx:end_idx]
        return agent_actions
```

### 步骤2: 创建__init__.py

```python
# rlinf/envs/your_marl_env/__init__.py

from .your_marl_env import YourMARLEnv

__all__ = ["YourMARLEnv"]
```

## 注册环境

在 `rlinf/envs/__init__.py` 中添加你的环境：

```python
def get_env_cls(simulator_type, env_cfg=None):
    # ... 现有代码 ...
    elif simulator_type == "your_marl_env":
        from rlinf.envs.your_marl_env.your_marl_env import YourMARLEnv
        return YourMARLEnv
    else:
        raise NotImplementedError(f"Simulator type {simulator_type} not implemented")
```

## 单智能体环境适配

如果你已有单智能体环境，框架会自动适配，但你可能需要调整奖励分配策略。

### 自动适配机制

框架会自动检测环境返回格式：
- 如果返回字典格式 → 直接使用
- 如果返回tensor格式 → 自动转换为多智能体格式

### 自定义奖励分配

如果需要自定义奖励分配策略，修改 `rlinf/workers/env/multi_agent_env_worker.py` 中的 `env_interact_step` 方法：

```python
# 在 env_interact_step 方法中
if not isinstance(chunk_rewards, dict):
    # 自定义奖励分配策略
    # 选项1: 平均分配
    chunk_rewards = {
        agent_id: chunk_rewards / self.num_agents 
        for agent_id in self.agent_ids
    }
    
    # 选项2: 按智能体贡献分配（需要环境提供额外信息）
    # chunk_rewards = self._custom_reward_allocation(chunk_rewards, infos)
```

## 完整示例

### 示例1: 简单的多智能体网格世界

```python
# rlinf/envs/simple_marl/simple_marl_env.py

import gymnasium as gym
import torch
import numpy as np
from omegaconf import DictConfig

class SimpleMARLEnv(gym.Env):
    """简单的多智能体网格世界环境"""
    
    def __init__(self, cfg: DictConfig, rank: int, num_envs: int, ret_device: str = "cpu"):
        self.cfg = cfg
        self.rank = rank
        self.num_envs = num_envs
        self.ret_device = ret_device
        
        self.num_agents = cfg.marl.num_agents
        self.agent_ids = cfg.marl.get("agent_ids", [f"agent_{i}" for i in range(self.num_agents)])
        
        # 环境参数
        self.grid_size = cfg.get("grid_size", 10)
        self.obs_dim = cfg.actor.model.obs_dim
        self.action_dim = cfg.actor.model.action_dim
        
        # 初始化智能体位置
        self.agent_positions = torch.randint(0, self.grid_size, (num_envs, self.num_agents, 2))
        self.target_positions = torch.randint(0, self.grid_size, (num_envs, 2))
        
        self._generator = torch.Generator()
        self._generator.manual_seed(cfg.seed + rank)
    
    def reset(self, options={}):
        """重置环境"""
        # 重置智能体位置
        self.agent_positions = torch.randint(0, self.grid_size, (self.num_envs, self.num_agents, 2))
        self.target_positions = torch.randint(0, self.grid_size, (self.num_envs, 2))
        
        # 获取观察
        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            # 观察包括：自己的位置、目标位置、其他智能体的位置
            agent_pos = self.agent_positions[:, i, :]  # [num_envs, 2]
            target_pos = self.target_positions  # [num_envs, 2]
            
            # 拼接观察
            obs = torch.cat([agent_pos, target_pos], dim=1)  # [num_envs, 4]
            
            # 如果obs_dim > 4，用零填充
            if self.obs_dim > 4:
                padding = torch.zeros(self.num_envs, self.obs_dim - 4)
                obs = torch.cat([obs, padding], dim=1)
            
            obs_dict[agent_id] = obs.to(self.ret_device)
        
        return obs_dict, {}
    
    def step(self, actions):
        """执行动作"""
        if isinstance(actions, dict):
            agent_actions = actions
        else:
            # 分割动作
            agent_actions = {}
            for i, agent_id in enumerate(self.agent_ids):
                agent_actions[agent_id] = actions[:, i * self.action_dim:(i + 1) * self.action_dim]
        
        # 更新位置（简化：动作直接是位置增量）
        for i, agent_id in enumerate(self.agent_ids):
            action = agent_actions[agent_id]
            # 假设action是位置增量
            self.agent_positions[:, i, :] += action[:, :2].clamp(-1, 1)
            self.agent_positions[:, i, :] = self.agent_positions[:, i, :].clamp(0, self.grid_size - 1)
        
        # 计算奖励
        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        truncations_dict = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            agent_pos = self.agent_positions[:, i, :]
            target_pos = self.target_positions
            
            # 观察
            obs = torch.cat([agent_pos, target_pos], dim=1)
            if self.obs_dim > 4:
                padding = torch.zeros(self.num_envs, self.obs_dim - 4)
                obs = torch.cat([obs, padding], dim=1)
            obs_dict[agent_id] = obs.to(self.ret_device)
            
            # 奖励：距离目标的负距离
            distance = torch.norm(agent_pos - target_pos, dim=1)
            rewards_dict[agent_id] = (-distance).to(self.ret_device)
            
            # Done：到达目标
            reached = distance < 1.0
            dones_dict[agent_id] = reached.to(self.ret_device)
            truncations_dict[agent_id] = torch.zeros_like(reached).to(self.ret_device)
        
        infos = {}
        return obs_dict, rewards_dict, dones_dict, truncations_dict, infos
    
    def chunk_step(self, chunk_actions):
        """执行chunk动作"""
        # 实现类似上面的step，但处理chunked actions
        num_chunks = chunk_actions[self.agent_ids[0]].shape[1]
        
        obs_dict = {}
        rewards_dict = {}
        terminations_dict = {}
        truncations_dict = {}
        
        for chunk_idx in range(num_chunks):
            current_chunk_actions = {
                agent_id: actions[:, chunk_idx, :] 
                for agent_id, actions in chunk_actions.items()
            }
            
            chunk_obs, chunk_rewards, chunk_dones, chunk_truncations, _ = self.step(current_chunk_actions)
            
            if chunk_idx == 0:
                for agent_id in self.agent_ids:
                    obs_dict[agent_id] = chunk_obs[agent_id].unsqueeze(1)
                    rewards_dict[agent_id] = chunk_rewards[agent_id].unsqueeze(1)
                    terminations_dict[agent_id] = chunk_dones[agent_id].unsqueeze(1)
                    truncations_dict[agent_id] = chunk_truncations[agent_id].unsqueeze(1)
            else:
                for agent_id in self.agent_ids:
                    obs_dict[agent_id] = torch.cat([obs_dict[agent_id], chunk_obs[agent_id].unsqueeze(1)], dim=1)
                    rewards_dict[agent_id] = torch.cat([rewards_dict[agent_id], chunk_rewards[agent_id].unsqueeze(1)], dim=1)
                    terminations_dict[agent_id] = torch.cat([terminations_dict[agent_id], chunk_dones[agent_id].unsqueeze(1)], dim=1)
                    truncations_dict[agent_id] = torch.cat([truncations_dict[agent_id], chunk_truncations[agent_id].unsqueeze(1)], dim=1)
        
        return obs_dict, rewards_dict, terminations_dict, truncations_dict, {}
```

## 配置环境

在配置文件中指定环境类型：

```yaml
env:
  train:
    simulator_type: "your_marl_env"  # 或 "simple_marl"
    total_num_envs: 64
    max_steps_per_rollout_epoch: 100
    seed: 42
    # 其他环境特定配置
    grid_size: 10  # 示例配置
```

## 测试环境

创建测试脚本验证环境：

```python
# test_env.py

from omegaconf import OmegaConf
from rlinf.envs import get_env_cls

cfg = OmegaConf.create({
    "marl": {
        "num_agents": 2,
        "agent_ids": ["agent_0", "agent_1"]
    },
    "actor": {
        "model": {
            "obs_dim": 24,
            "action_dim": 8
        }
    },
    "seed": 42
})

env_cls = get_env_cls("your_marl_env", cfg)
env = env_cls(cfg, rank=0, num_envs=4)

# 测试reset
obs_dict, infos = env.reset()
print("Reset obs shape:", {k: v.shape for k, v in obs_dict.items()})

# 测试step
actions = {agent_id: torch.randn(4, 8) for agent_id in ["agent_0", "agent_1"]}
obs, rewards, dones, truncations, infos = env.step(actions)
print("Step rewards:", {k: v.shape for k, v in rewards.items()})
```

## 常见问题

### Q1: 如何处理不同智能体的不同观察空间？

如果智能体观察空间不同，确保所有观察tensor的batch维度相同，但特征维度可以不同。框架会分别处理每个智能体的观察。

### Q2: 如何实现部分可观测性？

在 `_get_agent_observation` 中只返回该智能体可见的信息。

### Q3: 如何实现智能体间通信？

可以在观察中包含其他智能体的信息，或实现专门的通信机制。

### Q4: 如何处理异构智能体（不同动作空间）？

确保每个智能体的动作tensor维度匹配其动作空间，框架会分别处理。

## 下一步

1. 实现你的多智能体环境
2. 在 `rlinf/envs/__init__.py` 中注册
3. 在配置文件中指定 `simulator_type`
4. 运行训练测试

参考现有环境实现（如 `maniskill_env.py`）获取更多细节。

