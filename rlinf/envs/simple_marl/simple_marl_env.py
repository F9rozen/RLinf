# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
简单的多智能体环境实现

这是一个示例多智能体环境，展示如何实现符合MARL框架要求的环境接口。
环境是一个2D网格世界，多个智能体需要协作或竞争到达目标位置。
"""

import gymnasium as gym
import torch
from omegaconf import DictConfig
from typing import Any, Dict


class SimpleMARLEnv(gym.Env):
    """
    简单的多智能体环境
    
    这是一个2D网格世界，多个智能体需要到达各自的目标位置。
    每个智能体只能看到自己的位置和目标位置。
    适用于测试和演示MARL训练框架。
    """
    
    def __init__(self, cfg: DictConfig, rank: int, num_envs: int, ret_device: str = "cpu"):
        """
        初始化环境
        
        Args:
            cfg: 配置对象，必须包含marl配置
            rank: 进程rank
            num_envs: 并行环境数量
            ret_device: 返回数据的设备
        """
        self.cfg = cfg
        self.rank = rank
        self.num_envs = num_envs
        self.ret_device = ret_device
        self.seed = cfg.get("seed", 42) + rank
        
        # 多智能体配置
        self.num_agents = cfg.marl.num_agents
        self.agent_ids = cfg.marl.get("agent_ids", [f"agent_{i}" for i in range(self.num_agents)])
        assert len(self.agent_ids) == self.num_agents, (
            f"agent_ids长度({len(self.agent_ids)})必须等于num_agents({self.num_agents})"
        )
        
        # 环境参数
        self.grid_size = cfg.get("grid_size", 10)
        self.max_steps = cfg.get("max_steps", 100)
        self.obs_dim = cfg.actor.model.obs_dim
        self.action_dim = cfg.actor.model.action_dim
        
        # 初始化状态
        self.agent_positions = None
        self.target_positions = None
        self.step_count = None
        
        # 随机数生成器
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        
        # 重置状态ID
        self.reset_state_ids = list(range(num_envs))
    
    def reset(self, options: Dict[str, Any] = {}):
        """
        重置环境
        
        Returns:
            obs_dict: {agent_id: obs_tensor} 每个智能体的观察
            infos: 环境信息字典
        """
        # 随机初始化智能体位置
        self.agent_positions = torch.randint(
            0, self.grid_size, 
            (self.num_envs, self.num_agents, 2),
            generator=self._generator
        ).float()
        
        # 随机初始化目标位置
        self.target_positions = torch.randint(
            0, self.grid_size,
            (self.num_envs, 2),
            generator=self._generator
        ).float()
        
        # 重置步数计数
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long)
        
        # 获取观察
        obs_dict = self._get_observations()
        
        infos = {}
        return obs_dict, infos
    
    def step(self, actions: Dict[str, torch.Tensor]):
        """
        执行一步动作
        
        Args:
            actions: {agent_id: action_tensor} 每个智能体的动作
                    action_tensor shape: [num_envs, action_dim]
        
        Returns:
            obs_dict: 新的观察
            rewards_dict: 每个智能体的奖励
            dones_dict: 每个智能体的done标志
            truncations_dict: 每个智能体的truncation标志
            infos: 环境信息
        """
        # 更新智能体位置
        # 假设动作是位置增量（简化处理）
        for i, agent_id in enumerate(self.agent_ids):
            action = actions[agent_id]  # [num_envs, action_dim]
            # 使用动作的前2维作为位置增量
            delta = action[:, :2].clamp(-1.0, 1.0)
            self.agent_positions[:, i, :] += delta
            # 限制在网格范围内
            self.agent_positions[:, i, :] = self.agent_positions[:, i, :].clamp(0, self.grid_size - 1)
        
        # 更新步数
        self.step_count += 1
        
        # 计算奖励和done
        obs_dict = self._get_observations()
        rewards_dict = self._compute_rewards()
        dones_dict, truncations_dict = self._compute_dones()
        
        infos = {
            "step_count": self.step_count.clone(),
            "agent_positions": self.agent_positions.clone(),
            "target_positions": self.target_positions.clone(),
        }
        
        return obs_dict, rewards_dict, dones_dict, truncations_dict, infos
    
    def chunk_step(self, chunk_actions: Dict[str, torch.Tensor]):
        """
        执行chunk动作（用于多步rollout）
        
        Args:
            chunk_actions: {agent_id: action_tensor}
                          action_tensor shape: [num_envs, num_chunks, action_dim]
        
        Returns:
            obs_dict: {agent_id: obs_tensor} shape: [num_envs, num_chunks, obs_dim]
            rewards_dict: {agent_id: reward_tensor} shape: [num_envs, num_chunks]
            terminations_dict: {agent_id: done_tensor} shape: [num_envs, num_chunks]
            truncations_dict: {agent_id: truncation_tensor} shape: [num_envs, num_chunks]
            infos: 环境信息
        """
        num_chunks = chunk_actions[self.agent_ids[0]].shape[1]
        
        obs_dict = {}
        rewards_dict = {}
        terminations_dict = {}
        truncations_dict = {}
        
        # 对每个chunk执行step
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
                for agent_id in self.agent_ids:
                    obs_dict[agent_id] = chunk_obs[agent_id].unsqueeze(1)  # [num_envs, 1, obs_dim]
                    rewards_dict[agent_id] = chunk_rewards[agent_id].unsqueeze(1)  # [num_envs, 1]
                    terminations_dict[agent_id] = chunk_dones[agent_id].unsqueeze(1)  # [num_envs, 1]
                    truncations_dict[agent_id] = chunk_truncations[agent_id].unsqueeze(1)  # [num_envs, 1]
            else:
                for agent_id in self.agent_ids:
                    obs_dict[agent_id] = torch.cat([obs_dict[agent_id], chunk_obs[agent_id].unsqueeze(1)], dim=1)
                    rewards_dict[agent_id] = torch.cat([rewards_dict[agent_id], chunk_rewards[agent_id].unsqueeze(1)], dim=1)
                    terminations_dict[agent_id] = torch.cat([terminations_dict[agent_id], chunk_dones[agent_id].unsqueeze(1)], dim=1)
                    truncations_dict[agent_id] = torch.cat([truncations_dict[agent_id], chunk_truncations[agent_id].unsqueeze(1)], dim=1)
        
        # 使用最后一个chunk的infos
        infos = chunk_infos
        
        return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """获取所有智能体的观察"""
        obs_dict = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            # 智能体自己的位置
            agent_pos = self.agent_positions[:, i, :]  # [num_envs, 2]
            
            # 目标位置
            target_pos = self.target_positions  # [num_envs, 2]
            
            # 相对位置（目标相对于智能体的位置）
            relative_pos = target_pos - agent_pos  # [num_envs, 2]
            
            # 距离
            distance = torch.norm(relative_pos, dim=1, keepdim=True)  # [num_envs, 1]
            
            # 拼接观察
            obs = torch.cat([agent_pos, target_pos, relative_pos, distance], dim=1)  # [num_envs, 7]
            
            # 如果obs_dim > 7，用零填充
            if self.obs_dim > 7:
                padding = torch.zeros(self.num_envs, self.obs_dim - 7, device=obs.device)
                obs = torch.cat([obs, padding], dim=1)
            elif self.obs_dim < 7:
                # 如果obs_dim < 7，截断
                obs = obs[:, :self.obs_dim]
            
            obs_dict[agent_id] = obs.to(self.ret_device)
        
        return obs_dict
    
    def _compute_rewards(self) -> Dict[str, torch.Tensor]:
        """计算每个智能体的奖励"""
        rewards_dict = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            agent_pos = self.agent_positions[:, i, :]  # [num_envs, 2]
            target_pos = self.target_positions  # [num_envs, 2]
            
            # 奖励：负距离（距离越近奖励越高）
            distance = torch.norm(agent_pos - target_pos, dim=1)  # [num_envs]
            reward = -distance
            
            # 到达目标的额外奖励
            reached = distance < 0.5
            reward[reached] += 10.0
            
            rewards_dict[agent_id] = reward.to(self.ret_device)
        
        return rewards_dict
    
    def _compute_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """计算done和truncation标志"""
        dones_dict = {}
        truncations_dict = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            agent_pos = self.agent_positions[:, i, :]  # [num_envs, 2]
            target_pos = self.target_positions  # [num_envs, 2]
            
            # 到达目标
            distance = torch.norm(agent_pos - target_pos, dim=1)  # [num_envs]
            reached = distance < 0.5
            
            # 超时
            timeout = self.step_count >= self.max_steps
            
            # Done: 到达目标
            dones_dict[agent_id] = reached.to(self.ret_device)
            
            # Truncation: 超时
            truncations_dict[agent_id] = (timeout & ~reached).to(self.ret_device)
        
        return dones_dict, truncations_dict
    
    def update_reset_state_ids(self):
        """更新重置状态ID（用于环境管理）"""
        # 可以在这里实现重置状态的选择逻辑
        pass

