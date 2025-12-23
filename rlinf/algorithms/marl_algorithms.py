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
多智能体强化学习算法模块

支持多种MARL算法：
- Independent PPO (IPPO): 独立学习，每个智能体独立训练
- Multi-Agent PPO (MAPPO): 集中训练分散执行
- MADDPG: 多智能体DDPG
- QMIX: 值分解方法
"""

from typing import Optional

import torch
import torch.nn.functional as F

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("mappo")
def compute_mappo_advantages_and_returns(
    rewards: dict[str, torch.Tensor],  # {agent_id: rewards}
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    values: Optional[dict[str, torch.Tensor]] = None,  # {agent_id: values}
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[dict[str, torch.Tensor]] = None,  # {agent_id: mask}
    dones: Optional[dict[str, torch.Tensor]] = None,  # {agent_id: dones}
    global_done: Optional[torch.Tensor] = None,  # 全局done标志
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    计算MAPPO的优势函数和回报
    
    MAPPO (Multi-Agent PPO) 使用集中式价值函数，但每个智能体独立计算优势
    这里实现的是每个智能体独立计算GAE，但可以使用全局信息
    
    Args:
        rewards: 每个智能体的奖励 {agent_id: reward_tensor}
        gamma: 折扣因子
        gae_lambda: GAE平滑因子
        values: 每个智能体的价值函数估计 {agent_id: value_tensor}
        normalize_advantages: 是否归一化优势
        normalize_returns: 是否归一化回报
        loss_mask: 损失掩码 {agent_id: mask_tensor}
        dones: done标志 {agent_id: done_tensor}
        global_done: 全局done标志
        
    Returns:
        (advantages, returns): 每个智能体的优势和回报
    """
    advantages = {}
    returns = {}
    
    agent_ids = list(rewards.keys())
    
    for agent_id in agent_ids:
        agent_rewards = rewards[agent_id]
        agent_values = values[agent_id] if values is not None else None
        agent_dones = dones[agent_id] if dones is not None else None
        agent_mask = loss_mask[agent_id] if loss_mask is not None else None
        
        T = agent_rewards.shape[0]
        agent_advantages = torch.zeros_like(agent_rewards)
        agent_returns = torch.zeros_like(agent_rewards)
        gae = 0

        critic_free = agent_values is None
        if critic_free:
            gae_lambda = 1
            gamma = 1

        # 使用全局done或智能体特定的done
        if agent_dones is None:
            agent_dones = global_done.unsqueeze(0).repeat(T, 1) if global_done is not None else torch.zeros_like(agent_rewards)

        for step in reversed(range(T)):
            if critic_free:
                delta = agent_rewards[step]
            else:
                next_value = agent_values[step + 1] if step < T - 1 else torch.zeros_like(agent_values[step])
                done_mask = ~agent_dones[step + 1] if step < T - 1 else ~agent_dones[step]
                delta = (
                    agent_rewards[step]
                    + gamma * next_value * done_mask
                    - agent_values[step]
                )

            done_mask = ~agent_dones[step + 1] if step < T - 1 else ~agent_dones[step]
            gae = delta + gamma * gae_lambda * done_mask * gae
            agent_returns[step] = gae if critic_free else gae + agent_values[step]

        agent_advantages = agent_returns - agent_values[:-1] if not critic_free else agent_returns

        if normalize_advantages:
            agent_advantages = safe_normalize(agent_advantages, loss_mask=agent_mask)
        if normalize_returns:
            agent_returns = safe_normalize(agent_returns, loss_mask=agent_mask)

        advantages[agent_id] = agent_advantages
        returns[agent_id] = agent_returns

    return advantages, returns


@register_advantage("ippo")
def compute_ippo_advantages_and_returns(
    rewards: dict[str, torch.Tensor],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    values: Optional[dict[str, torch.Tensor]] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[dict[str, torch.Tensor]] = None,
    dones: Optional[dict[str, torch.Tensor]] = None,
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    计算IPPO (Independent PPO) 的优势函数和回报
    
    IPPO中每个智能体完全独立学习，不共享任何信息
    
    Args:
        rewards: 每个智能体的奖励 {agent_id: reward_tensor}
        gamma: 折扣因子
        gae_lambda: GAE平滑因子
        values: 每个智能体的价值函数估计 {agent_id: value_tensor}
        normalize_advantages: 是否归一化优势
        normalize_returns: 是否归一化回报
        loss_mask: 损失掩码 {agent_id: mask_tensor}
        dones: done标志 {agent_id: done_tensor}
        
    Returns:
        (advantages, returns): 每个智能体的优势和回报
    """
    # IPPO与MAPPO的主要区别在于是否使用全局信息
    # 这里实现上类似，但在实际应用中IPPO不使用其他智能体的信息
    return compute_mappo_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        gae_lambda=gae_lambda,
        values=values,
        normalize_advantages=normalize_advantages,
        normalize_returns=normalize_returns,
        loss_mask=loss_mask,
        dones=dones,
        global_done=None,  # IPPO不使用全局done
        **kwargs,
    )


@register_advantage("maddpg")
def compute_maddpg_advantages(
    rewards: dict[str, torch.Tensor],
    q_values: dict[str, torch.Tensor],  # Q值 {agent_id: q_value}
    next_q_values: Optional[dict[str, torch.Tensor]] = None,
    gamma: float = 0.99,
    dones: Optional[dict[str, torch.Tensor]] = None,
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    计算MADDPG的优势函数
    
    MADDPG使用Q-learning，优势 = Q(s,a) - V(s)
    其中V(s)可以通过Q值的期望或独立的价值网络计算
    
    Args:
        rewards: 每个智能体的奖励 {agent_id: reward_tensor}
        q_values: 当前Q值 {agent_id: q_value_tensor}
        next_q_values: 下一步Q值 {agent_id: next_q_value_tensor}
        gamma: 折扣因子
        dones: done标志 {agent_id: done_tensor}
        
    Returns:
        (advantages, returns): 每个智能体的优势和回报
    """
    advantages = {}
    returns = {}
    
    agent_ids = list(rewards.keys())
    
    for agent_id in agent_ids:
        agent_rewards = rewards[agent_id]
        agent_q = q_values[agent_id]
        agent_next_q = next_q_values[agent_id] if next_q_values is not None else None
        agent_dones = dones[agent_id] if dones is not None else None
        
        if agent_next_q is not None:
            # 计算TD目标
            agent_dones_expanded = agent_dones.unsqueeze(-1) if agent_dones is not None else torch.zeros_like(agent_rewards)
            targets = agent_rewards + gamma * agent_next_q * (~agent_dones_expanded)
            advantages[agent_id] = agent_q - targets.detach()
            returns[agent_id] = targets
        else:
            # 如果没有next_q，使用当前Q值作为优势
            advantages[agent_id] = agent_q
            returns[agent_id] = agent_q
    
    return advantages, returns


def compute_centralized_value(
    individual_values: dict[str, torch.Tensor],
    method: str = "sum",
) -> torch.Tensor:
    """
    计算集中式价值函数
    
    用于集中训练分散执行的方法（如MAPPO, QMIX）
    
    Args:
        individual_values: 每个智能体的价值 {agent_id: value_tensor}
        method: 聚合方法 ("sum", "mean", "max", "min")
        
    Returns:
        集中式价值
    """
    values_list = list(individual_values.values())
    stacked_values = torch.stack(values_list, dim=0)  # [num_agents, ...]
    
    if method == "sum":
        return stacked_values.sum(dim=0)
    elif method == "mean":
        return stacked_values.mean(dim=0)
    elif method == "max":
        return stacked_values.max(dim=0)[0]
    elif method == "min":
        return stacked_values.min(dim=0)[0]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_qmix_value(
    individual_q_values: dict[str, torch.Tensor],
    mixing_network: torch.nn.Module,
    state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    使用QMIX混合网络计算全局Q值
    
    Args:
        individual_q_values: 每个智能体的Q值 {agent_id: q_value}
        mixing_network: QMIX混合网络
        state: 全局状态（可选）
        
    Returns:
        混合后的全局Q值
    """
    q_values_list = list(individual_q_values.values())
    stacked_q = torch.stack(q_values_list, dim=-1)  # [..., num_agents]
    
    if state is not None:
        total_q = mixing_network(stacked_q, state)
    else:
        total_q = mixing_network(stacked_q)
    
    return total_q


def compute_multi_agent_policy_loss(
    advantages: dict[str, torch.Tensor],
    old_logprobs: dict[str, torch.Tensor],
    new_logprobs: dict[str, torch.Tensor],
    clip_ratio: float = 0.2,
    loss_mask: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    """
    计算多智能体策略损失（PPO风格）
    
    Args:
        advantages: 每个智能体的优势 {agent_id: advantage_tensor}
        old_logprobs: 旧策略的对数概率 {agent_id: logprob_tensor}
        new_logprobs: 新策略的对数概率 {agent_id: logprob_tensor}
        clip_ratio: PPO裁剪比例
        loss_mask: 损失掩码 {agent_id: mask_tensor}
        
    Returns:
        每个智能体的策略损失 {agent_id: loss_tensor}
    """
    policy_losses = {}
    
    for agent_id in advantages.keys():
        agent_adv = advantages[agent_id]
        agent_old_logprob = old_logprobs[agent_id]
        agent_new_logprob = new_logprobs[agent_id]
        agent_mask = loss_mask[agent_id] if loss_mask is not None else None
        
        # 计算重要性采样比率
        ratio = torch.exp(agent_new_logprob - agent_old_logprob)
        
        # PPO裁剪
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(ratio * agent_adv, clipped_ratio * agent_adv)
        
        if agent_mask is not None:
            policy_loss = policy_loss * agent_mask
        
        policy_losses[agent_id] = policy_loss
    
    return policy_losses


def compute_multi_agent_value_loss(
    returns: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    loss_mask: Optional[dict[str, torch.Tensor]] = None,
    clip_value: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """
    计算多智能体价值函数损失
    
    Args:
        returns: 每个智能体的回报 {agent_id: return_tensor}
        values: 每个智能体的价值估计 {agent_id: value_tensor}
        loss_mask: 损失掩码 {agent_id: mask_tensor}
        clip_value: 价值裁剪（可选）
        
    Returns:
        每个智能体的价值损失 {agent_id: loss_tensor}
    """
    value_losses = {}
    
    for agent_id in returns.keys():
        agent_returns = returns[agent_id]
        agent_values = values[agent_id]
        agent_mask = loss_mask[agent_id] if loss_mask is not None else None
        
        # 价值损失
        value_loss = (agent_values - agent_returns) ** 2
        
        if clip_value is not None:
            # 裁剪价值更新
            clipped_values = agent_returns + torch.clamp(
                agent_values - agent_returns, -clip_value, clip_value
            )
            clipped_value_loss = (clipped_values - agent_returns) ** 2
            value_loss = torch.max(value_loss, clipped_value_loss)
        
        if agent_mask is not None:
            value_loss = value_loss * agent_mask
        
        value_losses[agent_id] = value_loss
    
    return value_losses

