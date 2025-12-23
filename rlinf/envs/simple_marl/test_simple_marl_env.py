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
测试SimpleMARLEnv环境

运行此脚本以验证环境实现是否正确。
"""

import torch
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls


def test_simple_marl_env():
    """测试SimpleMARLEnv环境"""
    print("=" * 60)
    print("测试 SimpleMARLEnv 环境")
    print("=" * 60)
    
    # 创建测试配置
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
        "seed": 42,
        "grid_size": 10,
        "max_steps": 100
    })
    
    print(f"\n配置:")
    print(f"  智能体数量: {cfg.marl.num_agents}")
    print(f"  智能体ID: {cfg.marl.agent_ids}")
    print(f"  观察维度: {cfg.actor.model.obs_dim}")
    print(f"  动作维度: {cfg.actor.model.action_dim}")
    print(f"  网格大小: {cfg.grid_size}")
    
    # 获取环境类
    print("\n1. 获取环境类...")
    try:
        env_cls = get_env_cls("simple_marl", cfg)
        print(f"   ✓ 成功获取环境类: {env_cls.__name__}")
    except Exception as e:
        print(f"   ✗ 获取环境类失败: {e}")
        return False
    
    # 创建环境实例
    print("\n2. 创建环境实例...")
    try:
        env = env_cls(cfg, rank=0, num_envs=4)
        print(f"   ✓ 成功创建环境实例")
        print(f"   环境数量: {env.num_envs}")
        print(f"   智能体数量: {env.num_agents}")
        print(f"   智能体ID: {env.agent_ids}")
    except Exception as e:
        print(f"   ✗ 创建环境失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试reset
    print("\n3. 测试reset方法...")
    try:
        obs_dict, infos = env.reset()
        print(f"   ✓ reset成功")
        print(f"   观察形状: { {k: v.shape for k, v in obs_dict.items()} }")
        print(f"   观察设备: { {k: v.device for k, v in obs_dict.items()} }")
        
        # 验证观察格式
        assert isinstance(obs_dict, dict), "观察必须是字典格式"
        assert len(obs_dict) == env.num_agents, f"观察字典长度({len(obs_dict)})必须等于智能体数量({env.num_agents})"
        for agent_id in env.agent_ids:
            assert agent_id in obs_dict, f"缺少智能体 {agent_id} 的观察"
            assert obs_dict[agent_id].shape == (env.num_envs, cfg.actor.model.obs_dim), (
                f"观察形状错误: 期望 ({env.num_envs}, {cfg.actor.model.obs_dim}), "
                f"实际 {obs_dict[agent_id].shape}"
            )
        print(f"   ✓ 观察格式验证通过")
    except Exception as e:
        print(f"   ✗ reset测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试step
    print("\n4. 测试step方法...")
    try:
        actions = {
            agent_id: torch.randn(env.num_envs, cfg.actor.model.action_dim)
            for agent_id in env.agent_ids
        }
        obs, rewards, dones, truncations, infos = env.step(actions)
        
        print(f"   ✓ step成功")
        print(f"   观察形状: { {k: v.shape for k, v in obs.items()} }")
        print(f"   奖励形状: { {k: v.shape for k, v in rewards.items()} }")
        print(f"   Done形状: { {k: v.shape for k, v in dones.items()} }")
        print(f"   Truncation形状: { {k: v.shape for k, v in truncations.items()} }")
        
        # 验证返回格式
        assert isinstance(rewards, dict), "奖励必须是字典格式"
        assert isinstance(dones, dict), "Done必须是字典格式"
        assert isinstance(truncations, dict), "Truncation必须是字典格式"
        
        for agent_id in env.agent_ids:
            assert agent_id in rewards, f"缺少智能体 {agent_id} 的奖励"
            assert agent_id in dones, f"缺少智能体 {agent_id} 的done标志"
            assert agent_id in truncations, f"缺少智能体 {agent_id} 的truncation标志"
            
            assert rewards[agent_id].shape == (env.num_envs,), (
                f"奖励形状错误: 期望 ({env.num_envs},), 实际 {rewards[agent_id].shape}"
            )
            assert dones[agent_id].shape == (env.num_envs,), (
                f"Done形状错误: 期望 ({env.num_envs},), 实际 {dones[agent_id].shape}"
            )
            assert dones[agent_id].dtype == torch.bool, (
                f"Done必须是bool类型, 实际 {dones[agent_id].dtype}"
            )
        
        print(f"   ✓ step返回格式验证通过")
    except Exception as e:
        print(f"   ✗ step测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试chunk_step
    print("\n5. 测试chunk_step方法...")
    try:
        num_chunks = 5
        chunk_actions = {
            agent_id: torch.randn(env.num_envs, num_chunks, cfg.actor.model.action_dim)
            for agent_id in env.agent_ids
        }
        
        # 重置环境
        env.reset()
        
        obs, rewards, terminations, truncations, infos = env.chunk_step(chunk_actions)
        
        print(f"   ✓ chunk_step成功")
        print(f"   观察形状: { {k: v.shape for k, v in obs.items()} }")
        print(f"   奖励形状: { {k: v.shape for k, v in rewards.items()} }")
        print(f"   Termination形状: { {k: v.shape for k, v in terminations.items()} }")
        print(f"   Truncation形状: { {k: v.shape for k, v in truncations.items()} }")
        
        # 验证chunk格式
        for agent_id in env.agent_ids:
            assert obs[agent_id].shape == (env.num_envs, num_chunks, cfg.actor.model.obs_dim), (
                f"Chunk观察形状错误: 期望 ({env.num_envs}, {num_chunks}, {cfg.actor.model.obs_dim}), "
                f"实际 {obs[agent_id].shape}"
            )
            assert rewards[agent_id].shape == (env.num_envs, num_chunks), (
                f"Chunk奖励形状错误: 期望 ({env.num_envs}, {num_chunks}), "
                f"实际 {rewards[agent_id].shape}"
            )
        
        print(f"   ✓ chunk_step格式验证通过")
    except Exception as e:
        print(f"   ✗ chunk_step测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试多步交互
    print("\n6. 测试多步交互...")
    try:
        env.reset()
        total_rewards = {agent_id: torch.zeros(env.num_envs) for agent_id in env.agent_ids}
        
        for step in range(10):
            actions = {
                agent_id: torch.randn(env.num_envs, cfg.actor.model.action_dim)
                for agent_id in env.agent_ids
            }
            obs, rewards, dones, truncations, infos = env.step(actions)
            
            for agent_id in env.agent_ids:
                total_rewards[agent_id] += rewards[agent_id]
        
        print(f"   ✓ 多步交互成功")
        avg_rewards = {k: v.mean().item() for k, v in total_rewards.items()}
        print(f"   累计奖励: {avg_rewards}")
    except Exception as e:
        print(f"   ✗ 多步交互测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！环境实现正确。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_simple_marl_env()
    exit(0 if success else 1)

