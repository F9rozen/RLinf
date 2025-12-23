#!/usr/bin/env python3
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
快速验证环境是否正确注册

这个脚本检查环境是否可以正常导入和注册。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

def verify_env_registration():
    """验证环境注册"""
    print("=" * 60)
    print("验证 SimpleMARLEnv 环境注册")
    print("=" * 60)
    
    # 1. 检查环境模块是否可以导入
    print("\n1. 检查环境模块导入...")
    try:
        from rlinf.envs.simple_marl.simple_marl_env import SimpleMARLEnv
        print(f"   ✓ 成功导入 SimpleMARLEnv")
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return False
    
    # 2. 检查环境是否在__init__中注册
    print("\n2. 检查环境注册...")
    try:
        from rlinf.envs import get_env_cls
        from omegaconf import OmegaConf
        
        cfg = OmegaConf.create({
            "marl": {"num_agents": 2, "agent_ids": ["agent_0", "agent_1"]},
            "actor": {"model": {"obs_dim": 24, "action_dim": 8}},
        })
        
        env_cls = get_env_cls("simple_marl", cfg)
        print(f"   ✓ 环境已成功注册为 'simple_marl'")
        print(f"   环境类: {env_cls.__name__}")
    except Exception as e:
        print(f"   ✗ 环境注册检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 检查环境类的基本属性
    print("\n3. 检查环境类属性...")
    try:
        assert hasattr(SimpleMARLEnv, 'reset'), "缺少reset方法"
        assert hasattr(SimpleMARLEnv, 'step'), "缺少step方法"
        assert hasattr(SimpleMARLEnv, 'chunk_step'), "缺少chunk_step方法"
        print(f"   ✓ 环境类包含所有必需方法")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 环境注册验证通过！")
    print("=" * 60)
    print("\n环境可以在配置文件中使用:")
    print('  env:\n    train:\n      simulator_type: "simple_marl"')
    return True

if __name__ == "__main__":
    success = verify_env_registration()
    sys.exit(0 if success else 1)

