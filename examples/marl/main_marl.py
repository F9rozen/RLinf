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
多智能体强化学习训练主程序

使用示例:
    python main_marl.py --config config/mappo_example.yaml
"""

import argparse
import logging

from omegaconf import OmegaConf

from rlinf.config import build_config, validate_cfg
from rlinf.runners.marl_runner import MARLRunner, MultiAgentActorGroup
from rlinf.scheduler.cluster import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.multi_agent_env_worker import MultiAgentEnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_actor_worker_group(cfg, agent_id, cluster, placement):
    """
    创建单个智能体的Actor Worker组
    
    使用FSDP Actor（适用于embodied场景）
    如果需要使用Megatron Actor，可以修改此函数
    """
    # 使用FSDP Actor
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=f"{cfg.actor.group_name}_{agent_id}",
        placement_strategy=placement,
    )
    return actor_group


def main():
    parser = argparse.ArgumentParser(description="多智能体强化学习训练")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径",
    )
    args = parser.parse_args()

    # 加载配置
    cfg = OmegaConf.load(args.config)
    cfg = build_config(cfg)
    cfg = validate_cfg(cfg)

    # 验证MARL配置
    assert hasattr(cfg, "marl"), "配置文件中必须包含marl部分"
    assert cfg.marl.num_agents > 0, "num_agents必须大于0"
    
    num_agents = cfg.marl.num_agents
    agent_ids = cfg.marl.get("agent_ids", [f"agent_{i}" for i in range(num_agents)])
    assert len(agent_ids) == num_agents, (
        f"agent_ids长度({len(agent_ids)})必须等于num_agents({num_agents})"
    )

    logger.info(f"初始化MARL训练，智能体数量: {num_agents}")
    logger.info(f"智能体ID: {agent_ids}")
    logger.info(f"MARL算法: {cfg.marl.get('algorithm', 'mappo')}")

    # 初始化集群和placement
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # 创建多智能体Actor组
    actor_workers = {}
    policy_sharing = cfg.marl.get("policy_sharing", "independent")
    
    if policy_sharing == "shared":
        # 共享策略：所有智能体共享一个Actor
        logger.info("使用共享策略模式")
        shared_actor_placement = component_placement.get_strategy("actor")
        shared_actor = create_actor_worker_group(cfg, "shared", cluster, shared_actor_placement)
        for agent_id in agent_ids:
            actor_workers[agent_id] = shared_actor
        actor_group = MultiAgentActorGroup(cfg, actor_workers, shared_actor=shared_actor)
    else:
        # 独立策略：每个智能体有独立的Actor
        logger.info("使用独立策略模式")
        for agent_id in agent_ids:
            # 尝试获取特定智能体的placement，如果没有则使用默认actor placement
            try:
                actor_placement = component_placement.get_strategy(f"actor_{agent_id}")
            except (KeyError, AttributeError):
                # 如果没有特定配置，使用默认actor配置
                actor_placement = component_placement.get_strategy("actor")
            actor_workers[agent_id] = create_actor_worker_group(cfg, agent_id, cluster, actor_placement)
        actor_group = MultiAgentActorGroup(cfg, actor_workers)

    # 创建环境Worker组
    env_placement = component_placement.get_strategy("env")
    env_group = MultiAgentEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )

    # 创建Rollout Worker组
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement,
    )

    # 创建Runner
    runner = MARLRunner(
        cfg=cfg,
        actor_group=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    # 初始化并运行
    runner.init_workers()
    logger.info("开始MARL训练...")
    runner.run()
    logger.info("MARL训练完成")


if __name__ == "__main__":
    main()

