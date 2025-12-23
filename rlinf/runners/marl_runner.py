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
多智能体强化学习训练运行器

支持多种MARL训练范式：
- Independent Learning: 每个智能体独立学习
- Centralized Training Decentralized Execution (CTDE): 集中训练分散执行
- Fully Centralized: 完全集中式训练
"""

import logging
import os
from collections import defaultdict

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.env.multi_agent_env_worker import MultiAgentEnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentActorGroup:
    """
    多智能体Actor组
    
    管理多个智能体的策略网络，支持：
    - 独立策略：每个智能体有独立的策略网络
    - 共享策略：所有智能体共享同一个策略网络
    - 部分共享：部分智能体共享策略网络
    """

    def __init__(
        self,
        cfg: DictConfig,
        actor_workers: dict[str, any],  # {agent_id: actor_worker}
        shared_actor: any = None,  # 共享的actor（如果使用共享策略）
    ):
        self.cfg = cfg
        self.actor_workers = actor_workers
        self.shared_actor = shared_actor
        self.agent_ids = list(actor_workers.keys())
        self.use_shared_policy = shared_actor is not None

    def set_global_step(self, step: int):
        """设置全局步数"""
        for actor in self.actor_workers.values():
            if hasattr(actor, "set_global_step"):
                actor.set_global_step(step)
        if self.shared_actor is not None and hasattr(self.shared_actor, "set_global_step"):
            self.shared_actor.set_global_step(step)

    def sync_model_to_rollout(self):
        """同步模型权重到rollout worker"""
        futures = []
        for agent_id, actor in self.actor_workers.items():
            if hasattr(actor, "sync_model_to_rollout"):
                futures.append(actor.sync_model_to_rollout())
        if self.shared_actor is not None and hasattr(self.shared_actor, "sync_model_to_rollout"):
            futures.append(self.shared_actor.sync_model_to_rollout())
        return futures

    def compute_advantages_and_returns(self, rollout_data: dict[str, any]) -> dict[str, any]:
        """
        计算每个智能体的优势函数和回报
        
        Args:
            rollout_data: 每个智能体的rollout数据 {agent_id: rollout_data}
            
        Returns:
            每个智能体的优势和回报 {agent_id: (advantages, returns)}
        """
        results = {}
        for agent_id, actor in self.actor_workers.items():
            if hasattr(actor, "compute_advantages_and_returns"):
                agent_rollout = rollout_data.get(agent_id, {})
                results[agent_id] = actor.compute_advantages_and_returns(agent_rollout)
        return results

    def run_training(self) -> dict[str, any]:
        """运行训练步骤"""
        training_results = {}
        for agent_id, actor in self.actor_workers.items():
            if hasattr(actor, "run_training"):
                training_results[agent_id] = actor.run_training()
        return training_results

    def recv_rollout_batch(self) -> dict[str, any]:
        """接收rollout批次数据"""
        rollout_batches = {}
        for agent_id, actor in self.actor_workers.items():
            if hasattr(actor, "recv_rollout_batch"):
                rollout_batches[agent_id] = actor.recv_rollout_batch()
        return rollout_batches

    def save_checkpoint(self, base_path: str):
        """保存检查点"""
        futures = []
        for agent_id, actor in self.actor_workers.items():
            if hasattr(actor, "save_checkpoint"):
                agent_path = os.path.join(base_path, f"actor_{agent_id}")
                os.makedirs(agent_path, exist_ok=True)
                futures.append(actor.save_checkpoint(agent_path))
        if self.shared_actor is not None and hasattr(self.shared_actor, "save_checkpoint"):
            shared_path = os.path.join(base_path, "shared_actor")
            os.makedirs(shared_path, exist_ok=True)
            futures.append(self.shared_actor.save_checkpoint(shared_path))
        return futures

    def load_checkpoint(self, base_path: str):
        """加载检查点"""
        futures = []
        for agent_id, actor in self.actor_workers.items():
            if hasattr(actor, "load_checkpoint"):
                agent_path = os.path.join(base_path, f"actor_{agent_id}")
                if os.path.exists(agent_path):
                    futures.append(actor.load_checkpoint(agent_path))
        if self.shared_actor is not None and hasattr(self.shared_actor, "load_checkpoint"):
            shared_path = os.path.join(base_path, "shared_actor")
            if os.path.exists(shared_path):
                futures.append(self.shared_actor.load_checkpoint(shared_path))
        return futures


class MARLRunner:
    """
    多智能体强化学习训练运行器
    
    协调多智能体环境、rollout worker和actor worker进行MARL训练
    """

    def __init__(
        self,
        cfg: DictConfig,
        actor_group: MultiAgentActorGroup,
        rollout: MultiStepRolloutWorker,
        env: MultiAgentEnvWorker,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor_group = actor_group
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is MARL step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

        # MARL特定配置
        self.num_agents = cfg.marl.num_agents
        self.agent_ids = cfg.marl.get("agent_ids", [f"agent_{i}" for i in range(self.num_agents)])
        self.marl_algorithm = cfg.marl.get("algorithm", "mappo")  # mappo, ippo, maddpg, qmix

    def init_workers(self):
        """初始化所有workers"""
        # create worker in order to decrease the maximum memory usage
        for actor in self.actor_group.actor_workers.values():
            if hasattr(actor, "init_worker"):
                actor.init_worker().wait()
        if self.actor_group.shared_actor is not None and hasattr(self.actor_group.shared_actor, "init_worker"):
            self.actor_group.shared_actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor_group.load_checkpoint(actor_checkpoint_path)
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        """更新rollout worker的模型权重"""
        actor_futures = self.actor_group.sync_model_to_rollout()
        rollout_futures = self.rollout.sync_model_from_actor()
        for future in actor_futures:
            if future is not None:
                future.wait()
        rollout_futures.wait()

    def generate_rollouts(self):
        """
        生成多智能体rollout
        
        返回每个智能体的环境指标
        """
        # 启动环境交互
        env_futures = self.env.interact()
        
        # 启动rollout生成
        rollout_futures = self.rollout.generate()
        
        # 每个智能体的actor接收rollout批次
        actor_futures = {}
        for agent_id, actor in self.actor_group.actor_workers.items():
            if hasattr(actor, "recv_rollout_batch"):
                future = actor.recv_rollout_batch()
                if future is not None:
                    actor_futures[agent_id] = future
        
        # 等待所有操作完成
        env_results = env_futures.wait()
        for agent_id, future in actor_futures.items():
            if future is not None:
                future.wait()
        rollout_futures.wait()

        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        return env_metrics

    def evaluate(self):
        """评估多智能体策略"""
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_results = env_futures.wait()
        rollout_futures.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        """主训练循环"""
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="MARL Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor_group.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            
            eval_metrics = {}
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    self.update_rollout_weights()
                    
                with self.timer("generate_rollouts"):
                    env_metrics = self.generate_rollouts()

                # compute advantages and returns for each agent
                with self.timer("cal_adv_and_returns"):
                    # 每个智能体的actor会从自己的channel接收rollout数据
                    # 这里直接调用actor的compute_advantages_and_returns方法
                    actor_futures = {}
                    for agent_id, actor in self.actor_group.actor_workers.items():
                        if hasattr(actor, "compute_advantages_and_returns"):
                            # 调用actor的方法，它会从channel获取数据
                            future = actor.compute_advantages_and_returns()
                            if future is not None:
                                actor_futures[agent_id] = future
                    
                    actor_rollout_metrics = {}
                    for agent_id, future in actor_futures.items():
                        if future is not None:
                            result = future.wait()
                            # 处理返回结果，可能是tuple或dict
                            if isinstance(result, tuple):
                                actor_rollout_metrics[agent_id] = result[0] if len(result) > 0 else {}
                            elif isinstance(result, list) and len(result) > 0:
                                actor_rollout_metrics[agent_id] = result[0] if isinstance(result[0], dict) else {}
                            else:
                                actor_rollout_metrics[agent_id] = result if isinstance(result, dict) else {}

                # actor training for each agent
                with self.timer("actor_training"):
                    actor_training_futures = self.actor_group.run_training()
                    actor_training_metrics = {}
                    for agent_id, future in actor_training_futures.items():
                        if future is not None:
                            actor_training_metrics[agent_id] = future.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

            # 聚合和记录指标
            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            
            # 聚合每个智能体的rollout指标
            rollout_metrics = {}
            for agent_id, metrics in actor_rollout_metrics.items():
                for key, value in metrics.items():
                    rollout_metrics[f"rollout/{agent_id}/{key}"] = value
            
            # 聚合每个智能体的训练指标
            training_metrics = {}
            for agent_id, metrics in actor_training_metrics.items():
                for key, value in metrics.items():
                    training_metrics[f"train/{agent_id}/{key}"] = value
            
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            
            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def _save_checkpoint(self):
        """保存检查点"""
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        futures = self.actor_group.save_checkpoint(actor_save_path)
        for future in futures:
            if future is not None:
                future.wait()

    def set_max_steps(self):
        """设置最大训练步数"""
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        """当前epoch"""
        return self.global_step // self.num_steps_per_epoch

