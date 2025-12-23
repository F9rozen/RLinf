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

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class MultiAgentEnvOutput:
    """多智能体环境的输出数据结构"""

    def __init__(
        self,
        obs: dict[str, torch.Tensor],  # 每个智能体的观察: {agent_id: obs_tensor}
        rewards: dict[str, torch.Tensor],  # 每个智能体的奖励: {agent_id: reward_tensor}
        dones: dict[str, torch.Tensor],  # 每个智能体的done标志: {agent_id: done_tensor}
        global_done: torch.Tensor,  # 全局done标志（所有智能体都done）
        infos: dict[str, Any] = None,  # 额外信息
        final_obs: dict[str, torch.Tensor] = None,  # 最终观察
    ):
        self.obs = obs
        self.rewards = rewards
        self.dones = dones
        self.global_done = global_done
        self.infos = infos or {}
        self.final_obs = final_obs

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式以便通过channel传输"""
        return {
            "obs": self.obs,
            "rewards": self.rewards,
            "dones": self.dones,
            "global_done": self.global_done,
            "infos": self.infos,
            "final_obs": self.final_obs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiAgentEnvOutput":
        """从字典创建MultiAgentEnvOutput"""
        return cls(
            obs=data["obs"],
            rewards=data["rewards"],
            dones=data["dones"],
            global_done=data["global_done"],
            infos=data.get("infos"),
            final_obs=data.get("final_obs"),
        )


class MultiAgentEnvWorker(Worker):
    """多智能体环境Worker，支持多智能体环境的交互"""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        # 多智能体配置
        self.num_agents = cfg.marl.num_agents
        self.agent_ids = cfg.marl.get("agent_ids", [f"agent_{i}" for i in range(self.num_agents)])
        assert len(self.agent_ids) == self.num_agents, (
            f"agent_ids长度({len(self.agent_ids)})必须等于num_agents({self.num_agents})"
        )

        self.simulator_list = []
        self.last_obs_list = []
        self.last_dones_list = []
        self.eval_simulator_list = []

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        assert (
            self._component_placement.get_world_size("rollout")
            % self._component_placement.get_world_size("env")
            == 0
        )
        # gather_num: number of rollout for each env process
        self.gather_num = self._component_placement.get_world_size(
            "rollout"
        ) // self._component_placement.get_world_size("env")
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        # only need rank0 to create channel
        if self._rank == 0:
            self.channel = self.create_channel(cfg.env.channel.name)
        else:
            self.channel = self.connect_channel(cfg.env.channel.name)

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload

        train_env_cls = get_env_cls(
            self.cfg.env.train.simulator_type, self.cfg.env.train
        )
        eval_env_cls = get_env_cls(self.cfg.env.eval.simulator_type, self.cfg.env.eval)

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.cfg.env.train,
                        rank=self._rank,
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=train_env_cls,
                        enable_offload=enable_offload,
                    )
                )
        if self.enable_eval:
            for stage_id in range(self.stage_num):
                self.eval_simulator_list.append(
                    EnvManager(
                        self.cfg.env.eval,
                        rank=self._rank,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=eval_env_cls,
                        enable_offload=enable_offload,
                    )
                )

        if not self.only_eval:
            self._init_simulator()

    def _init_simulator(self):
        if self.cfg.env.train.auto_reset:
            for i in range(self.stage_num):
                self.simulator_list[i].start_simulator()
                extracted_obs, _ = self.simulator_list[i].reset()
                # 多智能体环境：假设环境返回的obs是字典格式 {agent_id: obs}
                # 如果不是，需要适配层转换
                if not isinstance(extracted_obs, dict):
                    # 单智能体环境适配：将单智能体obs转换为多智能体格式
                    extracted_obs = {agent_id: extracted_obs for agent_id in self.agent_ids}

                dones = {}
                for agent_id in self.agent_ids:
                    dones[agent_id] = (
                        torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                global_done = torch.zeros((self.train_num_envs_per_stage,), dtype=bool)

                self.last_obs_list.append(extracted_obs)
                self.last_dones_list.append(dones)
                self.simulator_list[i].stop_simulator()

    def env_interact_step(
        self, chunk_actions: dict[str, torch.Tensor], stage_id: int
    ) -> tuple[MultiAgentEnvOutput, dict[str, Any]]:
        """
        多智能体环境交互步骤
        
        Args:
            chunk_actions: 每个智能体的动作 {agent_id: action_tensor}
            stage_id: pipeline stage id
            
        Returns:
            MultiAgentEnvOutput, env_info
        """
        # 准备动作：将多智能体动作转换为环境需要的格式
        # 这里假设环境接受字典格式的动作，或者需要拼接
        prepared_actions = {}
        for agent_id, agent_actions in chunk_actions.items():
            prepared_actions[agent_id] = prepare_actions(
                raw_chunk_actions=agent_actions,
                simulator_type=self.cfg.env.train.simulator_type,
                model_type=self.cfg.actor.model.model_type,
                num_action_chunks=self.cfg.actor.model.num_action_chunks,
                action_dim=self.cfg.actor.model.action_dim,
                policy=self.cfg.actor.model.get("policy_setup", None),
            )

        env_info = {}

        # 调用环境step方法
        # 假设环境支持多智能体接口，返回多智能体格式的观察、奖励、done等
        # 如果环境不支持，需要适配层
        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.simulator_list[stage_id].chunk_step(prepared_actions)
        )

        # 处理多智能体输出
        if not isinstance(extracted_obs, dict):
            # 单智能体环境适配：假设所有智能体共享相同的观察
            extracted_obs = {agent_id: extracted_obs for agent_id in self.agent_ids}

        if not isinstance(chunk_rewards, dict):
            # 单智能体环境适配：假设奖励需要分配给各个智能体
            # 这里可以根据具体需求调整分配策略
            chunk_rewards = {
                agent_id: chunk_rewards / self.num_agents for agent_id in self.agent_ids
            }

        chunk_dones = {}
        for agent_id in self.agent_ids:
            if isinstance(chunk_terminations, dict):
                term = chunk_terminations.get(agent_id, torch.zeros_like(chunk_terminations[list(chunk_terminations.keys())[0]]))
                trunc = chunk_truncations.get(agent_id, torch.zeros_like(chunk_truncations[list(chunk_truncations.keys())[0]]))
            else:
                term = chunk_terminations
                trunc = chunk_truncations
            chunk_dones[agent_id] = torch.logical_or(term, trunc)

        # 计算全局done：所有智能体都done时全局done为True
        global_done = torch.ones((self.train_num_envs_per_stage,), dtype=bool)
        for agent_id in self.agent_ids:
            global_done = global_done & chunk_dones[agent_id][:, -1]

        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if global_done.any():
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif global_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][global_done].cpu()

        env_output = MultiAgentEnvOutput(
            obs=extracted_obs,
            rewards=chunk_rewards,
            dones=chunk_dones,
            global_done=global_done,
            infos=infos,
            final_obs=infos.get("final_observation"),
        )
        return env_output, env_info

    def recv_chunk_actions(self) -> dict[str, torch.Tensor]:
        """接收所有智能体的动作"""
        chunk_actions = {}
        for agent_id in self.agent_ids:
            agent_actions = []
            for gather_id in range(self.gather_num):
                action_key = f"{self._action_queue_name}_{agent_id}_{gather_id + self._rank * self.gather_num}"
                agent_actions.append(self.channel.get(key=action_key))
            chunk_actions[agent_id] = np.concatenate(agent_actions, axis=0)
        return chunk_actions

    def send_env_batch(self, env_batch: dict[str, Any], mode="train"):
        """发送环境批次数据"""
        # 为每个gather_id和每个智能体发送数据
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            for agent_id in self.agent_ids:
                obs_key = f"{self._obs_queue_name}_{agent_id}_{gather_id + self._rank * self.gather_num}"
                # 提取该智能体的观察
                agent_obs_batch = {
                    "obs": env_batch_i["obs"][agent_id] if isinstance(env_batch_i["obs"], dict) else env_batch_i["obs"],
                    "dones": env_batch_i["dones"][agent_id] if isinstance(env_batch_i["dones"], dict) else env_batch_i["dones"],
                    "global_done": env_batch_i.get("global_done"),
                    "rewards": env_batch_i["rewards"][agent_id] if isinstance(env_batch_i["rewards"], dict) else env_batch_i["rewards"],
                }
                self.channel.put(item=agent_obs_batch, key=obs_key)

    def split_env_batch(self, env_batch, gather_id, mode):
        """分割环境批次"""
        env_batch_i = {}
        for key, value in env_batch.items():
            if isinstance(value, dict):
                # 多智能体数据：对每个智能体的数据分别分割
                env_batch_i[key] = {}
                for agent_id, agent_value in value.items():
                    if isinstance(agent_value, torch.Tensor):
                        env_batch_i[key][agent_id] = agent_value.chunk(self.gather_num, dim=0)[gather_id].contiguous()
                    else:
                        env_batch_i[key][agent_id] = agent_value
            elif isinstance(value, torch.Tensor):
                env_batch_i[key] = value.chunk(self.gather_num, dim=0)[gather_id].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage}, got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage}, got {length}"
                    )
                env_batch_i[key] = value[
                    gather_id * length // self.gather_num : (gather_id + 1) * length // self.gather_num
                ]
            else:
                env_batch_i[key] = value
        return env_batch_i

    def interact(self):
        """多智能体环境交互主循环"""
        for simulator in self.simulator_list:
            simulator.start_simulator()

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)
        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_output_list = []
            if not self.cfg.env.train.auto_reset:
                for stage_id in range(self.stage_num):
                    self.simulator_list[stage_id].is_start = True
                    extracted_obs, infos = self.simulator_list[stage_id].reset()
                    
                    # 适配多智能体格式
                    if not isinstance(extracted_obs, dict):
                        extracted_obs = {agent_id: extracted_obs for agent_id in self.agent_ids}

                    dones = {}
                    for agent_id in self.agent_ids:
                        dones[agent_id] = (
                            torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                            .unsqueeze(1)
                            .repeat(1, self.cfg.actor.model.num_action_chunks)
                        )
                    global_done = torch.zeros((self.train_num_envs_per_stage,), dtype=bool)

                    env_output = MultiAgentEnvOutput(
                        obs=extracted_obs,
                        dones=dones,
                        global_done=global_done,
                        final_obs=infos.get("final_observation"),
                    )
                    env_output_list.append(env_output)
            else:
                for stage_id in range(self.stage_num):
                    env_output = MultiAgentEnvOutput(
                        obs=self.last_obs_list[stage_id],
                        rewards=None,
                        dones=self.last_dones_list[stage_id],
                        global_done=torch.zeros((self.train_num_envs_per_stage,), dtype=bool),
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.stage_num):
                env_output: MultiAgentEnvOutput = env_output_list[stage_id]
                self.send_env_batch(env_output.to_dict())

            for _ in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    chunk_actions = self.recv_chunk_actions()
                    env_output, env_info = self.env_interact_step(chunk_actions, stage_id)
                    self.send_env_batch(env_output.to_dict())
                    env_output_list[stage_id] = env_output
                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_dones_list = [env_output.dones for env_output in env_output_list]
            self.finish_rollout()

        for simulator in self.simulator_list:
            simulator.stop_simulator()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    def finish_rollout(self, mode="train"):
        """完成rollout后的清理工作"""
        if mode == "train":
            if self.cfg.env.train.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.simulator_list[i].flush_video()
            for i in range(self.stage_num):
                self.simulator_list[i].update_reset_state_ids()
        elif mode == "eval":
            if self.cfg.env.eval.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.eval_simulator_list[i].flush_video()
            if not self.cfg.env.eval.auto_reset:
                for i in range(self.stage_num):
                    self.eval_simulator_list[i].update_reset_state_ids()

    def evaluate(self):
        """多智能体环境评估"""
        eval_metrics = defaultdict(list)

        for stage_id in range(self.stage_num):
            self.eval_simulator_list[stage_id].start_simulator()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            for stage_id in range(self.stage_num):
                self.eval_simulator_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_simulator_list[stage_id].reset()
                
                # 适配多智能体格式
                if not isinstance(extracted_obs, dict):
                    extracted_obs = {agent_id: extracted_obs for agent_id in self.agent_ids}

                env_output = MultiAgentEnvOutput(
                    obs=extracted_obs,
                    global_done=torch.zeros((self.eval_num_envs_per_stage,), dtype=bool),
                    final_obs=infos.get("final_observation"),
                )
                self.send_env_batch(env_output.to_dict(), mode="eval")

            for eval_step in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    chunk_actions = self.recv_chunk_actions()
                    env_output, env_info = self.env_interact_step(chunk_actions, stage_id)

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)
                    if eval_step == n_chunk_steps - 1:
                        continue
                    self.send_env_batch(env_output.to_dict(), mode="eval")

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            self.eval_simulator_list[stage_id].stop_simulator()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

