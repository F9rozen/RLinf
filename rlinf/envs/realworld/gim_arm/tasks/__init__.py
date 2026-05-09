# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.gim_arm.gim_arm_env import GimArmEnv, GimArmRobotConfig
from rlinf.envs.realworld.gim_arm.tasks.peg_insertion import (
    GimArmPegInsertionEnv as GimArmPegInsertionEnv,
)


def create_gim_arm_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    """Factory for :class:`RealWorldEnv` / ``gym.make`` (matches Franka task factories)."""
    del env_cfg  # GimArmEnv does not use RL wrappers from ``apply_single_arm_wrappers``.
    config = GimArmRobotConfig(**override_cfg)
    return GimArmEnv(config, worker_info, hardware_info, env_idx)


register(
    id="GimArmEnv-v1",
    entry_point="rlinf.envs.realworld.gim_arm.tasks:create_gim_arm_env",
)

register(
    id="GimArmPegInsertionEnv-v1",
    entry_point="rlinf.envs.realworld.gim_arm.tasks:GimArmPegInsertionEnv",
)
