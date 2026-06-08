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

"""Build offline-SFT-equivalent multi_anchor samples from DAgger replay trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.rollout_temporal_obs import (
    restore_task_descriptions_from_obs,
)
from rlinf.data.datasets.dreamzero.sampling_strategy import (
    EmptyTemporalSampleError,
    MultiAnchorTemporalConfig,
    require_multi_anchor_temporal_indices,
)
from rlinf.data.embodied_io_struct import Trajectory


@dataclass(frozen=True)
class DaggerOfflineSftLayout:
    """Parameters for aligning DAgger replay windows with LeRobot multi_anchor SFT."""

    max_chunk_size: int
    action_horizon: int
    num_action_chunks: int
    macro_stride: int = 24
    state_horizon: int = 1


def _clip(value: int, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


def _trajectory_chunk_steps(trajectory: Trajectory) -> int:
    curr_obs = trajectory.curr_obs or {}
    main = curr_obs.get("main_images")
    if main is None:
        raise ValueError("Trajectory curr_obs is missing main_images.")
    return int(main.shape[0])


def _resolve_chunk_step_for_env_step(env_step: int, num_action_chunks: int) -> tuple[int, int]:
    """Map an env-step index to ``(curr_obs chunk index, intra-chunk frame index)``."""
    n = int(num_action_chunks)
    if env_step <= 0:
        return 0, 0
    chunk_idx = env_step // n + 1
    intra = env_step % n
    return chunk_idx, intra


def _get_video_frame(
    trajectory: Trajectory,
    env_step: int,
    batch_idx: int,
    *,
    env_key: str,
    num_action_chunks: int,
) -> np.ndarray:
    curr_obs = trajectory.curr_obs
    if curr_obs is None or env_key not in curr_obs:
        raise KeyError(f"Trajectory curr_obs missing {env_key!r}.")
    tensor = curr_obs[env_key]
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    num_chunk_steps = int(tensor.shape[0])
    chunk_idx, intra = _resolve_chunk_step_for_env_step(env_step, num_action_chunks)
    chunk_idx = _clip(chunk_idx, 0, num_chunk_steps - 1)
    max_intra = int(tensor.shape[2]) - 1 if tensor.ndim >= 5 else 0
    intra = _clip(intra, 0, max_intra)
    frame = tensor[chunk_idx, batch_idx, intra]
    return frame.detach().cpu().numpy().astype(np.uint8, copy=False)


def _get_state_row(
    trajectory: Trajectory,
    env_step: int,
    batch_idx: int,
    *,
    num_action_chunks: int,
) -> np.ndarray:
    curr_obs = trajectory.curr_obs or {}
    states = curr_obs.get("states")
    if states is None:
        raise KeyError("Trajectory curr_obs is missing states.")
    if not torch.is_tensor(states):
        states = torch.as_tensor(states)
    num_chunk_steps = int(states.shape[0])
    chunk_idx, intra = _resolve_chunk_step_for_env_step(env_step, num_action_chunks)
    chunk_idx = _clip(chunk_idx, 0, num_chunk_steps - 1)
    if states.ndim == 2:
        return states[chunk_idx, batch_idx].detach().cpu().numpy().astype(np.float32)
    max_intra = int(states.shape[2]) - 1
    intra = _clip(intra, 0, max_intra)
    row = states[chunk_idx, batch_idx, intra]
    return row.detach().cpu().numpy().astype(np.float32, copy=False)


def _get_action_row(
    trajectory: Trajectory,
    env_step: int,
    batch_idx: int,
    *,
    num_action_chunks: int,
    env_action_dim: int,
) -> np.ndarray:
    forward_inputs = trajectory.forward_inputs or {}
    action_key = "expert_action" if "expert_action" in forward_inputs else "action"
    if action_key not in forward_inputs:
        raise KeyError("Trajectory forward_inputs missing expert_action/action.")
    tensor = forward_inputs[action_key]
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    num_chunk_steps = int(tensor.shape[0])
    chunk_idx = max(0, env_step // int(num_action_chunks))
    chunk_idx = _clip(chunk_idx, 0, num_chunk_steps - 1)
    flat = tensor[chunk_idx, batch_idx].detach().cpu().numpy().astype(np.float32)
    width = int(num_action_chunks * env_action_dim)
    if flat.size < width:
        padded = np.zeros(width, dtype=np.float32)
        padded[: flat.size] = flat.reshape(-1)
        flat = padded
    actions = flat.reshape(int(num_action_chunks), env_action_dim)
    intra = _clip(env_step % int(num_action_chunks), 0, int(num_action_chunks) - 1)
    return actions[intra].astype(np.float32, copy=False)


def _resolve_language(
    trajectory: Trajectory,
    chunk_anchor: int,
    batch_idx: int,
    *,
    language_model_key: str,
) -> str:
    curr_obs = trajectory.curr_obs or {}
    obs = {k: v[chunk_anchor] if torch.is_tensor(v) else v for k, v in curr_obs.items()}
    obs = restore_task_descriptions_from_obs(obs)
    prompts = obs.get("task_descriptions")
    if prompts is None:
        return ""
    if isinstance(prompts, (list, tuple)):
        if len(prompts) == 0:
            return ""
        item = prompts[batch_idx] if batch_idx < len(prompts) else prompts[-1]
        if isinstance(item, (list, tuple)):
            return str(item[-1] if item else "")
        return str(item)
    return str(prompts)


def build_offline_sft_raw_from_trajectory(
    trajectory: Trajectory,
    chunk_anchor: int,
    batch_idx: int,
    *,
    layout: RolloutObsLayout,
    language_model_key: str,
    action_model_key: str,
    layout_cfg: DaggerOfflineSftLayout,
    env_action_dim: int,
) -> dict[str, Any]:
    """Build a single-sample modality dict matching ``DreamZeroLeRobotDataset`` output."""
    num_chunk_steps = _trajectory_chunk_steps(trajectory)
    if chunk_anchor < 0 or chunk_anchor >= num_chunk_steps:
        raise EmptyTemporalSampleError(
            f"chunk_anchor {chunk_anchor} out of range for trajectory len {num_chunk_steps}"
        )

    n = int(layout_cfg.num_action_chunks)
    ep_len = max(1, num_chunk_steps * n)
    anchor_env = max(0, chunk_anchor * n - 1)
    language = np.zeros(ep_len, dtype=np.int64)

    temporal_cfg = MultiAnchorTemporalConfig(
        max_chunk_size=int(layout_cfg.max_chunk_size),
        macro_stride=int(layout_cfg.macro_stride),
        action_horizon=int(layout_cfg.action_horizon),
    )
    temporal = require_multi_anchor_temporal_indices(
        anchor_env,
        language,
        ep_len,
        temporal_cfg,
        episode_index=chunk_anchor,
    )

    sample: dict[str, Any] = {}
    for env_key, model_key in layout.video_fields:
        frames = [
            _get_video_frame(
                trajectory,
                int(env_step),
                batch_idx,
                env_key=env_key,
                num_action_chunks=n,
            )
            for env_step in temporal.video
        ]
        sample[model_key] = np.stack(frames, axis=0)

    state_rows = [
        _get_state_row(
            trajectory,
            int(env_step),
            batch_idx,
            num_action_chunks=n,
        )
        for env_step in temporal.state
    ]
    for _, model_target in layout.state_fields:
        if isinstance(model_target, str):
            sample[model_target] = np.stack(state_rows, axis=0)
        else:
            raise NotImplementedError(
                "Multi-field state layouts are not supported for DAgger offline SFT align."
            )

    action_rows = [
        _get_action_row(
            trajectory,
            int(env_step),
            batch_idx,
            num_action_chunks=n,
            env_action_dim=env_action_dim,
        )
        for env_step in temporal.action
    ]
    sample[action_model_key] = np.stack(action_rows, axis=0)
    sample[language_model_key] = _resolve_language(
        trajectory, chunk_anchor, batch_idx, language_model_key=language_model_key
    )
    return sample


def sample_valid_window_index(
    trajectory: Trajectory,
    *,
    layout_cfg: DaggerOfflineSftLayout,
    rng: np.random.Generator,
    max_attempts: int = 32,
) -> tuple[int, int]:
    """Return ``(chunk_anchor, batch_idx)`` that yields a valid multi_anchor window."""
    curr_obs = trajectory.curr_obs or {}
    main = curr_obs.get("main_images")
    if main is None:
        raise ValueError("Trajectory curr_obs is missing main_images.")
    num_chunk_steps = int(main.shape[0])
    batch_size = int(main.shape[1])
    n = int(layout_cfg.num_action_chunks)
    ep_len = max(1, num_chunk_steps * n)
    temporal_cfg = MultiAnchorTemporalConfig(
        max_chunk_size=int(layout_cfg.max_chunk_size),
        macro_stride=int(layout_cfg.macro_stride),
        action_horizon=int(layout_cfg.action_horizon),
    )
    language = np.zeros(ep_len, dtype=np.int64)

    for _ in range(max_attempts):
        chunk_anchor = int(rng.integers(0, num_chunk_steps))
        batch_idx = int(rng.integers(0, batch_size))
        anchor_env = max(0, chunk_anchor * n - 1)
        try:
            require_multi_anchor_temporal_indices(
                anchor_env,
                language,
                ep_len,
                temporal_cfg,
                episode_index=chunk_anchor,
            )
            return chunk_anchor, batch_idx
        except EmptyTemporalSampleError:
            continue
    raise EmptyTemporalSampleError(
        f"Failed to find valid multi_anchor window in trajectory "
        f"(chunk_steps={num_chunk_steps}, batch={batch_size}) after {max_attempts} attempts."
    )
