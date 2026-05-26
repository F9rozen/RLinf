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

"""Temporal subsampling for DreamZero rollout observations (AR inference)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Offsets relative to the last frame in a chunk (0 = most recent / last micro-step).
ROLLOUT_AR_FRAME_OFFSETS_FROM_LAST: tuple[int, ...] = (-15, -10, -5, 0)


def resolve_ar_frame_indices(
    num_frames: int,
    offsets: tuple[int, ...] = ROLLOUT_AR_FRAME_OFFSETS_FROM_LAST,
) -> list[int]:
    """Map AR offsets to absolute indices in ``[0, num_frames - 1]`` (chronological order)."""
    if num_frames <= 0:
        return []
    last = num_frames - 1
    indices: list[int] = []
    for off in offsets:
        idx = last if off == 0 else last + off
        indices.append(int(np.clip(idx, 0, last)))
    return indices


def _index_along_time(value: Any, indices: list[int], time_axis: int = 1) -> Any:
    if not indices:
        return value
    if isinstance(value, torch.Tensor):
        index = torch.tensor(indices, device=value.device, dtype=torch.long)
        return value.index_select(time_axis, index)
    arr = np.asarray(value)
    return np.take(arr, indices, axis=time_axis)


def _has_temporal_dim(env_obs: dict[str, Any], key: str = "main_images") -> bool:
    if key not in env_obs or env_obs[key] is None:
        return False
    arr = env_obs[key]
    if isinstance(arr, torch.Tensor):
        return arr.ndim == 5
    return np.asarray(arr).ndim == 5


def select_rollout_temporal_obs(
    env_obs: dict[str, Any],
    *,
    ar_first_step: bool,
    frame_offsets: tuple[int, ...] = ROLLOUT_AR_FRAME_OFFSETS_FROM_LAST,
) -> dict[str, Any]:
    """Select frames for causal WAN rollout.

    Video and state use the same time indices:
    - First AR step: last frame only → ``[B, 1, ...]``
    - Later steps: offsets ``(-15, -10, -5, 0)`` (for ``T=16`` → ``[0, 5, 10, 15]`` → ``[B, 4, ...]``)
    """
    if not _has_temporal_dim(env_obs):
        return env_obs

    out = dict(env_obs)
    main = env_obs["main_images"]
    if isinstance(main, torch.Tensor):
        num_frames = int(main.shape[1])
    else:
        num_frames = int(np.asarray(main).shape[1])

    if ar_first_step:
        frame_indices = [num_frames - 1]
    else:
        frame_indices = resolve_ar_frame_indices(num_frames, frame_offsets)

    for key in ("main_images", "wrist_images", "extra_view_images", "states"):
        if key in env_obs and env_obs[key] is not None:
            value = env_obs[key]
            arr = value if isinstance(value, torch.Tensor) else np.asarray(value)
            if arr.ndim == 3 and key == "states":
                # [B, T, D]
                out[key] = _index_along_time(value, frame_indices, time_axis=1)
            elif arr.ndim == 5:
                out[key] = _index_along_time(value, frame_indices, time_axis=1)

    return out
