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

"""Temporal helpers for DreamZero rollout / DAgger replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Packed UTF-8 instructions for replay (splittable ``[B, 2+max_bytes]`` tensor).
TASK_DESCRIPTION_PACKED_KEY: str = "task_description_packed"
TASK_DESCRIPTION_MAX_BYTES: int = 512

# Offsets relative to the last frame in a chunk (0 = most recent micro-step).
ROLLOUT_AR_FRAME_OFFSETS_FROM_LAST: tuple[int, ...] = (-15, -10, -5, 0)


def resolve_ar_frame_indices(
    num_frames: int,
    offsets: tuple[int, ...] = ROLLOUT_AR_FRAME_OFFSETS_FROM_LAST,
) -> list[int]:
    """Map AR offsets to absolute indices in ``[0, num_frames - 1]``."""
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


def select_rollout_tail_frame_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the last frame along the chunk time axis (``T=1``).

    Expects ``main_images`` shaped ``[B, T, H, W, C]`` (and matching ``wrist_images`` /
    ``states``). Returns the same keys with ``T=1``.
    """
    if not _has_temporal_dim(env_obs):
        return env_obs

    out = dict(env_obs)
    main = env_obs["main_images"]
    if isinstance(main, torch.Tensor):
        last_idx = int(main.shape[1]) - 1
    else:
        last_idx = int(np.asarray(main).shape[1]) - 1
    frame_indices = [last_idx]

    for key in ("main_images", "wrist_images", "extra_view_images", "states"):
        if key not in env_obs or env_obs[key] is None:
            continue
        value = env_obs[key]
        arr = value if isinstance(value, torch.Tensor) else np.asarray(value)
        if arr.ndim == 3 and key == "states":
            out[key] = _index_along_time(value, frame_indices, time_axis=1)
        elif arr.ndim == 5:
            out[key] = _index_along_time(value, frame_indices, time_axis=1)
    return out


def select_rollout_temporal_obs(
    env_obs: dict[str, Any],
    *,
    ar_first_step: bool,
    frame_offsets: tuple[int, ...] = ROLLOUT_AR_FRAME_OFFSETS_FROM_LAST,
) -> dict[str, Any]:
    """Select frames for causal WAN rollout (AR subsampling)."""
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
        if key not in env_obs or env_obs[key] is None:
            continue
        value = env_obs[key]
        arr = value if isinstance(value, torch.Tensor) else np.asarray(value)
        if arr.ndim == 3 and key == "states":
            out[key] = _index_along_time(value, frame_indices, time_axis=1)
        elif arr.ndim == 5:
            out[key] = _index_along_time(value, frame_indices, time_axis=1)

    return out


def _infer_env_obs_batch_size(env_obs: dict[str, Any]) -> int:
    main = env_obs.get("main_images")
    if main is None:
        return 1
    if isinstance(main, torch.Tensor):
        return int(main.shape[0])
    return int(np.asarray(main).shape[0])


def pack_task_descriptions(
    descriptions: list[str] | Any,
    *,
    batch_size: int,
    max_bytes: int = TASK_DESCRIPTION_MAX_BYTES,
) -> torch.Tensor:
    """Encode ``task_descriptions`` as ``[B, 2+max_bytes]`` uint8 (length-prefixed UTF-8)."""
    if isinstance(descriptions, str):
        desc_list = [descriptions] * batch_size
    elif descriptions is None:
        desc_list = [""] * batch_size
    else:
        desc_list = [str(d if d is not None else "") for d in list(descriptions)]
        if len(desc_list) != batch_size:
            raise ValueError(
                f"task_descriptions length {len(desc_list)} != batch_size {batch_size}."
            )

    buf = torch.zeros(len(desc_list), 2 + max_bytes, dtype=torch.uint8)
    for i, desc in enumerate(desc_list):
        raw = desc.encode("utf-8")[:max_bytes]
        n = len(raw)
        if n > 0xFFFF:
            raise ValueError(f"Task description too long ({n} bytes > 65535).")
        buf[i, 0] = (n >> 8) & 0xFF
        buf[i, 1] = n & 0xFF
        if n:
            buf[i, 2 : 2 + n] = torch.tensor(list(raw), dtype=torch.uint8)
    return buf.contiguous()


def unpack_task_descriptions(packed: torch.Tensor) -> list[str]:
    """Decode tensors produced by :func:`pack_task_descriptions`."""
    if packed.dim() == 1:
        packed = packed.unsqueeze(0)
    descriptions: list[str] = []
    for row in packed.cpu().numpy():
        n = (int(row[0]) << 8) | int(row[1])
        if n <= 0:
            descriptions.append("")
            continue
        descriptions.append(bytes(row[2 : 2 + n]).decode("utf-8", errors="replace"))
    return descriptions


def stash_task_descriptions_in_obs(env_obs: dict[str, Any]) -> None:
    """Replace ``task_descriptions`` with a stackable uint8 tensor for replay."""
    if "task_descriptions" not in env_obs:
        return
    descriptions = env_obs.pop("task_descriptions")
    batch_size = _infer_env_obs_batch_size(env_obs)
    if (
        isinstance(descriptions, (list, tuple))
        and len(descriptions) > 0
        and isinstance(descriptions[0], (list, tuple))
    ):
        descriptions = [seq[-1] if seq else "" for seq in descriptions]
    env_obs[TASK_DESCRIPTION_PACKED_KEY] = pack_task_descriptions(
        descriptions,
        batch_size=batch_size,
    )


def restore_task_descriptions_from_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Decode ``task_description_packed`` back to ``task_descriptions``."""
    if TASK_DESCRIPTION_PACKED_KEY not in env_obs:
        return env_obs
    out = dict(env_obs)
    packed = out.pop(TASK_DESCRIPTION_PACKED_KEY)
    if isinstance(packed, torch.Tensor):
        out["task_descriptions"] = unpack_task_descriptions(packed)
    else:
        out["task_descriptions"] = unpack_task_descriptions(
            torch.as_tensor(packed, dtype=torch.uint8)
        )
    return out
