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

"""RLinf :class:`DreamTransform` subclass with embodiment-specific multi-view concat."""

from __future__ import annotations

from typing import Any

import numpy as np
from einops import rearrange
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)


def resolve_registry_tag(embodiment_tag: Any) -> str:
    """Map Groot ``EmbodimentTag`` or registry tag string to ``_EMBODIMENT_REGISTRY`` key."""
    if hasattr(embodiment_tag, "value"):
        return str(embodiment_tag.value)
    return str(embodiment_tag)


def concat_generic_grid_views(images: np.ndarray) -> np.ndarray:
    """Fallback 2x2 grid for unknown embodiments."""
    v, t, c, h, w = images.shape
    concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
    if v > 0:
        concat_images[0, :, :, :h, :w] = images[0]
    if v > 1:
        concat_images[0, :, :, h:, :w] = images[1]
    if v > 2:
        concat_images[0, :, :, :h, w:] = images[2]
    return concat_images


def concat_multiview_video(embodiment_tag: Any, images: Any) -> np.ndarray:
    """Concat multi-view frames using the embodiment registered in ``_EMBODIMENT_REGISTRY``."""
    from rlinf.data.datasets.dreamzero.data_transforms import (
        _EMBODIMENT_REGISTRY,
        _require_embodiment,
    )

    arr = np.asarray(images)
    tag = resolve_registry_tag(embodiment_tag)
    if tag in _EMBODIMENT_REGISTRY:
        return _require_embodiment(tag).concat_multiview_video(arr)
    return concat_generic_grid_views(arr)


class DreamTransform(DreamTransformBase):
    """DreamTransform that delegates multi-view layout to ``data_transforms`` registry."""

    def _prepare_state(self, data: dict):
        """Use ``action_horizon`` for training; rollout keeps one state per video frame."""
        if "state" not in data:
            return super()._prepare_state(data)

        state = data["state"]
        if self.training:
            return super()._prepare_state(data)

        n_state_dims = state.shape[-1]
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            state = np.pad(
                state,
                ((0, 0), (0, self.max_state_dim - n_state_dims)),
                "constant",
            )
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True
        return state, state_mask, state.shape[0]

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        """Collate with RLinf prompt wrapping (supports all registered embodiments)."""
        import tree

        from rlinf.data.datasets.dreamzero.dreamzero import DreamZeroCollator

        data.pop("lapa_action", None)
        data.pop("dream_actions", None)
        data_split = [
            tree.map_structure(lambda x: x[i], data) for i in range(batch_size)
        ]
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return DreamZeroCollator.collate_batch(
            data_split_processed,
            self.tokenizer,
            self.embodiment_tag_mapping,
        )

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        if images.shape[0] > 1:
            return concat_multiview_video(self.embodiment_tag, images)
        return images
