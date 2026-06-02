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

import logging
from typing import Any, Optional

import numpy as np
import torch
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.model.dreamzero.base_vla import VLA
from tianshou.data import Batch

from rlinf.data.datasets.dreamzero.data_transforms import (
    collect_dreamzero_dataset_keys,
    convert_rollout_env_obs,
    rollout_obs_layout_for_embodiment,
)
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform
from rlinf.data.datasets.dreamzero.rollout_temporal_obs import (
    restore_task_descriptions_from_obs,
    select_rollout_tail_frame_obs,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.dreamzero.dreamzero_config import DreamZeroConfig


class DreamZeroPolicy(VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    # CausalWanModel has to be wrapped to avoid a FSDP2 bug
    # when using with gradient checkpointing
    _no_split_modules = [
        "T5SelfAttention",  # text encoder
        "AttentionBlock",  # vae
        "CausalWanModel",  # action head
        "CausalWanAttentionBlock",  # action head layer
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config
        embodiment_tag = config.embodiment_tag
        if embodiment_tag is None:
            raise ValueError(
                "DreamZeroPolicy requires config.embodiment_tag (set in get_model)."
            )
        self._rollout_obs_layout = rollout_obs_layout_for_embodiment(embodiment_tag)
        _, _, action_keys, _ = collect_dreamzero_dataset_keys(
            config.data_transforms, embodiment_tag
        )
        self._action_keys = tuple(action_keys)

    _DAGGER_FORWARD_RESERVED_KEYS: frozenset[str] = frozenset(
        {"action", "expert_action", "model_action", "prev_logprobs", "prev_values"}
    )

    # This method is called in FSDPModelManager.setup_model_and_optimizer
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            enabled = True
            use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

            if diffusion_model is None:
                raise ValueError("DreamZero policy must have action_head.")

            if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                diffusion_model._set_gradient_checkpointing(diffusion_model, enabled)
            elif hasattr(diffusion_model, "gradient_checkpointing"):
                diffusion_model.gradient_checkpointing = enabled

            setattr(
                diffusion_model, "gradient_checkpointing_use_reentrant", use_reentrant
            )

            logging.warning(
                "DreamZero gradient checkpointing is enabled. If you encounter errors "
                "or memory leaks, consider: (1) upgrading to PyTorch 2.10 or later; "
                "(2) using use_reentrant=True to avoid issues when CUDA graphs and "
                "gradient checkpointing are used together."
            )

        except Exception:
            pass

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Run the forward modality pipeline on rollout observations.

        Input ``batch.obs`` is already in DreamZero modality keys (e.g.
        ``video.image``, ``state.state``, language key) from
        ``_observation_convert``. This method delegates to
        ``config.data_transforms``, built in ``get_model`` from Hydra cfg and
        ``metadata.json`` (via ``load_dreamzero_dataset_metadata`` +
        ``data_transforms.set_metadata``).

        Pipeline (libero_sim example, see ``libero_sim._build_composed_transform``):

        1. Video / state / action preprocessing and normalization
           (``StateActionTransform`` uses q99 stats from metadata).
        2. ``ConcatTransform.apply``: concat per-key tensors into flat
           ``state`` / ``action`` vectors. Per-key widths come from metadata
           (e.g. ``action.actions`` shape ``[7]`` for Libero).
        3. ``DreamTransform.apply``: pad state/action to ``max_state_dim`` /
           ``max_action_dim`` (typically 32 from yaml) so the WAN action head
           always sees a fixed width. Extra padded dims are zeros and masked
           during training; at inference the model still outputs width 32.

        The returned ``batch.normalized_obs`` is the dict consumed by
        ``lazy_joint_video_action_causal`` (tokens, video, padded actions, etc.).
        """
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Invert model actions back to environment-scale per-modality tensors.

        ``batch.normalized_action`` is ``action_pred`` from the WAN head, shape
        ``[..., max_action_dim]`` (e.g. 32), matching the padded width from
        ``DreamTransform.apply``. Environment DOF is smaller (e.g. Libero 7);
        that width is **not** taken from Hydra ``action_dim`` on the policy—it
        comes from ``metadata.json`` loaded at build time:

        - ``get_model`` calls ``data_transforms.set_metadata(metadata)``.
        - ``ConcatTransform.set_metadata`` sets ``action_dims["action.actions"]``
          from ``metadata.modalities.action.<key>.shape[0]`` (7 for libero_sim).
        - On ``unapply``, transforms run in reverse order:
          ``DreamTransform.unapply`` (passthrough) →
          ``ConcatTransform.unapply`` slices ``[..., 0:env_dim]`` per
          ``action_concat_order`` → ``StateActionTransform.unapply`` reverses
          q99 normalization.

        Output is a dict like ``{"action.actions": tensor}`` with **env** width
        (7 for Libero). ``predict_action_batch`` then merges keys via
        ``_actions_from_unapply`` for the sim.

        If ``relative_action`` / ``relative_action_per_horizon`` is enabled,
        optionally adds the last ``state.*`` from ``obs`` (converted rollout
        obs passed from ``predict_action_batch``) to obtain absolute actions.
        """
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": batch.normalized_action.cpu()}
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.config.relative_action
        relative_action_per_horizon = self.config.relative_action_per_horizon
        relative_action_keys = self.config.relative_action_keys
        if (
            (relative_action or relative_action_per_horizon)
            and relative_action_keys
            and obs is not None
        ):
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                # Try to find the state data - check multiple possible key formats
                last_state = None

                # Format 1: Direct key like "state.joint_position"
                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and "state" in obs:
                        state_data = obs["state"]
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep

                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(
                        last_state, axis=-2
                    )  # Add horizon dimension

                # Add state to relative action to get absolute action
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )

        batch.act = unnormalized_action
        return batch

    def _process_batch(self, batch: Batch) -> dict[str, Any]:
        """Process batch."""
        # Normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # If the normalized input is still a Batch, flatten it into a pure dict
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        # Do dtype cast if needed
        target_dtype = next(self.parameters()).dtype
        for k, v in normalized_input.items():
            if (
                torch.is_tensor(v)
                and v.dtype == torch.float32
                and target_dtype != torch.float32
            ):
                normalized_input[k] = v.to(dtype=target_dtype)
        return normalized_input

    def _observation_convert(self, env_obs: dict) -> dict:
        """Map RLinf rollout observations to DreamZero modality keys."""
        return convert_rollout_env_obs(self.config.embodiment_tag, env_obs)

    @staticmethod
    def _last_frame_task_descriptions(
        task_descriptions: Any,
    ) -> list[str] | str | None:
        if task_descriptions is None:
            return None
        if (
            isinstance(task_descriptions, (list, tuple))
            and len(task_descriptions) > 0
            and isinstance(task_descriptions[0], (list, tuple))
        ):
            return [seq[-1] for seq in task_descriptions]
        return task_descriptions

    @staticmethod
    def _env_obs_has_temporal_dim(env_obs: dict[str, Any]) -> bool:
        main = env_obs.get("main_images")
        if main is None:
            return False
        if torch.is_tensor(main):
            return main.ndim == 5
        return np.asarray(main).ndim == 5

    # DAgger SFT video frame count for the WAN action head.  Must yield a VAE latent
    # count ``T_latent = 1 + (T - 1) // 4`` such that:
    #   1) ``(T_latent - 1) % num_frame_per_block == 0`` (timestep block reshape).
    #   2) ``action_horizon / (T_latent - 1) == num_action_per_block // num_frame_per_block``.
    # With ``action_horizon=24, num_frame_per_block=2, num_action_per_block=24``,
    # the only valid T_latent is 3, which gives T = 9 frames.
    _DAGGER_SFT_NUM_FRAMES: int = 9

    def _build_replay_env_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Keep 9 uniformly sampled video frames and the last proprio step."""
        replay = dict(env_obs)
        main = replay.get("main_images")
        if main is not None:
            if torch.is_tensor(main):
                nf = int(main.shape[1])
                if nf > self._DAGGER_SFT_NUM_FRAMES:
                    indices = np.linspace(0, nf - 1, self._DAGGER_SFT_NUM_FRAMES, dtype=int)
                    replay["main_images"] = main[:, indices].contiguous()
                    for key in ("wrist_images", "extra_view_images"):
                        if key in replay and replay[key] is not None:
                            val = replay[key]
                            if isinstance(val, torch.Tensor) and val.ndim >= 5:
                                replay[key] = val[:, indices].contiguous()
            else:
                arr = np.asarray(main)
                nf = int(arr.shape[1])
                if nf > self._DAGGER_SFT_NUM_FRAMES:
                    indices = np.linspace(0, nf - 1, self._DAGGER_SFT_NUM_FRAMES, dtype=int)
                    replay["main_images"] = arr[:, indices]
                    for key in ("wrist_images", "extra_view_images"):
                        if key in replay and replay[key] is not None:
                            val = np.asarray(replay[key])
                            if val.ndim >= 5:
                                replay[key] = val[:, indices]
        states = replay.get("states")
        if states is not None:
            if torch.is_tensor(states):
                if states.ndim >= 3 and int(states.shape[1]) > 1:
                    replay["states"] = states[:, -1:].contiguous()
            else:
                arr = np.asarray(states)
                if arr.ndim >= 3 and int(arr.shape[1]) > 1:
                    replay["states"] = arr[:, -1:]
        if "task_descriptions" in replay:
            replay["task_descriptions"] = self._last_frame_task_descriptions(
                replay["task_descriptions"]
            )
        return replay

    def _normalize_env_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        converted_obs = self._observation_convert(env_obs)
        return self._process_batch(Batch(obs=converted_obs))

    def _dagger_env_action_dim(self) -> int:
        """Environment action DOF from dataset metadata (e.g. 7 for Libero)."""
        try:
            metadata = self.config.data_transforms.dataset_metadata
            action_key = self._action_keys[0].split(".", 1)[-1]
            return int(metadata.modalities.action[action_key].shape[0])
        except Exception:
            return 7

    def _dagger_get_rollout_layout(self) -> tuple[int, int, int, int]:
        """Return ``(action_horizon, max_action_dim, num_env_chunks, env_action_dim)``."""
        action_horizon = int(getattr(self.config, "action_horizon", 24) or 24)
        max_action_dim = int(getattr(self.config, "action_dim", 32) or 32)
        num_chunks = int(getattr(self.config, "num_action_chunks", 16) or 16)
        env_action_dim = self._dagger_env_action_dim()
        transform = self.config.data_transforms
        if transform is not None:
            for t in getattr(transform, "transforms", []):
                if hasattr(t, "action_horizon") and t.action_horizon:
                    action_horizon = int(t.action_horizon)
                if hasattr(t, "max_action_dim") and t.max_action_dim:
                    max_action_dim = int(t.max_action_dim)
                if hasattr(t, "num_chunks") and t.num_chunks:
                    num_chunks = int(t.num_chunks)
        return action_horizon, max_action_dim, num_chunks, env_action_dim

    def _normalize_env_action_chunk(
        self, env_actions: torch.Tensor, *, max_action_dim: int
    ) -> torch.Tensor:
        """Normalize one env-scale action chunk through the same transforms as SFT data."""
        if not self._action_keys:
            raise ValueError("DreamZero DAgger: missing action modality keys.")
        action_key = self._action_keys[0]
        transform = self.config.data_transforms
        if transform is None:
            raise ValueError("DreamZero DAgger: data_transforms is not configured.")

        chunk = env_actions.detach().cpu().float()
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        data: dict[str, Any] = {action_key: chunk.numpy().astype(np.float32, copy=False)}
        action_keys = set(self._action_keys)

        for step in transform.transforms:
            if isinstance(step, DreamTransform):
                break
            if isinstance(step, ConcatTransform):
                data = step.apply(data)
                continue
            apply_to = set(getattr(step, "apply_to", []) or [])
            if not apply_to.intersection(action_keys):
                continue
            data = step.apply(data)

        if "action" not in data:
            raise ValueError(
                "DreamZero DAgger: action transforms did not produce `action`; "
                f"available keys: {sorted(data)}."
            )

        normalized = data["action"]
        if torch.is_tensor(normalized):
            normalized = normalized.detach().cpu().numpy()
        normalized = np.asarray(normalized, dtype=np.float32)
        env_dim = normalized.shape[-1]
        if env_dim < max_action_dim:
            normalized = np.pad(
                normalized,
                ((0, 0), (0, max_action_dim - env_dim)),
                mode="constant",
            )
        return torch.from_numpy(normalized).float()

    @staticmethod
    def _collapse_dagger_sample_batch(tensor: torch.Tensor) -> torch.Tensor:
        """Replay samples are already ``[N, ...]``; micro-batch split may add a leading ``1``."""
        return tensor.contiguous()

    def _prepare_dagger_chunk_actions(
        self, batch: dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        """Normalize env-scale replay actions to ``[B, num_chunks, max_action_dim]``."""
        action_horizon, max_action_dim, num_chunks, env_action_dim = self._dagger_get_rollout_layout()
        if "expert_action" in batch:
            actions = batch["expert_action"]
        elif "action" in batch:
            actions = batch["action"]
        else:
            raise ValueError(
                "DreamZero DAgger SFT requires `expert_action` or `action` "
                "in forward_inputs."
            )
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = actions.float()
        if actions.dim() > 1:
            actions = self._collapse_dagger_sample_batch(actions)

        expected_env_flat = num_chunks * env_action_dim
        expected_model_flat = action_horizon * env_action_dim

        def _reshape_env_actions(flat_actions: torch.Tensor) -> torch.Tensor:
            width = int(flat_actions.shape[-1])
            bsz = int(flat_actions.shape[0])
            if width == expected_env_flat:
                return flat_actions.reshape(bsz, num_chunks, env_action_dim)
            if width == expected_model_flat:
                return flat_actions.reshape(
                    bsz, action_horizon, env_action_dim
                )[:, :num_chunks, :]
            raise ValueError(
                "DreamZero DAgger: expected flat action width "
                f"{expected_env_flat} (= num_action_chunks * env_dim) or "
                f"{expected_model_flat} (= action_horizon * env_dim), got {width}."
            )

        if actions.dim() == 1:
            if int(actions.numel()) == expected_env_flat:
                actions = actions.unsqueeze(0)
            elif int(actions.numel()) == expected_model_flat:
                actions = actions.unsqueeze(0)
            elif int(actions.numel()) % expected_env_flat == 0:
                actions = actions.reshape(-1, expected_env_flat)
            elif int(actions.numel()) % expected_model_flat == 0:
                actions = actions.reshape(-1, expected_model_flat)
            else:
                raise ValueError(
                    "DreamZero DAgger: flat action length "
                    f"{int(actions.numel())} is not a multiple of "
                    f"{expected_env_flat} or {expected_model_flat}."
                )
        elif actions.dim() == 2:
            width = int(actions.shape[-1])
            if width not in (expected_env_flat, expected_model_flat):
                raise ValueError(
                    "DreamZero DAgger: expected action width "
                    f"{expected_env_flat} or {expected_model_flat}, got {width}."
                )
        else:
            raise ValueError(
                "DreamZero DAgger: env action must be flat "
                f"[B, {expected_env_flat}] or [B, {expected_model_flat}], "
                f"got {tuple(actions.shape)}."
            )

        bsz = int(actions.shape[0])
        env_chunks = _reshape_env_actions(actions)
        normalized = [
            self._normalize_env_action_chunk(env_chunks[i], max_action_dim=max_action_dim)
            for i in range(bsz)
        ]
        return torch.stack(normalized, dim=0).to(device=device, dtype=torch.float32)

    def _pad_actions_for_vla_forward(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Pad env-chunk actions ``[B, num_chunks, D]`` to model ``action_horizon``."""
        model_action_horizon = int(
            getattr(self.config, "action_horizon", None)
            or getattr(self, "action_horizon", 24)
            or 24
        )
        model_action_dim = int(
            getattr(self.config, "action_dim", None)
            or getattr(self, "action_dim", 32)
            or 32
        )
        bsz, num_valid_steps, action_width = actions.shape
        actions = actions[..., :model_action_dim]
        if action_width < model_action_dim:
            pad_dim = torch.zeros(
                bsz,
                num_valid_steps,
                model_action_dim - action_width,
                dtype=actions.dtype,
                device=actions.device,
            )
            actions = torch.cat([actions, pad_dim], dim=-1)
        if num_valid_steps < model_action_horizon:
            pad_time = torch.zeros(
                bsz,
                model_action_horizon - num_valid_steps,
                model_action_dim,
                dtype=actions.dtype,
                device=actions.device,
            )
            actions = torch.cat([actions, pad_time], dim=1)
        elif num_valid_steps > model_action_horizon:
            actions = actions[:, :model_action_horizon, :model_action_dim]
            num_valid_steps = model_action_horizon
        return actions.contiguous(), num_valid_steps

    def _merge_sft_observation_and_actions(
        self,
        observation: dict[str, Any],
        actions: torch.Tensor,
    ) -> dict[str, Any]:
        """Build ``VLA.forward`` inputs from one replay chunk (16 frames / 1 state / 16 actions)."""
        inputs = dict(observation)
        actions, num_valid_steps = self._pad_actions_for_vla_forward(actions)
        inputs["action"] = actions
        device = actions.device
        bsz = actions.shape[0]
        action_horizon = actions.shape[1]
        max_action_dim = actions.shape[2]

        if "action_mask" not in inputs:
            _, _, _, env_action_dim = self._dagger_get_rollout_layout()
            valid_dim = min(max_action_dim, env_action_dim)
            action_mask = torch.zeros(
                bsz, action_horizon, max_action_dim, dtype=torch.bool, device=device
            )
            action_mask[:, :num_valid_steps, :valid_dim] = True
            inputs["action_mask"] = action_mask

        if "has_real_action" not in inputs:
            inputs["has_real_action"] = torch.ones(bsz, dtype=torch.bool, device=device)

        if "embodiment_id" not in inputs:
            embodiment_id = 21
            transform = self.config.data_transforms
            if transform is not None:
                for t in getattr(transform, "transforms", []):
                    mapping = getattr(t, "embodiment_tag_mapping", None)
                    tag = getattr(self.config, "embodiment_tag", None)
                    if mapping is not None and tag is not None and tag in mapping:
                        embodiment_id = int(mapping[tag])
                        break
            inputs["embodiment_id"] = torch.full(
                (bsz,), embodiment_id, dtype=torch.long, device=device
            )

        if "state_mask" not in inputs and "state" in inputs and torch.is_tensor(
            inputs["state"]
        ):
            state = inputs["state"]
            state_mask = torch.zeros_like(state, dtype=torch.bool, device=device)
            valid_state_dim = 8
            try:
                metadata = self.config.data_transforms.dataset_metadata
                st = metadata.statistics.state["state"]
                valid_state_dim = len(np.asarray(st.q01))
            except Exception:
                valid_state_dim = min(
                    state.shape[-1],
                    int(getattr(self.config, "max_state_dim", state.shape[-1])),
                )
            valid_state_dim = min(valid_state_dim, state.shape[-1])
            state_mask[..., :valid_state_dim] = True
            inputs["state_mask"] = state_mask

        return inputs

    def _actions_from_unapply(self, act_dict: dict[str, Any]) -> np.ndarray:
        """Concatenate per-key unnormalized actions in dataset concat order."""
        parts: list[np.ndarray] = []
        for key in self._action_keys:
            if key not in act_dict:
                raise KeyError(
                    f"Unnormalized action missing {key!r}; "
                    f"available keys: {sorted(act_dict)}."
                )
            value = act_dict[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value))
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=-1)

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - wrist_images: [B,H,W,C] (optional, embodiment-specific)
                - extra_view_images: [B,N,H,W,C] (optional, e.g. oxe_droid)
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, action_dim]
            result: dict  # compatible with rollout interface"""
        build_dagger_replay = self._env_obs_has_temporal_dim(env_obs)
        inference_env_obs = env_obs
        if build_dagger_replay:
            inference_env_obs = select_rollout_tail_frame_obs(env_obs)
            if "task_descriptions" in env_obs:
                inference_env_obs["task_descriptions"] = (
                    self._last_frame_task_descriptions(env_obs["task_descriptions"])
                )

        converted_obs = self._observation_convert(inference_env_obs)
        batch = Batch(obs=converted_obs)
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        action_horizon, max_action_dim, num_chunks, _ = self._dagger_get_rollout_layout()
        if normalized_action.dim() == 3 and int(normalized_action.shape[1]) > action_horizon:
            normalized_action = normalized_action[:, :action_horizon, :max_action_dim]

        batch = self.unapply(
            Batch(normalized_action=normalized_action),
            obs=converted_obs,
        )
        actions = self._actions_from_unapply(batch.act)

        if actions.ndim == 2:
            actions = actions.reshape(
                actions.shape[0], action_horizon, -1
            )
        elif actions.ndim == 3 and int(actions.shape[1]) > action_horizon:
            actions = actions[:, :action_horizon]

        if self._rollout_obs_layout.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
            )

        # Model predicts up to action_horizon steps; env rollout executes num_chunks.
        if int(actions.shape[1]) > num_chunks:
            env_actions = actions[:, :num_chunks, :]
        else:
            env_actions = actions

        bsz = env_actions.shape[0]
        flat = (
            torch.as_tensor(env_actions, dtype=torch.float32)
            .reshape(bsz, -1)
            .cpu()
        )

        forward_inputs: dict[str, Any] = {}
        if build_dagger_replay:
            # Env execution uses flat env-scale action; obs for SFT comes from curr_obs.
            forward_inputs["action"] = flat
            # Ensure every key in _DAGGER_FORWARD_RESERVED_KEYS is always present so
            # stack_list_of_dict_tensor sees consistent keys across all entries.
            forward_inputs["model_action"] = flat
            forward_inputs["expert_action"] = flat

        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((bsz, 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return env_actions, result

    def _curr_obs_to_replay_env_obs(self, curr_obs: dict[str, Any]) -> dict[str, Any]:
        """Collapse replay batch dim and restore raw env obs for normalization."""
        env_obs: dict[str, Any] = {}
        for key, value in curr_obs.items():
            if torch.is_tensor(value):
                env_obs[key] = self._collapse_dagger_sample_batch(value).detach().cpu()
            else:
                env_obs[key] = value
        env_obs = restore_task_descriptions_from_obs(env_obs)
        if not self._env_obs_has_temporal_dim(env_obs):
            raise ValueError(
                "DreamZero DAgger SFT requires temporal curr_obs "
                "(main_images shaped [B, T, H, W, C])."
            )
        return self._build_replay_env_obs(env_obs)

    def _prepare_dagger_sft_observation(
        self,
        curr_obs: dict[str, Any],
        *,
        device: torch.device,
        param_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run raw replay env obs through the same normalizer as inference."""
        replay_env_obs = self._curr_obs_to_replay_env_obs(curr_obs)
        replay_normalized = self._normalize_env_obs(replay_env_obs)
        observation: dict[str, Any] = {}
        for key, value in replay_normalized.items():
            if not torch.is_tensor(value):
                continue
            tensor = value.to(device=device)
            if tensor.dtype == torch.float32 and param_dtype != torch.float32:
                tensor = tensor.to(dtype=param_dtype)
            observation[key] = tensor
        if not observation:
            raise ValueError(
                "DreamZero DAgger: normalizer produced no observation tensors."
            )
        if "state" in observation and observation["state"].dim() == 2:
            observation["state"] = observation["state"].unsqueeze(1)
        return observation

    def prepare_dagger_sft_batch(
        self,
        batch: dict[str, Any],
        *,
        curr_obs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rebuild one replay chunk via inference-time normalizer (video/state/action)."""
        device = next(self.parameters()).device
        param_dtype = next(self.parameters()).dtype

        if curr_obs is None:
            raise ValueError(
                "DreamZero DAgger SFT requires `curr_obs` with 16-frame video; "
                "forward_inputs observation tensors are not reused."
            )

        observation = self._prepare_dagger_sft_observation(
            curr_obs, device=device, param_dtype=param_dtype
        )
        actions = self._prepare_dagger_chunk_actions(batch, device)
        return {
            "_dagger_replay_batch": True,
            "observation": observation,
            "actions": actions,
        }

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data=None, **kwargs):
        # Mark the start of each training iteration so PyTorch knows when
        # to reclaim memory held by CUDA graphs from the previous iteration.
        torch.compiler.cudagraph_mark_step_begin()

        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")

        if data.get("_dagger_replay_batch"):
            inputs = self._merge_sft_observation_and_actions(
                data["observation"], data["actions"]
            )
            outputs = super().forward(inputs)
            loss = outputs.get("loss") if hasattr(outputs, "get") else None
            if loss is None:
                raise ValueError("sft_forward requires `loss` in the outputs.")
            return loss

        outputs = super().forward(data)
        if hasattr(outputs, "data"):
            outputs = outputs.data
        if "loss" not in outputs:
            raise ValueError("sft_forward requires `loss` in the outputs.")
        return dict(outputs)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        raise NotImplementedError
