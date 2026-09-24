"""DP3 dataset adapter for fixed-shape real-robot Zarr trajectories.

The converted Zarr stores arrays under ``data/`` using the names
``point_cloud``, ``state`` and ``action``.  DP3 expects observations named
``point_cloud`` and ``agent_pos``, so this class performs that small naming
adaptation and samples fixed-length sequences without crossing episode
boundaries.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import zarr

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


class GenericZarrDataset(BaseDataset):
    """Load cropped XYZ point clouds and robot state/action trajectories.

    Expected per-frame shapes in the converted Zarr are:

    Subclasses set the expected point-cloud, state, and action shapes.
    """

    expected_point_cloud_shape = (1024, 3)
    expected_state_shape = (8,)
    expected_action_shape = (9,)

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
        task_name: str | None = None,
        grouped_validation: bool = False,
        point_cloud_noise_std: float = 0.0,
    ):
        super().__init__()

        self.task_name = task_name
        zarr_path = self._resolve_zarr_path(zarr_path)
        # RGB images are intentionally absent: DP3 consumes XYZ + robot state.
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["state", "action", "point_cloud"],
        )
        self._validate_replay_buffer()

        # Split whole episodes, not individual frames, to prevent train/val
        # leakage between neighboring samples from the same demonstration.
        if grouped_validation:
            groups = zarr.open_group(zarr_path, mode="r").attrs.get("source_episode_names")
            if groups is None or len(groups) != self.replay_buffer.n_episodes:
                raise ValueError(
                    "grouped_validation requires one source_episode_names entry per episode"
                )
            group_names = list(dict.fromkeys(str(value) for value in groups))
            group_val_mask = get_val_mask(len(group_names), val_ratio, seed)
            validation_groups = {
                name for name, selected in zip(group_names, group_val_mask) if selected
            }
            val_mask = np.asarray(
                [str(value) in validation_groups for value in groups], dtype=bool
            )
        else:
            val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.point_cloud_noise_std = float(point_cloud_noise_std)
        if self.point_cloud_noise_std < 0:
            raise ValueError("point_cloud_noise_std must be non-negative")

    @staticmethod
    def _resolve_zarr_path(zarr_path: str) -> str:
        """Resolve task-relative paths even when ``train.py`` changes cwd."""
        path = Path(zarr_path).expanduser()
        if path.is_absolute() or path.exists():
            return str(path)

        # This module lives at
        # <project>/diffusion_policy_3d/dataset/, while task YAML paths are
        # relative to <project>.  The upstream train entry point changes the
        # working directory before Hydra starts, so cwd alone is unreliable.
        project_relative_path = Path(__file__).resolve().parents[2] / path
        if project_relative_path.exists():
            return str(project_relative_path)

        raise FileNotFoundError(
            f"DP3 Zarr dataset not found at {path!s} or "
            f"{project_relative_path!s}. Run the conversion script first or "
            "set task.dataset.zarr_path explicitly."
        )

    def _validate_replay_buffer(self) -> None:
        expected_shapes = {
            "point_cloud": self.expected_point_cloud_shape,
            "state": self.expected_state_shape,
            "action": self.expected_action_shape,
        }
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(self.replay_buffer[key].shape[1:])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Unexpected {key} shape {actual_shape}; expected {expected_shape}. "
                    "Check the final-Zarr conversion dimensions."
                )

    def get_validation_dataset(self) -> "GenericZarrDataset":
        val_dataset = copy.copy(self)
        val_dataset.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_dataset.train_mask = ~self.train_mask
        val_dataset.point_cloud_noise_std = 0.0
        return val_dataset

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        # Each last dimension is normalized independently.
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = np.asarray(self.replay_buffer["action"][:], dtype=np.float32)
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: dict[str, np.ndarray]) -> dict:
        point_cloud = sample["point_cloud"].astype(np.float32, copy=False)
        if self.point_cloud_noise_std:
            point_cloud = point_cloud + np.random.normal(
                0.0, self.point_cloud_noise_std, point_cloud.shape
            ).astype(np.float32)
        return {
            "obs": {
                "point_cloud": point_cloud,
                # DP3 calls the proprioceptive state "agent_pos".
                "agent_pos": sample["state"].astype(np.float32, copy=False),
            },
            "action": sample["action"].astype(np.float32, copy=False),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(index)
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)
