"""Data loader for NativeMEM LeRobot datasets with precomputed memory tokens."""

from collections.abc import Sequence
import logging
import random
from typing import Any, Literal

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.training.data_loader import (
    DataLoader,
    DataLoaderImpl,
    FakeDataset,
    TorchDataLoader,
    TransformedDataset,
    transform_dataset,
)
import openpi.transforms as _transforms


class MemLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobot dataset that assembles NativeMEM `memory` from cached `mem_token` features."""

    def __init__(
        self,
        repo_id: str,
        intervals: tuple[int, int] = (8, 16),
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(repo_id, **kwargs)
        self._intervals = intervals
        self._rng = random.Random(seed)

    def _get_memory_indices(self, curr_idx: int, ep_idx: int) -> list[int]:
        ep_start = self.episode_data_index["from"][ep_idx].item()
        interval = self._rng.randint(self._intervals[0], self._intervals[1])
        indices = list(range(curr_idx, ep_start - 1, -interval))
        if indices[-1] != ep_start:
            indices.append(ep_start)
        indices.reverse()
        return indices

    def _query_memory(self, indices: Sequence[int]) -> np.ndarray:
        items = self.hf_dataset.select(indices)
        tokens = [np.asarray(items[i]["mem_token"], dtype=np.float32) for i in range(len(indices))]
        return np.concatenate(tokens, axis=0)

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"]
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, value in query_result.items():
                item[key] = value

        item["memory"] = self._query_memory(self._get_memory_indices(idx, ep_idx))

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for camera_key in self.meta.camera_keys:
                item[camera_key] = self.image_transforms(item[camera_key])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        for key in [key for key in item if key.startswith("trans_tokens.") or key.startswith("prev_indices.")]:
            del item[key]

        return item


def create_mem_torch_dataset(
    data_config: Any,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    seed: int = 0,
) -> Any:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = MemLeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        intervals=data_config.mem_intervals,
        seed=seed,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_mem_torch_data_loader(
    data_config: Any,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    dataset = create_mem_torch_dataset(data_config, action_horizon, model_config, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info("local_batch_size: %s", local_batch_size)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )
    return DataLoaderImpl(data_config, data_loader)


def create_mem_data_loader(
    config: Any,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_config)
    return create_mem_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


create_data_loader = create_mem_data_loader
