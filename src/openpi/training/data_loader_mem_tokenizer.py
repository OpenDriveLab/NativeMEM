"""Data loader for MemTokenizer temporal pretraining."""

import abc
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import queue
import threading
from typing import TYPE_CHECKING, Iterator

import cv2
import h5py
import jax
import numpy as np
import torch
from typing_extensions import override
import tyro

from openpi_client import image_tools
import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
from openpi.shared import normalize as _normalize
import openpi.shared.download as _download
from openpi.training.data_loader_mem import DataLoaderImpl, FakeDataset, TorchDataLoader, TransformedDataset

if TYPE_CHECKING:
    import openpi.training.config_mem_tokenizer as _config
import openpi.transforms as _transforms


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

DATASET_TYPES: tuple[str, ...] = ("trossen", "sim", "arx")
_GRIPPER_INDICES = np.asarray([6, 13])


def _validate_norm_stats(
    dataset_type: str,
    norm_stats: dict[str, _transforms.NormStats],
) -> None:
    """Reject stats computed in raw robot gripper space instead of pi0 space."""
    for feature in ("state", "actions"):
        stats = norm_stats.get(feature)
        if stats is None or stats.q01 is None or stats.q99 is None:
            continue
        if stats.q01.shape[-1] <= int(_GRIPPER_INDICES[-1]):
            continue
        q01 = np.asarray(stats.q01)[_GRIPPER_INDICES]
        q99 = np.asarray(stats.q99)[_GRIPPER_INDICES]
        if np.any(q01 < -0.05) or np.any(q99 > 1.2):
            raise ValueError(
                f"{dataset_type!r} {feature} norm stats contain gripper quantiles outside pi0 space: "
                f"q01={q01}, q99={q99}. Regenerate them with "
                "`python scripts/prepare_mem_tokenizer_h5.py` with the correct dataset_type key."
            )


@dataclasses.dataclass(frozen=True)
class H5Dataset:
    """One `scripts/prepare_mem_tokenizer_h5.py` output and the directory holding its norm_stats.json."""

    path: str
    norm_stats_dir: str


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    # Norm stats per sample `norm_key` (the H5 path). Each entry maps {"state", "actions"} -> NormStats.
    norm_stats: dict[str, dict[str, _transforms.NormStats]] | None = None
    # The H5 files to train on, each normalized with its own stats.
    h5_datasets: tuple[H5Dataset, ...] = ()
    transforms: Sequence[_transforms.DataTransformFn] = dataclasses.field(default_factory=tuple)
    use_quantile_norm: bool = True
    load_in_memory: bool = False
    history_seq_len: int = 8
    stride_range: tuple[int, int] = (8, 16)
    default_prompt: str = ""
    gripper_type: aloha_policy.GripperType | None = None


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AlohaTemporalInputs(_transforms.DataTransformFn):
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        dataset_type = data.get("dataset_type", None)
        extras = {
            key: data[key]
            for key in ("image_seq", "image_seq_mask", "history_positions", "dataset_type", "norm_key")
            if key in data
        }
        canonical = aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi, gripper_type=dataset_type)(
            {
                key: data[key]
                for key in ("images", "state", "actions", "prompt")
                if key in data
            }
        )
        return {**canonical, **extras}


@dataclasses.dataclass(frozen=True)
class PerDatasetNormalize(_transforms.DataTransformFn):
    """Dispatches Normalize by `norm_key` in the sample, then drops it and `dataset_type`."""

    normalizers: dict  # norm_key -> _transforms.Normalize

    def __call__(self, data: dict) -> dict:
        if not self.normalizers:
            # No norm stats loaded (e.g. fake config) — match Normalize(None) no-op behavior.
            data.pop("dataset_type", None)
            data.pop("norm_key", None)
            return data
        norm_key = data.get("norm_key")
        normalize = self.normalizers.get(norm_key)
        if normalize is None:
            raise KeyError(f"No norm stats loaded for norm_key={norm_key!r}; available: {sorted(self.normalizers)}")
        out = normalize(data)
        # String fields; must not reach batching.
        out.pop("dataset_type", None)
        out.pop("norm_key", None)
        return out


def build_model_transforms(
    model_config: _model.BaseModelConfig,
    *,
    norm_stats: dict[str, dict[str, _transforms.NormStats]] | None,
    default_prompt: str,
    gripper_type: aloha_policy.GripperType | None,
) -> tuple[_transforms.DataTransformFn, ...]:
    # Build one Normalize per norm_key so we don't reinstantiate per sample.
    normalizers = {
        norm_key: _transforms.Normalize(stats, use_quantiles=True)
        for norm_key, stats in (norm_stats or {}).items()
        if stats is not None
    }
    return (
        AlohaTemporalInputs(adapt_to_pi=True),
        _transforms.DeltaActions(_transforms.make_bool_mask(6, -1, 6, -1)),
        PerDatasetNormalize(normalizers=normalizers),
        _transforms.InjectDefaultPrompt(default_prompt),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=True,
        ),
        _transforms.PadStatesAndActions(model_config.action_dim),
    )


# ---------------------------------------------------------------------------
# DataConfig factories
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    history_seq_len: int = 8
    stride_range: tuple[int, int] = (8, 16)
    load_in_memory: bool = False
    default_prompt: str = ""
    # None means use each H5 episode's dataset_type attr as the gripper type.
    gripper_type: aloha_policy.GripperType | None = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        norm_stats = None
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.repo_id,
            norm_stats=norm_stats,
            transforms=build_model_transforms(
                model_config,
                norm_stats=norm_stats,
                default_prompt=self.default_prompt,
                gripper_type=self.gripper_type,
            ),
            load_in_memory=self.load_in_memory,
            history_seq_len=self.history_seq_len,
            stride_range=self.stride_range,
            default_prompt=self.default_prompt,
            gripper_type=self.gripper_type,
        )


@dataclasses.dataclass(frozen=True)
class H5MemTokenizerDataConfig(DataConfigFactory):
    repo_id: str = "h5_datasets"
    # `scripts/prepare_mem_tokenizer_h5.py` outputs (one dataset_type each), each with its own norm stats.
    datasets: tuple[H5Dataset, ...] = ()

    def _load_h5_norm_stats(self) -> dict[str, dict[str, _transforms.NormStats]]:
        out = {}
        for dataset in self.datasets:
            stats = _normalize.load(pathlib.Path(_download.maybe_download(dataset.norm_stats_dir)))
            _validate_norm_stats(dataset.path, stats)
            out[dataset.path] = stats
        return out

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        if not self.datasets:
            raise ValueError("H5MemTokenizerDataConfig.datasets is empty; list the H5 files to train on.")
        norm_stats = self._load_h5_norm_stats()
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.repo_id,
            norm_stats=norm_stats,
            h5_datasets=self.datasets,
            transforms=build_model_transforms(
                model_config,
                norm_stats=norm_stats,
                default_prompt=self.default_prompt,
                gripper_type=self.gripper_type,
            ),
            load_in_memory=self.load_in_memory,
            history_seq_len=self.history_seq_len,
            stride_range=self.stride_range,
            default_prompt=self.default_prompt,
            gripper_type=self.gripper_type,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _decode_attr(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class H5MemTokenizerDataset(torch.utils.data.Dataset):
    _CAM_TO_IMAGE_KEY = {
        "cam_head": "base_0_rgb",
        "cam_left": "left_wrist_0_rgb",
        "cam_right": "right_wrist_0_rgb",
    }
    _CAM_TO_ALOHA_KEY = {
        "cam_head": "cam_high",
        "cam_left": "cam_left_wrist",
        "cam_right": "cam_right_wrist",
    }

    def __init__(
        self,
        h5_path: str,
        *,
        history_seq_len: int,
        action_horizon: int,
        stride_range: tuple[int, int],
        load_in_memory: bool = False,
        default_prompt: str = "",
        norm_key: str,
    ):
        self._h5_path = h5_path
        self._norm_key = norm_key
        self._history_seq_len = history_seq_len
        self._seq_len = history_seq_len + 1
        self._action_horizon = action_horizon
        self._stride_range = stride_range
        self._load_in_memory = load_in_memory
        self._default_prompt = default_prompt
        self._h5_file = None
        self._episodes: list[dict] = []
        # Rows of (global_index, episode_index, local_index, stride); an array keeps worker pickling cheap.
        self._valid_samples: np.ndarray
        self._episode_cache: dict[int, dict] | None = {} if load_in_memory else None

        with h5py.File(h5_path, "r") as h5_file:
            self._episodes = self._index_single_type(h5_file)

            if self._episode_cache is not None:
                for episode in self._episodes:
                    episode_group = h5_file[episode["name"]]
                    instructions_dataset = episode_group.get("instructions")
                    if instructions_dataset is None:
                        raise KeyError(f"Episode '{episode['name']}' is missing the 'instructions' dataset.")
                    self._episode_cache[episode["index"]] = {
                        "cam_head": episode_group["cam_head"][:],
                        "cam_left": episode_group["cam_left"][:],
                        "cam_right": episode_group["cam_right"][:],
                        "state": episode_group["state"][:],
                        "action": episode_group["action"][:],
                        "instructions": [
                            instruction.decode("utf-8") if isinstance(instruction, bytes) else str(instruction)
                            for instruction in instructions_dataset[:]
                        ],
                    }

        self._pre_calculate_sampling_indices()

    @staticmethod
    def _index_single_type(h5_file: h5py.File) -> list[dict]:
        """Index a `scripts/prepare_mem_tokenizer_h5.py` output from its top-level `episode_ends` alone.

        That script names episodes episode_{i:08d} in `episode_ends` order and writes one dataset_type per file,
        so opening every episode group is unnecessary.
        """
        lengths = np.diff(np.asarray(h5_file["episode_ends"][:], dtype=np.int64), prepend=0)
        last_name = f"episode_{len(lengths) - 1:08d}"
        dataset_type = _decode_attr(h5_file["episode_00000000"].attrs["dataset_type"])
        # Cheap guards against a merged multi-type file, whose episodes would get the wrong gripper conversion.
        last_type = _decode_attr(h5_file[last_name].attrs["dataset_type"])
        if last_type != dataset_type or h5_file[last_name]["state"].shape[0] != lengths[-1]:
            raise ValueError(
                f"{h5_file.filename} is not a single-type prepare_mem_tokenizer_h5.py output: "
                f"first/last dataset_type {dataset_type!r}/{last_type!r}, last episode length "
                f"{h5_file[last_name]['state'].shape[0]} vs episode_ends {lengths[-1]}"
            )
        return [
            {"index": index, "name": f"episode_{index:08d}", "length": int(length), "dataset_type": dataset_type}
            for index, length in enumerate(lengths)
        ]

    def _ensure_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self._h5_path, "r")

    def _get_episode_data(self, episode_index: int) -> dict:
        if self._episode_cache is not None:
            return self._episode_cache[episode_index]

        self._ensure_open()
        episode_name = self._episodes[episode_index]["name"]
        assert self._h5_file is not None
        episode_group = self._h5_file.get(episode_name)
        if not isinstance(episode_group, h5py.Group):
            raise TypeError(f"Expected '{episode_name}' to be an HDF5 group, got {type(episode_group)}")
        state_dataset = episode_group.get("state")
        action_dataset = episode_group.get("action")
        instructions_dataset = episode_group.get("instructions")
        if state_dataset is None or action_dataset is None or instructions_dataset is None:
            raise KeyError(f"Episode '{episode_name}' is missing one of state/action/instructions datasets.")
        return {
            "cam_head": episode_group.get("cam_head"),
            "cam_left": episode_group.get("cam_left"),
            "cam_right": episode_group.get("cam_right"),
            "state": state_dataset,
            "action": action_dataset,
            "instructions": [
                instruction.decode("utf-8") if isinstance(instruction, bytes) else str(instruction)
                for instruction in instructions_dataset[:]
            ],
        }

    def _pre_calculate_sampling_indices(self):
        # One vectorized stride draw for all frames (fixed seed, storage order): a per-frame
        # RandomState(global_index) took ~20 min on a busy node for 2.6M frames.
        lengths = np.asarray([episode["length"] for episode in self._episodes], dtype=np.int64)
        episode_index = np.repeat(np.arange(len(lengths)), lengths)
        local_index = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        episode_len = lengths[episode_index]
        stride = np.random.default_rng(0).integers(
            self._stride_range[0], self._stride_range[1] + 1, size=len(local_index)
        )
        valid = (local_index >= (self._history_seq_len - 1) * stride) & (
            local_index + self._action_horizon * 0.75 <= episode_len
        )
        global_index = np.arange(len(local_index))
        self._valid_samples = np.stack([global_index, episode_index, local_index, stride], axis=1)[valid]

    def __len__(self) -> int:
        return len(self._valid_samples)

    @staticmethod
    def _decode_jpeg(encoded_image) -> np.ndarray:
        if isinstance(encoded_image, np.ndarray):
            encoded = encoded_image
        else:
            encoded = np.frombuffer(encoded_image, np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise ValueError("Failed to decode image from HDF5.")
        return image

    def __getitem__(self, index: int) -> dict:
        global_index, episode_index, local_index, stride = (int(value) for value in self._valid_samples[index])
        episode = self._get_episode_data(episode_index)

        history_indices = [0]
        history_indices.extend(
            local_index - step * stride for step in range(self._history_seq_len - 1, 0, -1)
        )
        history_indices.append(local_index)
        history_positions = np.asarray([frame_index // stride for frame_index in history_indices], dtype=np.int32)

        image_seq = {}
        image_seq_mask = {}
        images = {}
        for camera_name, image_key in self._CAM_TO_IMAGE_KEY.items():
            sequence_frames = np.stack(
                [self._decode_jpeg(episode[camera_name][idx]) for idx in history_indices],
                axis=0,
            )
            sequence_frames = image_tools.resize_with_pad(sequence_frames, 224, 224)
            image_seq[image_key] = sequence_frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            image_seq_mask[image_key] = np.ones((self._seq_len,), dtype=np.bool_)

            current_image = self._decode_jpeg(episode[camera_name][local_index])
            images[self._CAM_TO_ALOHA_KEY[camera_name]] = np.transpose(current_image, (2, 0, 1))

        action_slice = np.asarray(episode["action"][local_index : local_index + self._action_horizon], dtype=np.float32)
        if action_slice.shape[0] < self._action_horizon:
            pad = np.repeat(action_slice[-1:], self._action_horizon - action_slice.shape[0], axis=0)
            action_slice = np.concatenate([action_slice, pad], axis=0)

        instructions = episode["instructions"] or [self._default_prompt]
        prompt_rng = np.random.RandomState(global_index)
        prompt = instructions[int(prompt_rng.randint(0, len(instructions)))] if instructions else self._default_prompt

        return {
            "images": images,
            "image_seq": image_seq,
            "image_seq_mask": image_seq_mask,
            "history_positions": history_positions,
            "state": np.asarray(episode["state"][local_index], dtype=np.float32),
            "actions": action_slice,
            "prompt": prompt,
            "dataset_type": self._episodes[episode_index]["dataset_type"],
            "norm_key": self._norm_key,
        }


# ---------------------------------------------------------------------------
# PrefetchIterator
# ---------------------------------------------------------------------------

class PrefetchIterator:
    """Prefetch batches on a background thread to overlap host work with device compute."""

    def __init__(self, iterator: Iterator, prefetch_size: int = 2):
        self._iterator = iterator
        self._prefetch_size = max(prefetch_size, 1)
        self._queue: queue.Queue = queue.Queue(maxsize=self._prefetch_size)
        self._sentinel = object()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for item in self._iterator:
                self._queue.put(item)
        except Exception as exc:
            self._queue.put(exc)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------------------
# Data loader factory
# ---------------------------------------------------------------------------

def create_mem_tokenizer_data_loader(
    config: "_config.TrainConfig",
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoaderImpl:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_config)

    if data_config.repo_id == "fake":
        dataset = FakeDataset(config.model, num_samples=1024)
    else:
        dataset = torch.utils.data.ConcatDataset(
            [
                H5MemTokenizerDataset(
                    h5_dataset.path,
                    history_seq_len=data_config.history_seq_len,
                    action_horizon=config.model.action_horizon,
                    stride_range=data_config.stride_range,
                    load_in_memory=data_config.load_in_memory,
                    default_prompt=data_config.default_prompt,
                    norm_key=h5_dataset.path,
                )
                for h5_dataset in data_config.h5_datasets
            ]
        )
    dataset = TransformedDataset(dataset, data_config.transforms)

    # Multi-host: `config.batch_size` is the global batch. Each process contributes a disjoint slice of
    # it, which `TorchDataLoader` assembles into the global array. Without the sampler every host would
    # replay the same samples, so the effective batch would be `jax.process_count()` times smaller.
    local_batch_size = config.batch_size // jax.process_count()
    sampler = None
    if jax.process_count() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=jax.process_count(),
            rank=jax.process_index(),
            shuffle=shuffle,
            drop_last=True,
            seed=config.seed,
        )
    logging.info("local_batch_size: %s (global %s)", local_batch_size, config.batch_size)

    return DataLoaderImpl(
        data_config,
        TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=sharding,
            shuffle=shuffle,
            sampler=sampler,
            num_batches=num_batches,
            num_workers=config.num_workers,
            seed=config.seed,
        ),
    )
