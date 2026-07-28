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
                f"`python scripts/compute_norm_stats_h5.py ... --gripper-type {dataset_type}`."
            )


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    # Norm stats per dataset_type. Each entry maps {"state", "actions"} -> NormStats.
    norm_stats: dict[str, dict[str, _transforms.NormStats]] | None = None
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
            for key in ("image_seq", "image_seq_mask", "history_positions", "dataset_type")
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
    """Dispatches Normalize by `dataset_type` in the sample, then drops that field."""

    normalizers: dict  # dataset_type -> _transforms.Normalize

    def __call__(self, data: dict) -> dict:
        if not self.normalizers:
            # No norm stats loaded (e.g. fake config) — match Normalize(None) no-op behavior.
            data.pop("dataset_type", None)
            return data
        dataset_type = data.get("dataset_type", None)
        normalize = self.normalizers.get(dataset_type)
        if normalize is None:
            raise KeyError(
                f"No norm stats loaded for dataset_type={dataset_type!r}; "
                f"available: {sorted(self.normalizers)}"
            )
        out = normalize(data)
        out.pop("dataset_type", None)  # string field; must not reach batching.
        return out


def build_model_transforms(
    model_config: _model.BaseModelConfig,
    *,
    norm_stats: dict[str, dict[str, _transforms.NormStats]] | None,
    default_prompt: str,
    gripper_type: aloha_policy.GripperType | None,
) -> tuple[_transforms.DataTransformFn, ...]:
    # Build one Normalize per dataset_type so we don't reinstantiate per sample.
    per_type_stats = norm_stats or {}
    normalizers = {
        dataset_type: _transforms.Normalize(stats, use_quantiles=True)
        for dataset_type, stats in per_type_stats.items()
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
    # Base directory containing one subdirectory per dataset_type (sim/ trossen/ arx/),
    # each holding a norm_stats.json. Local path or gs:// URI.
    norm_stats_assets_dir: str = "assets"

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        raise NotImplementedError

    def _load_norm_stats(self) -> dict[str, dict[str, _transforms.NormStats]] | None:
        base = pathlib.Path(_download.maybe_download(self.norm_stats_assets_dir))
        out: dict[str, dict[str, _transforms.NormStats]] = {}
        for dataset_type in DATASET_TYPES:
            try:
                stats = _normalize.load(base / dataset_type)
                _validate_norm_stats(dataset_type, stats)
                out[dataset_type] = stats
            except FileNotFoundError:
                logging.info("Norm stats not found in %s/%s", self.norm_stats_assets_dir, dataset_type)
        return out or None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        norm_stats = self._load_norm_stats()
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
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        norm_stats = self._load_norm_stats()
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
    ):
        self._h5_path = h5_path
        self._history_seq_len = history_seq_len
        self._seq_len = history_seq_len + 1
        self._action_horizon = action_horizon
        self._stride_range = stride_range
        self._load_in_memory = load_in_memory
        self._default_prompt = default_prompt
        self._h5_file = None
        self._episodes: list[dict] = []
        self._global_to_local: list[tuple[int, int]] = []
        self._local_to_global: dict[tuple[int, int], int] = {}
        self._valid_samples: list[tuple[int, int, int, int]] = []
        self._episode_cache: dict[int, dict] | None = {} if load_in_memory else None

        with h5py.File(h5_path, "r") as h5_file:
            episode_names = sorted(
                name
                for name in h5_file
                if name.startswith("episode_") and isinstance(h5_file.get(name), h5py.Group)
            )
            for episode_index, episode_name in enumerate(episode_names):
                episode_group = h5_file.get(episode_name)
                if not isinstance(episode_group, h5py.Group):
                    raise TypeError(f"Expected '{episode_name}' to be an HDF5 group, got {type(episode_group)}")
                state_dataset = episode_group.get("state")
                if state_dataset is None:
                    raise KeyError(f"Episode '{episode_name}' is missing the 'state' dataset.")
                episode_len = int(state_dataset.shape[0])
                dataset_type_attr = episode_group.attrs.get("dataset_type", None)
                if isinstance(dataset_type_attr, bytes):
                    dataset_type_attr = dataset_type_attr.decode("utf-8")
                dataset_type_attr = str(dataset_type_attr)
                self._episodes.append({
                    "index": episode_index,
                    "name": episode_name,
                    "length": episode_len,
                    "dataset_type": dataset_type_attr,
                })

                if self._episode_cache is not None:
                    instructions_dataset = episode_group.get("instructions")
                    if instructions_dataset is None:
                        raise KeyError(f"Episode '{episode_name}' is missing the 'instructions' dataset.")
                    self._episode_cache[episode_index] = {
                        "cam_head": episode_group.get("cam_head")[:],
                        "cam_left": episode_group.get("cam_left")[:],
                        "cam_right": episode_group.get("cam_right")[:],
                        "state": state_dataset[:],
                        "action": episode_group.get("action")[:],
                        "instructions": [
                            instruction.decode("utf-8") if isinstance(instruction, bytes) else str(instruction)
                            for instruction in instructions_dataset[:]
                        ],
                    }

                for local_index in range(episode_len):
                    global_index = len(self._global_to_local)
                    self._global_to_local.append((episode_index, local_index))
                    self._local_to_global[(episode_index, local_index)] = global_index

        self._pre_calculate_sampling_indices()

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
        for global_index, (episode_index, local_index) in enumerate(self._global_to_local):
            rng = np.random.RandomState(global_index)
            stride = int(rng.randint(self._stride_range[0], self._stride_range[1] + 1))
            episode_len = self._episodes[episode_index]["length"]
            if local_index < (self._history_seq_len - 1) * stride:
                continue
            if local_index + self._action_horizon * 0.75 > episode_len:
                continue
            self._valid_samples.append((global_index, episode_index, local_index, stride))

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
        global_index, episode_index, local_index, stride = self._valid_samples[index]
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
        dataset = H5MemTokenizerDataset(
            data_config.repo_id,
            history_seq_len=data_config.history_seq_len,
            action_horizon=config.model.action_horizon,
            stride_range=data_config.stride_range,
            load_in_memory=data_config.load_in_memory,
            default_prompt=data_config.default_prompt,
        )
    dataset = TransformedDataset(dataset, data_config.transforms)

    return DataLoaderImpl(
        data_config,
        TorchDataLoader(
            dataset,
            local_batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=config.num_workers,
            seed=config.seed,
        ),
    )
