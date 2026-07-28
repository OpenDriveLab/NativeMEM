import dataclasses
import gc
import logging
from pathlib import Path
import random
import shutil
from typing import Literal

import cv2
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import h5py
import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from openpi_client import image_tools
import openpi.models.model as _model
import openpi.models.siglip as _siglip
from openpi.models.siglip_encoder import SiglipEncoder
from openpi.models.mem_tokenizer_config import MemTokenizerConfig
import openpi.shared.download as download
from openpi.models.model import restore_params
from openpi.shared import array_typing as at
from openpi.training import weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 8
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
BATCH_SIZE = 64


def _stringify_keys(d):
    """Recursively convert dict keys to strings for robust merging."""
    if isinstance(d, dict):
        return {str(k): _stringify_keys(v) for k, v in d.items()}
    return d


def _restore_keys(ref, merged):
    """Restore integer keys where the reference pytree used them."""
    if not isinstance(ref, dict) or not isinstance(merged, dict):
        return merged
    result = {}
    ref_key_map = {str(k): k for k in ref}
    for k, v in merged.items():
        orig_key = ref_key_map.get(k, k)
        result[orig_key] = _restore_keys(ref.get(orig_key, {}), v)
    return result


def _require_checkpoint_prefixes(loaded_params, prefixes: tuple[str, ...]) -> None:
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    for prefix in prefixes:
        if any(key.startswith(prefix + "/") for key in flat_loaded):
            continue
        raise ValueError(
            "Checkpoint is missing required subtree "
            f"'{prefix}'. Expected a mem_tokenizer checkpoint with {prefixes}."
        )


class _MemTokenExtractor(nnx.Module):
    """Minimal module matching the mem_tokenizer encoder subtree."""

    def __init__(self, config: MemTokenizerConfig, rngs: nnx.Rngs):
        siglip_params = _siglip.decode_variant(config.siglip_variant)

        encoder = nnx_bridge.ToNNX(
            SiglipEncoder(
                patch_size=siglip_params["patch_size"],
                width=siglip_params["width"],
                depth=siglip_params["depth"],
                mlp_dim=siglip_params["mlp_dim"],
                num_heads=siglip_params["num_heads"],
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        encoder.lazy_init(jnp.zeros((1, 224, 224, 3), dtype=jnp.float32), train=False, rngs=rngs)
        self.encoder = encoder


@nnx.jit
def _extract_mem_tokens(model: _MemTokenExtractor, images: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return one memory token per frame-view pair with shape [B, 3, D]."""
    mem_tokens_per_view = []
    for key in _model.IMAGE_KEYS:
        mem_token, _, _ = model.encoder(images[key], train=False, extract_block=None, ids_keep=None)
        mem_tokens_per_view.append(mem_token)
    return jax.lax.stop_gradient(jnp.concatenate(mem_tokens_per_view, axis=1))


class MemTokenExtractor:
    """Extract per-view memory tokens from a pretrained memory tokenizer."""

    def __init__(self, params_path: str, model_config: MemTokenizerConfig):
        self.model_config = model_config
        self.token_dim = _siglip.decode_variant(self.model_config.siglip_variant)["width"]
        self.model = _MemTokenExtractor(model_config, rngs=nnx.Rngs(0))

        path = download.maybe_download(params_path)
        loaded_params = restore_params(path, restore_type=np.ndarray)
        _require_checkpoint_prefixes(loaded_params, ("encoder",))

        graphdef, state = nnx.split(self.model)
        params_shape = state.to_pure_dict()

        try:
            merged_params = weight_loaders._merge_params(loaded_params, params_shape, missing_regex=".*")
        except TypeError:
            str_params = _stringify_keys(params_shape)
            merged_params = weight_loaders._merge_params(loaded_params, str_params, missing_regex=".*")
            merged_params = _restore_keys(params_shape, merged_params)

        at.check_pytree_equality(
            expected=params_shape,
            got=merged_params,
            check_shapes=True,
            check_dtypes=False,
        )

        state.replace_by_pure_dict(merged_params)
        nnx.update(self.model, state)
        self.model = nnx.merge(graphdef, state)
        self.model.eval()

        logging.info("Loaded mem_tokenizer checkpoint from %s", params_path)

    @staticmethod
    def _prepare_images(decoded_images: np.ndarray) -> np.ndarray:
        """Resize decoded BGR images to 224x224 and normalize to [-1, 1]."""
        images = image_tools.resize_with_pad(decoded_images, 224, 224)
        return images.astype(np.float32) / 255.0 * 2.0 - 1.0

    def __call__(
        self,
        head_images: np.ndarray,
        left_images: np.ndarray,
        right_images: np.ndarray,
    ) -> np.ndarray:
        """Extract memory tokens for all decoded frames, returning [N, 3, D]."""
        images_by_key = {
            "base_0_rgb": self._prepare_images(head_images),
            "left_wrist_0_rgb": self._prepare_images(left_images),
            "right_wrist_0_rgb": self._prepare_images(right_images),
        }

        n_frames = images_by_key["base_0_rgb"].shape[0]
        all_tokens = []
        for start in range(0, n_frames, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_frames)
            batch_images = {key: jnp.array(value[start:end]) for key, value in images_by_key.items()}
            batch_size = batch_images["base_0_rgb"].shape[0]
            if batch_size < BATCH_SIZE:
                batch_images = {
                    key: jnp.pad(value, ((0, BATCH_SIZE - batch_size), (0, 0), (0, 0), (0, 0)), mode="edge")
                    for key, value in batch_images.items()
                }
                mem_tokens = _extract_mem_tokens(self.model, batch_images)[:batch_size]
            else:
                mem_tokens = _extract_mem_tokens(self.model, batch_images)
            all_tokens.append(np.array(mem_tokens))
        return np.concatenate(all_tokens, axis=0)


def _decode_images(compressed_images: np.ndarray) -> np.ndarray:
    """Decode one camera stream from the HDF5 file."""
    images = []
    for encoded in compressed_images:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode an image from HDF5.")
        images.append(image)
    return np.stack(images, axis=0)


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    token_dim: int,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    img_keys = list(_model.IMAGE_KEYS)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "mem_token": {
            "dtype": "float32",
            "shape": (len(img_keys), token_dim),
            "names": [img_keys],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_file: Path,
    tokenizer: MemTokenExtractor,
    instruction: str = "",
) -> LeRobotDataset:
    with h5py.File(hdf5_file, "r") as f:
        episode_ends = f["episode_ends"][:]
        episode_idx = f["episode_idx"][:]

        for ep_idx in range(len(episode_ends)):
            begin_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            end_idx = episode_ends[ep_idx]
            group_idx, _ = episode_idx[begin_idx]

            # Decode each JPEG once, then reuse the images for both token extraction
            # and the LeRobot dataset.
            head_images = _decode_images(f[f"episode_{group_idx:0>8d}/cam_head"][:])
            left_images = _decode_images(f[f"episode_{group_idx:0>8d}/cam_left"][:])
            right_images = _decode_images(f[f"episode_{group_idx:0>8d}/cam_right"][:])

            if "instructions" in f[f"episode_{group_idx:0>8d}"].keys():
                instructions = f[f"episode_{group_idx:0>8d}/instructions"][()]
                instructions = [
                    instr.decode("utf-8") if isinstance(instr, bytes) else str(instr) for instr in instructions
                ]
                instruction = random.choice(instructions)
            else:
                print(
                    f"Warning: No instruction found for episode {group_idx}, "
                    f"using default instruction: {instruction}."
                )

            action = f["action"][begin_idx:end_idx]
            state = f["state"][begin_idx:end_idx]

            num_frames = state.shape[0]
            mem_tokens = tokenizer(head_images, left_images, right_images)

            if mem_tokens.shape[0] != num_frames:
                raise ValueError(
                    f"memory token frame count mismatch: tokens={mem_tokens.shape[0]}, frames={num_frames}"
                )

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": instruction,
                    "mem_token": mem_tokens[i],
                    "observation.images.cam_high": head_images[i].transpose(2, 0, 1),
                    "observation.images.cam_right_wrist": right_images[i].transpose(2, 0, 1),
                    "observation.images.cam_left_wrist": left_images[i].transpose(2, 0, 1),
                }
                dataset.add_frame(frame)

            dataset.save_episode()
            logging.info("Converted episode %d with %d frames.", ep_idx, num_frames)
            gc.collect()

    return dataset


def port_aloha(
    data_path: Path,
    repo_id: str,
    params_path: str = "checkpoints/mem_tokenizer_pretrain/<exp>/<step>/params",
    instruction: str = "",
    *,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    siglip_variant: str = "So400m/14",
    dtype: str = "float32",
):
    init_logging()

    tokenizer = MemTokenExtractor(
        params_path=params_path,
        model_config=MemTokenizerConfig(
            siglip_variant=siglip_variant,
            dtype=dtype,
        ),
    )

    logging.info("Creating dataset at %s", repo_id)
    dataset = create_empty_dataset(
        repo_id,
        robot_type="aloha",
        token_dim=tokenizer.token_dim,
        mode=mode,
        dataset_config=dataset_config,
    )
    populate_dataset(
        dataset,
        data_path,
        tokenizer=tokenizer,
        instruction=instruction,
    )


if __name__ == "__main__":
    tyro.cli(port_aloha)


"""
Example usage:
python examples/robotwin/convert_h5_to_lerobot_mem_tokenizer.py \
    --data_path /path/to/train.h5 \
    --repo_id my_task_nativemem \
    --params_path checkpoints/mem_tokenizer_pretrain/my_encoder/49999/params \
    --instruction "task description"
"""
