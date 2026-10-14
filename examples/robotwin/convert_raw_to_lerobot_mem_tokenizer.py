"""Stage 1.5 in one step: raw RoboTwin/RMBench episodes -> LeRobot dataset with cached `mem_token` features.

Frames go through the same resize-with-pad + JPEG round trip as `scripts/prepare_mem_tokenizer_h5.py`, so the cached
tokens see the same pixels as Stage 1 memory-tokenizer training.

Example:
    python examples/robotwin/convert_raw_to_lerobot_mem_tokenizer.py \
        --data_dirs "data/RMBench/click_button/demo_clean/data" \
        --repo_id click_button_nativemem \
        --params_path checkpoints/mem_tokenizer_pretrain/my_encoder/49999/params
"""

import dataclasses
import gc
import glob
import logging
import pathlib
import random
import shutil
import sys
from typing import Literal

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
import tyro

# Running this file puts only examples/robotwin on sys.path; the repo root is needed for `scripts.*`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from openpi.models.mem_tokenizer_config import MemTokenizerConfig
import openpi.models.model as _model
from openpi.models.model import restore_params
import openpi.models.siglip as _siglip
from openpi.models.siglip_encoder import SiglipEncoder
from openpi.shared import array_typing as at
import openpi.shared.download as download
from openpi.training import weight_loaders
from scripts import prepare_mem_tokenizer_h5 as prepare


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
            f"Checkpoint is missing required subtree '{prefix}'. Expected a mem_tokenizer checkpoint with {prefixes}."
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


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    token_dim: int,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
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

    if pathlib.Path(HF_LEROBOT_HOME / repo_id).exists():
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


@dataclasses.dataclass
class Args:
    data_dirs: list[str]  # Raw `.../<task>/<demo>/data` dirs holding episode<N>.hdf5; glob patterns allowed.
    repo_id: str
    params_path: str
    mode: Literal["video", "image"] = "image"
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG
    siglip_variant: str = "So400m/14"
    dtype: str = "float32"
    # Must match the Stage 1 HDF5 preprocessing so cached tokens see the same pixels as tokenizer training.
    height: int = 224
    width: int = 224
    quality: int = 95
    instruction_key: str = "seen"
    instruction_map: pathlib.Path | None = None
    default_instruction: list[str] = dataclasses.field(default_factory=list)
    fallback_instruction: str | None = None
    # Seeds the per-episode choice among multiple instructions.
    seed: int = 0
    # Stage 2 finetunes on the first N episodes (by episode id) of each data dir; None keeps all.
    episode_num: int | None = 50


def expand_data_dirs(patterns: list[str]) -> list[pathlib.Path]:
    dirs = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise FileNotFoundError(f"Pattern {pattern!r} matched no directories")
        dirs.extend(pathlib.Path(match).expanduser() for match in matches)
    for dataset_dir in dirs:
        if not dataset_dir.is_dir():
            raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")
    return dirs


def main(args: Args) -> None:
    init_logging()
    dataset_dirs = expand_data_dirs(args.data_dirs)
    instruction_map = prepare.load_instruction_map(args.instruction_map, args.default_instruction)
    rng = random.Random(args.seed)

    tokenizer = MemTokenExtractor(
        params_path=args.params_path,
        model_config=MemTokenizerConfig(siglip_variant=args.siglip_variant, dtype=args.dtype),
    )
    logging.info("Creating dataset at %s", args.repo_id)
    dataset = create_empty_dataset(
        args.repo_id,
        robot_type="aloha",
        token_dim=tokenizer.token_dim,
        mode=args.mode,
        dataset_config=args.dataset_config,
    )

    num_episodes = 0
    for dataset_dir in dataset_dirs:
        # Numeric episode order, so the first N are episode0..episode{N-1} rather than episode0, episode1, episode10, ...
        episode_paths = sorted(
            {*dataset_dir.glob("*.hdf5"), *dataset_dir.glob("*.h5")},
            key=lambda path: int(prepare.EPISODE_ID_PATTERN.search(path.stem).group(1)),
        )
        if args.episode_num is not None:
            if len(episode_paths) < args.episode_num:
                raise ValueError(f"{dataset_dir} has {len(episode_paths)} episodes, fewer than {args.episode_num}")
            episode_paths = episode_paths[: args.episode_num]
        for episode_path in episode_paths:
            frames, state, action = prepare.read_raw_episode(episode_path, args.height, args.width, args.quality)
            instructions = prepare.load_episode_instructions(
                episode_path,
                dataset_dir,
                instruction_key=args.instruction_key,
                instruction_map=instruction_map,
                fallback_instruction=args.fallback_instruction,
            )
            instruction = rng.choice(instructions)

            # Decode the re-encoded JPEGs once; reuse them for token extraction and the LeRobot frames.
            head_images = np.stack([prepare.decode_image(f) for f in frames["cam_head"]])
            left_images = np.stack([prepare.decode_image(f) for f in frames["cam_left"]])
            right_images = np.stack([prepare.decode_image(f) for f in frames["cam_right"]])
            mem_tokens = tokenizer(head_images, left_images, right_images)

            for i in range(len(state)):
                dataset.add_frame(
                    {
                        "observation.state": state[i],
                        "action": action[i],
                        "mem_token": mem_tokens[i],
                        "observation.images.cam_high": head_images[i].transpose(2, 0, 1),
                        "observation.images.cam_right_wrist": right_images[i].transpose(2, 0, 1),
                        "observation.images.cam_left_wrist": left_images[i].transpose(2, 0, 1),
                    },
                    task=instruction,
                )
            dataset.save_episode()
            logging.info("Converted %s (%d frames) as episode %d.", episode_path, len(state), num_episodes)
            num_episodes += 1
            gc.collect()

    if num_episodes == 0:
        raise ValueError(f"No episodes found in {args.data_dirs}")
    dataset.stop_image_writer()
    if args.mode == "image":
        # save_episode() embeds every frame's PNG bytes into the parquet; images/ is only the writer's staging copy.
        shutil.rmtree(dataset.root / "images")
    logging.info("Wrote %d episodes to %s", num_episodes, args.repo_id)


if __name__ == "__main__":
    main(tyro.cli(Args))
