"""One-shot Stage 1 data prep: raw RoboTwin/RMBench episodes -> memory-tokenizer HDF5 + its norm stats.

`sources.json` lists raw `.../data` dirs (glob patterns allowed) under a single dataset_type:

    {"sim": ["data/RMBench/*/demo_clean/data"]}

Each output file is one training dataset with its own norm stats at <assets_dir>/<output stem>/norm_stats.json,
e.g. sim.h5 -> assets/sim/norm_stats.json; build one file per dataset_type and train on the list.

Example:
    python scripts/prepare_mem_tokenizer_h5.py --sources data/sources.json \
        --output_path data/sim.h5 --assets_dir assets
"""

import dataclasses
import glob
import json
import pathlib
import re

import cv2
import h5py
import numpy as np
import tqdm
import tyro

import openpi.policies.aloha_policy as aloha_policy
import openpi.shared.normalize as normalize
from openpi.training.data_loader_mem_tokenizer import DATASET_TYPES
import openpi.transforms as _transforms

# Output camera dataset name -> raw camera dataset.
CAMERA_DATASETS = {
    "cam_head": "/observation/head_camera/rgb",
    "cam_right": "/observation/right_camera/rgb",
    "cam_left": "/observation/left_camera/rgb",
}
ACTION_DATASET = "/joint_action/vector"
# Prompts for tasks whose raw data ships without instructions/episode<N>.json.
DEFAULT_INSTRUCTIONS = {
    "click_button": "press the buttons in the order of blue, pink and yellow",
    "click_button_rand": "press each of the three buttons once",
}
EPISODE_ID_PATTERN = re.compile(r"episode[_-]?(\d+)")


@dataclasses.dataclass
class Args:
    sources: pathlib.Path
    output_path: pathlib.Path
    overwrite: bool = False
    # Norm stats are written to <assets_dir>/<output_path stem>/norm_stats.json.
    assets_dir: pathlib.Path = pathlib.Path("assets")
    action_horizon: int = 50
    height: int = 224
    width: int = 224
    quality: int = 95
    instruction_key: str = "seen"
    instruction_map: pathlib.Path | None = None
    default_instruction: list[str] = dataclasses.field(default_factory=list)
    fallback_instruction: str | None = None


def resize_with_pad_cv2(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize one image with aspect ratio preserved and zero padding."""
    cur_h, cur_w = image.shape[:2]
    if (cur_h, cur_w) == (height, width):
        return image

    scale = min(width / cur_w, height / cur_h)
    resized_w = max(1, round(cur_w * scale))
    resized_h = max(1, round(cur_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    pad_h = (height - resized_h) // 2
    pad_w = (width - resized_w) // 2
    out[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
    return out


def decode_image(encoded_image) -> np.ndarray:
    image_array = np.asarray(encoded_image)
    if image_array.ndim == 3:
        return image_array

    if isinstance(encoded_image, bytes):
        encoded = np.frombuffer(encoded_image, np.uint8)
    else:
        encoded = np.asarray(encoded_image, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise ValueError("Failed to decode image from HDF5.")
    return image


def decode_resize_encode(camera_frames, height: int, width: int, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    encoded_images = np.empty(len(camera_frames), dtype=object)
    for index, frame in enumerate(camera_frames):
        ok, encoded = cv2.imencode(".jpg", resize_with_pad_cv2(decode_image(frame), height, width), encode_param)
        if not ok:
            raise ValueError(f"JPEG encoding failed at frame index {index}")
        encoded_images[index] = encoded.astype(np.uint8)
    return encoded_images


def read_raw_episode(
    episode_path: pathlib.Path, height: int, width: int, quality: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Read one raw episode as resized JPEG frames plus state[t] = vec[t], action[t] = vec[t + 1]."""
    with h5py.File(episode_path, "r") as h5_file:
        missing = [path for path in [*CAMERA_DATASETS.values(), ACTION_DATASET] if path not in h5_file]
        if missing:
            raise KeyError(f"Missing datasets {missing} in {episode_path}")

        # The last camera frame has no next action, so it is dropped.
        frames = {
            camera_name: decode_resize_encode(h5_file[source_path][:-1], height, width, quality)
            for camera_name, source_path in CAMERA_DATASETS.items()
        }
        vector = np.asarray(h5_file[ACTION_DATASET][:], dtype=np.float32)
        if len(vector) < 2:
            raise ValueError(f"Need at least two action frames in {episode_path}, got {len(vector)}")

    state = vector[:-1]
    action = vector[1:]
    camera_lengths = {camera_name: len(images) for camera_name, images in frames.items()}
    if set(camera_lengths.values()) != {len(state)}:
        raise ValueError(f"Camera/action length mismatch in {episode_path}: {camera_lengths}, action={len(state)}")
    return frames, state, action


def load_instruction_map(path: pathlib.Path | None, pairs: list[str]) -> dict[str, str]:
    instruction_map = dict(DEFAULT_INSTRUCTIONS)
    if path is not None:
        with path.open("r", encoding="utf-8") as instruction_file:
            loaded = json.load(instruction_file)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected instruction map JSON object, got {type(loaded)}")
        instruction_map.update({str(key): str(value) for key, value in loaded.items()})

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected --default_instruction TASK=TEXT, got {pair!r}")
        task_name, instruction = pair.split("=", 1)
        instruction_map[task_name] = instruction
    return instruction_map


def load_episode_instructions(
    episode_path: pathlib.Path,
    dataset_dir: pathlib.Path,
    *,
    instruction_key: str,
    instruction_map: dict[str, str],
    fallback_instruction: str | None,
) -> list[str]:
    """Instructions from <task>/<demo>/instructions/episode<N>.json, next to the <task>/<demo>/data dir."""
    match = EPISODE_ID_PATTERN.search(episode_path.stem)
    if match is not None:
        instruction_path = dataset_dir.parent / "instructions" / f"episode{int(match.group(1))}.json"
        if instruction_path.exists():
            with instruction_path.open("r", encoding="utf-8") as instruction_file:
                instructions = json.load(instruction_file).get(instruction_key)
            if isinstance(instructions, str):
                return [instructions]
            if instructions:
                return [str(instruction) for instruction in instructions]
        else:
            task_name = dataset_dir.parent.parent.name
            if task_name in instruction_map or fallback_instruction is not None:
                print(f"Warning: no instructions file {instruction_path} for {episode_path}; using default instruction: {instruction_map.get(task_name) or fallback_instruction}")
                return [instruction_map.get(task_name) or fallback_instruction]
            else:
                print(f"Warning: no instructions file {instruction_path} for {episode_path}; using task name as instruction")
                return [task_name]


def accumulate_episode_stats(
    stats: dict[str, normalize.RunningStats],
    state: np.ndarray,
    action: np.ndarray,
    *,
    gripper_type: aloha_policy.GripperType,
    action_horizon: int,
) -> None:
    """Update `stats` with one episode, using the training-time AlohaInputs -> DeltaActions transforms."""
    aloha_inputs = aloha_policy.AlohaInputs(adapt_to_pi=True, gripper_type=gripper_type)
    delta_actions = _transforms.DeltaActions(_transforms.make_bool_mask(6, -1, 6, -1))
    # AlohaInputs requires an "images" dict containing at least cam_high; images do not affect the stats.
    dummy_image = np.zeros((3, 1, 1), dtype=np.uint8)

    episode_len = len(state)
    for t in range(episode_len):
        # Same action window as the Stage 1 loader: the next action_horizon actions, padded with the last one.
        window = action[t : min(t + action_horizon, episode_len)]
        if window.shape[0] < action_horizon:
            pad = np.repeat(window[-1:], action_horizon - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)

        sample = {"state": state[t].copy(), "actions": window.copy(), "images": {"cam_high": dummy_image}}
        sample = delta_actions(aloha_inputs(sample))
        stats["state"].update(sample["state"][None, :])
        stats["actions"].update(sample["actions"])


def load_sources(path: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    with path.open("r", encoding="utf-8") as sources_file:
        sources = json.load(sources_file)
    unknown = sorted(set(sources) - set(DATASET_TYPES))
    if unknown:
        raise ValueError(f"Unknown dataset_type keys {unknown} in {path}; expected a subset of {DATASET_TYPES}")

    out = {}
    for dataset_type, patterns in sources.items():
        dirs = []
        for pattern in patterns:
            matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
            if not matches:
                raise FileNotFoundError(f"Pattern {pattern!r} for {dataset_type!r} matched no directories")
            dirs.extend(pathlib.Path(match).expanduser() for match in matches)
        for dataset_dir in dirs:
            if not dataset_dir.is_dir():
                raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")
        out[dataset_type] = dirs
    return out


def main(args: Args) -> None:
    sources = load_sources(args.sources)
    # One norm-stats file per output, so all its episodes must share one gripper convention.
    if len(sources) != 1:
        raise ValueError(f"{args.sources} must list exactly one dataset_type, got {sorted(sources)}")
    ((dataset_type, dataset_dirs),) = sources.items()
    instruction_map = load_instruction_map(args.instruction_map, args.default_instruction)
    stats_dir = args.assets_dir / args.output_path.stem
    print(f"Preparing {dataset_type} HDF5 dataset at {args.output_path} from sources {args.sources}")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_path} exists; pass --overwrite to replace it")
    if (stats_dir / "norm_stats.json").exists() and not args.overwrite:
        raise FileExistsError(f"{stats_dir / 'norm_stats.json'} exists; pass --overwrite to replace it")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))
    str_dtype = h5py.string_dtype(encoding="utf-8")
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    episode_ends = []
    episode_idx = []
    num_episodes = 0
    num_frames = 0
    with h5py.File(args.output_path, "w", driver="core", backing_store=True) as h5_file:
        for dataset_dir in dataset_dirs:
            episode_paths = sorted({*dataset_dir.glob("*.hdf5"), *dataset_dir.glob("*.h5")})
            for episode_path in tqdm.tqdm(episode_paths, desc=f"{dataset_type}: {dataset_dir}"):
                frames, state, action = read_raw_episode(episode_path, args.height, args.width, args.quality)
                instructions = load_episode_instructions(
                    episode_path,
                    dataset_dir,
                    instruction_key=args.instruction_key,
                    instruction_map=instruction_map,
                    fallback_instruction=args.fallback_instruction,
                )

                episode_len = len(state)
                episode_idx.extend((num_episodes, frame_idx) for frame_idx in range(episode_len))
                num_frames += episode_len
                episode_ends.append(num_frames)

                group = h5_file.create_group(f"episode_{num_episodes:0>8d}")
                group.attrs["dataset_type"] = dataset_type
                for camera_name, images in frames.items():
                    group.create_dataset(camera_name, data=images, dtype=vlen_u8)
                group.create_dataset("state", data=state, dtype=np.float32)
                group.create_dataset("action", data=action, dtype=np.float32)
                group.create_dataset("instructions", data=instructions, dtype=str_dtype)
                num_episodes += 1

                accumulate_episode_stats(
                    stats, state, action, gripper_type=dataset_type, action_horizon=args.action_horizon
                )

        if num_episodes == 0:
            raise ValueError(f"No episodes found in {args.sources}")
        meta_comp_kwargs = {"compression": "gzip", "compression_opts": 4}
        h5_file.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64), **meta_comp_kwargs)
        h5_file.create_dataset("episode_idx", data=np.asarray(episode_idx, dtype=np.int64), **meta_comp_kwargs)
        print(f"Writing {num_episodes} episodes and {num_frames} frames to {args.output_path}")

    normalize.save(stats_dir, {key: s.get_statistics() for key, s in stats.items()})
    print(f"{dataset_type}: {num_episodes} episodes, {num_frames} frames")
    print(f"HDF5 dataset: {args.output_path}; norm stats: {stats_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
