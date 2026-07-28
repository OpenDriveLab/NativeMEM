"""Convert RoboTwin/RMBench raw HDF5 episodes into memory-tokenizer HDF5 format."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import re

import cv2
import h5py
import numpy as np
from tqdm import tqdm

CAMERA_DATASETS = {
    "cam_head": "/observation/head_camera/rgb",
    "cam_right": "/observation/right_camera/rgb",
    "cam_left": "/observation/left_camera/rgb",
}
ACTION_DATASET = "/joint_action/vector"
DATASET_TYPES = ("trossen", "sim", "arx")
DEFAULT_INSTRUCTIONS = {
    "click_button": "press the buttons in the order of blue, pink and yellow",
    "click_button_rand": "press each of the three buttons once",
}
EPISODE_ID_PATTERN = re.compile(r"episode[_-]?(\d+)")


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


def encode_images(images: Sequence[np.ndarray], quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    encoded_images = np.empty(len(images), dtype=object)
    for index, image in enumerate(images):
        ok, encoded = cv2.imencode(".jpg", image, encode_param)
        if not ok:
            raise ValueError(f"JPEG encoding failed at frame index {index}")
        encoded_images[index] = encoded.astype(np.uint8)
    return encoded_images


def decode_resize_encode(camera_frames, height: int, width: int, quality: int) -> np.ndarray:
    images = []
    for frame in camera_frames:
        image = decode_image(frame)
        images.append(resize_with_pad_cv2(image, height, width))
    return encode_images(images, quality=quality)


def parse_episode_id(path: Path) -> int | None:
    match = EPISODE_ID_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def read_manifest(path: Path) -> list[Path]:
    dataset_dirs = []
    with path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                dataset_dirs.append(Path(stripped))
    return dataset_dirs


def load_instruction_map(path: Path | None, pairs: Sequence[str]) -> dict[str, str]:
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
    episode_path: Path,
    dataset_dir: Path,
    *,
    instruction_key: str,
    instruction_map: dict[str, str],
    fallback_instruction: str | None,
) -> list[str]:
    task_name = dataset_dir.parent.parent.name
    episode_id = parse_episode_id(episode_path)
    if episode_id is not None:
        instruction_path = dataset_dir.parent.parent / "instructions" / f"episode{episode_id}.json"
        if instruction_path.exists():
            with instruction_path.open("r", encoding="utf-8") as instruction_file:
                instruction_dict = json.load(instruction_file)
            instructions = instruction_dict.get(instruction_key)
            if isinstance(instructions, str):
                return [instructions]
            if instructions:
                return [str(instruction) for instruction in instructions]

    instruction = instruction_map.get(task_name) or fallback_instruction or task_name.replace("_", " ")
    return [instruction]


class TrajectoryDataset:
    def __init__(
        self,
        dataset_dir: Path,
        *,
        height: int,
        width: int,
        quality: int,
        instruction_key: str,
        instruction_map: dict[str, str],
        fallback_instruction: str | None,
    ):
        self.dataset_dir = dataset_dir
        self.data_files = sorted({*dataset_dir.glob("*.hdf5"), *dataset_dir.glob("*.h5")})
        self.height = height
        self.width = width
        self.quality = quality
        self.instruction_key = instruction_key
        self.instruction_map = instruction_map
        self.fallback_instruction = fallback_instruction

    def __len__(self) -> int:
        return len(self.data_files)

    def __iter__(self):
        for index in range(len(self)):
            yield self.extract_hdf5_data(index)

    def extract_hdf5_data(self, index: int):
        episode_path = self.data_files[index]
        with h5py.File(episode_path, "r") as h5_file:
            missing = [path for path in [*CAMERA_DATASETS.values(), ACTION_DATASET] if path not in h5_file]
            if missing:
                raise KeyError(f"Missing datasets {missing} in {episode_path}")

            cam_dict = {
                camera_name: decode_resize_encode(h5_file[source_path][:-1], self.height, self.width, self.quality)
                for camera_name, source_path in CAMERA_DATASETS.items()
            }
            vector = np.asarray(h5_file[ACTION_DATASET][:], dtype=np.float32)
            if len(vector) < 2:
                raise ValueError(f"Need at least two action frames in {episode_path}, got {len(vector)}")

        state = vector[:-1]
        action = vector[1:]
        episode_len = len(state)
        camera_lengths = {camera_name: len(camera_frames) for camera_name, camera_frames in cam_dict.items()}
        if set(camera_lengths.values()) != {episode_len}:
            raise ValueError(f"Camera/action length mismatch in {episode_path}: {camera_lengths}, action={episode_len}")

        instructions = load_episode_instructions(
            episode_path,
            self.dataset_dir,
            instruction_key=self.instruction_key,
            instruction_map=self.instruction_map,
            fallback_instruction=self.fallback_instruction,
        )
        return cam_dict, state, action, instructions


def collect_dataset_dirs(dataset_dirs: Sequence[Path], manifest: Path | None) -> list[Path]:
    collected = list(dataset_dirs)
    if manifest is not None:
        collected.extend(read_manifest(manifest))
    if not collected:
        raise ValueError("Provide at least one --dataset_dir or --dataset_manifest.")

    resolved = []
    for dataset_dir in collected:
        expanded_dir = dataset_dir.expanduser()
        if not expanded_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {expanded_dir}")
        if not expanded_dir.is_dir():
            raise NotADirectoryError(f"Expected dataset directory, got: {expanded_dir}")
        resolved.append(expanded_dir)
    return resolved


def extract_datasets(
    dataset_dirs: Sequence[Path],
    output_path: Path,
    *,
    dataset_type: str,
    height: int,
    width: int,
    quality: int,
    instruction_key: str,
    instruction_map: dict[str, str],
    fallback_instruction: str | None,
) -> None:
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_comp_kwargs = {"compression": "gzip", "compression_opts": 4}
    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))
    str_dtype = h5py.string_dtype(encoding="utf-8")
    episode_ends = []
    episode_idx = []
    global_episode_idx = 0
    total_frames = 0

    with h5py.File(output_path, "w") as h5_file:
        for dataset_dir in dataset_dirs:
            dataset = TrajectoryDataset(
                dataset_dir,
                height=height,
                width=width,
                quality=quality,
                instruction_key=instruction_key,
                instruction_map=instruction_map,
                fallback_instruction=fallback_instruction,
            )
            for cam_dict, state, action, instructions in tqdm(dataset, desc=f"Extracting {dataset_dir}"):
                episode_len = len(state)
                total_frames += episode_len
                episode_ends.append(total_frames)
                episode_idx.extend((global_episode_idx, frame_idx) for frame_idx in range(episode_len))

                group = h5_file.create_group(f"episode_{global_episode_idx:0>8d}")
                group.attrs["dataset_type"] = dataset_type
                group.create_dataset("cam_head", data=cam_dict["cam_head"], dtype=vlen_u8)
                group.create_dataset("cam_left", data=cam_dict["cam_left"], dtype=vlen_u8)
                group.create_dataset("cam_right", data=cam_dict["cam_right"], dtype=vlen_u8)
                group.create_dataset("state", data=state, dtype=np.float32)
                group.create_dataset("action", data=action, dtype=np.float32)
                group.create_dataset("instructions", data=instructions, dtype=str_dtype)
                global_episode_idx += 1

        for key in ("episode_ends", "episode_idx"):
            if key in h5_file:
                del h5_file[key]
        h5_file.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64), **meta_comp_kwargs)
        h5_file.create_dataset("episode_idx", data=np.asarray(episode_idx, dtype=np.int64), **meta_comp_kwargs)

    print(f"Wrote {global_episode_idx} episodes and {total_frames} frames to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        nargs="+",
        action="extend",
        default=[],
        help="Raw RoboTwin/RMBench data dir. Accepts one or more dirs and can be repeated.",
    )
    parser.add_argument("--dataset_manifest", type=Path, help="Text file with one raw data dir per line.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to write the memory-tokenizer HDF5 file.")
    parser.add_argument("--dataset_type", choices=DATASET_TYPES, default="sim", help="Episode dataset_type attribute.")
    parser.add_argument("--height", type=int, default=224, help="Output image height.")
    parser.add_argument("--width", type=int, default=224, help="Output image width.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality.")
    parser.add_argument("--instruction_key", default="seen", help="Instruction JSON key to read when present.")
    parser.add_argument("--instruction_map", type=Path, help="JSON object mapping task names to fallback prompts.")
    parser.add_argument(
        "--default_instruction",
        action="append",
        default=[],
        metavar="TASK=TEXT",
        help="Fallback prompt for one task. Can be passed multiple times.",
    )
    parser.add_argument("--fallback_instruction", help="Fallback prompt used when task-specific text is unavailable.")
    args = parser.parse_args()

    dataset_dirs = collect_dataset_dirs(args.dataset_dir, args.dataset_manifest)
    instruction_map = load_instruction_map(args.instruction_map, args.default_instruction)
    extract_datasets(
        dataset_dirs,
        args.output_path,
        dataset_type=args.dataset_type,
        height=args.height,
        width=args.width,
        quality=args.quality,
        instruction_key=args.instruction_key,
        instruction_map=instruction_map,
        fallback_instruction=args.fallback_instruction,
    )


if __name__ == "__main__":
    main()
