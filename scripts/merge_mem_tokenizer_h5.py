"""Merge memory-tokenizer HDF5 shards into one training file."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import re

import h5py
import numpy as np
from tqdm import tqdm

DATASET_TYPES = ("trossen", "sim", "arx")
EPISODE_PATTERN = re.compile(r"^episode_(\d+)$")


def iter_episode_names(h5_file: h5py.File) -> list[str]:
    episode_names = [key for key in h5_file if EPISODE_PATTERN.match(key)]
    return sorted(episode_names, key=lambda name: int(EPISODE_PATTERN.match(name).group(1)))


def get_episode_length(episode_group: h5py.Group) -> int:
    required_keys = ("cam_head", "cam_left", "cam_right", "state", "action")
    missing = [key for key in required_keys if key not in episode_group]
    if missing:
        raise KeyError(f"Missing datasets {missing} in {episode_group.name}")

    lengths = {key: len(episode_group[key]) for key in required_keys}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Inconsistent episode lengths in {episode_group.name}: {lengths}")
    return unique_lengths.pop()


def normalize_dataset_type(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    if value not in DATASET_TYPES:
        raise ValueError(f"dataset_type must be one of {DATASET_TYPES}, got {value!r}")
    return value


def infer_dataset_type(path: Path) -> str | None:
    prefix = path.stem.split("_", 1)[0]
    if prefix in DATASET_TYPES:
        return prefix
    return None


def resolve_dataset_types(input_paths: Sequence[Path], dataset_types: Sequence[str] | None) -> list[str | None]:
    if dataset_types is None:
        return [infer_dataset_type(path) for path in input_paths]
    if len(dataset_types) != len(input_paths):
        raise ValueError("When provided, --dataset_type must be passed once for each --input_path.")
    return [normalize_dataset_type(dataset_type) for dataset_type in dataset_types]


def merge_datasets(input_paths: Sequence[Path], output_path: Path, dataset_types: Sequence[str | None]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_files = [path.expanduser().resolve() for path in input_paths]
    if output_path.expanduser().resolve() in source_files:
        raise ValueError("Output path must be different from all source dataset paths.")

    episode_ends = []
    episode_idx = []
    next_episode_idx = 0
    total_frames = 0

    with h5py.File(output_path, "w") as dst_h5:
        for source_path, source_dataset_type in zip(source_files, dataset_types, strict=True):
            if not source_path.exists():
                raise FileNotFoundError(f"Source dataset not found: {source_path}")

            with h5py.File(source_path, "r") as src_h5:
                episode_names = iter_episode_names(src_h5)
                for episode_name in tqdm(episode_names, desc=f"Copying {source_path}"):
                    src_episode = src_h5[episode_name]
                    episode_length = get_episode_length(src_episode)
                    dst_name = f"episode_{next_episode_idx:0>8d}"
                    dst_h5.copy(src_episode, dst_h5, name=dst_name)
                    dst_episode = dst_h5[dst_name]

                    if source_dataset_type is None:
                        if "dataset_type" not in dst_episode.attrs:
                            raise ValueError(
                                f"{source_path}:{episode_name} has no dataset_type attr; "
                                "pass --dataset_type for this input."
                            )
                        dataset_type = normalize_dataset_type(dst_episode.attrs["dataset_type"])
                    else:
                        dataset_type = source_dataset_type
                    dst_episode.attrs["dataset_type"] = dataset_type

                    total_frames += episode_length
                    episode_ends.append(total_frames)
                    episode_idx.extend((next_episode_idx, frame_idx) for frame_idx in range(episode_length))
                    next_episode_idx += 1

        dst_h5.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
        dst_h5.create_dataset("episode_idx", data=np.asarray(episode_idx, dtype=np.int64))

        print(f"Merged {len(episode_ends)} episodes with {total_frames} frames into {output_path}")
        print("Keys in merged file:")
        episode_keys = []
        for key in dst_h5:
            if key.startswith("episode_"):
                episode_keys.append(key)
            else:
                print(f"\t{key}: {dst_h5[key].shape} {dst_h5[key].dtype}")
        preview_keys = episode_keys if len(episode_keys) <= 5 else [*episode_keys[:3], "...", *episode_keys[-2:]]
        for key in preview_keys:
            print(f"\t{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, action="append", required=True, help="Input HDF5 shard.")
    parser.add_argument(
        "--dataset_type",
        choices=DATASET_TYPES,
        action="append",
        help="Dataset type for the corresponding --input_path. Otherwise inferred or read from episode attrs.",
    )
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the combined HDF5 file.")
    args = parser.parse_args()

    dataset_types = resolve_dataset_types(args.input_path, args.dataset_type)
    merge_datasets(args.input_path, args.output_path, dataset_types)


if __name__ == "__main__":
    main()
