"""Compute normalization statistics directly from a merged H5 dataset.

Reads an H5 file produced by the merge script (top-level `episode_########`
groups, each containing `state` and `action` datasets) and writes a
`norm_stats.json` to the chosen output directory.

Applies the same per-sample transforms used at training time in
`data_loader_mem_tokenizer.py` before accumulating stats:
    AlohaInputs(adapt_to_pi=True) -> DeltaActions(make_bool_mask(6, -1, 6, -1))

Example:
    python scripts/compute_norm_stats_h5.py \
        --h5-path data/sim_all.h5 \
        --output-dir assets/sim \
        --gripper-type sim
"""

import pathlib

import h5py
import numpy as np
import tqdm
import tyro

import openpi.policies.aloha_policy as aloha_policy
import openpi.shared.normalize as normalize
import openpi.transforms as _transforms

DATASET_TYPES = ("trossen", "sim", "arx")


def iter_episode_keys(f: h5py.File) -> list[str]:
    keys = [k for k in f.keys() if k.startswith("episode_0")]
    keys.sort(key=lambda x: int(x.split("_")[-1]))
    return keys


def main(
    h5_path: str,
    output_dir: str,
    gripper_type: aloha_policy.GripperType | None = None,
    action_horizon: int = 50,
) -> None:
    h5_path = pathlib.Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    delta_actions = _transforms.DeltaActions(_transforms.make_bool_mask(6, -1, 6, -1))

    # AlohaInputs requires an "images" dict containing at least cam_high; supply
    # a 1x1 placeholder since images are not used for norm-stat computation.
    dummy_image = np.zeros((3, 1, 1), dtype=np.uint8)

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    with h5py.File(h5_path, "r") as f:
        episode_keys = iter_episode_keys(f)
        aloha_inputs = aloha_policy.AlohaInputs(adapt_to_pi=True, gripper_type=gripper_type)
        for ep_key in tqdm.tqdm(episode_keys, desc="Computing stats"):
            group = f[ep_key]
            state_arr = np.asarray(group["state"][()], dtype=np.float32)
            action_arr = np.asarray(group["action"][()], dtype=np.float32)
            if state_arr.ndim == 1:
                state_arr = state_arr[:, None]
            if action_arr.ndim == 1:
                action_arr = action_arr[:, None]

            episode_len = state_arr.shape[0]
            for t in range(episode_len):
                end = min(t + action_horizon, episode_len)
                window = action_arr[t:end]
                if window.shape[0] < action_horizon:
                    pad = np.repeat(window[-1:], action_horizon - window.shape[0], axis=0)
                    window = np.concatenate([window, pad], axis=0)

                sample = {
                    "state": state_arr[t].copy(),
                    "actions": window.copy(),
                    "images": {"cam_high": dummy_image},
                }
                sample = aloha_inputs(sample)
                sample = delta_actions(sample)

                stats["state"].update(sample["state"][None, :])
                stats["actions"].update(sample["actions"])

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = pathlib.Path(output_dir)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
