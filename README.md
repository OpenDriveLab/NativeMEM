# NativeMEM

<div id="top" align="left">
  <a href="https://opendrivelab.com/NativeMEM"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2607.06678"><img src="https://img.shields.io/badge/arXiv-2607.06678-b31b1b" alt="arXiv"></a>
</div>

> [!WARNING]
> This repository is a work-in-progress public code release. The code is being cleaned and validated, so APIs, configuration names, preprocessing scripts, and training commands may still change.

NativeMEM equips a pretrained vision-language-action (VLA) policy with long-term, real-time-updated visual memory. Its Native Memory Compression scheme repurposes the VLA's own vision encoder to compress each historical frame-view observation into a single native memory token, then appends those tokens to the VLA's original input sequence.

This repository retains the upstream `openpi` Python package namespace for compatibility with the underlying model, training, and policy infrastructure.

## Method overview

NativeMEM training has three main steps:

1. **Native Memory Compression:** initialize the memory tokenizer from the VLA's native vision encoder, freeze the pretrained VLA, and train only the memory branch with the VLA's original action-prediction objective. Each frame-view pair is summarized into one action-aligned memory token.
2. **Memory caching:** run the learned memory tokenizer over target-task demonstrations and store the per-frame, per-view `mem_token` features in a LeRobot dataset.
3. **Task-specific finetuning:** retrieve the cached memory tokens, append them to the standard observation and prompt tokens, and finetune the VLA backbone, action head, memory projection, and memory beginning-of-sequence token. The memory-tokenizer encoder remains fixed.

The cached LeRobot feature is named `mem_token`. The Stage 2 data pipeline collects selected frame-view tokens into `Observation.memory` and produces the corresponding `Observation.memory_mask`.

## Setup

NativeMEM requires Python 3.11 and a CUDA-capable NVIDIA GPU for practical training. Install the environment with [`uv`](https://docs.astral.sh/uv/):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Activate the environment before running any command in this README.
```bash
source .venv/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

Set `HF_LEROBOT_HOME` if LeRobot datasets should be stored outside the default cache:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot
```

## Prepare the data

### 1. Convert raw episodes to Stage 1 HDF5

Prepare RoboTwin/RMBench-style raw HDF5 episodes for memory-tokenizer training:

```bash
python scripts/prepare_mem_tokenizer_h5.py \
    --dataset_dir /path/to/task_a/demo/data /path/to/task_b/demo/data \
    --output_path /path/to/sim_all.h5 \
    --dataset_type sim
```

Supported dataset types are `sim`, `trossen`, and `arx`. Each output episode contains three camera streams, state, action, instructions, and a `dataset_type` attribute. When instruction JSON files are unavailable, provide fallback text with `--instruction_map`, repeated `--default_instruction TASK=TEXT`, or `--fallback_instruction`.

For a large task collection, `--dataset_manifest` accepts a text file containing one raw data directory per line.

### 2. Merge prepared shards

Merge separately prepared datasets into one Stage 1 training file:

```bash
python scripts/merge_mem_tokenizer_h5.py \
    --input_path /path/to/sim_all.h5 --dataset_type sim \
    --input_path /path/to/arx_all.h5 --dataset_type arx \
    --output_path /path/to/train.h5
```

Pass `--dataset_type` once for each input. It may be omitted when the type can be inferred from a filename such as `sim_all.h5` or from the episode attributes.

### 3. Compute normalization assets

Compute normalization statistics for every dataset type used in Stage 1. Store each result under `assets/<dataset_type>` so the loader can select the correct statistics per episode.

```bash
python scripts/compute_norm_stats_h5.py \
    --h5-path /path/to/sim_all.h5 \
    --output-dir assets/sim \
    --gripper-type sim

python scripts/compute_norm_stats_h5.py \
    --h5-path /path/to/arx_all.h5 \
    --output-dir assets/arx \
    --gripper-type arx
```

## Stage 1: Train the memory tokenizer

```bash
python scripts/train_mem_tokenizer.py mem_tokenizer_pretrain \
    --exp_name=my_mem_tokenizer \
    --data.repo_id=/path/to/train.h5 \
    --overwrite
```

## Stage 1.5: Extract memory tokens

Convert each task HDF5 file to a LeRobot dataset and extract one `mem_token` per frame-view pair with the trained memory tokenizer:

```bash
python examples/robotwin/convert_h5_to_lerobot_mem_tokenizer.py \
    --data_path /path/to/task_name.h5 \
    --repo_id task_name_nativemem \
    --params_path checkpoints/mem_tokenizer_pretrain/my_mem_tokenizer/49999/params \
    --instruction "task description"
```

The converter uses episode-level `instructions` when present; `--instruction` is the fallback. The resulting LeRobot dataset is stored under `$HF_LEROBOT_HOME/<repo_id>`.

## Stage 2: Train NativeMEM

Before training, update the `NativeMEMWeightLoader` path in [`src/openpi/training/config_nativemem.py`](src/openpi/training/config_nativemem.py) to the Stage 1 `params` directory:

```python
weight_loader=weight_loaders.NativeMEMWeightLoader(
    "checkpoints/mem_tokenizer_pretrain/my_mem_tokenizer/49999/params"
)
```

The provided `nativemem_pi05` config targets simulated ALOHA data: it uses `gripper_type="sim"` and loads normalization statistics from `assets/sim`. When training another robot type, update both the gripper type and normalization asset ID in the same config.

The same config maps `repo_id` prefixes to the maximum flattened memory length. Add a task prefix to `REPO_ID_MEMORY_MAX_LEN` when the dataset needs a value other than the model default. The first matching prefix wins.

Then launch Stage 2:

```bash
python scripts/train_nativemem.py nativemem_pi05 \
    --exp_name=my_nativemem \
    --data.repo_id=task_name_nativemem \
    --overwrite
```


## Repository layout

```text
scripts/prepare_mem_tokenizer_h5.py                         # Raw data -> Stage 1 HDF5
scripts/merge_mem_tokenizer_h5.py                           # Merge Stage 1 HDF5 shards
scripts/compute_norm_stats_h5.py                            # Compute HDF5 normalization statistics
scripts/train_mem_tokenizer.py                              # Stage 1 training entry point
examples/robotwin/convert_h5_to_lerobot_mem_tokenizer.py    # Cache mem_token features in LeRobot
scripts/train_nativemem.py                                  # Stage 2 training entry point
src/openpi/models/mem_tokenizer.py                          # Memory-tokenizer model
src/openpi/models/nativemem.py                              # NativeMEM policy model
src/openpi/training/config_mem_tokenizer.py                 # Stage 1 configuration
src/openpi/training/config_nativemem.py                     # Stage 2 configuration
```


## Acknowledgements

NativeMEM builds on the public [OpenPI](https://github.com/Physical-Intelligence/openpi) codebase and pi0.5 infrastructure. We thank the OpenPI authors and the broader open-source robotics community for making this work possible.

## License

See [`LICENSE`](LICENSE) and [`LICENSE_GEMMA.txt`](LICENSE_GEMMA.txt) for license information.
