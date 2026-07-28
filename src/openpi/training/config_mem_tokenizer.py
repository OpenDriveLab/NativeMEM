"""Training configuration for mem_tokenizer action pretraining."""

import dataclasses
import difflib

import flax.nnx as nnx
import tyro

from openpi.models.mem_tokenizer_config import MemTokenizerConfig
import openpi.training.optimizer as _optimizer
import openpi.training.data_loader_mem_tokenizer as temporal_data
import openpi.training.weight_loaders as weight_loaders

Filter = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi_mem_tokenizer"
    exp_name: str = tyro.MISSING

    model: MemTokenizerConfig = dataclasses.field(default_factory=MemTokenizerConfig)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float = 0.9999

    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    data: temporal_data.DataConfigFactory = dataclasses.field(default_factory=temporal_data.FakeDataConfig)
    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 16
    num_workers: int = 2
    num_train_steps: int = 100_000

    log_interval: int = 100
    save_interval: int = 10_000
    keep_period: int | None = 50_000

    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True

    fsdp_devices: int = 1

    @property
    def assets_dirs(self):
        return (temporal_data.pathlib.Path("./assets") / self.name).resolve()

    @property
    def checkpoint_dir(self):
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (temporal_data.pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def _base_model() -> MemTokenizerConfig:
    return MemTokenizerConfig(
        history_seq_len=8,
        current_patch_drop_prob=0.0,
        action_horizon=50,
    )


_model_default = _base_model()

_CONFIGS = [
    TrainConfig(
        name="mem_tokenizer_pretrain",
        model=MemTokenizerConfig(
            history_seq_len=8,
            action_horizon=50,
        ),
        weight_loader=weight_loaders.MemTokenizerWeightLoader(
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
        ),
        freeze_filter=_model_default.get_freeze_filter(),
        data=temporal_data.H5MemTokenizerDataConfig(
            load_in_memory=False,
            history_seq_len=_model_default.history_seq_len,
            stride_range=(8, 16),
        ),
        batch_size=16,
        num_workers=4,
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
