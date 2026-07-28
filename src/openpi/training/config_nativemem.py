"""Training configs for NativeMEM."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.nativemem_config as nativemem_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

# Maps repo_id prefix to MemoryTokenizer max_len. The first matching prefix wins.
REPO_ID_MEMORY_MAX_LEN: dict[str, int] = {
    "click_button": 256,
    "put_back_block": 256,
    "observe_and_pickup": 256,
    "swap_blocks": 256,
    "rearrange_blocks": 256,
    "cover_blocks": 512,
    "arx/click_button": 256,
    "arx/put_cube": 256,
    "arx/scan_code": 512,
}


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = True
    action_sequence_keys: Sequence[str] = ("action",)
    prompt_from_task: bool = False
    mem_intervals: tuple[int, int] = (8, 16)


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a transform group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    default_prompt: str | None = None
    memory_tokenizer_max_len: int | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not isinstance(model_config, nativemem_config.NativeMEMConfig):
            raise ValueError(f"Unsupported NativeMEM model config type: {type(model_config)}")

        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
                _transforms.TokenizeMemory(
                    _tokenizer.MemoryTokenizer(
                        max_len=self.memory_tokenizer_max_len or model_config.memory_tokenizer_max_len
                    )
                ),
            ],
        )


@dataclasses.dataclass(frozen=True)
class AlohaMemInputs(_transforms.DataTransformFn):
    adapt_to_pi: bool = True
    gripper_type: aloha_policy.GripperType = "trossen"

    def __call__(self, data: dict) -> dict:
        memory = data.get("memory")
        canonical = aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi, gripper_type=self.gripper_type)(
            {key: data[key] for key in ("images", "state", "actions", "prompt") if key in data}
        )
        if memory is not None:
            canonical["memory"] = memory
        return canonical


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    mem_intervals: tuple[int, int] = (8, 16)

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        raise NotImplementedError

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del model_config
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            mem_intervals=self.mem_intervals,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info("Loaded norm stats from %s", data_assets_dir)
            return norm_stats
        except FileNotFoundError:
            logging.info("Norm stats not found in %s/%s, skipping.", assets_dir, asset_id)
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs, model_config
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaMemDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    gripper_type: aloha_policy.GripperType = "trossen"
    memory_tokenizer_max_len: int | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "prompt": "prompt",
                        "actions": "action",
                        "memory": "memory",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[AlohaMemInputs(adapt_to_pi=self.adapt_to_pi, gripper_type=self.gripper_type)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi, gripper_type=self.gripper_type)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        resolved_max_len = self.memory_tokenizer_max_len
        if resolved_max_len is None:
            for prefix, max_len in REPO_ID_MEMORY_MAX_LEN.items():
                if (self.repo_id or "").startswith(prefix):
                    resolved_max_len = max_len
                    logging.info("Setting memory tokenizer max_len to %d for repo_id %s.", max_len, self.repo_id)
                    break
        if resolved_max_len is None:
            resolved_max_len = model_config.memory_tokenizer_max_len
            logging.warning(
                "repo_id %s does not match any prefix in REPO_ID_MEMORY_MAX_LEN; using the model default of %d.",
                self.repo_id,
                resolved_max_len,
            )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt,
            memory_tokenizer_max_len=resolved_max_len,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=True,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi_nativemem"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=nativemem_config.NativeMEMConfig)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 8
    num_train_steps: int = 50_000
    log_interval: int = 100
    save_interval: int = 10_000
    keep_period: int | None = 20_000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_BASE_MODEL = nativemem_config.NativeMEMConfig(pi05=True, memory_insert_type="self", siglip_variant="So400m/14")

_CONFIGS = [
    TrainConfig(
        name="nativemem_pi05",
        model=_BASE_MODEL,
        data=LeRobotAlohaMemDataConfig(
            assets=AssetsConfig(
                assets_dir="./assets",
                asset_id="sim",
            ),
            gripper_type="sim",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        # weight_loader=weight_loaders.NativeMEMWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        weight_loader=weight_loaders.NativeMEMWeightLoader("checkpoints/mem_tokenizer_pretrain/pretrain/49999/params"),
        num_train_steps = 20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-5,
            decay_steps=20_000,
            decay_lr=1e-6,
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
