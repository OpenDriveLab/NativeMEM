"""Configuration for mem_tokenizer action-prediction pretraining."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.mem_tokenizer import MemTokenizerModel


@dataclasses.dataclass(frozen=True)
class MemTokenizerConfig(_model.BaseModelConfig):
    """Config for mem_tokenizer action prediction with a frozen pi0.5 VLM."""

    dtype: str = "float32"

    siglip_variant: str = "So400m/14"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    memory_insert_type: str = "self"

    history_seq_len: int = 8

    # Bernoulli drop probability for current-frame patch tokens during training.
    # Drop = setting input_mask=False so the patch is excluded from attention,
    # forcing the LLM to lean on memory tokens for that view's information.
    current_patch_drop_prob: float = 0.0
    # Cosine warmup for current_patch_drop_prob: 0 for the first
    # drop_warmup_steps, then cosine-ramp to current_patch_drop_prob over
    # drop_ramp_steps, then constant. Avoids early-training loss collapse.
    drop_warmup_steps: int = 5000
    drop_ramp_steps: int = 10000

    max_token_len: int = 200

    action_dim: int = 32
    action_horizon: int = 50

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "MemTokenizerModel":
        from openpi.models.mem_tokenizer import MemTokenizerModel

        return MemTokenizerModel(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        t = self.history_seq_len + 1
        image_seq_spec = jax.ShapeDtypeStruct([batch_size, t, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_seq_mask_spec = jax.ShapeDtypeStruct([batch_size, t], jnp.bool_)
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                image_seq={
                    "base_0_rgb": image_seq_spec,
                    "left_wrist_0_rgb": image_seq_spec,
                    "right_wrist_0_rgb": image_seq_spec,
                },
                image_seq_masks={
                    "base_0_rgb": image_seq_mask_spec,
                    "left_wrist_0_rgb": image_seq_mask_spec,
                    "right_wrist_0_rgb": image_seq_mask_spec,
                },
                history_positions=jax.ShapeDtypeStruct([batch_size, t], jnp.int32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        return nnx.Not(nnx.Any(
            nnx_utils.PathRegex(r"encoder/.*"),
            nnx_utils.PathRegex(r"memory_proj_in/.*"),
            nnx_utils.PathRegex(r".*mem_bos.*"),
            # nnx_utils.PathRegex(r".*llm.*_1.*")
        ))
