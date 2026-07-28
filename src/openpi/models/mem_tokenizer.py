"""Temporal action pretraining with a trainable SigLIP encoder and frozen pi0.5 VLM."""

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at
from openpi.models.siglip_encoder import SiglipEncoder
from openpi.models.mem_tokenizer_config import MemTokenizerConfig


def _make_attn_mask(input_mask: jnp.ndarray, mask_ar: jnp.ndarray) -> jnp.ndarray:
    input_mask = jnp.asarray(input_mask, dtype=jnp.bool_)
    mask_ar = jnp.asarray(mask_ar, dtype=jnp.bool_)

    if mask_ar.ndim == 1:
        with jax.ensure_compile_time_eval():
            block_id = jnp.cumsum(mask_ar.astype(jnp.int32), axis=0)
        block_id = jnp.broadcast_to(block_id[None, :], input_mask.shape)
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        block_id = jnp.cumsum(mask_ar.astype(jnp.int32), axis=1)

    attn_mask = block_id[:, :, None] >= block_id[:, None, :]
    valid_mask = input_mask[:, :, None] & input_mask[:, None, :]
    return attn_mask & valid_mask


@at.typecheck
def _posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class MemTokenizerModel(_model.BaseModel):
    """Pretrain SigLIP memory tokens through the downstream action expert path."""

    def __init__(self, config: MemTokenizerConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.history_seq_len = config.history_seq_len
        self.n_views = len(_model.IMAGE_KEYS)
        self.memory_insert_type = config.memory_insert_type
        self.current_patch_drop_prob = config.current_patch_drop_prob
        self.drop_warmup_steps = config.drop_warmup_steps
        self.drop_ramp_steps = config.drop_ramp_steps
        assert self.memory_insert_type in ["self", "causal"], (
            f"Invalid memory_insert_type: {self.memory_insert_type}"
        )

        siglip_params = _siglip.decode_variant(config.siglip_variant)
        paligemma_cfg = _gemma.get_config(config.paligemma_variant)
        action_expert_cfg = _gemma.get_config(config.action_expert_variant)

        self.encoder_width = siglip_params["width"]
        self.paligemma_width = paligemma_cfg.width

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

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_cfg, action_expert_cfg],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_cfg.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(jnp.zeros((1, 224, 224, 3), dtype=jnp.float32), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.memory_proj_in = nnx.Linear(self.encoder_width, self.paligemma_width, rngs=rngs)
        self.mem_bos = nnx.Param(jax.random.normal(rngs.params(), (1, 1, self.paligemma_width)) * 0.02)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_cfg.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_cfg.width, action_expert_cfg.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_cfg.width, action_expert_cfg.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_cfg.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    def _preprocess_image_seq(
        self,
        rng: at.KeyArrayLike | None,
        image_seq: dict[str, jnp.ndarray],
        *,
        train: bool,
    ) -> dict[str, jnp.ndarray]:
        any_view = next(iter(image_seq.values()))
        batch_size, seq_len = any_view.shape[:2]
        flat_images = {name: value.reshape(batch_size * seq_len, *value.shape[2:]) for name, value in image_seq.items()}
        flat_observation = _model.Observation(
            images=flat_images,
            image_masks={name: jnp.ones((batch_size * seq_len,), dtype=jnp.bool_) for name in image_seq},
            state=jnp.zeros((batch_size * seq_len, self.action_dim), dtype=jnp.float32),
        )
        flat_observation = _model.preprocess_observation(rng, flat_observation, train=train)
        return {
            name: flat_observation.images[name].reshape(batch_size, seq_len, *flat_observation.images[name].shape[1:])
            for name in image_seq
        }

    def _encode_hist_cur(self, image_seq: dict[str, jnp.ndarray], *, train: bool) -> jnp.ndarray:
        """Encode history+current frames with the trainable student encoder."""
        any_view = next(iter(image_seq.values()))
        B, T = any_view.shape[:2]
        mem_tokens_per_view = []
        for name in _model.IMAGE_KEYS:
            flat_images = image_seq[name].reshape(B * T, *image_seq[name].shape[2:])
            mem_token, _patches, _ = self.encoder(flat_images, train=train, extract_block=None, ids_keep=None)
            mem_tokens_per_view.append(mem_token.reshape(B, T, 1, self.encoder_width))
        return jnp.concatenate(mem_tokens_per_view, axis=2)

    def _scheduled_drop_prob(self, step: jnp.ndarray | int | None) -> jnp.ndarray | float:
        """Cosine warmup: 0 for warmup_steps, then cos-ramp to v over ramp_steps."""
        v = self.current_patch_drop_prob
        if v <= 0.0 or step is None:
            return v
        step_f = jnp.asarray(step, dtype=jnp.float32)
        ramp = max(self.drop_ramp_steps, 1)
        progress = jnp.clip((step_f - self.drop_warmup_steps) / ramp, 0.0, 1.0)
        factor = 0.5 * (1.0 - jnp.cos(jnp.pi * progress))
        return v * factor

    def _embed_prefix(
        self,
        rng: at.KeyArrayLike | None,
        curr_frame: dict,
        tokenized_prompt: jnp.ndarray,
        tokenized_prompt_mask: jnp.ndarray,
        mem_token_seq: jnp.ndarray,
        *,
        train: bool,
        step: jnp.ndarray | int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        B = mem_token_seq.shape[0]
        tokens = []
        input_mask = []
        ar_mask = []

        drop_prob = self._scheduled_drop_prob(step) if train else 0.0
        # In jax we may have a tracer, so check the static config rather than the value.
        needs_rng = train and self.current_patch_drop_prob > 0.0
        if needs_rng and rng is None:
            raise ValueError("rng is required when current_patch_drop_prob > 0 in train mode.")

        n_img = 0
        for name in _model.IMAGE_KEYS:
            img_tokens, _ = self.PaliGemma.img(curr_frame[name], train=False)
            n_img += img_tokens.shape[1]
            patch_mask = jnp.ones((B, img_tokens.shape[1]), dtype=jnp.bool_)
            if needs_rng:
                rng, sub = jax.random.split(rng)
                keep = jax.random.bernoulli(sub, 1.0 - drop_prob, shape=patch_mask.shape)
                patch_mask = patch_mask & keep
            tokens.append(img_tokens)
            input_mask.append(patch_mask)
            ar_mask += [False] * img_tokens.shape[1]

        if tokenized_prompt is not None:
            prompt_emb = self.PaliGemma.llm(tokenized_prompt, method="embed")
            tokens.append(prompt_emb)
            input_mask.append(tokenized_prompt_mask)
            ar_mask += [False] * prompt_emb.shape[1]

        mem_bos = jnp.broadcast_to(self.mem_bos.value, (B, 1, self.paligemma_width))
        mem_flat = mem_token_seq.reshape(B, mem_token_seq.shape[1] * self.n_views, self.encoder_width)
        mem_tokens = self.memory_proj_in(mem_flat)

        tokens.append(mem_bos)
        input_mask.append(jnp.ones((B, 1), dtype=jnp.bool_))
        ar_mask += [False]

        tokens.append(mem_tokens)
        input_mask.append(jnp.ones((B, mem_tokens.shape[1]), dtype=jnp.bool_))
        if self.memory_insert_type == "causal":
            ar_mask += [True] + [False] * (mem_tokens.shape[1] - 1)
        else:
            ar_mask += [False] * mem_tokens.shape[1]

        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.asarray(ar_mask, dtype=jnp.bool_),
            n_img,
        )

    def embed_suffix(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del observation
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = _posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)

        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.asarray([True] + ([False] * (self.action_horizon - 1)), dtype=jnp.bool_)
        return action_tokens, input_mask, ar_mask, time_emb

    def _build_training_positions(
        self,
        prefix_mask: jnp.ndarray,
        suffix_mask: jnp.ndarray,
        n_img: int,
        history_positions: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if history_positions is None:
            # Fallback: dense consecutive positions for the whole prefix.
            prefix_positions = jnp.cumsum(prefix_mask.astype(jnp.int32), axis=1) - 1
            prefix_len = jnp.sum(prefix_mask.astype(jnp.int32), axis=1, keepdims=True)
            suffix_positions = prefix_len + jnp.cumsum(suffix_mask.astype(jnp.int32), axis=1) - 1
            return jnp.concatenate([prefix_positions, suffix_positions], axis=1)

        B = prefix_mask.shape[0]
        H = self.history_seq_len
        V = self.n_views
        # prefix layout (from _embed_prefix): [img_patches | prompt | BOS | mem_tokens]
        # BOS+mem always valid, count is fixed: 1 + (H+1)*V
        n_mem = 1 + (H + 1) * V

        # Image patches occupy a fixed-size head block of size n_img and always take
        # positions 0..n_img-1 — independent of any per-patch dropout, so BOS / memory
        # / suffix anchor at the same positions seen at inference time.
        img_positions = jnp.broadcast_to(jnp.arange(n_img, dtype=jnp.int32)[None, :], (B, n_img))

        # Prompt tokens (with padding) follow the image block.
        prompt_mask = prefix_mask[:, n_img:-n_mem]                                           # [B, prompt_max]
        prompt_positions = n_img + jnp.cumsum(prompt_mask.astype(jnp.int32), axis=1) - 1     # [B, prompt_max]
        prompt_len = jnp.sum(prompt_mask.astype(jnp.int32), axis=1, keepdims=True)           # [B, 1]
        img_prompt_len = n_img + prompt_len                                                  # [B, 1]

        # BOS at position img_prompt_len.
        bos_pos = img_prompt_len                                                             # [B, 1]

        # Frame t, view v → position img_prompt_len + 1 + hp[t]*V + v
        # history_positions: [B, H+1]; ep_start is always hp[0]=0 (attention sink)
        v_offsets = jnp.arange(V, dtype=jnp.int32)[None, None, :]                          # [1, 1, V]
        mem_positions = (
            img_prompt_len[:, :, None]              # [B, 1, 1]
            + 1
            + history_positions[:, :, None] * V    # [B, H+1, 1]
            + v_offsets                             # [1, 1, V]
        )                                           # [B, H+1, V]
        mem_positions = mem_positions.reshape(B, (H + 1) * V)                                # [B, (H+1)*V]

        prefix_positions = jnp.concatenate(
            [img_positions, prompt_positions, bos_pos, mem_positions], axis=1
        )                                                                                    # [B, prefix_len]

        # Suffix tokens continue stride-1 from the last (largest) memory position.
        last_mem_pos = jnp.max(mem_positions, axis=1, keepdims=True)                        # [B, 1]
        suffix_positions = last_mem_pos + jnp.cumsum(suffix_mask.astype(jnp.int32), axis=1)  # [B, suffix_len]

        return jnp.concatenate([prefix_positions, suffix_positions], axis=1)

    def _mem_token_similarity(self, mem_token_seq: jnp.ndarray) -> jnp.ndarray:
        batch_size = mem_token_seq.shape[0]
        if batch_size <= 1:
            return jnp.array(0.0, dtype=mem_token_seq.dtype)

        mem_token_flat = mem_token_seq.reshape(batch_size, -1)
        normalized_mem_token = mem_token_flat / (jnp.linalg.norm(mem_token_flat, axis=-1, keepdims=True) + 1e-8)
        sim_matrix = normalized_mem_token @ normalized_mem_token.T
        triu_idx = np.triu_indices(batch_size, k=1)
        return jnp.mean(sim_matrix[triu_idx])

    def _compute_losses(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        step: jnp.ndarray | int | None = None,
    ):
        preprocess_rng, seq_rng, noise_rng, time_rng, drop_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        if observation.image_seq is None:
            raise ValueError("MemTokenizerModel requires observation.image_seq for training.")

        hist_cur_frames = {
            name: observation.image_seq[name][:, : self.history_seq_len + 1] for name in _model.IMAGE_KEYS
        }
        hist_cur_frames = self._preprocess_image_seq(seq_rng, hist_cur_frames, train=train)
        mem_token_seq = self._encode_hist_cur(hist_cur_frames, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        curr_frame = {name: observation.images[name] for name in _model.IMAGE_KEYS}
        prefix_tokens, prefix_mask, prefix_ar_mask, n_img = self._embed_prefix(
            drop_rng,
            curr_frame,
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            mem_token_seq,
            train=train,
            step=step,
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _make_attn_mask(input_mask, ar_mask)
        positions = self._build_training_positions(prefix_mask, suffix_mask, n_img, observation.history_positions)

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=[1, 2])
        mem_token_sim = self._mem_token_similarity(mem_token_seq)

        metrics = {
            "action_loss": jnp.mean(action_loss),
            "mem_token_sim": mem_token_sim,
            "current_patch_drop_prob": jnp.asarray(
                self._scheduled_drop_prob(step) if train else 0.0, dtype=jnp.float32
            ),
        }
        return action_loss, metrics

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        total, _ = self._compute_losses(rng, observation, actions, train=train)
        return total

    def compute_loss_with_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        step: jnp.ndarray | int | None = None,
    ):
        return self._compute_losses(rng, observation, actions, train=train, step=step)

    @override
    def sample_actions(self, rng, observation, **kwargs):
        raise NotImplementedError("MemTokenizerModel does not implement action sampling.")
