import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    input_mask = jnp.asarray(input_mask, dtype=jnp.bool_)
    mask_ar = jnp.asarray(mask_ar, dtype=jnp.bool_)

    # Most call sites pass a static 1D autoregressive mask. Evaluate its cumsum at
    # trace time before broadcasting so XLA does not spend time constant-folding a
    # batch-sized reduce-window during compilation.
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
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
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


class Pi0RTC(_model.BaseModel):
    """Pi0 with Real-Time Chunking (RTC).

    Same architecture as :class:`openpi.models.pi0.Pi0`, but trained and sampled with
    an action prefix: the first `delay` actions of the chunk are treated as already-known
    ground truth (clean, t=0), and only the remaining "postfix" actions are denoised. This
    lets the policy condition new chunks on actions still executing from the previous chunk.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.max_delay = config.max_delay

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ah emb"] | None,
    ]:
        """Embed suffix tokens with per-token flow matching timesteps (RTC).

        Args:
            timestep: Per-token timestep of shape [b, ah]. In our convention,
                0.0 = clean data, 1.0 = pure noise.
        """
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)  # [b, ah, width]
        b, ah = timestep.shape
        emb_dim = self.action_in_proj.out_features
        # Per-token time embedding: flatten [b, ah] -> [b*ah], compute, reshape
        time_flat = timestep.reshape(-1)
        time_emb_flat = posemb_sincos(time_flat, emb_dim, min_period=4e-3, max_period=4.0)
        time_emb = time_emb_flat.reshape(b, ah, -1)  # [b, ah, width]

        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)  # Linear works on last dim
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb  # [b, ah, width] per-token conditioning
        else:
            # Per-token time concatenation; mix timestep + action information using an MLP (no adaRMS)
            action_time_tokens = jnp.concatenate([action_tokens, time_emb], axis=-1)  # [b, ah, 2*width]
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b = actions.shape[0]
        ah = self.action_horizon
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (b,)) * 0.999 + 0.001  # [b]

        # RTC: sample random delay per batch element
        delay = jax.random.randint(delay_rng, (b,), 0, self.max_delay)  # [b]
        prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]  # [b, ah]

        # Per-token time: prefix (ground truth) gets 0.0, postfix gets sampled time
        # Convention: t=0 means clean data, t=1 means pure noise
        time_per_token = jnp.where(prefix_mask, 0.0, time[:, None])  # [b, ah]

        # Noisy interpolation: x_t = t * noise + (1 - t) * actions
        # For prefix (t=0): x_t = actions (ground truth automatically)
        # For postfix: standard flow-matching interpolation
        time_expanded = time_per_token[:, :, None]  # [b, ah, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # target velocity

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask_attn, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_per_token)

        input_mask = jnp.concatenate([prefix_mask_attn, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask.astype(jnp.int32), axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # RTC: mask loss to postfix only
        postfix_mask = jnp.logical_not(prefix_mask)  # [b, ah]
        sq_err = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [b, ah]
        return sq_err * postfix_mask  # zero out prefix tokens

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        delay: at.Int[at.Array, " b"] | None = None,
    ) -> _model.Actions:
        """Sample actions with optional RTC action prefix conditioning.

        Args:
            action_prefix: Ground-truth action prefix padded to [b, ah, ad].
                Only the first `delay` actions per batch element are valid.
            delay: Number of prefix actions per batch element [b].
                If None (or action_prefix is None), no prefix conditioning is used.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Build prefix mask for RTC conditioning
        use_rtc = action_prefix is not None and delay is not None
        if use_rtc:
            rtc_prefix_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]  # [b, ah]
        else:
            rtc_prefix_mask = jnp.zeros((batch_size, self.action_horizon), dtype=jnp.bool_)
            action_prefix = jnp.zeros_like(noise)  # placeholder, won't be used

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask.astype(jnp.int32), axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # RTC: replace prefix with ground-truth actions
            x_t = jnp.where(rtc_prefix_mask[:, :, None], action_prefix, x_t)
            # Per-token time: prefix gets 0.0 (clean), postfix gets current time
            time_per_token = jnp.where(
                rtc_prefix_mask, 0.0, jnp.broadcast_to(time, (batch_size, self.action_horizon))
            )  # [b, ah]

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time_per_token
            )
            # `suffix_attn_mask` says how suffix tokens attend to each other.
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_cross_mask` says how suffix tokens attend to prefix tokens.
            prefix_cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `full_attn_mask` covers suffix queries over the full prefix + suffix KV sequence.
            full_attn_mask = jnp.concatenate([prefix_cross_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask.astype(jnp.int32), axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # Ensure the exact ground-truth prefix is returned for RTC-conditioned elements.
        return jnp.where(rtc_prefix_mask[:, :, None], action_prefix, x_0)
