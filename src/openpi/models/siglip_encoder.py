"""Reusable SigLIP encoder for NativeMEM memory-tokenizer pretraining."""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

import openpi.models.siglip as _siglip


class SiglipEncoder(nn.Module):
    """SigLIP ViT returning one memory summary token and the patch tokens.

    The module names mirror the pi0.5 SigLIP image tower so its weights can be
    remapped into this encoder during mem_tokenizer initialization.
    """

    patch_size: Sequence[int] = (14, 14)
    width: int = 1152
    depth: int = 27
    mlp_dim: int | None = 4304
    num_heads: int = 16
    posemb: str = "learn"
    dropout: float = 0.0
    scan: bool = True
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False, extract_block=None, ids_keep=None):
        image = jnp.asarray(image, jnp.float32)

        x = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        x = x + _siglip.get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if ids_keep is not None:
            batch_idx = jnp.arange(n)[:, None]
            x = x[batch_idx, ids_keep]

        # Keep the parameter key for compatibility with existing memory-tokenizer checkpoints.
        mem_query = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
        x = jnp.concatenate([jnp.tile(mem_query, [n, 1, 1]), x], axis=1)
        x = x.astype(self.dtype_mm)

        x, encoder_out = _siglip.Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)

        mem_token = x[:, :1]
        patch_tokens = x[:, 1:]

        intermediate = None
        if extract_block is not None:
            block_key = f"block{extract_block:02d}"
            intermediate = encoder_out[block_key]["+mlp"]

        return mem_token, patch_tokens, intermediate
