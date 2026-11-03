# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual-attention functional kernels with cross-sample fences."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from lilax._backend.kernels.attention import jax_flash_attention
from lilax._backend.kernels.tokamax_splash_attention import splash_attention_kernel
from lilax._backend.models.weaver.masks import (
    build_causal_understanding_mask,
    build_causal_understanding_splash_mask,
    build_full_generative_mask,
    build_full_generative_splash_mask,
)
import numpy as np


def causal_understanding_attention(
    q_und: jax.Array,
    k_und: jax.Array,
    v_und: jax.Array,
    causal_q_offsets: jax.Array,
    scale: float | None = None,
    mask_value: float = -1e9,
    attention_kernel: str = "dot_product",
    block_q: int = 128,
    block_kv: int = 128,
) -> jax.Array:
  """Kernel 1: Causal Understanding Attention with cross-sample fences."""
  n_und, num_heads, head_dim = q_und.shape
  if n_und == 0:
    return jnp.zeros_like(q_und)

  num_kv_heads = k_und.shape[1]
  if num_heads != num_kv_heads:
    repeat_factor = num_heads // num_kv_heads
    k_und = jnp.repeat(k_und, repeat_factor, axis=1)
    v_und = jnp.repeat(v_und, repeat_factor, axis=1)

  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  if attention_kernel == "flash":
    if n_und % block_q != 0:
      raise ValueError(
          "FlashAttention requires sequence length divisible by block_q"
          f" ({block_q}), but got N_und={n_und}. Use"
          " attention_kernel='dot_product' for unaligned lengths."
      )
    mask = build_causal_understanding_mask(causal_q_offsets, n_und)
    q_scaled = q_und * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)[jnp.newaxis, ...]
    k_reshaped = k_und.transpose(1, 0, 2)[jnp.newaxis, ...]
    v_reshaped = v_und.transpose(1, 0, 2)[jnp.newaxis, ...]
    flash_out = jax_flash_attention.flash_attention_block_masked(
        q=q_reshaped,
        k=k_reshaped,
        v=v_reshaped,
        segment_ids=None,
        block_q=block_q,
        block_kv=block_kv,
        mask=mask,
        mask_value=mask_value,
    )
    out = jnp.squeeze(flash_out, axis=0).transpose(1, 0, 2)
    return out.astype(q_und.dtype)

  elif attention_kernel == "splash":
    if n_und % block_q != 0:
      raise ValueError(
          "Splash Attention requires sequence length divisible by block size"
          f" ({block_q}), but got N_und={n_und}. Use"
          " attention_kernel='dot_product' for unaligned lengths."
      )
    splash_mask = build_causal_understanding_splash_mask(causal_q_offsets, n_und)
    sa_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        block_q=block_q,
        block_kv=block_kv,
    )
    splash_kernel_fn = splash_attention_kernel.make_splash_mha_single_device(
        mask=splash_mask,
        config=sa_config,
        downcast_smem_data=False,
        partial_mask_blocks_dtype=np.int32,
    )
    q_scaled = q_und * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)
    k_reshaped = k_und.transpose(1, 0, 2)
    v_reshaped = v_und.transpose(1, 0, 2)
    splash_out = splash_kernel_fn(q_reshaped, k_reshaped, v_reshaped)
    out = splash_out.transpose(1, 0, 2)
    return out.astype(q_und.dtype)

  elif attention_kernel == "dot_product":
    scores = jnp.einsum("qhd,khd->hqk", q_und, k_und) * scale
    mask = build_causal_understanding_mask(causal_q_offsets, n_und)
    scores = jnp.where(mask[None, :, :], scores, mask_value)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q_und.dtype)
    return jnp.einsum("hqk,khd->qhd", attn_weights, v_und)
  else:
    raise ValueError(
        f"Unsupported attention_kernel '{attention_kernel}'. " "Expected one of ['dot_product', 'flash', 'splash']."
    )


def full_generative_attention(
    q_gen: jax.Array,
    k_all: jax.Array,
    v_all: jax.Array,
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    scale: float | None = None,
    mask_value: float = -1e9,
    attention_kernel: str = "dot_product",
    block_q: int = 128,
    block_kv: int = 128,
) -> jax.Array:
  """Kernel 2: Full Generative Attention with cross-sample fences."""
  n_gen, num_heads, head_dim = q_gen.shape
  total_tokens = k_all.shape[0]
  if n_gen == 0 or total_tokens == 0:
    return jnp.zeros_like(q_gen)

  num_kv_heads = k_all.shape[1]
  if num_heads != num_kv_heads:
    repeat_factor = num_heads // num_kv_heads
    k_all = jnp.repeat(k_all, repeat_factor, axis=1)
    v_all = jnp.repeat(v_all, repeat_factor, axis=1)

  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  if attention_kernel == "flash":
    if n_gen % block_q != 0 or total_tokens % block_kv != 0:
      raise ValueError(
          "FlashAttention requires sequence lengths divisible by block sizes"
          f" (block_q={block_q}, block_kv={block_kv}), but got N_gen={n_gen},"
          f" N_total={total_tokens}. Use attention_kernel='dot_product' for"
          " unaligned lengths."
      )
    mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    q_scaled = q_gen * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)[jnp.newaxis, ...]
    k_reshaped = k_all.transpose(1, 0, 2)[jnp.newaxis, ...]
    v_reshaped = v_all.transpose(1, 0, 2)[jnp.newaxis, ...]
    flash_out = jax_flash_attention.flash_attention_block_masked(
        q=q_reshaped,
        k=k_reshaped,
        v=v_reshaped,
        segment_ids=None,
        block_q=block_q,
        block_kv=block_kv,
        mask=mask,
        mask_value=mask_value,
    )
    out = jnp.squeeze(flash_out, axis=0).transpose(1, 0, 2)
    return out.astype(q_gen.dtype)

  elif attention_kernel == "splash":
    if n_gen % block_q != 0 or total_tokens % block_kv != 0:
      raise ValueError(
          "Splash Attention requires sequence lengths divisible by block sizes"
          f" (block_q={block_q}, block_kv={block_kv}), but got N_gen={n_gen},"
          f" N_total={total_tokens}. Use attention_kernel='dot_product' for"
          " unaligned lengths."
      )
    splash_mask = build_full_generative_splash_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    sa_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        block_q=block_q,
        block_kv=block_kv,
    )
    splash_kernel_fn = splash_attention_kernel.make_splash_mha_single_device(
        mask=splash_mask,
        config=sa_config,
        downcast_smem_data=False,
        partial_mask_blocks_dtype=np.int32,
    )
    q_scaled = q_gen * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)
    k_reshaped = k_all.transpose(1, 0, 2)
    v_reshaped = v_all.transpose(1, 0, 2)
    splash_out = splash_kernel_fn(q_reshaped, k_reshaped, v_reshaped)
    out = splash_out.transpose(1, 0, 2)
    return out.astype(q_gen.dtype)

  elif attention_kernel == "dot_product":
    scores = jnp.einsum("qhd,khd->hqk", q_gen, k_all) * scale
    mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    scores = jnp.where(mask[None, :, :], scores, mask_value)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q_gen.dtype)
    return jnp.einsum("hqk,khd->qhd", attn_weights, v_all)
  else:
    raise ValueError(
        f"Unsupported attention_kernel '{attention_kernel}'. " "Expected one of ['dot_product', 'flash', 'splash']."
    )
