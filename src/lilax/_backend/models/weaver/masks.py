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

"""Sample-fenced causal & bidirectional attention masks and Pallas Splash masks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from lilax._backend.kernels.tokamax_splash_attention import splash_attention_mask
from lilax._backend.kernels.tokamax_splash_attention import splash_attention_mask_info
import numpy as np


def build_causal_understanding_mask(
    causal_q_offsets: jax.Array,
    num_und_tokens: int,
) -> jax.Array:
  """Builds sample-specific lower-triangular causal attention mask for Kernel 1.

  Cross-sample interference is strictly prohibited via offset fences:
  tokens can only attend to earlier tokens within the same sample.

  Args:
    causal_q_offsets: Array of shape [num_samples + 1] giving cumulative UND
      token counts.
    num_und_tokens: Total number of UND tokens N_und.

  Returns:
    A boolean array of shape [num_und_tokens, num_und_tokens], where True
    indicates valid attention and False indicates masked-out positions.
  """
  if num_und_tokens == 0:
    return jnp.zeros((0, 0), dtype=jnp.bool_)

  indices = jnp.arange(num_und_tokens)
  sample_ids = jnp.sum(indices[:, None] >= causal_q_offsets[None, 1:], axis=-1)

  same_sample = sample_ids[:, None] == sample_ids[None, :]
  is_causal = indices[:, None] >= indices[None, :]
  return same_sample & is_causal


def build_full_generative_mask(
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    num_gen_tokens: int,
    total_tokens: int,
) -> jax.Array:
  """Builds bidirectional full generative attention mask for Kernel 2.

  Enables generation queries to attend to all preceding context (both
  textual/UND and visual/GEN tokens) within the same sample while strictly
  prohibiting cross-sample interference.

  Args:
    full_q_offsets: Array of shape [num_samples + 1] giving cumulative GEN query
      counts.
    sample_kv_offsets: Array of shape [num_samples + 1] giving cumulative total
      token counts.
    num_gen_tokens: Total number of GEN query tokens N_gen.
    total_tokens: Total sequence length N_total (combined UND + GEN).

  Returns:
    A boolean array of shape [num_gen_tokens, total_tokens], where True
    indicates valid attention and False indicates masked-out positions.
  """
  if num_gen_tokens == 0 or total_tokens == 0:
    return jnp.zeros((num_gen_tokens, total_tokens), dtype=jnp.bool_)

  q_indices = jnp.arange(num_gen_tokens)
  kv_indices = jnp.arange(total_tokens)

  q_sample_ids = jnp.sum(q_indices[:, None] >= full_q_offsets[None, 1:], axis=-1)
  kv_sample_ids = jnp.sum(kv_indices[:, None] >= sample_kv_offsets[None, 1:], axis=-1)

  return q_sample_ids[:, None] == kv_sample_ids[None, :]


def build_causal_understanding_splash_mask(
    causal_q_offsets: jax.Array,
    num_und_tokens: int,
):
  """Builds a Pallas Splash Attention Mask for Kernel 1 (Causal Understanding)."""
  dense_mask = build_causal_understanding_mask(causal_q_offsets, num_und_tokens)
  return splash_attention_mask.NumpyMask(np.asarray(dense_mask, dtype=np.bool_))


def build_full_generative_splash_mask(
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    num_gen_tokens: int,
    total_tokens: int,
):
  """Builds a Pallas Splash Attention Mask for Kernel 2 (Full Generative)."""
  dense_mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, num_gen_tokens, total_tokens)
  return splash_attention_mask.NumpyMask(np.asarray(dense_mask, dtype=np.bool_))


def compile_weaver_splash_mask(
    mask_array: jax.Array | np.ndarray,
    block_q: int = 128,
    block_kv: int = 128,
):
  """Compiles a dense boolean mask into Pallas Splash Attention MaskInfo."""
  mask_np = np.asarray(mask_array, dtype=np.bool_)
  q_len, kv_len = mask_np.shape
  if q_len % block_q != 0 or kv_len % block_kv != 0:
    raise ValueError(
        f"Sequence lengths ({q_len}, {kv_len}) must be multiples of block sizes"
        f" ({block_q}, {block_kv}) for Splash Attention compilation."
    )
  return splash_attention_mask_info.process_dynamic_mask(mask_np, (block_q, block_kv))
