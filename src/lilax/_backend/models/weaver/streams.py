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

"""Splitting and re-interleaving of packed understanding and generation streams."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def unpack_streams(
    packed_tokens: jax.Array,
    packed_und_token_indexes: jax.Array,
    packed_gen_token_indexes: jax.Array,
    use_ragged_ops: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Unpacks a single 1D packed sequence into understanding and generation streams.

  Args:
    packed_tokens: Tensor of shape (N_total, D_model) containing all packed
      tokens.
    packed_und_token_indexes: 1D array of indices for UND tokens.
    packed_gen_token_indexes: 1D array of indices for GEN tokens.
    use_ragged_ops: Whether to use TPU SparseCore ragged gather kernels if
      available. Default is False (standard JAX indexing).

  Returns:
    und_tokens: Tensor of shape (N_und, D_model).
    gen_tokens: Tensor of shape (N_gen, D_model).
  """
  del use_ragged_ops
  und_tokens = packed_tokens[packed_und_token_indexes]
  gen_tokens = packed_tokens[packed_gen_token_indexes]
  return und_tokens, gen_tokens


def reinterleave_streams(
    und_tokens: jax.Array,
    gen_tokens: jax.Array,
    packed_und_token_indexes: jax.Array,
    packed_gen_token_indexes: jax.Array,
    total_tokens: int,
    use_ragged_ops: bool = False,
) -> jax.Array:
  """Re-interleaves sliced understanding and generation outputs into the global packed order.

  Args:
    und_tokens: Tensor of shape (N_und, D_model) from the understanding stream.
    gen_tokens: Tensor of shape (N_gen, D_model) from the generation stream.
    packed_und_token_indexes: 1D array of indices where UND tokens belong in the
      packed sequence.
    packed_gen_token_indexes: 1D array of indices where GEN tokens belong in the
      packed sequence.
    total_tokens: Total sequence length N_total.
    use_ragged_ops: Whether to use TPU SparseCore ragged scatter kernels if
      available. Default is False (standard JAX dynamic tensor scatter).

  Returns:
    A single merged tensor of shape (N_total, D_model) matching the original
    global layout.
  """
  del use_ragged_ops
  trailing_shape = und_tokens.shape[1:]
  output_tokens = jnp.zeros((total_tokens, *trailing_shape), dtype=und_tokens.dtype)
  if und_tokens.shape[0] > 0:
    output_tokens = output_tokens.at[packed_und_token_indexes].set(und_tokens)
  if gen_tokens.shape[0] > 0:
    output_tokens = output_tokens.at[packed_gen_token_indexes].set(gen_tokens)
  return output_tokens
