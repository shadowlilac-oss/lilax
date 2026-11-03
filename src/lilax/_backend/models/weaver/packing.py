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

"""Boundary metadata for packed multimodal Weaver sequences."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class WeaverPackingMetadata:
  """Boundary metadata and stream indexing for packed multimodal sequences.

  Attributes:
    causal_q_offsets: Cumulative offsets for understanding (UND) queries, shape
      [num_samples + 1]. Defines sample boundaries in the UND stream.
    full_q_offsets: Cumulative offsets for generation (GEN) queries, shape
      [num_samples + 1]. Defines sample boundaries in the GEN stream.
    sample_kv_offsets: Cumulative offsets for total tokens per sample in the
      interleaved sequence, shape [num_samples + 1]. Defines sample boundaries
      for the combined KV context.
    packed_und_token_indexes: Global token indices belonging to the UND stream,
      shape [num_und_tokens].
    packed_gen_token_indexes: Global token indices belonging to the GEN stream,
      shape [num_gen_tokens].
    max_causal_len: Maximum number of UND tokens across all samples.
    max_full_len: Maximum number of GEN tokens across all samples.
    max_sample_len: Maximum number of total tokens across all samples.
    num_samples: Number of batch instances B packed into the sequence.
    total_tokens: Total sequence length N_total = N_und + N_gen.
    num_und_tokens: Total number of UND tokens across all samples.
    num_gen_tokens: Total number of GEN tokens across all samples.
  """

  causal_q_offsets: jax.Array
  full_q_offsets: jax.Array
  sample_kv_offsets: jax.Array
  packed_und_token_indexes: jax.Array
  packed_gen_token_indexes: jax.Array
  max_causal_len: int
  max_full_len: int
  max_sample_len: int
  num_samples: int
  total_tokens: int
  num_und_tokens: int
  num_gen_tokens: int


def build_weaver_packing_metadata(
    sample_und_lens: Sequence[int],
    sample_gen_lens: Sequence[int],
    packed_und_token_indexes: jax.Array | None = None,
    packed_gen_token_indexes: jax.Array | None = None,
) -> WeaverPackingMetadata:
  """Constructs WeaverPackingMetadata from per-sample token counts.

  If index arrays are not provided, assumes default contiguous layout per
  sample: each sample b has its UND tokens followed by its GEN tokens.

  Args:
    sample_und_lens: Number of UND tokens for each sample in the batch.
    sample_gen_lens: Number of GEN tokens for each sample in the batch.
    packed_und_token_indexes: Optional explicit global indices for UND tokens.
    packed_gen_token_indexes: Optional explicit global indices for GEN tokens.

  Returns:
    A fully populated WeaverPackingMetadata dataclass.
  """
  num_samples = len(sample_und_lens)
  if len(sample_gen_lens) != num_samples:
    raise ValueError(
        f"Length of sample_und_lens ({num_samples}) does not match " f"sample_gen_lens ({len(sample_gen_lens)})"
    )

  und_lens = list(sample_und_lens)
  gen_lens = list(sample_gen_lens)
  sample_lens = [u + g for u, g in zip(und_lens, gen_lens)]

  causal_q_offsets_list = [0]
  full_q_offsets_list = [0]
  sample_kv_offsets_list = [0]

  for u, g, s in zip(und_lens, gen_lens, sample_lens):
    causal_q_offsets_list.append(causal_q_offsets_list[-1] + u)
    full_q_offsets_list.append(full_q_offsets_list[-1] + g)
    sample_kv_offsets_list.append(sample_kv_offsets_list[-1] + s)

  total_tokens = sample_kv_offsets_list[-1]
  num_und_tokens = causal_q_offsets_list[-1]
  num_gen_tokens = full_q_offsets_list[-1]

  if packed_und_token_indexes is None or packed_gen_token_indexes is None:
    und_idx_list = []
    gen_idx_list = []
    current_offset = 0
    for u, g in zip(und_lens, gen_lens):
      und_idx_list.extend(range(current_offset, current_offset + u))
      gen_idx_list.extend(range(current_offset + u, current_offset + u + g))
      current_offset += u + g
    packed_und_token_indexes = jnp.array(und_idx_list, dtype=jnp.int32)
    packed_gen_token_indexes = jnp.array(gen_idx_list, dtype=jnp.int32)
  else:
    packed_und_token_indexes = jnp.asarray(packed_und_token_indexes, dtype=jnp.int32)
    packed_gen_token_indexes = jnp.asarray(packed_gen_token_indexes, dtype=jnp.int32)

  max_causal = max(und_lens) if und_lens else 0
  max_full = max(gen_lens) if gen_lens else 0
  max_sample = max(sample_lens) if sample_lens else 0

  return WeaverPackingMetadata(
      causal_q_offsets=jnp.array(causal_q_offsets_list, dtype=jnp.int32),
      full_q_offsets=jnp.array(full_q_offsets_list, dtype=jnp.int32),
      sample_kv_offsets=jnp.array(sample_kv_offsets_list, dtype=jnp.int32),
      packed_und_token_indexes=packed_und_token_indexes,
      packed_gen_token_indexes=packed_gen_token_indexes,
      max_causal_len=max_causal,
      max_full_len=max_full,
      max_sample_len=max_sample,
      num_samples=num_samples,
      total_tokens=total_tokens,
      num_und_tokens=num_und_tokens,
      num_gen_tokens=num_gen_tokens,
  )
