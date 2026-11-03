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

"""3D M-RoPE position ids and rotary position embedding helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rotate_half(x: jax.Array) -> jax.Array:
  """Rotates half the hidden dimensions of the input tensor."""
  half_dim = x.shape[-1] // 2
  x1 = x[..., :half_dim]
  x2 = x[..., half_dim:]
  return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int | None = 1,
) -> jax.Array:
  """Applies Rotary Position Embedding (RoPE / 3D M-RoPE) to an activation tensor."""
  if unsqueeze_dim is not None:
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)

  x_f32 = x.astype(jnp.float32)
  cos_f32 = cos.astype(jnp.float32)
  sin_f32 = sin.astype(jnp.float32)

  rotated = (x_f32 * cos_f32) + (rotate_half(x_f32) * sin_f32)
  return rotated.astype(x.dtype)


def compute_3d_mrope_cos_sin(
    position_ids_3d: jax.Array,
    head_dim: int,
    mrope_section: tuple[int, int, int] = (24, 20, 20),
    rope_theta: float = 5_000_000.0,
) -> tuple[jax.Array, jax.Array]:
  """Computes interleaved 3D M-RoPE cosine and sine tensors from 3D coordinates."""
  if sum(mrope_section) != head_dim // 2:
    raise ValueError(f"mrope_section {mrope_section} sum must equal head_dim // 2" f" ({head_dim // 2})")

  freq_dim = head_dim // 2
  inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, freq_dim, 1, dtype=jnp.float32) * 2 / head_dim))

  pos_temporal = position_ids_3d[:, 0, None].astype(jnp.float32)
  freqs = pos_temporal * inv_freq[None, :]

  pos_height = position_ids_3d[:, 1, None].astype(jnp.float32)
  pos_width = position_ids_3d[:, 2, None].astype(jnp.float32)

  h_limit = mrope_section[1] * 3
  w_limit = mrope_section[2] * 3

  h_indices = jnp.arange(1, min(h_limit, freq_dim), 3)
  w_indices = jnp.arange(2, min(w_limit, freq_dim), 3)

  freqs = freqs.at[:, h_indices].set(pos_height * inv_freq[h_indices])
  freqs = freqs.at[:, w_indices].set(pos_width * inv_freq[w_indices])

  emb = jnp.concatenate([freqs, freqs], axis=-1)
  return jnp.cos(emb), jnp.sin(emb)


def build_weaver_3d_position_ids(batch_size: int, s_und: int, t_lat: int, h_patch: int, w_patch: int) -> jax.Array:
  """Construct packed 3D M-RoPE coordinates for text and vision samples."""
  pos_und = jnp.broadcast_to(jnp.arange(s_und, dtype=jnp.int32)[:, None], (s_und, 3))
  t_grid = jnp.repeat(jnp.arange(t_lat, dtype=jnp.int32) + s_und, h_patch * w_patch)
  h_grid = jnp.tile(jnp.repeat(jnp.arange(h_patch, dtype=jnp.int32), w_patch), t_lat)
  w_grid = jnp.tile(jnp.arange(w_patch, dtype=jnp.int32), t_lat * h_patch)
  pos_gen = jnp.stack([t_grid, h_grid, w_grid], axis=-1)
  sample_pos = jnp.concatenate([pos_und, pos_gen], axis=0)
  return jnp.tile(sample_pos, (batch_size, 1))
