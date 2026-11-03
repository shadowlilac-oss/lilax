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

"""Sinusoidal diffusion timestep embeddings and the Weaver timestep embedder."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from lilax._backend.models.weaver.dtypes import _resolve_jax_dtype
from lilax._backend.models.weaver.leaf_layers import _make_linear


def get_weaver_timestep_embedding(
    timesteps: jax.Array, num_channels: int = 256, *, flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0, scale: float = 1.0, max_period: int = 10000,
) -> jax.Array:
  """Compute sinusoidal timestep embeddings in float32."""
  half_dim = num_channels // 2
  exponent = -math.log(float(max_period)) * jnp.arange(0, half_dim, dtype=jnp.float32)
  exponent = exponent / (half_dim - downscale_freq_shift)
  emb = jnp.exp(exponent)
  emb = timesteps[..., None].astype(jnp.float32) * emb
  emb = scale * emb
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
  if flip_sin_to_cos:
    emb = jnp.concatenate([emb[..., half_dim:], emb[..., :half_dim]], axis=-1)
  if num_channels % 2 == 1:
    emb = jnp.pad(emb, ((0, 0),) * (emb.ndim - 1) + ((0, 1),))
  return emb


class WeaverTimeEmbedder(nnx.Module):
  """Sinusoidal and two-layer SiLU diffusion timestep embedder."""

  def __init__(
      self, in_channels: int = 256, time_embed_dim: int = 4096, *, timestep_scale: float = 0.001,
      max_period: int = 10000, use_bias: bool = True,
      in_kernel_axes: tuple[None | str, ...] = ("time_embed", "embed"),
      out_kernel_axes: tuple[None | str, ...] = ("embed", "embed"),
      sharding_hook: Callable[[Any, str], Any] | None = None, dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32, kernel_init: Callable[..., jax.Array] | None = None, rngs: nnx.Rngs,
  ):
    self.in_channels = in_channels
    self.time_embed_dim = time_embed_dim
    self.timestep_scale = timestep_scale
    self.max_period = max_period
    self.use_bias = use_bias
    self.dtype = _resolve_jax_dtype(dtype)
    self.weight_dtype = _resolve_jax_dtype(weight_dtype)
    self.mlp_0 = _make_linear(
        in_channels, time_embed_dim, use_bias=use_bias, dtype=jnp.float32, weight_dtype=self.weight_dtype,
        kernel_init=kernel_init, kernel_axes=in_kernel_axes, sharding_hook=sharding_hook,
        hook_name="time_mlp_0_kernel_init", rngs=rngs,
    )
    self.mlp_2 = _make_linear(
        time_embed_dim, time_embed_dim, use_bias=use_bias, dtype=jnp.float32, weight_dtype=self.weight_dtype,
        kernel_init=kernel_init, kernel_axes=out_kernel_axes, sharding_hook=sharding_hook,
        hook_name="time_mlp_2_kernel_init", rngs=rngs,
    )

  @property
  def linear_1(self) -> nnx.Linear:
    return self.mlp_0

  @property
  def linear_2(self) -> nnx.Linear:
    return self.mlp_2

  def __call__(self, timesteps: jax.Array) -> jax.Array:
    scaled_t = timesteps.astype(jnp.float32) * self.timestep_scale
    t_emb = get_weaver_timestep_embedding(
        scaled_t, num_channels=self.in_channels, flip_sin_to_cos=True, downscale_freq_shift=0.0,
        max_period=self.max_period,
    )
    hidden = jax.nn.silu(self.mlp_0(t_emb))
    return self.mlp_2(hidden).astype(self.dtype)
