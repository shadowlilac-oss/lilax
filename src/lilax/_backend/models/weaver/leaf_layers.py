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

"""Pure ``flax.nnx`` leaf layer builders with logical-axis annotations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import inspect
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


def _adapt_kernel_init(
    kernel_init: Callable[..., jax.Array] | None,
) -> nnx.Initializer:
  """Adapts a 3-arg NNX initializer or 5-arg NdInitializer to a 3-arg NNX initializer."""
  if kernel_init is None:
    return nnx.initializers.lecun_normal()

  num_params = len(inspect.signature(kernel_init).parameters)

  def _wrapped(key: jax.Array, shape: Sequence[int], dtype: Any = jnp.float32) -> jax.Array:
    if num_params >= 5:
      in_axis = (0,)
      out_axis = tuple(range(1, len(shape)))
      return kernel_init(key, shape, dtype, in_axis, out_axis)
    return kernel_init(key, shape, dtype)

  return _wrapped


def _make_embed(
    num_embeddings: int,
    features: int,
    *,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    embedding_init: Callable[..., jax.Array] | None = None,
    kernel_axes: tuple[None | str, ...] = ("vocab", "embed"),
    sharding_hook: Callable[[Any, str], Any] | None = None,
    hook_name: str = "embed_kernel_init",
    rngs: nnx.Rngs,
) -> nnx.Embed:
  """Constructs a pure ``nnx.Embed`` with logical-axis annotations."""
  base_init = _adapt_kernel_init(embedding_init or nnx.initializers.normal(stddev=0.02))
  if sharding_hook is not None:
    base_init = sharding_hook(base_init, hook_name)
  meta_init = nnx.with_metadata(
      base_init,
      out_sharding=kernel_axes,
      kernel_axes=kernel_axes,
      eager_sharding=False,
  )
  embed = nnx.Embed(
      num_embeddings=num_embeddings,
      features=features,
      dtype=dtype,
      param_dtype=weight_dtype,
      embedding_init=meta_init,
      rngs=rngs,
  )
  embed.kernel_axes = kernel_axes
  return embed


def _make_linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = False,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_init: Callable[..., jax.Array] | None = None,
    kernel_axes: tuple[None | str, ...] = (),
    sharding_hook: Callable[[Any, str], Any] | None = None,
    hook_name: str = "linear_kernel_init",
    rngs: nnx.Rngs,
) -> nnx.Linear:
  """Constructs a pure ``nnx.Linear`` with logical-axis annotations."""
  base_init = _adapt_kernel_init(kernel_init)
  if sharding_hook is not None:
    base_init = sharding_hook(base_init, hook_name)
  meta_init = nnx.with_metadata(
      base_init,
      out_sharding=kernel_axes,
      kernel_axes=kernel_axes,
      eager_sharding=False,
  )
  bias_axes = kernel_axes[-1:] if kernel_axes else ()
  bias_init = nnx.with_metadata(
      nnx.initializers.zeros_init(),
      out_sharding=bias_axes,
      kernel_axes=bias_axes,
      eager_sharding=False,
  )
  layer = nnx.Linear(
      in_features=in_features,
      out_features=out_features,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=weight_dtype,
      kernel_init=meta_init,
      bias_init=bias_init,
      rngs=rngs,
  )
  layer.kernel_axes = kernel_axes
  return layer


def _make_rmsnorm(
    num_features: int,
    *,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: tuple[None | str, ...] = (),
    sharding_hook: Callable[[Any, str], Any] | None = None,
    hook_name: str = "norm_scale_init",
    rngs: nnx.Rngs,
) -> nnx.RMSNorm:
  """Constructs a pure ``nnx.RMSNorm`` with logical-axis annotations."""
  base_scale_init: Any = nnx.initializers.ones_init()
  if sharding_hook is not None:
    hooked = sharding_hook(1, hook_name)
    if callable(hooked):
      base_scale_init = hooked
  meta_scale_init = nnx.with_metadata(
      base_scale_init,
      out_sharding=kernel_axes,
      kernel_axes=kernel_axes,
      eager_sharding=False,
  )
  norm = nnx.RMSNorm(
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      param_dtype=weight_dtype,
      scale_init=meta_scale_init,
      rngs=rngs,
  )
  norm.kernel_axes = kernel_axes
  return norm
