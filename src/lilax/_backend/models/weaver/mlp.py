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

"""Weaver feed-forward network (pure Flax NNX)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
from jax import sharding
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from lilax._backend.models.weaver.config import WeaverConfig
from lilax._backend.models.weaver.leaf_layers import _make_linear


class WeaverMLP(nnx.Module):
  """Feed-forward network for a single Weaver tower using pure ``nnx.Linear``.

  Supports two activation variants:
  * ``silu``  - gated SwiGLU: ``down(silu(gate(x)) * up(x))`` (weaver-mini)
  * ``relu2`` - squared ReLU: ``down(relu(up(x)) ** 2)`` (weaver-max)
  """

  def __init__(
      self, hidden_size: int | WeaverConfig, intermediate_size: int | None = None, *,
      hidden_act: str = "silu", use_bias: bool = False,
      up_kernel_axes: tuple[None | str, ...] = ("embed", "mlp"),
      down_kernel_axes: tuple[None | str, ...] = ("mlp", "embed"),
      sharding_hook: Callable[[Any, str], Any] | None = None, mesh: sharding.Mesh | None = None,
      dtype: Any = jnp.float32, weight_dtype: Any = jnp.float32,
      kernel_init: Callable[..., jax.Array] | None = None, rngs: nnx.Rngs,
  ):
    """Initializes the Weaver feed-forward block."""
    del mesh
    if isinstance(hidden_size, WeaverConfig):
      cfg = hidden_size
      hidden_size = cfg.hidden_size
      intermediate_size = cfg.intermediate_size
      hidden_act = cfg.hidden_act
      up_kernel_axes = cfg.mlp_up_kernel_axes
      down_kernel_axes = cfg.mlp_down_kernel_axes
      dtype = cfg.dtype
      weight_dtype = cfg.weight_dtype

    if intermediate_size is None:
      raise ValueError("intermediate_size is required when hidden_size is an int.")

    if hidden_act not in ("silu", "relu2"):
      raise ValueError(f"Weaver only supports `hidden_act` values 'silu' and 'relu2', got {hidden_act!r}.")

    self.hidden_act = hidden_act
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.up_kernel_axes = up_kernel_axes
    self.down_kernel_axes = down_kernel_axes
    self.sharding_hook = sharding_hook or (lambda x, name: x)

    self.gate_proj = (
        _make_linear(
            hidden_size, intermediate_size, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
            kernel_init=kernel_init, kernel_axes=up_kernel_axes, sharding_hook=sharding_hook,
            hook_name="gate_proj_kernel_init", rngs=rngs,
        )
        if hidden_act == "silu"
        else None
    )
    self.up_proj = _make_linear(
        hidden_size, intermediate_size, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=up_kernel_axes, sharding_hook=sharding_hook,
        hook_name="up_proj_kernel_init", rngs=rngs,
    )
    self.down_proj = _make_linear(
        intermediate_size, hidden_size, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=down_kernel_axes, sharding_hook=sharding_hook,
        hook_name="down_proj_kernel_init", rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies the feed-forward transformation.

    Input shape and layout:
      x: Activation array of shape ``[N, hidden_size]``, where ``N`` is the
        number of tokens in the tower (e.g. ``N_und`` or ``N_gen``).

    Output shape and layout:
      Transformed activations of shape ``[N, hidden_size]``.
    """
    hook = self.sharding_hook
    if self.hidden_act == "relu2":
      hidden = jnp.square(jax.nn.relu(self.up_proj(x)))
    else:
      hidden = jax.nn.silu(self.gate_proj(x)) * self.up_proj(x)
    hidden = checkpoint_name(hook(hidden, "mlpwi"), "mlpwi")
    return hook(self.down_proj(hidden), "post_mlp")
