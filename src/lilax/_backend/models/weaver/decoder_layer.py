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

r"""Weaver dual-pathway Mixture-of-Transformers decoder layer.

    ========================================================================
                   WEAVER DECODER LAYER DATA FLOW
    ========================================================================

      und_seq [U, M]                                gen_seq [G, M]
           │                                              │
           ├──────────────┐                ┌──────────────┤
           │      input_layernorm   input_layernorm_moe_gen
           │              │                │              │
           │              └──── WeaverDualAttention ──────┘
           │                    (causal und | full gen)
           │              ┌────────────────┬──────────────┐
           │              │                │              │
           ▼              ▼                ▼              ▼
      residual_und = und_seq + und_attn   residual_gen = gen_seq + gen_attn
           │                                              │
    post_attention_layernorm          post_attention_layernorm_moe_gen
           │                                              │
          mlp                                        mlp_moe_gen
           │                                              │
           ▼                                              ▼
      und_out = residual_und + mlp_out    gen_out = residual_gen + mlp_out
      [U, M]                                              [G, M]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
from jax import sharding
import jax.numpy as jnp
from lilax._backend.models.weaver.attention import WeaverDualAttention
from lilax._backend.models.weaver.config import WeaverConfig
from lilax._backend.models.weaver.leaf_layers import _make_rmsnorm
from lilax._backend.models.weaver.mlp import WeaverMLP
from lilax._backend.models.weaver.packing import WeaverPackingMetadata


class WeaverMoTDecoderLayer(nnx.Module):
  """Dual-pathway Weaver Mixture-of-Transformers decoder layer."""

  def __init__(
      self,
      hidden_size: int | WeaverConfig,
      head_dim: int | None = None,
      num_attention_heads: int | None = None,
      num_key_value_heads: int | None = None,
      intermediate_size: int | None = None,
      *,
      attention_bias: bool = False,
      rms_norm_eps: float = 1e-6,
      hidden_act: str = "silu",
      qk_norm_for_text: bool = True,
      qk_norm_for_diffusion: bool = True,
      use_und_k_norm_for_gen: bool = False,
      attention_kernel: str = "dot_product",
      use_ragged_ops: bool = False,
      block_q: int = 128,
      block_kv: int = 128,
      q_kernel_axes: tuple[None | str, ...] = ("embed", "heads"),
      kv_kernel_axes: tuple[None | str, ...] = ("embed", "kv_heads"),
      o_kernel_axes: tuple[None | str, ...] = ("heads", "embed"),
      qk_norm_kernel_axes: tuple[None | str, ...] = (),
      mlp_up_kernel_axes: tuple[None | str, ...] = ("embed", "mlp"),
      mlp_down_kernel_axes: tuple[None | str, ...] = ("mlp", "embed"),
      norm_kernel_axes: tuple[None | str, ...] = ("norm",),
      sharding_hook: Callable[[Any, str], Any] | None = None,
      mesh: sharding.Mesh | None = None,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_init: Callable[..., jax.Array] | None = None,
      rngs: nnx.Rngs,
  ):
    del mesh
    if not isinstance(hidden_size, int):
      cfg = hidden_size if isinstance(hidden_size, WeaverConfig) else WeaverConfig.from_maxtext_config(hidden_size)
      hidden_size = cfg.hidden_size
      head_dim = cfg.head_dim
      num_attention_heads = cfg.num_attention_heads
      num_key_value_heads = cfg.num_key_value_heads
      intermediate_size = cfg.intermediate_size
      attention_bias = cfg.attention_bias
      rms_norm_eps = cfg.rms_norm_eps
      hidden_act = cfg.hidden_act
      qk_norm_for_text = cfg.qk_norm_for_text
      qk_norm_for_diffusion = cfg.qk_norm_for_diffusion
      use_und_k_norm_for_gen = cfg.use_und_k_norm_for_gen
      attention_kernel = cfg.attention_kernel
      use_ragged_ops = cfg.use_ragged_ops
      block_q = cfg.block_q
      block_kv = cfg.block_kv
      q_kernel_axes = cfg.q_kernel_axes
      kv_kernel_axes = cfg.kv_kernel_axes
      o_kernel_axes = cfg.o_kernel_axes
      qk_norm_kernel_axes = cfg.qk_norm_kernel_axes
      mlp_up_kernel_axes = cfg.mlp_up_kernel_axes
      mlp_down_kernel_axes = cfg.mlp_down_kernel_axes
      norm_kernel_axes = cfg.norm_kernel_axes
      dtype = cfg.dtype
      weight_dtype = cfg.weight_dtype

    if head_dim is None or num_attention_heads is None or num_key_value_heads is None or intermediate_size is None:
      raise ValueError(
          "head_dim, num_attention_heads, num_key_value_heads, and intermediate_size "
          "are required when hidden_size is an int."
      )

    self.hidden_size = hidden_size
    self.head_dim = head_dim
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.intermediate_size = intermediate_size
    self.rms_norm_eps = rms_norm_eps
    self.hidden_act = hidden_act
    self.dtype = dtype
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    self.self_attn = WeaverDualAttention(
        dim=hidden_size, num_heads=num_attention_heads, num_kv_heads=num_key_value_heads, head_dim=head_dim,
        qk_norm_for_text=qk_norm_for_text, qk_norm_for_diffusion=qk_norm_for_diffusion,
        use_und_k_norm_for_gen=use_und_k_norm_for_gen, use_bias=attention_bias, rms_norm_eps=rms_norm_eps,
        attention_kernel=attention_kernel, use_ragged_ops=use_ragged_ops, block_q=block_q, block_kv=block_kv,
        q_kernel_axes=q_kernel_axes, kv_kernel_axes=kv_kernel_axes, o_kernel_axes=o_kernel_axes,
        qk_norm_kernel_axes=qk_norm_kernel_axes, sharding_hook=sharding_hook, dtype=dtype,
        weight_dtype=weight_dtype, kernel_init=kernel_init, rngs=rngs,
    )

    def _norm() -> nnx.RMSNorm:
      return _make_rmsnorm(
          hidden_size, epsilon=rms_norm_eps, dtype=dtype, weight_dtype=weight_dtype,
          kernel_axes=norm_kernel_axes, sharding_hook=sharding_hook, hook_name="norm_scale_init", rngs=rngs,
      )

    def _mlp() -> WeaverMLP:
      return WeaverMLP(
          hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act,
          up_kernel_axes=mlp_up_kernel_axes, down_kernel_axes=mlp_down_kernel_axes, sharding_hook=sharding_hook,
          dtype=dtype, weight_dtype=weight_dtype, kernel_init=kernel_init, rngs=rngs,
      )

    self.input_layernorm = _norm()
    self.post_attention_layernorm = _norm()
    self.mlp = _mlp()
    self.input_layernorm_moe_gen = _norm()
    self.post_attention_layernorm_moe_gen = _norm()
    self.mlp_moe_gen = _mlp()

  @classmethod
  def from_config(
      cls, config: WeaverConfig | Any, *, rngs: nnx.Rngs,
      sharding_hook: Callable[[Any, str], Any] | None = None,
      kernel_init: Callable[..., jax.Array] | None = None,
  ) -> WeaverMoTDecoderLayer:
    return cls(config, sharding_hook=sharding_hook, kernel_init=kernel_init, rngs=rngs)

  @property
  def to_q(self) -> nnx.Linear:
    return self.self_attn.q_proj

  @property
  def to_k(self) -> nnx.Linear:
    return self.self_attn.k_proj

  @property
  def to_v(self) -> nnx.Linear:
    return self.self_attn.v_proj

  @property
  def to_out(self) -> nnx.Linear:
    return self.self_attn.o_proj

  @property
  def norm_q(self) -> nnx.RMSNorm | None:
    return self.self_attn.q_norm

  @property
  def norm_k(self) -> nnx.RMSNorm | None:
    return self.self_attn.k_norm

  @property
  def add_q_proj(self) -> nnx.Linear:
    return self.self_attn.q_proj_gen

  @property
  def add_k_proj(self) -> nnx.Linear:
    return self.self_attn.k_proj_gen

  @property
  def add_v_proj(self) -> nnx.Linear:
    return self.self_attn.v_proj_gen

  @property
  def to_add_out(self) -> nnx.Linear:
    return self.self_attn.o_proj_gen

  @property
  def norm_added_q(self) -> nnx.RMSNorm | None:
    return self.self_attn.q_norm_gen

  @property
  def norm_added_k(self) -> nnx.RMSNorm | None:
    return self.self_attn.k_norm_gen

  @property
  def k_norm_und_for_gen(self) -> nnx.RMSNorm | None:
    return self.self_attn.k_norm_und_for_gen

  def __call__(
      self, und_seq: jax.Array, gen_seq: jax.Array, metadata: WeaverPackingMetadata,
      cos: jax.Array | None = None, sin: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    hook = self.sharding_hook
    und_norm = hook(self.input_layernorm(und_seq), "attn_input")
    gen_norm = hook(self.input_layernorm_moe_gen(gen_seq), "attn_input")
    und_attn_out, gen_attn_out = self.self_attn((und_norm, gen_norm), metadata, cos, sin)
    residual_und = hook(und_seq + und_attn_out, "post_attn")
    residual_gen = hook(gen_seq + gen_attn_out, "gen_post_attn")
    mlp_in_und = hook(self.post_attention_layernorm(residual_und), "mlp_input")
    mlp_in_gen = hook(self.post_attention_layernorm_moe_gen(residual_gen), "mlp_input")
    mlp_out_und = self.mlp(mlp_in_und)
    mlp_out_gen = self.mlp_moe_gen(mlp_in_gen)
    return hook(residual_und + mlp_out_und, "post_mlp"), hook(residual_gen + mlp_out_gen, "gen_post_mlp")
