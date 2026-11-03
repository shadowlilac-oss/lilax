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

"""Weaver dual-pathway attention module (pure Flax NNX)."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any, Literal, overload

from flax import nnx
import jax
from jax import sharding
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from lilax._backend.models.weaver.attention_kernels import causal_understanding_attention, full_generative_attention
from lilax._backend.models.weaver.config import WeaverConfig
from lilax._backend.models.weaver.leaf_layers import _make_linear, _make_rmsnorm
from lilax._backend.models.weaver.packing import WeaverPackingMetadata
from lilax._backend.models.weaver.rotary import apply_rotary_pos_emb
from lilax._backend.models.weaver.streams import reinterleave_streams, unpack_streams


class WeaverDualAttention(nnx.Module):
  """Dual-pathway packed attention for Weaver using pure ``flax.nnx`` layers.

  Composes ``nnx.Linear`` and ``nnx.RMSNorm`` directly with zero wrapper layer
  classes.
  """

  def __init__(
      self,
      dim: int | WeaverConfig,
      num_heads: int | None = None,
      num_kv_heads: int | None = None,
      head_dim: int | None = None,
      *,
      qk_norm_for_text: bool = True,
      qk_norm_for_diffusion: bool = True,
      use_und_k_norm_for_gen: bool = False,
      use_bias: bool = False,
      rms_norm_eps: float = 1e-6,
      attention_kernel: str = "dot_product",
      use_ragged_ops: bool = False,
      block_q: int = 128,
      block_kv: int = 128,
      q_kernel_axes: tuple[None | str, ...] = ("embed", "heads"),
      kv_kernel_axes: tuple[None | str, ...] = ("embed", "kv_heads"),
      o_kernel_axes: tuple[None | str, ...] = ("heads", "embed"),
      qk_norm_kernel_axes: tuple[None | str, ...] = (),
      sharding_hook: Callable[[Any, str], Any] | None = None,
      mesh: sharding.Mesh | None = None,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_init: Callable[..., jax.Array] | None = None,
      rngs: nnx.Rngs,
  ):
    """Initializes ``WeaverDualAttention`` from explicit arguments or ``WeaverConfig``."""
    del mesh
    if isinstance(dim, WeaverConfig):
      cfg = dim
      dim = cfg.hidden_size
      num_heads = cfg.num_attention_heads
      num_kv_heads = cfg.num_key_value_heads
      head_dim = cfg.head_dim
      qk_norm_for_text = cfg.qk_norm_for_text
      qk_norm_for_diffusion = cfg.qk_norm_for_diffusion
      use_und_k_norm_for_gen = cfg.use_und_k_norm_for_gen
      use_bias = cfg.attention_bias
      rms_norm_eps = cfg.rms_norm_eps
      attention_kernel = cfg.attention_kernel
      use_ragged_ops = cfg.use_ragged_ops
      block_q = cfg.block_q
      block_kv = cfg.block_kv
      q_kernel_axes = cfg.q_kernel_axes
      kv_kernel_axes = cfg.kv_kernel_axes
      o_kernel_axes = cfg.o_kernel_axes
      qk_norm_kernel_axes = cfg.qk_norm_kernel_axes
      dtype = cfg.dtype
      weight_dtype = cfg.weight_dtype

    if num_heads is None or num_kv_heads is None or head_dim is None:
      raise ValueError("num_heads, num_kv_heads, and head_dim are required when dim is an int.")

    if attention_kernel not in ("dot_product", "flash", "splash"):
      raise ValueError(
          f"Unsupported attention_kernel '{attention_kernel}'. "
          "Supported kernels are ['dot_product', 'flash', 'splash']."
      )

    self.attention_kernel = attention_kernel
    self.use_ragged_ops = use_ragged_ops
    self.block_q = block_q
    self.block_kv = block_kv
    self.dim = dim
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.qk_norm_for_text = qk_norm_for_text
    self.qk_norm_for_diffusion = qk_norm_for_diffusion
    self.use_und_k_norm_for_gen = use_und_k_norm_for_gen
    self.q_kernel_axes = q_kernel_axes
    self.kv_kernel_axes = kv_kernel_axes
    self.o_kernel_axes = o_kernel_axes
    self.qk_norm_kernel_axes = qk_norm_kernel_axes
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.scaling = 1.0 / math.sqrt(head_dim)

    # Understanding Pathway Projections (pure nnx.Linear)
    self.q_proj = _make_linear(
        dim, num_heads * head_dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=q_kernel_axes, sharding_hook=sharding_hook,
        hook_name="q_proj_kernel_init", rngs=rngs,
    )
    self.k_proj = _make_linear(
        dim, num_kv_heads * head_dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=kv_kernel_axes, sharding_hook=sharding_hook,
        hook_name="k_proj_kernel_init", rngs=rngs,
    )
    self.v_proj = _make_linear(
        dim, num_kv_heads * head_dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=kv_kernel_axes, sharding_hook=sharding_hook,
        hook_name="v_proj_kernel_init", rngs=rngs,
    )
    self.o_proj = _make_linear(
        num_heads * head_dim, dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=o_kernel_axes, sharding_hook=sharding_hook,
        hook_name="o_proj_kernel_init", rngs=rngs,
    )

    # Understanding QK Normalization (pure nnx.RMSNorm)
    if qk_norm_for_text:
      self.q_norm = _make_rmsnorm(
          head_dim, epsilon=rms_norm_eps, dtype=dtype, weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes, sharding_hook=sharding_hook, hook_name="qk_norm_scale_init", rngs=rngs,
      )
      self.k_norm = _make_rmsnorm(
          head_dim, epsilon=rms_norm_eps, dtype=dtype, weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes, sharding_hook=sharding_hook, hook_name="qk_norm_scale_init", rngs=rngs,
      )
    else:
      self.q_norm = None
      self.k_norm = None

    # Generation Pathway Projections (pure nnx.Linear)
    self.q_proj_gen = _make_linear(
        dim, num_heads * head_dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=q_kernel_axes, sharding_hook=sharding_hook,
        hook_name="q_proj_gen_kernel_init", rngs=rngs,
    )
    self.k_proj_gen = _make_linear(
        dim, num_kv_heads * head_dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=kv_kernel_axes, sharding_hook=sharding_hook,
        hook_name="k_proj_gen_kernel_init", rngs=rngs,
    )
    self.v_proj_gen = _make_linear(
        dim, num_kv_heads * head_dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=kv_kernel_axes, sharding_hook=sharding_hook,
        hook_name="v_proj_gen_kernel_init", rngs=rngs,
    )
    self.o_proj_gen = _make_linear(
        num_heads * head_dim, dim, use_bias=use_bias, dtype=dtype, weight_dtype=weight_dtype,
        kernel_init=kernel_init, kernel_axes=o_kernel_axes, sharding_hook=sharding_hook,
        hook_name="o_proj_gen_kernel_init", rngs=rngs,
    )

    # Generation QK Normalization (pure nnx.RMSNorm)
    if qk_norm_for_diffusion:
      self.q_norm_gen = _make_rmsnorm(
          head_dim, epsilon=rms_norm_eps, dtype=dtype, weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes, sharding_hook=sharding_hook, hook_name="qk_norm_scale_init", rngs=rngs,
      )
      self.k_norm_gen = _make_rmsnorm(
          head_dim, epsilon=rms_norm_eps, dtype=dtype, weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes, sharding_hook=sharding_hook, hook_name="qk_norm_scale_init", rngs=rngs,
      )
    else:
      self.q_norm_gen = None
      self.k_norm_gen = None

    # Cross-Attention UND K Normalization (pure nnx.RMSNorm)
    if use_und_k_norm_for_gen and qk_norm_for_diffusion and not qk_norm_for_text:
      self.k_norm_und_for_gen = _make_rmsnorm(
          head_dim, epsilon=rms_norm_eps, dtype=dtype, weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes, sharding_hook=sharding_hook, hook_name="qk_norm_scale_init", rngs=rngs,
      )
    else:
      self.k_norm_und_for_gen = None

  @classmethod
  def from_config(
      cls,
      config: WeaverConfig,
      *,
      rngs: nnx.Rngs,
      sharding_hook: Callable[[Any, str], Any] | None = None,
      kernel_init: Callable[..., jax.Array] | None = None,
  ) -> WeaverDualAttention:
    """Constructs ``WeaverDualAttention`` from a ``WeaverConfig``."""
    return cls(config, sharding_hook=sharding_hook, kernel_init=kernel_init, rngs=rngs)

  @overload
  def __call__(
      self, tokens: jax.Array, metadata: WeaverPackingMetadata, cos: jax.Array | None = None,
      sin: jax.Array | None = None, *, reinterleave: Literal[True] = True,
  ) -> jax.Array:
    ...

  @overload
  def __call__(
      self, tokens: tuple[jax.Array, jax.Array], metadata: WeaverPackingMetadata, cos: jax.Array | None = None,
      sin: jax.Array | None = None, *, reinterleave: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    ...

  @overload
  def __call__(
      self, tokens: jax.Array, metadata: WeaverPackingMetadata, cos: jax.Array | None = None,
      sin: jax.Array | None = None, *, reinterleave: Literal[False],
  ) -> tuple[jax.Array, jax.Array]:
    ...

  def __call__(
      self,
      tokens: jax.Array | tuple[jax.Array, jax.Array],
      metadata: WeaverPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: bool = True,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Executes dual-pathway attention on understanding and generation tokens.

    Input shapes and layouts:
      tokens:
        * Packed mode: 2D array of shape ``[N_total, dim]`` in global packed
          order, where ``N_total = N_und + N_gen``.
        * Unpacked mode: Tuple ``(und_tokens, gen_tokens)`` with shapes
          ``[N_und, dim]`` and ``[N_gen, dim]``.
      metadata: ``WeaverPackingMetadata`` containing cumulative sample
        boundaries (``causal_q_offsets``: ``[B + 1]``, ``full_q_offsets``:
        ``[B + 1]``, ``sample_kv_offsets``: ``[B + 1]``) and 1D index arrays
        (``packed_und_token_indexes``: ``[N_und]``,
        ``packed_gen_token_indexes``: ``[N_gen]``).
      cos: Optional 3D M-RoPE cosine table of shape ``[N_total, head_dim]``.
      sin: Optional 3D M-RoPE sine table of shape ``[N_total, head_dim]``.
      reinterleave: When True and ``tokens`` is a 2D array, re-interleaves the
        outputs back into global packed order.

    Output shapes and layouts:
      * When ``reinterleave=True`` and ``tokens`` is 2D: Transformed packed array
        of shape ``[N_total, dim]`` in global packed order.
      * When ``reinterleave=False`` or ``tokens`` is a tuple: Tuple
        ``(und_out, gen_out)`` of shapes ``[N_und, dim]`` and ``[N_gen, dim]``.
    """
    hook = self.sharding_hook
    if isinstance(tokens, tuple):
      und_tokens, gen_tokens = tokens
      should_reinterleave = False
    else:
      und_tokens, gen_tokens = unpack_streams(
          tokens, metadata.packed_und_token_indexes, metadata.packed_gen_token_indexes,
          use_ragged_ops=self.use_ragged_ops,
      )
      should_reinterleave = reinterleave

    n_und = und_tokens.shape[0]
    n_gen = gen_tokens.shape[0]
    total_tokens = metadata.total_tokens

    # 1. Linear Projections
    if n_und > 0:
      q_und = self.q_proj(und_tokens).reshape(n_und, self.num_heads, self.head_dim)
      k_und = self.k_proj(und_tokens).reshape(n_und, self.num_kv_heads, self.head_dim)
      v_und = self.v_proj(und_tokens).reshape(n_und, self.num_kv_heads, self.head_dim)
    else:
      q_und = jnp.zeros((0, self.num_heads, self.head_dim), dtype=self.dtype)
      k_und = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)
      v_und = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)

    if n_gen > 0:
      q_gen = self.q_proj_gen(gen_tokens).reshape(n_gen, self.num_heads, self.head_dim)
      k_gen = self.k_proj_gen(gen_tokens).reshape(n_gen, self.num_kv_heads, self.head_dim)
      v_gen = self.v_proj_gen(gen_tokens).reshape(n_gen, self.num_kv_heads, self.head_dim)
    else:
      q_gen = jnp.zeros((0, self.num_heads, self.head_dim), dtype=self.dtype)
      k_gen = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)
      v_gen = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)

    # 2. QK Normalization
    if self.q_norm is not None and n_und > 0:
      q_und = self.q_norm(q_und)
    if self.k_norm is not None and n_und > 0:
      k_und = self.k_norm(k_und)
    if self.q_norm_gen is not None and n_gen > 0:
      q_gen = self.q_norm_gen(q_gen)
    if self.k_norm_gen is not None and n_gen > 0:
      k_gen = self.k_norm_gen(k_gen)

    # 3. Apply 3D M-RoPE
    k_und_pre_rope = k_und
    if cos is not None and sin is not None:
      if n_und > 0:
        cos_und = cos[metadata.packed_und_token_indexes]
        sin_und = sin[metadata.packed_und_token_indexes]
        q_und = apply_rotary_pos_emb(q_und, cos_und, sin_und, unsqueeze_dim=1)
        k_und = apply_rotary_pos_emb(k_und, cos_und, sin_und, unsqueeze_dim=1)
      if n_gen > 0:
        cos_gen = cos[metadata.packed_gen_token_indexes]
        sin_gen = sin[metadata.packed_gen_token_indexes]
        q_gen = apply_rotary_pos_emb(q_gen, cos_gen, sin_gen, unsqueeze_dim=1)
        k_gen = apply_rotary_pos_emb(k_gen, cos_gen, sin_gen, unsqueeze_dim=1)

    q_und = checkpoint_name(hook(q_und, "query"), "query_proj")
    k_und = checkpoint_name(hook(k_und, "key"), "key_proj")
    v_und = checkpoint_name(hook(v_und, "value"), "value_proj")
    q_gen = checkpoint_name(hook(q_gen, "gen_query"), "gen_query_proj")
    k_gen = checkpoint_name(hook(k_gen, "gen_key"), "gen_key_proj")
    v_gen = checkpoint_name(hook(v_gen, "gen_value"), "gen_value_proj")

    # 4. Cross-Attention UND K Normalization (if configured)
    if self.k_norm_und_for_gen is not None and n_und > 0:
      k_und_norm_for_gen = self.k_norm_und_for_gen(k_und_pre_rope)
      if cos is not None and sin is not None:
        k_und_for_gen = apply_rotary_pos_emb(k_und_norm_for_gen, cos_und, sin_und, unsqueeze_dim=1)
      else:
        k_und_for_gen = k_und_norm_for_gen
    else:
      k_und_for_gen = k_und

    # 5. Assemble Interleaved K_all and V_all for Kernel 2
    k_all = jnp.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    v_all = jnp.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    if n_und > 0:
      k_all = k_all.at[metadata.packed_und_token_indexes].set(k_und_for_gen)
      v_all = v_all.at[metadata.packed_und_token_indexes].set(v_und)
    if n_gen > 0:
      k_all = k_all.at[metadata.packed_gen_token_indexes].set(k_gen)
      v_all = v_all.at[metadata.packed_gen_token_indexes].set(v_gen)

    # 6. Kernel 1: Causal Understanding Attention
    if n_und > 0:
      und_attn_context = causal_understanding_attention(
          q_und=q_und, k_und=k_und, v_und=v_und, causal_q_offsets=metadata.causal_q_offsets,
          scale=self.scaling, attention_kernel=self.attention_kernel, block_q=self.block_q, block_kv=self.block_kv,
      )
      und_attn_context = checkpoint_name(hook(und_attn_context, "attn_out"), "attention_out")
      und_attn_flat = und_attn_context.reshape(n_und, self.num_heads * self.head_dim)
      und_out = hook(self.o_proj(und_attn_flat), "post_attn")
    else:
      und_out = jnp.zeros((0, self.dim), dtype=self.dtype)

    # 7. Kernel 2: Full Generative Attention
    if n_gen > 0:
      gen_attn_context = full_generative_attention(
          q_gen=q_gen, k_all=k_all, v_all=v_all, full_q_offsets=metadata.full_q_offsets,
          sample_kv_offsets=metadata.sample_kv_offsets, scale=self.scaling, attention_kernel=self.attention_kernel,
          block_q=self.block_q, block_kv=self.block_kv,
      )
      gen_attn_context = checkpoint_name(hook(gen_attn_context, "gen_attn_out"), "gen_attention_out")
      gen_attn_flat = gen_attn_context.reshape(n_gen, self.num_heads * self.head_dim)
      gen_out = hook(self.o_proj_gen(gen_attn_flat), "gen_post_attn")
    else:
      gen_out = jnp.zeros((0, self.dim), dtype=self.dtype)

    if should_reinterleave:
      return reinterleave_streams(
          und_tokens=und_out, gen_tokens=gen_out, packed_und_token_indexes=metadata.packed_und_token_indexes,
          packed_gen_token_indexes=metadata.packed_gen_token_indexes, total_tokens=metadata.total_tokens,
          use_ragged_ops=self.use_ragged_ops,
      )

    return und_out, gen_out


WeaverAttention = WeaverDualAttention
