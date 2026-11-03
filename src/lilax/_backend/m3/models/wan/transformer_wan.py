"""Wan diffusion transformer attention components."""

import contextlib
import dataclasses
import math
from typing import Any, Dict, Optional, Tuple, Union

from flax import nnx
import jax
from jax import Array
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import tokamax

from lilax._backend.common import common_types
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask
from lilax._backend.m3.core.sharding import Sharding, TensorType
from lilax._backend.utils import max_logging
from .transformer_attention_dot_product import _dot_product_attention
from .transformer_attention_flash import _pad_axis, _splash_block_sizes, unflatten_heads
from .transformer_embeddings import apply_rope, embed_model_conditioning, get_1d_rotary_pos_embed, get_sinusoidal_embeddings
from .transformer_model_execution import combine_conditioning, compute_layer_kv_cache, run_transformer_blocks
from .transformer_timestep_embedding import TimestepEmbedding
from .utils_wan import (BlockSizes, CONTEXT, CROSS_ATTN_HEAD, CROSS_ATTN_KV_LENGTH, CROSS_ATTN_Q_LENGTH,
                        SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, SELF_ATTN_Q_LENGTH, TpuType, constrain, get_activation,
                        get_tpu_type, param_metadata, safe_getattr, tokamax_splash)

Array = common_types.Array
DType = common_types.DType
AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
EMBED = common_types.EMBED

for _function in (_dot_product_attention, _pad_axis, _splash_block_sizes, unflatten_heads, apply_rope,
                  get_1d_rotary_pos_embed, get_sinusoidal_embeddings):
  _function.__module__ = __name__
del _function
TimestepEmbedding.__module__ = __name__


class FP32LayerNorm(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, eps: float, elementwise_affine: bool):
    self.layer_norm = nnx.LayerNorm(rngs=rngs, num_features=dim, epsilon=eps, use_bias=elementwise_affine,
                                    use_scale=elementwise_affine, param_dtype=jnp.float32, dtype=jnp.float32)

  def __call__(self, inputs: jax.Array) -> jax.Array:
    origin_dtype = inputs.dtype
    return self.layer_norm(inputs.astype(dtype=jnp.float32)).astype(dtype=origin_dtype)


class SimpleFeedForward(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, dim_out: Optional[int] = None, mult: int = 4,
               activation_fn: str = "gelu", dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: Optional[jax.lax.Precision] = None, sharding_specs: Optional[Any] = None,
               sharding: Sharding | None = None):
    inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim
    net_0_kernel = safe_getattr(sharding_specs, "net_0_kernel", ("embed", "mlp"))
    net_0_bias = safe_getattr(sharding_specs, "net_0_bias", ("mlp",))
    net_2_kernel = safe_getattr(sharding_specs, "net_2_kernel", ("mlp", "embed"))
    net_2_bias = safe_getattr(sharding_specs, "net_2_bias", ("embed",))
    self.net_0 = nnx.Linear(dim, inner_dim, rngs=rngs, use_bias=True, dtype=dtype, param_dtype=weights_dtype,
                            precision=precision, kernel_init=nnx.initializers.lecun_normal(),
                            kernel_metadata=param_metadata(sharding, "ffn_net_0_kernel", net_0_kernel),
                            bias_init=nnx.initializers.zeros,
                            bias_metadata=param_metadata(sharding, "ffn_net_0_bias", net_0_bias))
    self.act = get_activation(activation_fn)
    self.net_2 = nnx.Linear(inner_dim, dim_out, rngs=rngs, use_bias=True, dtype=dtype, param_dtype=weights_dtype,
                            precision=precision, kernel_init=nnx.initializers.lecun_normal(),
                            kernel_metadata=param_metadata(sharding, "ffn_net_2_kernel", net_2_kernel),
                            bias_init=nnx.initializers.zeros,
                            bias_metadata=param_metadata(sharding, "ffn_net_2_bias", net_2_bias))

  def __call__(self, hidden_states: Array) -> Array:
    hidden_states = self.net_0(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.net_2(hidden_states)
    return hidden_states


def _check_attention_inputs(query: Array, key: Array, value: Array) -> None:
  assert key.ndim == value.ndim, "k, v must have same rank."
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
  assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
  assert query.shape[-1] == key.shape[-1], "q, k depths must match."


_ATTENTION_KERNELS = ("dot_product", "flash")


def _padding_segment_ids(q_len: int, q_padded_len: int, kv_len: int, kv_padded_len: int, attention_mask: Array | None):
  q_ids = (jnp.arange(q_padded_len) < q_len).astype(jnp.int32)
  kv_ids = (jnp.arange(kv_padded_len) < kv_len).astype(jnp.int32)
  if attention_mask is not None:
    kv_ids = kv_ids * _pad_axis(attention_mask.astype(jnp.int32), 1, kv_padded_len)
    q_ids = jnp.broadcast_to(q_ids, (attention_mask.shape[0], q_padded_len))
  return tokamax_splash.SegmentIds(q=q_ids, kv=kv_ids)


def _flash_attention(query: Array, key: Array, value: Array, heads: int, dtype: jnp.dtype,
                     flash_block_sizes: BlockSizes | None, q_spec: P, kv_spec: P, mask_spec: P | None,
                     context_axis: str, attention_mask: Array | None, mask_padding_tokens: bool,
                     residual_checkpoint_name: str | None) -> Array:
  mesh = jax.sharding.get_abstract_mesh()
  num_context_shards = mesh.shape[context_axis] if context_axis in mesh.shape else 1
  query, key, value = (x if x.ndim == 4 else unflatten_heads(x, heads) for x in (query, key, value))
  q_len, kv_len = query.shape[2], key.shape[2]
  block_sizes = dataclasses.replace(_splash_block_sizes(flash_block_sizes, q_len, kv_len, dtype),
                                    residual_checkpoint_name=residual_checkpoint_name,
                                    use_base2_exp=False, use_experimental_scheduler=False)
  q_block = max(block_sizes.block_q, block_sizes.block_q_dkv or 0)
  kv_block = max(block_sizes.block_kv, block_sizes.block_kv_dkv or 0)
  query = _pad_axis(query, 2, num_context_shards)
  if kv_spec[2] is not None:
    key, value = (_pad_axis(x, 2, num_context_shards) for x in (key, value))
  if attention_mask is not None:
    batch, padded_kv_len = query.shape[0], key.shape[2]
    attention_mask = jnp.broadcast_to(attention_mask, (batch, attention_mask.shape[1]))[:, :padded_kv_len]
    attention_mask = _pad_axis(attention_mask, 1, padded_kv_len)

  def local_attention(q, k, v, mask=None):
    local_q_len, local_kv_len, head_dim = q.shape[2], k.shape[2], q.shape[3]
    q = _pad_axis(_pad_axis(q, 2, q_block), 3, 128)
    k = _pad_axis(_pad_axis(k, 2, kv_block), 3, 128)
    v = _pad_axis(_pad_axis(v, 2, kv_block), 3, 128)
    kernel = tokamax_splash.make_splash_mha(mask=tokamax_splash_mask.FullMask(_shape=(q.shape[2], k.shape[2])),
                                            config=block_sizes, q_seq_shards=1, save_residuals=False)
    segment_ids = None
    if mask_padding_tokens or mask is not None:
      segment_ids = _padding_segment_ids(local_q_len, q.shape[2], local_kv_len, k.shape[2], mask)
    out = jax.vmap(kernel, in_axes=(0, 0, 0, None if mask is None else 0))(q, k, v, segment_ids)
    return out[:, :, :local_q_len, :head_dim].astype(q.dtype)

  in_specs, args = (q_spec, kv_spec, kv_spec), (query, key, value)
  if attention_mask is not None:
    in_specs += (mask_spec if mask_spec is not None else P(kv_spec[0], kv_spec[2]),)
    args += (attention_mask,)
  out = jax.shard_map(local_attention, mesh=mesh, in_specs=in_specs, out_specs=q_spec, check_vma=False)(*args)
  out = jnp.transpose(out[:, :, :q_len, :], (0, 2, 1, 3))
  return out.reshape(*out.shape[:2], -1)


def _apply_attention(query: Array, key: Array, value: Array, heads: int, attention_kernel: str,
                     flash_min_seq_length: int, use_memory_efficient_attention: bool, scale: float, dtype: jnp.dtype,
                     q_spec: P, kv_spec: P, flash_block_sizes: BlockSizes, mask_padding_tokens: bool = True,
                     residual_checkpoint_name: str | None = None, attention_mask: Array = None,
                     mask_spec: P | None = None, out_spec: P | None = None, context_axis: str = CONTEXT):
  _check_attention_inputs(query, key, value)
  if attention_kernel not in _ATTENTION_KERNELS:
    raise ValueError(f"Unexpected attention kernel {attention_kernel=}.")
  seq_len_idx = 2 if query.ndim == 4 else 1
  use_flash = (attention_kernel == "flash" and not use_memory_efficient_attention
               and min(query.shape[seq_len_idx], key.shape[seq_len_idx]) >= flash_min_seq_length)
  if attention_mask is not None:
    if attention_mask.ndim != 2:
      raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
    attention_mask = attention_mask.astype(jnp.bool_)
  if use_flash:
    with jax.named_scope("kernel_flash"):
      out = _flash_attention(query, key * scale, value, heads, dtype, flash_block_sizes, q_spec, kv_spec,
                             mask_spec, context_axis, attention_mask, mask_padding_tokens, residual_checkpoint_name)
  else:
    with jax.named_scope("kernel_dot_product"):
      out = _dot_product_attention(query, key, value, heads, scale, dtype, use_memory_efficient_attention, attention_mask)
  return out if out_spec is None else jax.lax.with_sharding_constraint(out, out_spec)


def _activation_spec(sharding: Sharding | None, tensor_name: str, axes: AxisNames) -> P:
  if sharding is None:
    return P(*(None,) * len(axes))
  return sharding(tensor_name, axes, TensorType.ACTIVATION)


class AttentionOp(nnx.Module):
  def __init__(self, sharding: Sharding | None, attention_kernel: str, scale: float, heads: int, dim_head: int,
               use_memory_efficient_attention: bool = False, axis_names_q: AxisNames = (BATCH, HEAD, LENGTH, D_KV),
               axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV), flash_min_seq_length: int = 0,
               flash_block_sizes: BlockSizes = None, dtype: DType = jnp.float32, mask_padding_tokens: bool = True,
               residual_checkpoint_name: str | None = None, use_base2_exp: bool = False,
               use_experimental_scheduler: bool = False):
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.scale, self.heads, self.dim_head = scale, heads, dim_head
    self.attention_kernel = attention_kernel
    self.use_memory_efficient_attention = use_memory_efficient_attention
    self.axis_names_q, self.axis_names_kv = axis_names_q, axis_names_kv
    self.q_spec = _activation_spec(sharding, "attention_q", axis_names_q)
    self.kv_spec = _activation_spec(sharding, "attention_kv", axis_names_kv)
    self.mask_spec = _activation_spec(sharding, "attention_mask", (axis_names_kv[0], axis_names_kv[2]))
    self.out_spec = None
    self.context_axis = getattr(sharding, "context_axis", CONTEXT)
    self.flash_min_seq_length = flash_min_seq_length
    self.flash_block_sizes = flash_block_sizes
    self.dtype = dtype
    self.mask_padding_tokens = mask_padding_tokens
    self.residual_checkpoint_name = residual_checkpoint_name

  def apply_attention(self, query: Array, key: Array, value: Array, attention_mask: Array = None):
    return _apply_attention(query=query, key=key, value=value, heads=self.heads, attention_kernel=self.attention_kernel,
                            flash_min_seq_length=self.flash_min_seq_length,
                            use_memory_efficient_attention=self.use_memory_efficient_attention, scale=self.scale,
                            dtype=self.dtype, q_spec=self.q_spec, kv_spec=self.kv_spec,
                            flash_block_sizes=self.flash_block_sizes, mask_padding_tokens=self.mask_padding_tokens,
                            residual_checkpoint_name=self.residual_checkpoint_name, attention_mask=attention_mask,
                            mask_spec=self.mask_spec, out_spec=self.out_spec, context_axis=self.context_axis)


class WanAttention(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, query_dim: int, cross_attention_dim: Optional[int] = None,
               heads: int = 8, dim_head: int = 64, dropout: float = 0.0, eps: float = 1e-6,
               qk_norm: str = "rms_norm_across_heads", use_memory_efficient_attention: bool = False,
               attention_kernel: str = "flash", flash_min_seq_length: int = 0, flash_block_sizes: BlockSizes = None,
               sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               query_axis_names: AxisNames = (BATCH, LENGTH, HEAD), key_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
               value_axis_names: AxisNames = (BATCH, LENGTH, HEAD), out_axis_names: AxisNames = (BATCH, LENGTH, EMBED),
               precision: jax.lax.Precision = None, qkv_bias: bool = False, is_self_attention: bool = True,
               mask_padding_tokens: bool = True, residual_checkpoint_name: str | None = None,
               enable_jax_named_scopes: bool = False, added_kv_proj_dim: Optional[int] = None,
               image_seq_len: Optional[int] = None, attention_config: Optional[dict] = None):
    attention_config = {"use_base2_exp": False, "use_experimental_scheduler": False, "ulysses_shards": -1,
                        "ulysses_attention_chunks": 1, **(attention_config or {})}
    self.dim_head, self.heads, self.sharding = dim_head, heads, sharding
    self.inner_dim = dim_head * heads
    scale = dim_head**-0.5
    self.qk_norm = qk_norm
    self.query_axis_names, self.key_axis_names = query_axis_names, key_axis_names
    self.value_axis_names, self.out_axis_names = value_axis_names, out_axis_names
    self.enable_jax_named_scopes = enable_jax_named_scopes
    cross_attention_uses_local_kv = not is_self_attention and attention_kernel == "flash"
    if is_self_attention:
      axis_names_q = (BATCH, SELF_ATTN_HEAD, SELF_ATTN_Q_LENGTH, D_KV)
      axis_names_kv = (BATCH, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, D_KV)
    else:
      axis_names_q = (BATCH, CROSS_ATTN_HEAD, CROSS_ATTN_Q_LENGTH, D_KV)
      axis_names_kv = (BATCH, CROSS_ATTN_HEAD, None if cross_attention_uses_local_kv else CROSS_ATTN_KV_LENGTH, D_KV)
    self.added_kv_proj_dim, self.image_seq_len = added_kv_proj_dim, image_seq_len
    tpu_type = get_tpu_type()
    self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128
    self.attention_op = AttentionOp(sharding=sharding, attention_kernel=attention_kernel, scale=scale, heads=heads,
                                    dim_head=dim_head, use_memory_efficient_attention=use_memory_efficient_attention,
                                    axis_names_q=axis_names_q, axis_names_kv=axis_names_kv,
                                    flash_min_seq_length=flash_min_seq_length, flash_block_sizes=flash_block_sizes,
                                    dtype=dtype, mask_padding_tokens=mask_padding_tokens,
                                    residual_checkpoint_name=residual_checkpoint_name,
                                    use_base2_exp=attention_config["use_base2_exp"],
                                    use_experimental_scheduler=attention_config["use_experimental_scheduler"])
    kernel_axes = ("embed", "heads")
    self.query = nnx.Linear(rngs=rngs, in_features=self.inner_dim, out_features=self.inner_dim,
                            kernel_init=nnx.initializers.lecun_normal(),
                            kernel_metadata=param_metadata(sharding, "attn_q_kernel", kernel_axes), dtype=dtype,
                            param_dtype=weights_dtype, precision=precision, bias_init=nnx.initializers.zeros,
                            bias_metadata=param_metadata(sharding, "attn_q_bias", ("heads",)))
    self.key = nnx.Linear(rngs=rngs, in_features=self.inner_dim, out_features=self.inner_dim,
                          kernel_init=nnx.initializers.lecun_normal(),
                          kernel_metadata=param_metadata(sharding, "attn_k_kernel", kernel_axes), dtype=dtype,
                          param_dtype=weights_dtype, precision=precision, bias_init=nnx.initializers.zeros,
                          bias_metadata=param_metadata(sharding, "attn_k_bias", ("heads",)))
    self.value = nnx.Linear(rngs=rngs, in_features=self.inner_dim, out_features=self.inner_dim,
                            kernel_init=nnx.initializers.lecun_normal(),
                            kernel_metadata=param_metadata(sharding, "attn_v_kernel", kernel_axes), dtype=dtype,
                            param_dtype=weights_dtype, precision=precision, bias_init=nnx.initializers.zeros,
                            bias_metadata=param_metadata(sharding, "attn_v_bias", ("heads",)))
    self.proj_attn = nnx.Linear(rngs=rngs, in_features=self.inner_dim, out_features=self.inner_dim,
                                kernel_init=nnx.initializers.lecun_normal(),
                                kernel_metadata=param_metadata(sharding, "attn_out_kernel", ("heads", "embed")),
                                dtype=dtype, param_dtype=weights_dtype, precision=precision,
                                bias_init=nnx.initializers.zeros,
                                bias_metadata=param_metadata(sharding, "attn_out_bias", ("embed",)))
    self.drop_out = nnx.Dropout(dropout, deterministic=False)
    self.norm_q = nnx.data(None)
    self.norm_k = nnx.data(None)
    if qk_norm is not None:
      self.norm_q = nnx.RMSNorm(num_features=self.inner_dim, rngs=rngs, epsilon=eps, dtype=dtype,
                                scale_init=nnx.initializers.ones,
                                scale_metadata=param_metadata(sharding, "attn_q_norm_scale", ("norm",)),
                                param_dtype=weights_dtype)
      self.norm_k = nnx.RMSNorm(num_features=self.inner_dim, rngs=rngs, dtype=dtype, scale_init=nnx.initializers.ones,
                                scale_metadata=param_metadata(sharding, "attn_k_norm_scale", ("norm",)),
                                param_dtype=weights_dtype)
    self.add_k_proj = nnx.data(None)
    self.add_v_proj = nnx.data(None)
    self.norm_added_k = nnx.data(None)
    if self.added_kv_proj_dim is not None:
      self.add_k_proj = nnx.Linear(self.added_kv_proj_dim, self.inner_dim, rngs=rngs, dtype=dtype,
                                   param_dtype=weights_dtype, precision=precision, bias_init=nnx.initializers.zeros,
                                   bias_metadata=param_metadata(sharding, "attn_add_k_bias", ("embed",)))
      self.add_v_proj = nnx.Linear(self.added_kv_proj_dim, self.inner_dim, rngs=rngs, dtype=dtype,
                                   param_dtype=weights_dtype, precision=precision, bias_init=nnx.initializers.zeros,
                                   bias_metadata=param_metadata(sharding, "attn_add_v_bias", ("embed",)))
      self.norm_added_k = nnx.RMSNorm(num_features=self.inner_dim, rngs=rngs, epsilon=eps, dtype=dtype,
                                      param_dtype=weights_dtype, scale_init=nnx.initializers.ones,
                                      scale_metadata=param_metadata(sharding, "attn_add_k_norm_scale", ("norm",)))

  def _apply_rope(self, xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array) -> Tuple[jax.Array, jax.Array]:
    cos = jnp.real(freqs_cis).astype(xq.dtype)
    sin = jnp.imag(freqs_cis).astype(xq.dtype)
    xq_reshaped = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.reshape(*xk.shape[:-1], -1, 2)
    xq_0, xq_1 = xq_reshaped[..., 0], xq_reshaped[..., 1]
    xk_0, xk_1 = xk_reshaped[..., 0], xk_reshaped[..., 1]
    xq_out_0, xq_out_1 = xq_0 * cos - xq_1 * sin, xq_0 * sin + xq_1 * cos
    xk_out_0, xk_out_1 = xk_0 * cos - xk_1 * sin, xk_0 * sin + xk_1 * cos
    return jnp.stack([xq_out_0, xq_out_1], axis=-1).reshape(xq.shape), jnp.stack([xk_out_0, xk_out_1], axis=-1).reshape(xk.shape)

  def conditional_named_scope(self, name: str):
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(self, hidden_states: jax.Array, encoder_hidden_states: jax.Array = None,
               rotary_emb: Optional[jax.Array] = None, encoder_attention_mask: Optional[jax.Array] = None,
               deterministic: bool = True, rngs: nnx.Rngs = None,
               cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None) -> jax.Array:
    dtype = hidden_states.dtype
    is_self_attention = encoder_hidden_states is None
    if encoder_hidden_states is None:
      encoder_hidden_states = hidden_states
    is_i2v_cross_attention = self.added_kv_proj_dim is not None and not is_self_attention
    if not is_i2v_cross_attention:
      encoder_attention_mask = None
    if not is_i2v_cross_attention:
      with jax.named_scope("query_proj"):
        query_proj = self.query(hidden_states)
      if self.qk_norm:
        with self.conditional_named_scope("attn_q_norm"):
          query_proj = self.norm_q(query_proj)
      if not is_self_attention and cached_kv is not None and "text" in cached_kv:
        key_proj, value_proj = cached_kv["text"]
      else:
        with jax.named_scope("key_proj"):
          key_proj = self.key(encoder_hidden_states)
        with jax.named_scope("value_proj"):
          value_proj = self.value(encoder_hidden_states)
        if self.qk_norm:
          with self.conditional_named_scope("attn_k_norm"):
            key_proj = self.norm_k(key_proj)
      if rotary_emb is not None:
        with self.conditional_named_scope("attn_rope"):
          query_proj = unflatten_heads(query_proj, self.heads)
          key_proj = unflatten_heads(key_proj, self.heads)
          value_proj = unflatten_heads(value_proj, self.heads)
          query_proj, key_proj = self._apply_rope(query_proj, key_proj, rotary_emb)
      query_proj = checkpoint_name(query_proj, "query_proj")
      key_proj = checkpoint_name(key_proj, "key_proj")
      value_proj = checkpoint_name(value_proj, "value_proj")
      with jax.named_scope("apply_attention"):
        attn_output = self.attention_op.apply_attention(query_proj, key_proj, value_proj,
                                                        attention_mask=encoder_attention_mask)
    else:
      with self.conditional_named_scope("proj_query"):
        query_proj_raw = self.query(hidden_states)
      if self.added_kv_proj_dim is not None:
        alignment = self.alignment
        image_seq_len_actual = self.image_seq_len if self.image_seq_len is not None else 257
        padded_img_len = ((image_seq_len_actual + alignment - 1) // alignment) * alignment
        encoder_hidden_states_img = encoder_hidden_states[:, :padded_img_len, :]
        encoder_hidden_states_text = encoder_hidden_states[:, padded_img_len:, :]
        if encoder_attention_mask is not None:
          encoder_attention_mask_img = encoder_attention_mask[:, :padded_img_len]
        else:
          encoder_attention_mask_img = None
      else:
        encoder_hidden_states_img = None
        encoder_hidden_states_text = encoder_hidden_states
        encoder_attention_mask_img = None
      if self.qk_norm:
        with self.conditional_named_scope("attn_q_norm"):
          query_proj_text = self.norm_q(query_proj_raw)
      else:
        query_proj_text = query_proj_raw
      if cached_kv is not None and "text" in cached_kv:
        key_proj_text, value_proj_text = cached_kv["text"]
      else:
        with self.conditional_named_scope("proj_key"):
          key_proj_text = self.key(encoder_hidden_states_text)
        if self.qk_norm:
          with self.conditional_named_scope("attn_k_norm"):
            key_proj_text = self.norm_k(key_proj_text)
        with self.conditional_named_scope("proj_value"):
          value_proj_text = self.value(encoder_hidden_states_text)
      if encoder_hidden_states_img is not None:
        if cached_kv is not None and "image" in cached_kv:
          key_proj_img, value_proj_img = cached_kv["image"]
        else:
          with self.conditional_named_scope("add_proj_k"):
            key_proj_img = self.add_k_proj(encoder_hidden_states_img)
          with self.conditional_named_scope("norm_add_k"):
            key_proj_img = self.norm_added_k(key_proj_img)
          with self.conditional_named_scope("add_proj_v"):
            value_proj_img = self.add_v_proj(encoder_hidden_states_img)
        query_proj_img = query_proj_raw
        query_proj_text = checkpoint_name(query_proj_text, "query_proj")
        key_proj_text = checkpoint_name(key_proj_text, "key_proj_text")
        value_proj_text = checkpoint_name(value_proj_text, "value_proj_text")
        key_proj_img = checkpoint_name(key_proj_img, "key_proj_img")
        value_proj_img = checkpoint_name(value_proj_img, "value_proj_img")
        query_proj_img = checkpoint_name(query_proj_img, "query_proj_img")
        with self.conditional_named_scope("cross_attn_text_apply"):
          attn_output_text = self.attention_op.apply_attention(query_proj_text, key_proj_text, value_proj_text)
        with self.conditional_named_scope("cross_attn_img_apply"):
          attn_output_img = self.attention_op.apply_attention(query_proj_img, key_proj_img, value_proj_img,
                                                              attention_mask=encoder_attention_mask_img)
        attn_output = attn_output_text + attn_output_img
      else:
        query_proj_text = checkpoint_name(query_proj_text, "query_proj")
        key_proj_text = checkpoint_name(key_proj_text, "key_proj_text")
        value_proj_text = checkpoint_name(value_proj_text, "value_proj_text")
        with self.conditional_named_scope("cross_attn_text_apply"):
          attn_output = self.attention_op.apply_attention(query_proj_text, key_proj_text, value_proj_text)
    attn_output = attn_output.astype(dtype=dtype)
    attn_output = checkpoint_name(attn_output, "attn_output")
    with jax.named_scope("proj_attn"):
      hidden_states = self.proj_attn(attn_output)
      if self.drop_out.rate > 0:
        hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    return hidden_states

  def compute_kv(self, encoder_hidden_states: jax.Array, encoder_attention_mask: Optional[jax.Array] = None
                 ) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    is_i2v_cross_attention = self.added_kv_proj_dim is not None
    if not is_i2v_cross_attention:
      with jax.named_scope("key_proj"):
        key_proj = self.key(encoder_hidden_states)
      with jax.named_scope("value_proj"):
        value_proj = self.value(encoder_hidden_states)
      if self.qk_norm:
        with self.conditional_named_scope("attn_k_norm"):
          key_proj = self.norm_k(key_proj)
      return {"text": (key_proj, value_proj)}
    alignment = self.alignment
    image_seq_len_actual = self.image_seq_len if self.image_seq_len is not None else 257
    padded_img_len = ((image_seq_len_actual + alignment - 1) // alignment) * alignment
    if encoder_attention_mask is None:
      padded_img_len = image_seq_len_actual
    encoder_hidden_states_img = encoder_hidden_states[:, :padded_img_len, :]
    encoder_hidden_states_text = encoder_hidden_states[:, padded_img_len:, :]
    with self.conditional_named_scope("proj_key"):
      key_proj_text = self.key(encoder_hidden_states_text)
    if self.qk_norm:
      with self.conditional_named_scope("attn_k_norm"):
        key_proj_text = self.norm_k(key_proj_text)
    with self.conditional_named_scope("proj_value"):
      value_proj_text = self.value(encoder_hidden_states_text)
    if encoder_hidden_states_img is not None:
      with self.conditional_named_scope("add_proj_k"):
        key_proj_img = self.add_k_proj(encoder_hidden_states_img)
      with self.conditional_named_scope("norm_add_k"):
        key_proj_img = self.norm_added_k(key_proj_img)
      with self.conditional_named_scope("add_proj_v"):
        value_proj_img = self.add_v_proj(encoder_hidden_states_img)
      return {"text": (key_proj_text, value_proj_text), "image": (key_proj_img, value_proj_img)}
    return {"text": (key_proj_text, value_proj_text)}


class Timesteps(nnx.Module):
  def __init__(self, dim: int = 32, flip_sin_to_cos: bool = False, freq_shift: float = 1.0, scale: int = 1):
    self.dim, self.flip_sin_to_cos, self.freq_shift, self.scale = dim, flip_sin_to_cos, freq_shift, scale

  def __call__(self, timesteps):
    return get_sinusoidal_embeddings(timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos,
                                     freq_shift=self.freq_shift)


class WanImageEmbedding(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int, dtype: jnp.dtype,
               weights_dtype: jnp.dtype, precision: jax.lax.Precision, pos_embed_seq_len=None,
               alignment: Optional[int] = None, flash_min_seq_length: int = 4096, sharding: Sharding | None = None):
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=in_features, elementwise_affine=True, eps=1e-6)
    self.ff = SimpleFeedForward(rngs=rngs, dim=in_features, dim_out=out_features, mult=1, activation_fn="gelu",
                                dtype=dtype, weights_dtype=weights_dtype, precision=precision, sharding=sharding)
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=out_features, elementwise_affine=True, eps=1e-6)
    if alignment is None:
      tpu_type = get_tpu_type()
      self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128
    else:
      self.alignment = alignment
    self.flash_min_seq_length = flash_min_seq_length
    if pos_embed_seq_len is not None:
      self.pos_embed = nnx.Param(jnp.zeros((1, pos_embed_seq_len, in_features), dtype=dtype))
    else:
      self.pos_embed = nnx.data(None)

  def __call__(self, encoder_hidden_states_image: jax.Array) -> tuple[jax.Array, jax.Array]:
    hidden_states = encoder_hidden_states_image
    B, current_seq_len, _ = hidden_states.shape
    if self.pos_embed is not None:
      pe_len = self.pos_embed.value.shape[1]
      add_len = min(current_seq_len, pe_len)
      hidden_states = hidden_states.at[:, :add_len, :].add(self.pos_embed.value[:, :add_len, :])
      if current_seq_len > pe_len:
        max_logging.log(f"[WARN] Input seq_len {current_seq_len} > pos_embed len {pe_len}")
    hidden_states = self.norm1(hidden_states)
    hidden_states = self.ff(hidden_states)
    hidden_states = self.norm2(hidden_states)
    B, current_seq_len, D_out = hidden_states.shape
    use_flash_attn = current_seq_len >= self.flash_min_seq_length
    if use_flash_attn:
      num_blocks = (current_seq_len + self.alignment - 1) // self.alignment
      target_seq_len = num_blocks * self.alignment
    else:
      target_seq_len = current_seq_len
    attention_mask = jnp.ones((B, current_seq_len), dtype=jnp.int32)
    if current_seq_len < target_seq_len:
      padding_size = target_seq_len - current_seq_len
      padding = jnp.zeros((B, padding_size, D_out), dtype=hidden_states.dtype)
      hidden_states = jnp.concatenate([hidden_states, padding], axis=1)
      padding_mask = jnp.zeros((B, padding_size), dtype=jnp.int32)
      attention_mask = jnp.concatenate([attention_mask, padding_mask], axis=1)
    if not use_flash_attn:
      attention_mask = None
    return hidden_states, attention_mask


class PixArtAlphaTextProjection(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, in_features: int, hidden_size: int, out_features: int = None,
               act_fn: str = "gelu_tanh", dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None, sharding_specs: Optional[Any] = None,
               sharding: Sharding | None = None):
    linear_1_kernel = safe_getattr(sharding_specs, "emb_linear_1_kernel", ("embed", "mlp"))
    linear_1_bias = safe_getattr(sharding_specs, "emb_linear_1_bias", ("mlp",))
    linear_2_kernel = safe_getattr(sharding_specs, "emb_linear_2_kernel", ("mlp", "embed"))
    linear_2_bias = safe_getattr(sharding_specs, "emb_linear_2_bias", ("embed",))
    if out_features is None:
      out_features = hidden_size
    self.linear_1 = nnx.Linear(rngs=rngs, in_features=in_features, out_features=hidden_size, use_bias=True,
                               dtype=dtype, param_dtype=weights_dtype, precision=precision,
                               kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "text_embed_linear_1_kernel", linear_1_kernel),
                               bias_init=nnx.initializers.zeros,
                               bias_metadata=param_metadata(sharding, "text_embed_linear_1_bias", linear_1_bias))
    self.act_1 = get_activation(act_fn)
    self.linear_2 = nnx.Linear(rngs=rngs, in_features=hidden_size, out_features=out_features, use_bias=True,
                               dtype=dtype, param_dtype=weights_dtype, precision=precision,
                               kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "text_embed_linear_2_kernel", linear_2_kernel),
                               bias_init=nnx.initializers.zeros,
                               bias_metadata=param_metadata(sharding, "text_embed_linear_2_bias", linear_2_bias))

  def __call__(self, caption):
    hidden_states = self.linear_1(caption)
    hidden_states = self.act_1(hidden_states)
    return self.linear_2(hidden_states)


def get_frequencies(max_seq_len: int, theta: int, attention_head_dim: int):
  h_dim = w_dim = 2 * (attention_head_dim // 6)
  t_dim = attention_head_dim - h_dim - w_dim
  freqs = []
  for dim in [t_dim, h_dim, w_dim]:
    freqs.append(get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype=jnp.float32))
  freqs = jnp.concatenate(freqs, axis=1)
  t_size = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
  hw_size = attention_head_dim // 6
  cumulative_sizes = np.cumsum([t_size, hw_size, hw_size])
  return jnp.split(freqs, cumulative_sizes[:-1].tolist(), axis=1)


class WanRotaryPosEmbed(nnx.Module):
  def __init__(self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0):
    self.attention_head_dim, self.patch_size, self.max_seq_len, self.theta = attention_head_dim, patch_size, max_seq_len, theta

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    freqs_split = get_frequencies(self.max_seq_len, self.theta, self.attention_head_dim)
    freqs_f = jnp.expand_dims(jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))
    freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))
    freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))
    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    return jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))


class WanTimeTextImageEmbedding(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int,
               image_embed_dim: Optional[int] = None, pos_embed_seq_len: Optional[int] = None,
               dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None, flash_min_seq_length: int = 4096, sharding: Sharding | None = None):
    self.timesteps_proj = Timesteps(dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0)
    self.time_embedder = TimestepEmbedding(rngs=rngs, in_channels=time_freq_dim, time_embed_dim=dim, dtype=dtype,
                                           weights_dtype=weights_dtype, precision=precision, sharding=sharding)
    self.act_fn = get_activation("silu")
    self.time_proj = nnx.Linear(rngs=rngs, in_features=dim, out_features=time_proj_dim, dtype=jnp.float32,
                                param_dtype=weights_dtype, precision=precision, kernel_init=nnx.initializers.xavier_uniform(),
                                kernel_metadata=param_metadata(sharding, "time_proj_kernel", ("embed", "mlp")),
                                bias_init=nnx.initializers.zeros,
                                bias_metadata=param_metadata(sharding, "time_proj_bias", ("mlp",)))
    self.text_embedder = PixArtAlphaTextProjection(rngs=rngs, in_features=text_embed_dim, hidden_size=dim,
                                                  act_fn="gelu_tanh", sharding=sharding)
    self.image_embedder = nnx.data(None)
    if image_embed_dim is not None:
      self.image_embedder = WanImageEmbedding(rngs=rngs, in_features=image_embed_dim, out_features=dim,
                                              pos_embed_seq_len=pos_embed_seq_len, dtype=dtype,
                                              weights_dtype=weights_dtype, precision=precision,
                                              flash_min_seq_length=flash_min_seq_length, sharding=sharding)

  def __call__(self, timestep: jax.Array, encoder_hidden_states: jax.Array,
               encoder_hidden_states_image: Optional[jax.Array] = None, skip_embeddings: bool = False):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    with jax.named_scope("time_proj"):
      timestep_proj = self.time_proj(self.act_fn(temb))
    if not skip_embeddings:
      encoder_hidden_states = self.text_embedder(encoder_hidden_states)
      encoder_attention_mask = None
      if encoder_hidden_states_image is not None:
        encoder_hidden_states_image, encoder_attention_mask = self.image_embedder(encoder_hidden_states_image)
    else:
      encoder_attention_mask = None
      if encoder_hidden_states_image is not None and encoder_hidden_states_image.shape[-1] != encoder_hidden_states.shape[-1]:
        img_dim = encoder_hidden_states_image.shape[-1]
        text_dim = encoder_hidden_states.shape[-1]
        if img_dim < text_dim:
          pad_shape = (encoder_hidden_states_image.shape[0], encoder_hidden_states_image.shape[1], text_dim - img_dim)
          encoder_hidden_states_image = jnp.concatenate([encoder_hidden_states_image,
                                                         jnp.zeros(pad_shape, dtype=encoder_hidden_states_image.dtype)], axis=-1)
        else:
          pad_shape = (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], img_dim - text_dim)
          encoder_hidden_states = jnp.concatenate([encoder_hidden_states,
                                                   jnp.zeros(pad_shape, dtype=encoder_hidden_states.dtype)], axis=-1)
    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image, encoder_attention_mask


class ApproximateGELU(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim_in: int, dim_out: int, bias: bool, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None,
               sharding: Sharding | None = None):
    self.proj = nnx.Linear(rngs=rngs, in_features=dim_in, out_features=dim_out, use_bias=bias, dtype=dtype,
                           param_dtype=weights_dtype, precision=precision, kernel_init=nnx.initializers.xavier_uniform(),
                           kernel_metadata=param_metadata(sharding, "ffn_proj_in_kernel", ("embed", "mlp")),
                           bias_init=nnx.initializers.zeros,
                           bias_metadata=param_metadata(sharding, "ffn_proj_in_bias", ("mlp",)))

  def __call__(self, x: jax.Array) -> jax.Array:
    with jax.named_scope("gelu"):
      x = self.proj(x)
    return nnx.gelu(x)


class WanFeedForward(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, dim_out: Optional[int] = None, mult: int = 4,
               dropout: float = 0.0, activation_fn: str = "geglu", final_dropout: bool = False,
               inner_dim: int = None, bias: bool = True, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None,
               enable_jax_named_scopes: bool = False, sharding: Sharding | None = None):
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim
    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.act_fn = nnx.data(None)
    if activation_fn == "gelu-approximate":
      self.act_fn = ApproximateGELU(rngs=rngs, dim_in=dim, dim_out=inner_dim, bias=bias, dtype=dtype,
                                    weights_dtype=weights_dtype, precision=precision, sharding=sharding)
    else:
      raise NotImplementedError(f"{activation_fn} is not implemented.")
    self.drop_out = nnx.Dropout(dropout, deterministic=False)
    self.proj_out = nnx.Linear(rngs=rngs, in_features=inner_dim, out_features=dim_out, use_bias=bias, dtype=dtype,
                               param_dtype=weights_dtype, precision=precision, kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "ffn_proj_out_kernel", ("mlp", "embed")))

  def conditional_named_scope(self, name: str):
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(self, hidden_states: jax.Array, deterministic: bool = True, rngs: nnx.Rngs = None) -> jax.Array:
    hidden_states = self.act_fn(hidden_states)
    hidden_states = checkpoint_name(hidden_states, "ffn_activation")
    if self.drop_out.rate > 0:
      hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    with jax.named_scope("proj_out"):
      return self.proj_out(hidden_states)


class WanTransformerBlock(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, ffn_dim: int, num_heads: int, qk_norm: str = "rms_norm_across_heads",
               cross_attn_norm: bool = False, eps: float = 1e-6, added_kv_proj_dim: Optional[int] = None,
               image_seq_len: Optional[int] = None, flash_min_seq_length: int = 4096, flash_block_sizes: BlockSizes = None,
               sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None, attention: str = "dot_product", dropout: float = 0.0,
               mask_padding_tokens: bool = True, enable_jax_named_scopes: bool = False,
               attention_config: Optional[dict] = None):
    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.sharding = sharding
    attention_config = {"use_base2_exp": False, "use_experimental_scheduler": False, "ulysses_shards": -1,
                        "ulysses_attention_chunks": 1, **(attention_config or {})}
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.attn1 = WanAttention(rngs=rngs, query_dim=dim, heads=num_heads, dim_head=dim // num_heads, qk_norm=qk_norm,
                              eps=eps, flash_min_seq_length=flash_min_seq_length, flash_block_sizes=flash_block_sizes,
                              sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision,
                              attention_kernel=attention, dropout=dropout, is_self_attention=True,
                              mask_padding_tokens=mask_padding_tokens, residual_checkpoint_name="self_attn",
                              enable_jax_named_scopes=enable_jax_named_scopes, attention_config=attention_config)
    self.attn2 = WanAttention(rngs=rngs, query_dim=dim, heads=num_heads, dim_head=dim // num_heads, qk_norm=qk_norm,
                              eps=eps, added_kv_proj_dim=added_kv_proj_dim, image_seq_len=image_seq_len,
                              flash_min_seq_length=flash_min_seq_length, flash_block_sizes=flash_block_sizes,
                              sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision,
                              attention_kernel=attention, dropout=dropout, is_self_attention=False,
                              mask_padding_tokens=mask_padding_tokens, residual_checkpoint_name="cross_attn",
                              enable_jax_named_scopes=enable_jax_named_scopes, attention_config=attention_config)
    assert cross_attn_norm is True
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)
    self.ffn = WanFeedForward(rngs=rngs, dim=dim, inner_dim=ffn_dim, activation_fn="gelu-approximate", dtype=dtype,
                              weights_dtype=weights_dtype, precision=precision, dropout=dropout,
                              enable_jax_named_scopes=enable_jax_named_scopes, sharding=sharding)
    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(jax.random.normal(key, (1, 6, dim)) / dim**0.5)

  def conditional_named_scope(self, name: str):
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(self, hidden_states: jax.Array, encoder_hidden_states: jax.Array, temb: jax.Array,
               rotary_emb: jax.Array, deterministic: bool = True, rngs: nnx.Rngs = None,
               encoder_attention_mask: Optional[jax.Array] = None,
               cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None):
    with self.conditional_named_scope("transformer_block"):
      if temb.ndim == 4:
        adaln = jnp.expand_dims(self.adaln_scale_shift_table, 0)
        combined = adaln + temb.astype(jnp.float32)
        parts = jnp.split(combined, 6, axis=2)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (part.squeeze(2) for part in parts)
      else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = jnp.split(
            self.adaln_scale_shift_table + temb.astype(jnp.float32), 6, axis=1)
      hidden_states = constrain(self.sharding, hidden_states, "block_hidden", (BATCH, LENGTH, HEAD))
      hidden_states = checkpoint_name(hidden_states, "hidden_states")
      encoder_hidden_states = constrain(self.sharding, encoder_hidden_states, "block_encoder_hidden", (BATCH, LENGTH, D_KV))
      with self.conditional_named_scope("self_attn"):
        with self.conditional_named_scope("self_attn_norm"):
          norm_hidden_states = (self.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(hidden_states.dtype)
        with self.conditional_named_scope("self_attn_attn"):
          attn_output = self.attn1(hidden_states=norm_hidden_states, encoder_hidden_states=norm_hidden_states,
                                   rotary_emb=rotary_emb, deterministic=deterministic, rngs=rngs)
        with self.conditional_named_scope("self_attn_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(hidden_states.dtype)
      with self.conditional_named_scope("cross_attn"):
        with self.conditional_named_scope("cross_attn_norm"):
          norm_hidden_states = self.norm2(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
        with self.conditional_named_scope("cross_attn_attn"):
          attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                                   deterministic=deterministic, rngs=rngs, encoder_attention_mask=encoder_attention_mask,
                                   cached_kv=cached_kv)
        with self.conditional_named_scope("cross_attn_residual"):
          hidden_states = hidden_states + attn_output
      with self.conditional_named_scope("mlp"):
        with self.conditional_named_scope("mlp_norm"):
          norm_hidden_states = (self.norm3(hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa).astype(hidden_states.dtype)
        with self.conditional_named_scope("mlp_ffn"):
          ff_output = self.ffn(norm_hidden_states, deterministic=deterministic, rngs=rngs)
        with self.conditional_named_scope("mlp_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa).astype(hidden_states.dtype)
      return hidden_states

  def compute_kv(self, encoder_hidden_states: jax.Array, encoder_attention_mask: Optional[jax.Array] = None
                 ) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    return self.attn2.compute_kv(encoder_hidden_states, encoder_attention_mask)


class WanModel(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, model_type="t2v", patch_size: Tuple[int] = (1, 2, 2),
               num_attention_heads: int = 40, attention_head_dim: int = 128, in_channels: int = 16,
               out_channels: int = 16, text_dim: int = 4096, freq_dim: int = 256, ffn_dim: int = 13824,
               num_layers: int = 40, dropout: float = 0.0, cross_attn_norm: bool = True,
               qk_norm: Optional[str] = "rms_norm_across_heads", eps: float = 1e-6, image_dim: Optional[int] = None,
               added_kv_proj_dim: Optional[int] = None, rope_max_seq_len: int = 1024,
               pos_embed_seq_len: Optional[int] = None, image_seq_len: Optional[int] = None,
               flash_min_seq_length: int = 4096, flash_block_sizes: BlockSizes = None,
               sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None, attention: str = "dot_product", mask_padding_tokens: bool = True,
               scan_layers: bool = True, enable_jax_named_scopes: bool = False, attention_config: Optional[dict] = None):
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.in_channels = in_channels
    self.patch_size = tuple(patch_size)
    self.num_layers, self.scan_layers = num_layers, scan_layers
    self.enable_jax_named_scopes, self.sharding = enable_jax_named_scopes, sharding
    attention_config = {"use_base2_exp": False, "use_experimental_scheduler": False, "ulysses_shards": -1,
                        "ulysses_attention_chunks": 1, **(attention_config or {})}
    self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
    self.patch_embedding = nnx.Conv(in_channels, inner_dim, rngs=rngs, kernel_size=patch_size, strides=patch_size,
                                    dtype=dtype, param_dtype=weights_dtype, precision=precision,
                                    kernel_init=nnx.initializers.xavier_uniform(),
                                    kernel_metadata=param_metadata(sharding, "patch_embedding_kernel",
                                                                   (None, None, None, None, "conv_out")))
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs, dim=inner_dim, time_freq_dim=freq_dim, time_proj_dim=inner_dim * 6, text_embed_dim=text_dim,
        image_embed_dim=image_dim, pos_embed_seq_len=pos_embed_seq_len, flash_min_seq_length=flash_min_seq_length,
        sharding=sharding)

    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
    def init_block(rngs):
      return WanTransformerBlock(rngs=rngs, dim=inner_dim, ffn_dim=ffn_dim, num_heads=num_attention_heads,
                                 qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps,
                                 flash_min_seq_length=flash_min_seq_length, flash_block_sizes=flash_block_sizes,
                                 sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision,
                                 attention=attention, dropout=dropout, mask_padding_tokens=mask_padding_tokens,
                                 enable_jax_named_scopes=enable_jax_named_scopes, added_kv_proj_dim=added_kv_proj_dim,
                                 image_seq_len=image_seq_len, attention_config=attention_config)

    if scan_layers:
      self.blocks = init_block(rngs)
    else:
      blocks = []
      for _ in range(num_layers):
        block = WanTransformerBlock(rngs=rngs, dim=inner_dim, ffn_dim=ffn_dim, num_heads=num_attention_heads,
                                    qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps,
                                    added_kv_proj_dim=added_kv_proj_dim, image_seq_len=image_seq_len,
                                    flash_min_seq_length=flash_min_seq_length, flash_block_sizes=flash_block_sizes,
                                    sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision,
                                    attention=attention, dropout=dropout, mask_padding_tokens=mask_padding_tokens,
                                    enable_jax_named_scopes=enable_jax_named_scopes, attention_config=attention_config)
        blocks.append(block)
      self.blocks = nnx.data(blocks)
    self.norm_out = FP32LayerNorm(rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False)
    self.proj_out = nnx.Linear(rngs=rngs, in_features=inner_dim, out_features=out_channels * math.prod(patch_size),
                               dtype=dtype, param_dtype=weights_dtype, precision=precision,
                               kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "proj_out_kernel", ("embed", None)))
    key = rngs.params()
    self.scale_shift_table = nnx.Param(jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5,
                                      kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(),
                                                                        (None, None, "embed")))

  @staticmethod
  def train_tflops_per_sample(transformer_config: dict, latent_shape: tuple, text_len: int) -> float:
    d = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    f = transformer_config["ffn_dim"]
    p_t, p_h, p_w = transformer_config.get("patch_size", (1, 2, 2))
    _, _, frames, height, width = latent_shape
    s = (frames // p_t) * (height // p_h) * (width // p_w)
    per_token_layer = 12 * d * d + 4 * d * f + 4 * s * d + 4 * text_len * d
    return 3 * transformer_config["num_layers"] * per_token_layer * s / 1e12

  @staticmethod
  def text_embedding_shape(transformer_config: dict, batch: int, max_sequence_length: int) -> tuple:
    return (batch, max_sequence_length, transformer_config["text_dim"])

  def conditional_named_scope(self, name: str):
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def compute_kv_cache(self, encoder_hidden_states: jax.Array, encoder_hidden_states_image: Optional[jax.Array] = None,
                       timestep: Optional[jax.Array] = None
                       ) -> Tuple[Dict[str, Tuple[jax.Array, jax.Array]], Optional[jax.Array]]:
    if timestep is None:
      batch_size = encoder_hidden_states.shape[0]
      timestep = jnp.zeros((batch_size,), dtype=jnp.int32)
    with self.conditional_named_scope("condition_embedder"):
      _, _, encoder_hidden_states, encoder_hidden_states_image, encoder_attention_mask = self.condition_embedder(
          timestep, encoder_hidden_states, encoder_hidden_states_image)
    encoder_hidden_states, encoder_attention_mask = combine_conditioning(
        encoder_hidden_states, encoder_hidden_states_image, encoder_attention_mask)
    kv_cache = compute_layer_kv_cache(self.blocks, self.scan_layers, encoder_hidden_states, encoder_attention_mask)
    return kv_cache, encoder_attention_mask

  @jax.named_scope("WanModel")
  def __call__(self, hidden_states: jax.Array, timestep: jax.Array, encoder_hidden_states: jax.Array,
               encoder_hidden_states_image: Optional[jax.Array] = None, return_dict: bool = True,
               attention_kwargs: Optional[Dict[str, Any]] = None, deterministic: bool = True,
               rngs: Optional[nnx.Rngs] = None, skip_blocks: bool = False, cached_residual: Optional[jax.Array] = None,
               return_residual: bool = False, kv_cache: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
               rotary_emb: Optional[jax.Array] = None, encoder_attention_mask: Optional[jax.Array] = None
               ) -> Union[jax.Array, Tuple[jax.Array, jax.Array], Dict[str, jax.Array]]:
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    with self.conditional_named_scope("rotary_embedding"):
      if rotary_emb is None:
        rotary_emb = self.rope(hidden_states)
    with self.conditional_named_scope("patch_embedding"):
      hidden_states = self.patch_embedding(hidden_states)
      hidden_states = jax.lax.collapse(hidden_states, 1, -1)
    per_token_t = timestep.ndim == 2
    with self.conditional_named_scope("condition_embedder"):
      temb, timestep_proj, encoder_hidden_states_out, encoder_hidden_states_image_out, encoder_attention_mask_out = (
          embed_model_conditioning(self.condition_embedder, timestep, encoder_hidden_states,
                                   encoder_hidden_states_image, kv_cache is not None))
    if encoder_attention_mask is None:
      encoder_attention_mask = encoder_attention_mask_out
    encoder_hidden_states, encoder_attention_mask = combine_conditioning(
        encoder_hidden_states_out, encoder_hidden_states_image_out, encoder_attention_mask, extend_mask=kv_cache is None)
    encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)

    hidden_states_before_blocks = hidden_states
    if skip_blocks:
      if cached_residual is None:
        raise ValueError("cached_residual must be provided when skip_blocks is True")
      hidden_states = hidden_states + cached_residual
    else:
      hidden_states = run_transformer_blocks(
          self.blocks, self.scan_layers, self.num_layers, hidden_states, encoder_hidden_states, timestep_proj,
          rotary_emb, deterministic, rngs, encoder_attention_mask, kv_cache)
    residual_x = hidden_states - hidden_states_before_blocks
    if per_token_t:
      combined_head = jnp.expand_dims(self.scale_shift_table, 0) + jnp.expand_dims(temb, 2)
      shift, scale = jnp.split(combined_head, 2, axis=2)
      shift = shift.squeeze(2)
      scale = scale.squeeze(2)
    else:
      shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)
    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    with jax.named_scope("proj_out"):
      hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                                          p_t, p_h, p_w, -1)
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    hidden_states = hidden_states.reshape(batch_size, -1, num_frames, height, width)
    if return_residual:
      return hidden_states, residual_x
    return hidden_states
