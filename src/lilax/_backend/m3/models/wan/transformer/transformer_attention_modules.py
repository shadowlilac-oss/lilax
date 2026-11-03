"""Wan transformer attention modules and operations."""

import contextlib
import dataclasses
from typing import Dict, Optional, Tuple

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask

from lilax._backend.common import common_types
from lilax._backend.m3.core.sharding import Sharding, TensorType
from .transformer_attention_dot_product import _dot_product_attention
from .transformer_attention_flash import _pad_axis, _splash_block_sizes, unflatten_heads
from ..utils_wan import (BlockSizes, CONTEXT, CROSS_ATTN_HEAD, CROSS_ATTN_KV_LENGTH, CROSS_ATTN_Q_LENGTH,
                        SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, SELF_ATTN_Q_LENGTH, TpuType, get_tpu_type,
                        param_metadata, tokamax_splash)

Array = common_types.Array
DType = common_types.DType
AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
EMBED = common_types.EMBED


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
