"""GLM-5.3-Flash autoregressive language model primitives."""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any, Literal, cast
import typing

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.typing import DTypeLike
from lilax._backend.m3.configs import glm5_3_flash_config as glm5_config
from lilax._backend.m3.core import sharding as m3_sharding
import optax


def _make_lin(dt: DTypeLike, p_dt: DTypeLike, sharding, rngs: nnx.Rngs):
  """Construct a Linear layer factory with sharding metadata."""

  def _lin(in_features: int, out_features: int, name: str, axes: tuple[str | None, ...]):
    kernel_metadata = {"sharding": sharding(name, axes)} if sharding else {}
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=False,
        dtype=dt,
        param_dtype=p_dt,
        precision=jax.lax.Precision.HIGHEST,
        kernel_metadata=kernel_metadata,
        rngs=rngs,
    )

  return _lin


def l2norm(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
  """Normalize along an axis using epsilon inside the square root."""
  input_dtype = x.dtype
  acc_dtype = jnp.promote_types(input_dtype, jnp.float32)
  x_acc = x.astype(acc_dtype)
  norm = jnp.sqrt(jnp.sum(x_acc * x_acc, axis=axis, keepdims=True) + eps)
  out = x_acc / norm
  return out.astype(input_dtype)


class Glm5RMSNormGated(nnx.Module):
  """RMSNorm with sigmoid gating."""

  def __init__(
      self,
      hidden_size: int,
      eps: float,
      *,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.hidden_size = hidden_size
    self.eps = eps
    self.param_dtype = param_dtype
    self.sharding = sharding
    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=param_dtype), out_sharding=spec)

  def __call__(self, hidden_states: jax.Array, gate: jax.Array) -> jax.Array:
    input_dtype = hidden_states.dtype
    hidden_fp32 = hidden_states.astype(jnp.float32)
    gate_fp32 = gate.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_fp32), axis=-1, keepdims=True)
    normed = hidden_fp32 * jax.lax.rsqrt(variance + self.eps)
    weight = self.weight[...].astype(jnp.float32)
    out = (weight * normed) * jax.nn.sigmoid(gate_fp32)
    return out.astype(input_dtype)


class Glm5ShortConv1d(nnx.Module):
  """Causal depthwise convolution with SiLU activation."""

  def __init__(
      self,
      dim: int,
      kernel_size: int = 4,
      *,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs | None = None,
  ):
    if rngs is None:
      raise ValueError("Glm5ShortConv1d requires rngs for weight initialization.")
    self.dim = dim
    self.kernel_size = kernel_size
    self.param_dtype = param_dtype
    self.sharding = sharding
    spec = sharding("conv1d", ("heads", None, None)) if sharding else None
    bound = 1.0 / math.sqrt(kernel_size)
    weight = jax.random.uniform(
        rngs.params(), (dim, 1, kernel_size), dtype=jnp.float32, minval=-bound, maxval=bound
    ).astype(param_dtype)
    self.weight = nnx.Param(weight, out_sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    input_dtype = x.dtype
    batch, sequence, _ = x.shape
    kernel_size = self.kernel_size
    x_fp32 = x.astype(jnp.float32)
    padding = jnp.zeros((batch, kernel_size - 1, self.dim), dtype=jnp.float32)
    x_padded = jnp.concatenate([padding, x_fp32], axis=1)
    weight_fp32 = self.weight[...].astype(jnp.float32)
    out = sum(x_padded[:, index : index + sequence, :] * weight_fp32[:, 0, index] for index in range(kernel_size))
    return jax.nn.silu(out).astype(input_dtype)


class Glm5KdaAttention(nnx.Module):
  """Kimi Delta Attention for GLM-5.3-Flash."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sharding = sharding
    kda_impl = config.kda_implementation
    valid = {"xla", "xla_chunked", "mosaic"}
    if not isinstance(kda_impl, str) or kda_impl not in valid:
      raise ValueError(f"kda_implementation must be one of {valid}, got {kda_impl!r}.")
    if kda_impl in {"xla_chunked", "mosaic"} and config.linear_lower_bound is None:
      raise ValueError("linear_lower_bound cannot be None when kda_implementation is" f" {kda_impl!r}.")
    self.kda_implementation = kda_impl
    self.hidden_size = config.hidden_size
    self.num_heads = config.linear_num_key_heads
    self.head_dim = config.linear_key_head_dim
    self.qkv_dim = self.num_heads * self.head_dim
    self.dtype = getattr(jnp, config.dtype)
    self.param_dtype = getattr(jnp, config.weight_dtype)
    linear = _make_lin(self.dtype, self.param_dtype, sharding, rngs)

    def conv(dim):
      return Glm5ShortConv1d(
          dim=dim, kernel_size=config.linear_conv_kernel_dim, param_dtype=jnp.float32, sharding=sharding, rngs=rngs
      )

    hidden_size, qkv_dim = self.hidden_size, self.qkv_dim
    head_dim, num_heads = self.head_dim, self.num_heads
    self.q_proj = linear(hidden_size, qkv_dim, "q_proj", ("embed", "heads"))
    self.k_proj = linear(hidden_size, qkv_dim, "k_proj", ("embed", "heads"))
    self.v_proj = linear(hidden_size, qkv_dim, "v_proj", ("embed", "heads"))
    self.q_conv1d, self.k_conv1d, self.v_conv1d = conv(qkv_dim), conv(qkv_dim), conv(qkv_dim)
    self.f_a_proj = linear(hidden_size, head_dim, "f_a_proj", ("embed", None))
    self.f_b_proj = linear(head_dim, qkv_dim, "f_b_proj", (None, "heads"))
    a_spec = sharding("A_log", ("heads",)) if sharding else None
    dt_spec = sharding("dt_bias", ("heads",)) if sharding else None
    if config.linear_lower_bound is not None:
      a_log_init = jnp.zeros((num_heads,), dtype=jnp.float32)
    else:
      a_init = jax.random.uniform(rngs.params(), (num_heads,), dtype=jnp.float32, minval=1.0, maxval=16.0)
      a_log_init = jnp.log(a_init)
    self.A_log = nnx.Param(a_log_init, out_sharding=a_spec)
    log_dt = jax.random.uniform(
        rngs.params(), (qkv_dim,), dtype=jnp.float32, minval=math.log(1e-3), maxval=math.log(1e-1)
    )
    dt = jnp.maximum(jnp.exp(log_dt), 1e-4)
    dt_bias_init = (dt + jnp.log(-jnp.expm1(-dt))).astype(jnp.float32)
    self.dt_bias = nnx.Param(dt_bias_init, out_sharding=dt_spec)
    self.b_proj = linear(hidden_size, num_heads, "b_proj", ("embed", "heads"))
    self.g_a_proj = linear(hidden_size, head_dim, "g_a_proj", ("embed", None))
    self.g_b_proj = linear(head_dim, qkv_dim, "g_b_proj", (None, "heads"))
    self.o_norm = Glm5RMSNormGated(
        hidden_size=self.head_dim, eps=config.rms_norm_eps, param_dtype=self.param_dtype, sharding=sharding
    )
    self.o_proj = linear(qkv_dim, hidden_size, "o_proj", ("heads", "embed"))

  def __call__(self, hidden_states: jax.Array, decoder_segment_ids: jax.Array | None = None) -> jax.Array:
    batch, sequence, _ = hidden_states.shape
    mask = None
    if decoder_segment_ids is not None:
      mask = (decoder_segment_ids != 0)[..., None]
      hidden_states = jnp.where(mask, hidden_states, 0)
    query = self.q_conv1d(self.q_proj(hidden_states))
    key = self.k_conv1d(self.k_proj(hidden_states))
    value = self.v_conv1d(self.v_proj(hidden_states))
    forget = self.f_b_proj(self.f_a_proj(hidden_states))
    acc_dtype = jnp.promote_types(hidden_states.dtype, jnp.float32)
    beta_value = self.b_proj(hidden_states).astype(acc_dtype)
    beta = jax.nn.sigmoid(beta_value).astype(hidden_states.dtype)
    if mask is not None:
      query, key, value = (jnp.where(mask, item, 0) for item in (query, key, value))
      beta = jnp.where(mask, beta, 0)
    head_shape = (batch, sequence, self.num_heads, self.head_dim)

    def transpose_heads(item):
      return jnp.transpose(item.reshape(head_shape), (2, 0, 1, 3))

    query_heads, key_heads, value_heads, forget_heads = (
        transpose_heads(item) for item in (query, key, value, forget)
    )
    query_heads = l2norm(query_heads, axis=-1, eps=1e-6)
    key_heads = l2norm(key_heads, axis=-1, eps=1e-6)
    beta_heads = jnp.transpose(beta, (2, 0, 1))
    from tokamax._src.ops.kda import api as tokamax_kda
    from tokamax._src.ops.kda import pallas_mosaic_tpu as _tokamax_kda_mosaic

    impl = cast(tokamax_kda.Implementation, self.kda_implementation)

    def run_kda(query_arg, key_arg, value_arg, forget_arg, beta_arg, a_log_arg, dt_bias_arg):
      out, _ = tokamax_kda.kimi_delta_attention(
          query=query_arg,
          key=key_arg,
          value=value_arg,
          gate=forget_arg,
          beta=beta_arg,
          a_log=a_log_arg,
          delta_time_bias=dt_bias_arg,
          use_gate_in_kernel=True,
          lower_bound=self.config.linear_lower_bound,
          use_qk_l2norm=False,
          implementation=impl,
      )
      return out

    if self.sharding is not None and not jax.sharding.get_abstract_mesh().empty:
      head_spec = self.sharding("kda_head_act", ("heads", "batch", None, None), m3_sharding.TensorType.ACTIVATION)
      beta_spec = self.sharding("kda_beta_act", ("heads", "batch", None), m3_sharding.TensorType.ACTIVATION)
      param_spec = self.sharding("kda_head_param", ("heads",))
      attn_out = jax.shard_map(
          run_kda,
          in_specs=(head_spec, head_spec, head_spec, head_spec, beta_spec, param_spec, param_spec),
          out_specs=head_spec,
          check_vma=False,
      )(query_heads, key_heads, value_heads, forget_heads, beta_heads, self.A_log[...], self.dt_bias[...])
    else:
      attn_out = run_kda(query_heads, key_heads, value_heads, forget_heads, beta_heads, self.A_log[...], self.dt_bias[...])
    attn_out = jnp.transpose(attn_out, (1, 2, 0, 3))
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(head_shape)
    normed = self.o_norm(attn_out, gate)
    output = self.o_proj(normed.reshape(batch, sequence, self.qkv_dim))
    if mask is not None:
      output = jnp.where(mask, output, 0)
    if self.sharding:
      output = self.sharding.constrain(output, "kda_out", ("batch", "length", "embed"))
    return output


class Glm5RMSNorm(nnx.Module):
  """RMSNorm matching Glm5NextTextRMSNorm."""

  def __init__(self, d: int, *, eps: float, param_dtype: DTypeLike = jnp.float32,
               sharding: m3_sharding.Sharding | None = None):
    self.eps = eps
    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(jnp.ones((d,), dtype=param_dtype), out_sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    in_dt, x32 = x.dtype, x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normed = (x32 * jax.lax.rsqrt(variance + self.eps)).astype(in_dt)
    return self.weight[...].astype(in_dt) * normed


class Glm5LayerNorm(nnx.Module):
  """LayerNorm with weight and bias."""

  def __init__(self, d: int, *, eps: float, param_dtype: DTypeLike = jnp.float32,
               sharding: m3_sharding.Sharding | None = None):
    self.eps = eps
    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(jnp.ones((d,), dtype=param_dtype), out_sharding=spec)
    self.bias = nnx.Param(jnp.zeros((d,), dtype=param_dtype), out_sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    in_dt, x32 = x.dtype, x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    variance = jnp.var(x32, axis=-1, keepdims=True)
    normed = (x32 - mean) * jax.lax.rsqrt(variance + self.eps)
    weight = self.weight[...].astype(jnp.float32)
    bias = self.bias[...].astype(jnp.float32)
    return (normed * weight + bias).astype(in_dt)


class Glm5Indexer(nnx.Module):
  """DeepSeek Sparse Attention indexer with k-pool compression."""

  def __init__(self, config: glm5_config.Glm5Config, *, sharding: m3_sharding.Sharding | None = None, rngs: nnx.Rngs):
    self.config = config
    hidden_size, head_dim = config.hidden_size, config.index_head_dim
    num_heads, query_rank = config.index_n_heads, config.q_lora_rank
    self.head_dim = head_dim
    self.index_topk = config.index_topk
    self.index_kpool = config.index_kpool
    self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
    dtype = getattr(jnp, config.dtype)
    param_dtype = getattr(jnp, config.weight_dtype)
    linear = _make_lin(dtype, param_dtype, sharding, rngs)
    self.wq_b = linear(query_rank, num_heads * head_dim, "wq_b", ("q_lora", "heads"))
    self.wk = linear(hidden_size, head_dim, "wk", ("embed", None))
    self.k_norm = Glm5LayerNorm(head_dim, eps=1e-6, param_dtype=param_dtype, sharding=sharding)
    self.weights_proj = linear(hidden_size, num_heads, "weights_proj", ("embed", None))
    ape_spec = sharding("index_kpool_compress_ape", (None, None)) if sharding else None
    ape_init = jnp.zeros((self.index_kpool, head_dim), dtype=param_dtype)
    self.index_kpool_compress_ape = nnx.Param(ape_init, out_sharding=ape_spec)
    gate_spec = sharding("index_kpool_compress_gate", (None, "embed")) if sharding else None
    gate_init = jnp.ones((head_dim, hidden_size), dtype=param_dtype)
    self.index_kpool_compress_gate = nnx.Param(gate_init, out_sharding=gate_spec)

  def __call__(self, hidden_states: jax.Array, q_resid: jax.Array,
               attention_mask: jax.Array | None = None) -> jax.Array:
    hidden_states = jax.lax.stop_gradient(hidden_states)
    q_resid = jax.lax.stop_gradient(q_resid)
    batch, sequence = hidden_states.shape[:2]
    if attention_mask is None:
      attention_mask = jnp.ones((batch, sequence), dtype=jnp.bool_)
    else:
      attention_mask = attention_mask.astype(jnp.bool_)
    query = self.wq_b(q_resid).reshape(batch, sequence, -1, self.head_dim)
    key = self.k_norm(self.wk(hidden_states))
    gate_weight = self.index_kpool_compress_gate[...].astype(hidden_states.dtype)
    gate = jnp.einsum("bsd,hd->bsh", hidden_states, gate_weight, precision=jax.lax.Precision.HIGHEST)
    num_pools = (sequence + self.index_kpool - 1) // self.index_kpool
    any_valid = jnp.any(attention_mask, axis=-1)
    first_key = jnp.where(any_valid, jnp.argmax(attention_mask, axis=-1), sequence)
    total_pooled = num_pools * self.index_kpool
    offsets = jnp.arange(total_pooled).reshape(1, num_pools, self.index_kpool)
    raw_pool_indices = first_key[:, None, None] + offsets
    batch_index = jnp.arange(batch)[:, None, None]
    safe_index = jnp.clip(raw_pool_indices, 0, sequence - 1)
    group_keys = key[batch_index, safe_index]
    group_gate = gate[batch_index, safe_index]
    group_valid = attention_mask[batch_index, safe_index] & (raw_pool_indices < sequence)
    pool_valid = jnp.all(group_valid, axis=-1)
    pool_indices = jnp.where(group_valid, raw_pool_indices, -1)
    ape = self.index_kpool_compress_ape[...].astype(jnp.float32)
    gate_ape = group_gate.astype(jnp.float32) + ape[None, None]
    logits = jnp.where(group_valid[..., None], gate_ape, -jnp.inf)
    probs = jnp.nan_to_num(jax.nn.softmax(logits, axis=2), nan=0.0)
    pool_keys = jnp.sum(probs.astype(group_keys.dtype) * group_keys, axis=2)
    query_f32, pool_f32 = query.astype(jnp.float32), pool_keys.astype(jnp.float32)
    raw_scores = jnp.einsum("bshd,bpd->bshp", query_f32, pool_f32, precision=jax.lax.Precision.HIGHEST)
    scores = jax.nn.relu(raw_scores * (self.head_dim**-0.5))
    scale_weight = query.shape[2] ** -0.5
    weights = self.weights_proj(hidden_states).astype(jnp.float32) * scale_weight
    index_scores = jnp.einsum("bsh,bshp->bsp", weights, scores, precision=jax.lax.Precision.HIGHEST)
    sequence_index = jnp.arange(sequence)[None, :, None]
    pool_end = jnp.clip(raw_pool_indices[..., -1], 0, sequence - 1)[:, None, :]
    valid_candidates = (pool_end <= sequence_index) & pool_valid[:, None, :]
    index_scores = jnp.where(valid_candidates, index_scores, jnp.finfo(jnp.float32).min)
    selection_count = min(self.index_topk // self.index_kpool, num_pools)
    _, selected = jax.lax.top_k(index_scores, selection_count)
    selected_valid = valid_candidates[batch_index, sequence_index, selected]
    selected_indices = pool_indices[batch_index, selected]
    flat_length = selection_count * self.index_kpool
    valid_repeated = jnp.repeat(selected_valid[..., None], self.index_kpool, axis=-1)
    valid_mask = valid_repeated.reshape(batch, sequence, flat_length)
    topk_indices = jnp.where(valid_mask, selected_indices.reshape(batch, sequence, flat_length), -1)
    if self.index_kpool_always_select_tail:
      max_tail = self.index_kpool - 1
      if max_tail > 0:
        visible_count = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1)
        tail_count = visible_count % self.index_kpool
        tail_offsets = jnp.arange(max_tail)
        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets[None, None, :]
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & (tail_indices < sequence)
        tail_safe = jnp.clip(tail_indices, 0, sequence - 1)
        tail_visible = (tail_safe <= sequence_index) & attention_mask[batch_index, tail_safe]
        tail_masked = jnp.where(tail_valid & tail_visible, tail_indices, -1)
        topk_indices = jnp.concatenate([topk_indices, tail_masked], axis=-1)
      out_width = self.index_topk + max_tail
    else:
      out_width = self.index_topk
    width_difference = out_width - topk_indices.shape[-1]
    if width_difference > 0:
      topk_indices = jnp.pad(topk_indices, ((0, 0), (0, 0), (0, width_difference)), constant_values=-1)
    result = jnp.where(attention_mask[..., None], topk_indices[..., :out_width], -1)
    return result.astype(jnp.int32)


class Glm5DsaMlaAttention(nnx.Module):
  """NoPE multi-head latent attention with DSA indexing."""

  def __init__(self, config: glm5_config.Glm5Config, layer_idx: int, *, indexer_type: str,
               sharding: m3_sharding.Sharding | None = None, rngs: nnx.Rngs):
    if indexer_type != "full":
      raise ValueError(f"Shared indexer unsupported: {indexer_type!r}.")
    self.config, self.layer_idx = config, layer_idx
    self.indexer_type = indexer_type
    self.sharding = sharding
    if config.qk_rope_head_dim != 0 or config.rope_parameters is not None:
      raise ValueError(f"NoPE violation: {config.qk_rope_head_dim=}.")
    if config.indexer_loss_scaling_factor != 0.0:
      raise ValueError(f"indexer_loss_scaling_factor != 0 is not yet supported: {config.indexer_loss_scaling_factor}.")
    hidden_size, num_heads = config.hidden_size, config.num_attention_heads
    query_rank, kv_rank = config.q_lora_rank, config.kv_lora_rank
    query_dim, value_dim = config.qk_nope_head_dim, config.v_head_dim
    self.scaling = query_dim**-0.5
    self.num_heads = num_heads
    self.qk_nope_head_dim = query_dim
    self.v_head_dim = value_dim
    dtype, param_dtype = getattr(jnp, config.dtype), getattr(jnp, config.weight_dtype)
    linear = _make_lin(dtype, param_dtype, sharding, rngs)

    def norm(dim):
      return Glm5RMSNorm(dim, eps=config.rms_norm_eps, param_dtype=param_dtype, sharding=sharding)

    self.q_a_proj = linear(hidden_size, query_rank, "q_a_proj", ("embed", None))
    self.q_a_layernorm = norm(query_rank)
    self.q_b_proj = linear(query_rank, num_heads * query_dim, "q_b_proj", ("q_lora", "heads"))
    self.kv_a_proj_with_mqa = linear(hidden_size, kv_rank, "kv_a_proj_with_mqa", ("embed", None))
    self.kv_a_layernorm = norm(kv_rank)
    self.kv_b_proj = linear(kv_rank, num_heads * (query_dim + value_dim), "kv_b_proj", ("kv_lora", "heads"))
    self.o_proj = linear(num_heads * value_dim, hidden_size, "o_proj", ("heads", "embed"))
    self.indexer = Glm5Indexer(config, sharding=sharding, rngs=rngs) if config.use_indexer else None

  def _build_causal_padding_mask(self, attn_mask: jax.Array | None, b: int, s: int, dtype: DTypeLike) -> jax.Array:
    pos = jnp.arange(s)
    causal = pos[None, None, None, :] <= pos[None, None, :, None]
    if attn_mask is not None:
      visible = causal & attn_mask[:, None, None, :] & attn_mask[:, None, :, None]
    else:
      visible = jnp.broadcast_to(causal, (b, 1, s, s))
    return jnp.where(visible, 0.0, jnp.finfo(dtype).min).astype(dtype)

  def build_attention_mask_from_topk(self, topk_indices: jax.Array, query_states: jax.Array, kv_len: int) -> jax.Array:
    b, s = topk_indices.shape[:2]
    valid = (topk_indices >= 0) & (topk_indices < kv_len)
    target_idx = jnp.where(valid, topk_indices, kv_len)
    b_idx = jnp.arange(b)[:, None, None]
    s_idx = jnp.arange(s)[None, :, None]
    buf = jnp.zeros((b, s, kv_len + 1), dtype=jnp.bool_)
    mask = buf.at[b_idx, s_idx, target_idx].set(True)[:, None, :, :kv_len]
    return jnp.where(mask, 0.0, jnp.finfo(query_states.dtype).min).astype(query_states.dtype)

  def _call_dense_mask(self, q: jax.Array, k: jax.Array, v: jax.Array, mask_or_topk: jax.Array,
                       decoder_segment_ids: jax.Array | None, s: int | None = None) -> jax.Array:
    if mask_or_topk.ndim == 4:
      mask = mask_or_topk
    else:
      if s is None:
        s = q.shape[2]
      mask = self.build_attention_mask_from_topk(mask_or_topk, q, s)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2), precision=jax.lax.Precision.HIGHEST) * self.scaling + mask
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = jnp.matmul(weights, v, precision=jax.lax.Precision.HIGHEST)
    if decoder_segment_ids is not None:
      out = jnp.where((decoder_segment_ids != 0)[:, None, :, None], out, 0.0)
    return out

  @typing.overload
  def __call__(self, hidden_states: jax.Array, decoder_segment_ids: jax.Array | None = None, *,
               _force_indexer: bool = False, return_topk_indices: Literal[False] = False) -> jax.Array: ...

  @typing.overload
  def __call__(self, hidden_states: jax.Array, decoder_segment_ids: jax.Array | None = None, *,
               _force_indexer: bool = False, return_topk_indices: Literal[True]) -> tuple[jax.Array, jax.Array | None]: ...

  @typing.overload
  def __call__(self, hidden_states: jax.Array, decoder_segment_ids: jax.Array | None = None, *,
               _force_indexer: bool = False,
               return_topk_indices: bool = False) -> jax.Array | tuple[jax.Array, jax.Array | None]: ...

  def __call__(self, hidden_states: jax.Array, decoder_segment_ids: jax.Array | None = None, *,
               _force_indexer: bool = False,
               return_topk_indices: bool = False) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    b, s = hidden_states.shape[:2]
    attn_mask = decoder_segment_ids != 0 if decoder_segment_ids is not None else None
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_heads, qk_dim = self.num_heads, self.qk_nope_head_dim
    q = self.q_b_proj(q_resid).reshape(b, s, q_heads, qk_dim).transpose(0, 2, 1, 3)
    kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
    kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
    kv = kv.reshape(b, s, q_heads, qk_dim + self.v_head_dim).transpose(0, 2, 1, 3)
    k, v = jnp.split(kv, [qk_dim], axis=-1)
    if not _force_indexer and (
        not self.config.use_indexer or (s <= self.config.index_topk and self.config.index_kpool_always_select_tail)
    ):
      mask = self._build_causal_padding_mask(attn_mask, b, s, q.dtype)
      topk_idx = (self.indexer(hidden_states, q_resid, attention_mask=attn_mask)
                  if return_topk_indices and self.config.use_indexer and self.indexer is not None else None)
    else:
      if self.indexer is None:
        raise ValueError("indexer is None but indexer path was requested.")
      topk_idx = self.indexer(hidden_states, q_resid, attention_mask=attn_mask)
      mask = self.build_attention_mask_from_topk(topk_idx, q, s)
    if self.config.dsa_attention_implementation == "dense_mask":
      out = self._call_dense_mask(q, k, v, mask, decoder_segment_ids)
    else:
      raise ValueError("Unsupported Glm5Config.dsa_attention_implementation:"
                       f" {self.config.dsa_attention_implementation!r}.")
    out = self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, s, -1))
    if self.sharding:
      out = self.sharding.constrain(out, "mla_out", ("batch", "length", "embed"))
    if return_topk_indices:
      return out, topk_idx
    return out


class Glm5RouterBias(nnx.Variable):
  """Router bias state excluded from gradients."""


class Glm5MLP(nnx.Module):
  """Clamped SwiGLU dense MLP."""

  def __init__(self, config: glm5_config.Glm5Config, intermediate_size: int, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    self.config = config
    self.hidden_size = hidden_size = config.hidden_size
    self.intermediate_size = intermediate_size
    self.swiglu_limit = config.mlp_clamp_limit
    dtype, param_dtype = getattr(jnp, config.dtype), getattr(jnp, config.weight_dtype)
    linear = _make_lin(dtype, param_dtype, sharding, rngs)
    self.gate_proj = linear(hidden_size, intermediate_size, "gate_proj", ("embed", "mlp"))
    self.up_proj = linear(hidden_size, intermediate_size, "up_proj", ("embed", "mlp"))
    self.down_proj = linear(intermediate_size, hidden_size, "down_proj", ("mlp", "embed"))

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = jnp.minimum(self.gate_proj(x), self.swiglu_limit)
    up = jnp.clip(self.up_proj(x), -self.swiglu_limit, self.swiglu_limit)
    return self.down_proj(jax.nn.silu(gate) * up)


class Glm5TopkRouter(nnx.Module):
  """Top-k sigmoid router with non-Param correction bias."""

  def __init__(self, config: glm5_config.Glm5Config, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    if rngs is None:
      raise ValueError("rngs must be provided.")
    self.config = config
    if (scoring_func := config.scoring_func) != "sigmoid":
      raise ValueError(f"scoring_func must be 'sigmoid': {scoring_func!r}.")
    self.hidden_dim = hidden_size = config.hidden_size
    self.num_experts = num_experts = config.n_routed_experts
    self.top_k, self.num_group = config.num_experts_per_tok, config.n_group
    self.topk_group = config.topk_group
    self.routed_scaling_factor = config.routed_scaling_factor
    self.norm_topk_prob = config.norm_topk_prob
    param_dtype = getattr(jnp, config.router_param_dtype)
    weight_spec = sharding("router_weight", ("embed", None)) if sharding else None
    bias_spec = sharding("router_bias", (None,)) if sharding else None
    weight_init = nnx.initializers.normal(0.02)(rngs.params(), (hidden_size, num_experts), param_dtype)
    self.weight = nnx.Param(weight_init, out_sharding=weight_spec)
    self.e_score_correction_bias = Glm5RouterBias(jnp.zeros((num_experts,), dtype=param_dtype), sharding=bias_spec)

  def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    hidden_flat = hidden_states.reshape(-1, self.hidden_dim).astype(jnp.float32)
    router_logits = jnp.matmul(hidden_flat, self.weight[...].astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    scores = jax.nn.sigmoid(router_logits)
    bias = self.e_score_correction_bias[...].astype(jnp.float32)
    scores_for_choice = jax.lax.stop_gradient(scores + bias)
    experts_per_group = self.num_experts // self.num_group
    grouped = scores_for_choice.reshape(-1, self.num_group, experts_per_group)
    group_scores = jnp.sum(jax.lax.top_k(grouped, 2)[0], axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, self.topk_group)
    group_arange = jnp.arange(self.num_group)[None, :]
    group_mask = jnp.any(group_arange[:, :, None] == group_idx[:, None, :], axis=-1)
    score_mask = jnp.repeat(group_mask, experts_per_group, axis=-1)
    masked_scores = jnp.where(score_mask, scores_for_choice, jnp.finfo(jnp.float32).min)
    _, topk_indices = jax.lax.top_k(masked_scores, self.top_k)
    topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)
    if self.norm_topk_prob:
      denom = jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
      topk_weights = topk_weights / denom
    topk_weights = topk_weights * self.routed_scaling_factor
    return router_logits, topk_weights, topk_indices, scores


class Glm5Experts(nnx.Module):
  """Replicated expert weights and ragged-dot dispatch."""

  def __init__(self, config: glm5_config.Glm5Config, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    if rngs is None:
      raise ValueError("rngs must be provided.")
    self.config = config
    experts, hidden_size = config.n_routed_experts, config.hidden_size
    intermediate = config.moe_intermediate_size
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate
    self.num_experts, self.top_k = experts, config.num_experts_per_tok
    self.swiglu_limit = config.mlp_clamp_limit
    self.sharding = sharding
    self.ep_axis = sharding.ep_axis if isinstance(sharding, glm5_config.Glm5Sharding) else "expert"
    if isinstance(sharding, glm5_config.Glm5Sharding) and sharding.ep_size > 1:
      raise NotImplementedError(f"Glm5Experts expert parallelism ({self.ep_axis}={sharding.ep_size} >"
                                " 1) requires shard_map EP dispatch (follow-up F34).")
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
      raise NotImplementedError("Glm5Experts expert parallelism"
                                f" ({self.ep_axis}={mesh.shape[self.ep_axis]} > 1) requires shard_map"
                                " EP dispatch (follow-up F34).")
    param_dtype = getattr(jnp, config.weight_dtype)
    gate_up_spec = sharding("gate_up_proj", ("expert", "embed", None, "mlp")) if sharding else None
    down_spec = sharding("down_proj", ("expert", "mlp", "embed")) if sharding else None
    gate_up_init = nnx.initializers.normal(0.02)(rngs.params(), (experts, hidden_size, 2, intermediate), param_dtype)
    self.gate_up_proj = nnx.Param(gate_up_init, sharding=gate_up_spec)
    down_init = nnx.initializers.normal(0.02)(rngs.params(), (experts, intermediate, hidden_size), param_dtype)
    self.down_proj = nnx.Param(down_init, out_sharding=down_spec)

  def __call__(self, hidden_states: jax.Array, topk_indices: jax.Array, topk_weights: jax.Array) -> jax.Array:
    if isinstance(self.sharding, glm5_config.Glm5Sharding) and self.sharding.ep_size > 1:
      raise NotImplementedError(f"Glm5Experts.__call__ with {self.ep_axis} > 1 requires shard_map EP"
                                f" dispatch (follow-up F34); got ep_size={self.sharding.ep_size}.")
    mesh = None
    x_sharding = jax.typeof(hidden_states).sharding
    param_sharding = jax.typeof(self.gate_up_proj.value).sharding
    if isinstance(x_sharding, jax.sharding.NamedSharding):
      mesh = x_sharding.mesh
    elif isinstance(param_sharding, jax.sharding.NamedSharding):
      mesh = param_sharding.mesh
    elif not jax.sharding.get_abstract_mesh().empty:
      mesh = jax.sharding.get_abstract_mesh()
    if mesh is not None and not mesh.empty and self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
      raise NotImplementedError(f"Glm5Experts.__call__ with {self.ep_axis} > 1 requires shard_map"
                                " EP dispatch (follow-up F34); got mesh"
                                f" {self.ep_axis}={mesh.shape[self.ep_axis]}.")
    flat_idx = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1, 1).astype(jnp.float32)
    perm = jnp.argsort(flat_idx)
    sorted_hidden = hidden_states[perm // self.top_k]
    inverse_perm = jnp.argsort(perm)
    group_sizes = jnp.bincount(flat_idx, length=self.num_experts)
    dtype = hidden_states.dtype
    gate_up = self.gate_up_proj[...].astype(dtype)
    gate = jax.lax.ragged_dot(sorted_hidden, gate_up[:, :, 0, :], group_sizes=group_sizes,
                              preferred_element_type=dtype, precision=jax.lax.Precision.HIGHEST)
    up = jax.lax.ragged_dot(sorted_hidden, gate_up[:, :, 1, :], group_sizes=group_sizes,
                            preferred_element_type=dtype, precision=jax.lax.Precision.HIGHEST)
    gate = jnp.minimum(gate, self.swiglu_limit)
    up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
    activation = jax.nn.silu(gate) * up
    down_weight = self.down_proj[...].astype(dtype)
    sorted_out = jax.lax.ragged_dot(activation, down_weight, group_sizes=group_sizes,
                                    preferred_element_type=dtype, precision=jax.lax.Precision.HIGHEST)
    count = hidden_states.shape[0]
    weighted = (sorted_out[inverse_perm].astype(jnp.float32) * flat_weights).reshape(count, self.top_k, -1)
    return jnp.sum(weighted, axis=1).astype(dtype)


class Glm5MoE(nnx.Module):
  """Routed and shared experts."""

  def __init__(self, config: glm5_config.Glm5Config, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    self.config = config
    self.ep_axis = sharding.ep_axis if isinstance(sharding, glm5_config.Glm5Sharding) else "expert"
    if isinstance(sharding, glm5_config.Glm5Sharding) and sharding.ep_size > 1:
      raise NotImplementedError(f"Glm5MoE expert parallelism ({self.ep_axis}={sharding.ep_size} > 1)"
                                " requires all-to-all dispatch (planned for G8); single-host bringup"
                                f" requires {self.ep_axis}=1.")
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
      raise NotImplementedError("Glm5MoE expert parallelism"
                                f" ({self.ep_axis}={mesh.shape[self.ep_axis]} > 1) requires"
                                " all-to-all dispatch (planned for G8); single-host bringup requires"
                                f" {self.ep_axis}=1.")
    self.experts = Glm5Experts(config, rngs=rngs, sharding=sharding)
    self.gate = Glm5TopkRouter(config, rngs=rngs, sharding=sharding)
    self.shared_experts = Glm5MLP(config, config.moe_intermediate_size * config.n_shared_experts,
                                  rngs=rngs, sharding=sharding)

  def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    orig_shape = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, orig_shape[-1])
    router_logits, topk_weights, topk_indices, router_probs = self.gate(hidden_flat)
    expert_out = self.experts(hidden_flat, topk_indices, topk_weights).reshape(orig_shape)
    shared_out = self.shared_experts(hidden_states)
    topk_reshaped = topk_indices.reshape(orig_shape[:-1] + (self.config.num_experts_per_tok,))
    weights_reshaped = topk_weights.reshape(orig_shape[:-1] + (self.config.num_experts_per_tok,))
    probs_reshaped = router_probs.reshape(orig_shape[:-1] + (self.config.n_routed_experts,))
    return expert_out + shared_out, (router_logits, topk_reshaped, weights_reshaped, probs_reshaped)


def init_moe_lb_stats(num_experts: int) -> tuple[jax.Array, jax.Array, jax.Array]:
  zeros = jnp.zeros((num_experts,), dtype=jnp.float32)
  return zeros, zeros, jnp.zeros((), dtype=jnp.float32)


def accumulate_moe_lb_stats(
    stats: tuple[jax.Array, jax.Array, jax.Array],
    layer_gate: jax.Array,
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  tokens_sum, prob_sum, total_rows = stats
  gate_2d = layer_gate.reshape(-1, layer_gate.shape[-1]).astype(jnp.float32)
  routing_weights = jax.nn.softmax(gate_2d, axis=-1)
  _, selected_experts = jax.lax.top_k(routing_weights, top_k)
  selected_flat = selected_experts.reshape(-1)
  if attention_mask is None:
    counts = jnp.bincount(selected_flat, length=num_experts)
    probs = routing_weights
    rows = jnp.asarray(routing_weights.shape[0], dtype=jnp.float32)
  else:
    flat_mask = attention_mask.reshape(-1).astype(jnp.float32)
    mask_repeated = jnp.repeat(flat_mask, top_k)
    counts = jnp.bincount(selected_flat, weights=mask_repeated, length=num_experts)
    probs = routing_weights * flat_mask[:, None]
    rows = jnp.sum(flat_mask)
  return tokens_sum + counts.astype(jnp.float32), prob_sum + jnp.sum(probs, axis=0), total_rows + rows


def finalize_moe_lb_loss(stats: tuple[jax.Array, jax.Array, jax.Array], num_experts: int) -> jax.Array:
  tokens_sum, prob_sum, total_rows = stats
  safe_rows = jnp.maximum(total_rows, 1.0)
  tokens_per_expert, routing_prob = tokens_sum / safe_rows, prob_sum / safe_rows
  return jnp.sum(tokens_per_expert * routing_prob) * num_experts


def compute_moe_load_balancing_loss(
    gate_logits: Sequence[jax.Array],
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
  if not gate_logits:
    raise ValueError("gate_logits must be non-empty.")
  stats = init_moe_lb_stats(num_experts)
  for layer_gate in gate_logits:
    stats = accumulate_moe_lb_stats(stats, layer_gate, num_experts, top_k, attention_mask=attention_mask)
  return finalize_moe_lb_loss(stats, num_experts)


class Glm5UnweightedRMSNorm(nnx.Module):
  """Unweighted RMSNorm."""

  def __init__(self, eps: float):
    self.eps = eps

  def __call__(self, x: jax.Array) -> jax.Array:
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(variance + self.eps)).astype(x.dtype)


class Glm5HyperConnection(nnx.Module):
  """Manifold-constrained hyper-connections."""

  def __init__(self, config: glm5_config.Glm5Config, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    if rngs is None:
      raise ValueError("rngs must be provided to Glm5HyperConnection.")
    self.config, self.hc_mult = config, config.mhc_expansion_rate
    self.hc_sinkhorn_iters = config.mhc_sinkhorn_iters
    self.input_norm = Glm5UnweightedRMSNorm(eps=config.rms_norm_eps)
    param_dtype = getattr(jnp, config.weight_dtype)
    mix = (2 + self.hc_mult) * self.hc_mult
    fn_init = nnx.initializers.normal(0.02)(rngs.params(), (self.hc_mult * config.hidden_size, mix), param_dtype)
    self.fn = nnx.Param(fn_init, out_sharding=sharding("mhc_fn", ("embed", None)) if sharding else None)
    self.base = nnx.Param(jnp.zeros((mix,), dtype=param_dtype),
                          out_sharding=sharding("mhc_base", (None,)) if sharding else None)
    self.scale = nnx.Param(jnp.ones((3,), dtype=param_dtype),
                           out_sharding=sharding("mhc_scale", (None,)) if sharding else None)

  def __call__(self, hidden_streams: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    hc, eps = self.hc_mult, self.config.mhc_eps
    batch, sequence = hidden_streams.shape[:2]
    flat = self.input_norm(hidden_streams.reshape(batch, sequence, -1).astype(jnp.float32))
    fn_weight = self.fn[...].astype(jnp.float32)
    proj = jnp.matmul(flat, fn_weight, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)
    pre_weight, post_weight, comb_weight = jnp.split(proj, [hc, 2 * hc], axis=-1)
    pre_bias, post_bias, comb_bias = jnp.split(self.base[...].astype(jnp.float32), [hc, 2 * hc], axis=-1)
    pre_scale, post_scale, comb_scale = self.scale[...].astype(jnp.float32)
    pre = jax.nn.sigmoid(pre_weight * pre_scale + pre_bias) + eps
    post = 2.0 * jax.nn.sigmoid(post_weight * post_scale + post_bias)
    comb_logits = comb_weight.reshape(batch, sequence, hc, hc) * comb_scale + comb_bias.reshape(hc, hc)
    comb = jax.nn.softmax(comb_logits, axis=-1) + eps
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(self.hc_sinkhorn_iters - 1):
      for axis in (-1, -2):
        comb = comb / (jnp.sum(comb, axis=axis, keepdims=True) + eps)
    collapsed = jnp.sum(pre[..., None] * hidden_streams.astype(jnp.float32), axis=2).astype(hidden_streams.dtype)
    return post, comb, collapsed

  def apply_residual(self, residual: jax.Array, sublayer_out: jax.Array, post: jax.Array, comb: jax.Array) -> jax.Array:
    dtype = residual.dtype
    branch = post.astype(dtype)[..., None] * sublayer_out[:, :, None, :]
    return branch + jnp.matmul(jnp.swapaxes(comb.astype(dtype), -1, -2), residual,
                               precision=jax.lax.Precision.HIGHEST)


class Glm5HyperHead(nnx.Module):
  """Collapse hyper-connection streams by their mean."""

  def __call__(self, hidden_streams: jax.Array) -> jax.Array:
    return jnp.mean(hidden_streams, axis=2)


class Glm5DecoderLayer(nnx.Module):
  """GLM-5.3-Flash decoder layer."""

  def __init__(self, config: glm5_config.Glm5Config, layer_idx: int, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    if not 0 <= layer_idx < config.num_hidden_layers:
      raise ValueError(f"layer_idx {layer_idx} out of range [0, {config.num_hidden_layers}).")
    self.config, self.layer_idx = config, layer_idx
    if config.layer_types is None or config.indexer_types is None:
      raise ValueError("config.layer_types and config.indexer_types must not be None.")
    block_type = config.layer_types[layer_idx]
    self.block_type = block_type
    self.layer_type = block_type
    param_dtype = getattr(jnp, config.weight_dtype)
    if block_type == "linear_attention":
      self.self_attn = Glm5KdaAttention(config, sharding=sharding, rngs=rngs)
    elif block_type == "full_attention":
      self.self_attn = Glm5DsaMlaAttention(config, layer_idx, indexer_type=config.indexer_types[layer_idx],
                                           sharding=sharding, rngs=rngs)
    else:
      raise ValueError(f"Unknown block_type: {block_type}")
    self.is_moe = layer_idx >= config.first_k_dense_replace
    self.mlp = (Glm5MoE(config, sharding=sharding, rngs=rngs) if self.is_moe
                else Glm5MLP(config, config.intermediate_size, sharding=sharding, rngs=rngs))

    def norm() -> Glm5RMSNorm:
      return Glm5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, param_dtype=param_dtype, sharding=sharding)

    self.input_layernorm, self.post_attention_layernorm = norm(), norm()
    self.attn_hc = Glm5HyperConnection(config, sharding=sharding, rngs=rngs)
    self.ffn_hc = Glm5HyperConnection(config, sharding=sharding, rngs=rngs)

  def __call__(self, hidden_streams: jax.Array, decoder_segment_ids: jax.Array | None = None, *,
               return_aux: bool) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None,
                                          jax.Array | None, jax.Array | None]:
    post, comb, collapsed = self.attn_hc(hidden_streams)
    need_indexer = return_aux or self.config.enable_debug_stats
    if isinstance(self.self_attn, Glm5DsaMlaAttention):
      attn_out = self.self_attn(self.input_layernorm(collapsed), decoder_segment_ids=decoder_segment_ids,
                                return_topk_indices=need_indexer)
    else:
      attn_out = self.self_attn(self.input_layernorm(collapsed), decoder_segment_ids=decoder_segment_ids)
    indexer_topk_indices = None
    if isinstance(attn_out, tuple):
      attn_out, indexer_topk_indices = attn_out
    hidden_streams = self.attn_hc.apply_residual(hidden_streams, attn_out, post, comb)
    post, comb, collapsed = self.ffn_hc(hidden_streams)
    normed_mlp_in = self.post_attention_layernorm(collapsed)
    if isinstance(self.mlp, Glm5MoE):
      mlp_out, (router_logits, router_topk_indices, router_weights, router_probs) = self.mlp(normed_mlp_in)
    else:
      mlp_out = self.mlp(normed_mlp_in)
      router_logits = router_topk_indices = router_weights = router_probs = None
    return (self.ffn_hc.apply_residual(hidden_streams, mlp_out, post, comb), router_logits,
            router_topk_indices, indexer_topk_indices, router_weights if return_aux else None,
            router_probs if return_aux else None)


def layer_to_block_slot(layer_idx: int, first_k_dense_replace: int, num_hidden_layers: int) -> tuple[str, int, int | None]:
  if num_hidden_layers <= 0:
    raise ValueError(f"num_hidden_layers must be positive: {num_hidden_layers}")
  if layer_idx < 0 or layer_idx >= num_hidden_layers:
    raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_hidden_layers})")
  if first_k_dense_replace < 0 or first_k_dense_replace > num_hidden_layers:
    raise ValueError(f"first_k_dense_replace {first_k_dense_replace} invalid for" f" {num_hidden_layers} layers")
  if layer_idx < first_k_dense_replace:
    return ("prefix", layer_idx, None)
  super_block_layers = 4 * ((num_hidden_layers - first_k_dense_replace) // 4)
  relative = layer_idx - first_k_dense_replace
  if relative < super_block_layers:
    return ("super_block", relative // 4, relative % 4)
  return ("suffix", relative - super_block_layers, None)


class Glm5SuperBlock(nnx.Module):
  """Four-layer super-block of decoder layers."""

  def __init__(self, config: glm5_config.Glm5Config, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    self.config = config
    first = config.first_k_dense_replace
    self.slot_0 = Glm5DecoderLayer(config, first + 0, rngs=rngs, sharding=sharding)
    self.slot_1 = Glm5DecoderLayer(config, first + 1, rngs=rngs, sharding=sharding)
    self.slot_2 = Glm5DecoderLayer(config, first + 2, rngs=rngs, sharding=sharding)
    self.slot_3 = Glm5DecoderLayer(config, first + 3, rngs=rngs, sharding=sharding)

  def __call__(self, hidden_streams: jax.Array, decoder_segment_ids: jax.Array | None = None,
               remat_policy: Any = None, cur_lb: tuple[jax.Array, jax.Array, jax.Array] | None = None,
               return_router_logits: bool = True, *, return_aux: bool) -> tuple[
                   jax.Array, jax.Array | None, jax.Array | None, tuple[jax.Array | None, ...],
                   tuple[jax.Array, jax.Array, jax.Array] | None, jax.Array | None, jax.Array | None]:
    num_experts = self.config.n_routed_experts
    top_k = self.config.num_experts_per_tok
    attn_mask = (decoder_segment_ids != 0) if decoder_segment_ids is not None else None

    def call_slot(slot, streams, lb):
      streams, router, indices, indexer, weights, probs = slot(
          streams, decoder_segment_ids=decoder_segment_ids, return_aux=return_aux
      )
      if router is None or indices is None:
        raise ValueError("Super-block slots must all be MoE with non-None router logits and indices.")
      if lb is not None:
        lb = accumulate_moe_lb_stats(lb, router, num_experts=num_experts, top_k=top_k, attention_mask=attn_mask)
      return (streams, lb, router if return_router_logits else None,
              indices if (return_router_logits or return_aux) else None, indexer,
              weights if return_aux else None, probs if return_aux else None)

    slots = (self.slot_0, self.slot_1, self.slot_2, self.slot_3)
    router_logits, router_indices, router_weights, router_probs, indexer_topks = [], [], [], [], []
    for slot in slots:
      if remat_policy is not None:
        if not callable(remat_policy):
          raise ValueError(f"remat_policy must be None or callable: {remat_policy!r}.")
        exec_slot = jax.checkpoint(lambda streams, lb, sl=slot: call_slot(sl, streams, lb),
                                   policy=remat_policy, prevent_cse=True)
      else:
        def exec_slot(streams: jax.Array, lb: Any, sl: Any = slot) -> Any:
          return call_slot(sl, streams, lb)
      hidden_streams, cur_lb, router, indices, indexer, weights, probs = exec_slot(hidden_streams, cur_lb)
      if return_router_logits:
        router_logits.append(router)
      if return_router_logits or return_aux:
        router_indices.append(indices)
      indexer_topks.append(indexer)
      if return_aux:
        router_weights.append(weights)
        router_probs.append(probs)
    logits_stack = jnp.stack(router_logits, axis=0) if return_router_logits else None
    indices_stack = jnp.stack(router_indices, axis=0) if (return_router_logits or return_aux) else None
    weights_stack = jnp.stack(router_weights, axis=0) if return_aux else None
    probs_stack = jnp.stack(router_probs, axis=0) if return_aux else None
    return hidden_streams, logits_stack, indices_stack, tuple(indexer_topks), cur_lb, weights_stack, probs_stack


def create_stacked_super_block(config: glm5_config.Glm5Config, num_super_blocks: int, *, rngs: nnx.Rngs,
                               sharding: m3_sharding.Sharding | None = None, place_on_devices: bool) -> Glm5SuperBlock:
  if num_super_blocks <= 0:
    raise ValueError(f"num_super_blocks must be positive, got {num_super_blocks}")
  first, layer_types, indexer_types = config.first_k_dense_replace, config.layer_types, config.indexer_types
  if layer_types is None or indexer_types is None:
    raise ValueError(f"layer_types and indexer_types must not be None, got {layer_types=}, {indexer_types=}")
  if first + 4 * num_super_blocks > config.num_hidden_layers:
    raise ValueError(f"Required {first + 4 * num_super_blocks} layers for {num_super_blocks}"
                     f" super blocks with first_k_dense_replace={first} exceeds"
                     f" num_hidden_layers={config.num_hidden_layers}.")
  for block in range(num_super_blocks):
    for slot in range(4):
      idx = first + 4 * block + slot
      if layer_types[idx] != layer_types[first + slot] or indexer_types[idx] != indexer_types[first + slot]:
        raise ValueError(f"Non-periodic schedule at layer {idx}: {layer_types[idx]!r} / {indexer_types[idx]!r}.")

  def init_single(forked_rngs: nnx.Rngs) -> Glm5SuperBlock:
    return Glm5SuperBlock(config, rngs=forked_rngs, sharding=sharding)

  forked_rngs = rngs.split(num_super_blocks)
  stacked_block = nnx.vmap(init_single, in_axes=0, out_axes=0)(forked_rngs)
  graphdef, state = nnx.split(stacked_block)

  def update_var(path, var):
    if isinstance(var, nnx.Variable):
      orig_sh = var.get_metadata("out_sharding", None)
      if orig_sh is None:
        if sharding is not None:
          raise ValueError(f"Expected sharding metadata on stacked variable at {path}, got None.")
        if place_on_devices:
          raise ValueError("place_on_devices=True requires a non-empty NamedSharding mesh"
                           f" on stacked parameters at {path}.")
        return var
      orig_spec = orig_sh.spec if isinstance(orig_sh, jax.sharding.NamedSharding) else orig_sh
      if not isinstance(orig_spec, jax.sharding.PartitionSpec):
        raise ValueError(f"Unsupported sharding metadata type at {path}: {type(orig_sh)}")
      new_spec = jax.sharding.PartitionSpec(None, *orig_spec)
      raw_val = var.get_raw_value()
      if place_on_devices:
        raw_sh = raw_val.sharding if isinstance(raw_val, jax.Array) else None
        mesh = (orig_sh.mesh if isinstance(orig_sh, jax.sharding.NamedSharding)
                else (raw_sh.mesh if isinstance(raw_sh, jax.sharding.NamedSharding) else None))
        if mesh is None or mesh.empty:
          raise ValueError("place_on_devices=True requires a non-empty NamedSharding mesh"
                           f" on stacked parameters at {path}.")
        new_sh = jax.sharding.NamedSharding(mesh, new_spec)
        placed_val = jax.device_put(raw_val, new_sh)
        return var.replace(placed_val, out_sharding=new_spec)
      return var.replace(raw_val, out_sharding=new_spec)
    return var

  updated_state = jax.tree_util.tree_map_with_path(update_var, state, is_leaf=lambda value: isinstance(value, nnx.Variable))
  return nnx.merge(graphdef, updated_state)


def _slice_state(state: Any, i: int) -> Any:
  def slice_leaf(value: Any) -> Any:
    if isinstance(value, nnx.Variable):
      raw_val = value.get_raw_value()[i]
      orig_sh = (value.get_metadata("out_sharding", None) if hasattr(value, "get_metadata")
                 else (value.out_sharding if hasattr(value, "out_sharding") else None))
      if isinstance(orig_sh, jax.sharding.PartitionSpec) and len(orig_sh) > 0:
        return value.replace(raw_val, out_sharding=jax.sharding.PartitionSpec(*orig_sh[1:]))
      return value.replace(raw_val)
    return value[i]

  return jax.tree.map(slice_leaf, state, is_leaf=lambda value: isinstance(value, nnx.Variable))


def scan_super_blocks(super_block: Glm5SuperBlock, hidden_streams: jax.Array, *, scan_layers: bool,
                      decoder_segment_ids: jax.Array | None = None, remat_policy: Any = None,
                      return_router_logits: bool, lb_stats: tuple[jax.Array, jax.Array, jax.Array] | None = None,
                      return_aux: bool) -> tuple[
                          jax.Array, list[jax.Array], tuple[jax.Array, jax.Array, jax.Array], list[jax.Array],
                          list[tuple[jax.Array | None, ...]] | None, list[jax.Array], list[jax.Array]]:
  if super_block.slot_0.attn_hc.base.ndim != 2:
    raise ValueError("super_block is not stacked: scan_super_blocks requires a stacked"
                     " Glm5SuperBlock with a leading num_super_blocks axis, got"
                     f" slot_0.attn_hc.base.ndim={super_block.slot_0.attn_hc.base.ndim}")
  cfg = super_block.config
  cur_lb_init = lb_stats if lb_stats is not None else init_moe_lb_stats(cfg.n_routed_experts)
  graphdef, params, non_params = nnx.split(super_block, nnx.Param, ...)
  leaves = jax.tree.leaves(params)
  if not leaves:
    raise ValueError("super_block has no parameters.")
  num_super_blocks = leaves[0].shape[0]

  def step_fn(carry_in: Any, xs_i: tuple[Any, Any]) -> tuple[Any, Any]:
    params_i, non_params_i = xs_i
    carry, cur_lb = carry_in
    block = nnx.merge(graphdef, params_i, non_params_i)
    carry, logits, indices, indexer, cur_lb, weights, probs = block(
        carry, decoder_segment_ids=decoder_segment_ids, remat_policy=remat_policy, cur_lb=cur_lb,
        return_router_logits=return_router_logits, return_aux=return_aux
    )
    return (carry, cur_lb), (logits if return_router_logits else None,
                             indices if (return_router_logits or return_aux) else None,
                             indexer if (return_aux or cfg.enable_debug_stats) else None,
                             weights if return_aux else None, probs if return_aux else None)

  if remat_policy is not None and not callable(remat_policy):
    raise ValueError(f"remat_policy must be None or callable: {remat_policy!r}.")
  init_carry = (hidden_streams, cur_lb_init)
  if scan_layers:
    (final_streams, final_lb), (scanned_r, scanned_rt, scanned_idx, scanned_rw, scanned_rp) = jax.lax.scan(
        step_fn, init_carry, (params, non_params)
    )
    all_router_logits = ([scanned_r[block, slot] for block in range(num_super_blocks) for slot in range(4)]
                         if return_router_logits and scanned_r is not None else [])
    all_router_topk = ([scanned_rt[block, slot] for block in range(num_super_blocks) for slot in range(4)]
                       if (return_router_logits or return_aux) and scanned_rt is not None else [])
    all_idx = ([tuple(scanned_idx[slot][block] if scanned_idx[slot] is not None else None for slot in range(4))
                for block in range(num_super_blocks)] if scanned_idx is not None else None)
    all_routing_weights = ([scanned_rw[block, slot] for block in range(num_super_blocks) for slot in range(4)]
                           if return_aux and scanned_rw is not None else [])
    all_router_probs = ([scanned_rp[block, slot] for block in range(num_super_blocks) for slot in range(4)]
                        if return_aux and scanned_rp is not None else [])
  else:
    carry_state = init_carry
    all_router_logits, all_router_topk, all_idx_list, all_routing_weights, all_router_probs = [], [], [], [], []
    for idx in range(num_super_blocks):
      params_i, non_params_i = _slice_state(params, idx), _slice_state(non_params, idx)
      carry_state, (r_step, rt_step, idx_step, rw_step, rp_step) = step_fn(carry_state, (params_i, non_params_i))
      if return_router_logits and r_step is not None:
        for slot in range(4):
          all_router_logits.append(r_step[slot])
      if (return_router_logits or return_aux) and rt_step is not None:
        for slot in range(4):
          all_router_topk.append(rt_step[slot])
      if (return_aux or cfg.enable_debug_stats) and idx_step is not None:
        all_idx_list.append(idx_step)
      if return_aux and rw_step is not None:
        for slot in range(4):
          all_routing_weights.append(rw_step[slot])
      if return_aux and rp_step is not None:
        for slot in range(4):
          all_router_probs.append(rp_step[slot])
    final_streams, final_lb = carry_state
    all_idx = all_idx_list if all_idx_list else None
  return (final_streams, all_router_logits, final_lb, all_router_topk, all_idx,
          all_routing_weights, all_router_probs)


def apply_glm5_remat(model: Any, *, policy: Any) -> Any:
  if policy is not None and not callable(policy):
    raise ValueError(f"policy must be None or callable: {policy!r}.")
  model.remat_policy = policy
  return model


class Glm5Mtp(nnx.Module):
  """Multi-token prediction block for GLM-5.3-Flash."""

  def __init__(self, config: glm5_config.Glm5Config, *, sharding: m3_sharding.Sharding | None = None,
               rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    hidden_size, eps = config.hidden_size, config.rms_norm_eps
    dtype, param_dtype = getattr(jnp, config.dtype), getattr(jnp, config.weight_dtype)
    self.enorm = Glm5RMSNorm(hidden_size, eps=eps, param_dtype=param_dtype, sharding=sharding)
    self.hnorm = Glm5RMSNorm(hidden_size, eps=eps, param_dtype=param_dtype, sharding=sharding)
    eh_metadata = {"sharding": sharding("eh_proj", ("embed", None))} if sharding else {}
    self.eh_proj = nnx.Linear(2 * hidden_size, hidden_size, use_bias=False, dtype=dtype, param_dtype=param_dtype,
                              precision=jax.lax.Precision.HIGHEST, kernel_metadata=eh_metadata, rngs=rngs)
    self.input_layernorm = Glm5RMSNorm(hidden_size, eps=eps, param_dtype=param_dtype, sharding=sharding)
    self.self_attn = Glm5DsaMlaAttention(config, layer_idx=glm5_config.MTP_LAYER_INDEX,
                                         indexer_type=config.mtp_indexer_type, sharding=sharding, rngs=rngs)
    self.post_attention_layernorm = Glm5RMSNorm(hidden_size, eps=eps, param_dtype=param_dtype, sharding=sharding)
    self.mlp = Glm5MoE(config, sharding=sharding, rngs=rngs)
    self.shared_head_norm = Glm5RMSNorm(hidden_size, eps=eps, param_dtype=param_dtype, sharding=sharding)

  def __call__(self, hidden_states: jax.Array, next_token_ids: jax.Array, embed_tokens: nnx.Embed,
               lm_head: nnx.Linear | None, decoder_segment_ids: jax.Array | None) -> tuple[jax.Array, jax.Array, jax.Array]:
    emb = embed_tokens(next_token_ids)
    x = self.eh_proj(jnp.concatenate([self.enorm(emb), self.hnorm(hidden_states)], axis=-1))
    attn_out = self.self_attn(self.input_layernorm(x), decoder_segment_ids=decoder_segment_ids)
    if isinstance(attn_out, tuple):
      attn_out = attn_out[0]
    y = x + attn_out
    mlp_out, (router_logits, router_topk, _, _) = self.mlp(self.post_attention_layernorm(y))
    normed_z = self.shared_head_norm(y + mlp_out)
    if lm_head is None:
      return normed_z, router_logits, router_topk
    head_dtype = lm_head.kernel.dtype
    return lm_head(normed_z.astype(head_dtype)), router_logits, router_topk


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Glm5Output:
  logits: jax.Array | None
  hidden_states: jax.Array
  aux: dict[str, Any]


class Glm5Embedding(nnx.Embed):
  """Token embedding module for GLM-5.3-Flash."""

  def __init__(self, config: glm5_config.Glm5Config, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None):
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    super().__init__(num_embeddings=config.vocab_size, features=config.hidden_size, dtype=getattr(jnp, config.dtype),
                     param_dtype=getattr(jnp, config.weight_dtype),
                     embedding_metadata={"sharding": sharding("embed", ("vocab", "embed"))} if sharding else {},
                     rngs=rngs)


def _validate_generic_maxtext_kwargs(kwargs: dict[str, Any]) -> None:
  for key, value in kwargs.items():
    if key == "enable_dropout":
      if value:
        raise ValueError(f"Dropout is not supported in GLM-5.3-Flash, but enable_dropout={value!r}.")
    elif key in ("decoder_target_tokens", "decoder_target_mask"):
      if value is not None:
        raise ValueError(f"{key} is not supported in GLM-5.3-Flash forward pass, got {value!r}.")
    elif key.startswith("encoder_"):
      if value is not None:
        raise ValueError(f"Encoder inputs are not supported in decoder-only GLM-5.3-Flash: {key}={value!r}.")
    else:
      raise TypeError(f"Glm5Model received unexpected keyword argument: {key!r}")


class Glm5Model(nnx.Module):
  """GLM-5.3-Flash autoregressive language model with hybrid layers."""

  def __init__(self, config: glm5_config.Glm5Config, mesh: Mesh | None = None, *, rngs: nnx.Rngs,
               sharding: m3_sharding.Sharding | None = None, place_on_devices: bool):
    if not isinstance(place_on_devices, bool):
      raise TypeError("place_on_devices must be a bool, got" f" {type(place_on_devices).__name__}")
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    self.config = glm5_config.Glm5Config.from_maxtext_config(config)
    self.mesh = mesh
    if sharding is None and mesh is not None:
      sharding = glm5_config.Glm5Sharding.from_mesh(mesh)
    self.sharding = sharding
    self.remat_policy: Any = None
    cfg = self.config
    if cfg.first_k_dense_replace < 0 or (cfg.first_k_dense_replace == 0 and cfg.num_hidden_layers > 1):
      raise ValueError("first_k_dense_replace must be >= 1 unless num_hidden_layers == 1.")
    dt, p_dt = getattr(jnp, cfg.dtype), getattr(jnp, cfg.weight_dtype)
    self.embed_tokens = Glm5Embedding(cfg, rngs=rngs, sharding=sharding)
    num_prefix = min(cfg.num_hidden_layers, cfg.first_k_dense_replace)
    self.prefix_layers = nnx.List([Glm5DecoderLayer(cfg, i, rngs=rngs, sharding=sharding) for i in range(num_prefix)])
    rem = cfg.num_hidden_layers - num_prefix
    num_super_blocks, num_suffix = rem // 4, rem % 4
    self.super_block = (create_stacked_super_block(cfg, num_super_blocks, rngs=rngs, sharding=sharding,
                                                   place_on_devices=place_on_devices)
                        if num_super_blocks > 0 else None)
    suffix_start = num_prefix + 4 * num_super_blocks
    self.suffix_layers = nnx.List(
        [Glm5DecoderLayer(cfg, suffix_start + i, rngs=rngs, sharding=sharding) for i in range(num_suffix)]
    )
    self.hc_head = Glm5HyperHead()
    self.norm = Glm5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=p_dt, sharding=sharding)
    lm_meta = {"sharding": sharding("lm_head", ("embed", "vocab"))} if sharding else {}
    self.lm_head = nnx.Linear(cfg.hidden_size, cfg.vocab_size, use_bias=False, dtype=dt, param_dtype=p_dt,
                              precision=jax.lax.Precision.HIGHEST, kernel_metadata=lm_meta, rngs=rngs)
    self.mtp = Glm5Mtp(cfg, sharding=sharding, rngs=rngs) if cfg.num_nextn_predict_layers > 0 else None

  def forward_with_aux(self, decoder_input_tokens: jax.Array, decoder_positions: jax.Array | None = None,
                       decoder_segment_ids: jax.Array | None = None, *, compute_logits: bool = True,
                       return_aux: bool = False, **kwargs: Any) -> Glm5Output:
    _validate_generic_maxtext_kwargs(kwargs)
    del decoder_positions
    x = self.embed_tokens(decoder_input_tokens)
    hidden_streams = jnp.repeat(x[:, :, None, :], self.config.mhc_expansion_rate, axis=2)
    if self.sharding is not None:
      hidden_streams = self.sharding.constrain(hidden_streams, "hidden_streams", ("batch", "length", None, "embed"))
    cfg = self.config

    def _sync_cfg(module):
      if module.config.enable_debug_stats != cfg.enable_debug_stats:
        module.config = dataclasses.replace(module.config, enable_debug_stats=cfg.enable_debug_stats)

    for layer in self.prefix_layers:
      _sync_cfg(layer)
    if self.super_block is not None:
      _sync_cfg(self.super_block)
      for slot in range(4):
        _sync_cfg(getattr(self.super_block, f"slot_{slot}"))
    for layer in self.suffix_layers:
      _sync_cfg(layer)

    attention_mask = (decoder_segment_ids != 0) if decoder_segment_ids is not None else None
    keep_debug = bool(cfg.enable_debug_stats)
    keep_router_logits = keep_debug
    router_logits_list: list[jax.Array] = []
    router_topk_list: list[jax.Array] = []
    routing_weights_list: list[jax.Array] = []
    router_probs_list: list[jax.Array] = []
    dsa_topk_indices_list: list[jax.Array] = []
    layer_aux_list: list[dict[str, Any]] = []
    lb_stats = init_moe_lb_stats(cfg.n_routed_experts)
    layer_0_act: jax.Array | None = None

    def _run_layers(layers: Sequence[Glm5DecoderLayer], streams: jax.Array,
                    cur_lb: tuple[jax.Array, jax.Array, jax.Array]):
      nonlocal layer_0_act
      for layer in layers:
        if self.remat_policy is not None:
          fn = jax.checkpoint(lambda s, lyr=layer: lyr(s, decoder_segment_ids=decoder_segment_ids,
                                                       return_aux=return_aux),
                              policy=self.remat_policy, prevent_cse=True)
          streams, r_logits, r_topk, idx_topk, r_weights, r_probs = fn(streams)
        else:
          streams, r_logits, r_topk, idx_topk, r_weights, r_probs = layer(
              streams, decoder_segment_ids=decoder_segment_ids, return_aux=return_aux)
        if idx_topk is not None and cfg.use_indexer:
          dsa_topk_indices_list.append(idx_topk)
        if layer_0_act is None:
          layer_0_act = streams
        if r_logits is not None:
          cur_lb = accumulate_moe_lb_stats(cur_lb, r_logits, num_experts=cfg.n_routed_experts,
                                            top_k=cfg.num_experts_per_tok, attention_mask=attention_mask)
          if keep_router_logits:
            router_logits_list.append(r_logits)
          if (keep_router_logits or return_aux) and r_topk is not None:
            router_topk_list.append(r_topk)
          if return_aux and r_weights is not None:
            routing_weights_list.append(r_weights)
          if return_aux and r_probs is not None:
            router_probs_list.append(r_probs)
        if return_aux:
          layer_dict: dict[str, Any] = {}
          if r_topk is not None:
            layer_dict["router_indices"] = r_topk
            layer_dict["routing_weights"] = r_weights
            layer_dict["router_probs"] = r_probs
          if layer.layer_type == "full_attention" and cfg.use_indexer:
            if idx_topk is None:
              raise RuntimeError(f"Layer {layer.layer_idx} has full_attention but idx is None.")
            layer_dict["selected_block_indices"] = idx_topk
          layer_aux_list.append(layer_dict)
      return streams, cur_lb

    if self.prefix_layers:
      hidden_streams, lb_stats = _run_layers(self.prefix_layers, hidden_streams, lb_stats)
    if self.super_block is not None:
      layer_types = cfg.layer_types
      if layer_types is None:
        raise ValueError("cfg.layer_types must not be None when super_block is present.")
      hidden_streams, sb_r, lb_stats, sb_rt, sb_idx, sb_rw, sb_rp = scan_super_blocks(
          self.super_block, hidden_streams, decoder_segment_ids=decoder_segment_ids, scan_layers=cfg.scan_layers,
          remat_policy=self.remat_policy, return_router_logits=keep_router_logits, lb_stats=lb_stats,
          return_aux=return_aux)
      if sb_idx is not None and cfg.use_indexer:
        for block_idx in sb_idx:
          if block_idx is None:
            continue
          for slot_idx in block_idx:
            if slot_idx is not None:
              dsa_topk_indices_list.append(slot_idx)
      if keep_router_logits:
        router_logits_list.extend(sb_r)
      if keep_router_logits or return_aux:
        router_topk_list.extend(sb_rt)
      if return_aux:
        routing_weights_list.extend(sb_rw)
        router_probs_list.extend(sb_rp)
        num_sb = len(sb_idx) if sb_idx is not None else (len(sb_rt) // 4)
        for flat_slot_idx in range(4 * num_sb):
          block, slot = divmod(flat_slot_idx, 4)
          slot_dict: dict[str, Any] = {"router_indices": sb_rt[flat_slot_idx],
                                      "routing_weights": sb_rw[flat_slot_idx], "router_probs": sb_rp[flat_slot_idx]}
          if layer_types[cfg.first_k_dense_replace + slot] == "full_attention" and cfg.use_indexer:
            if sb_idx is None or sb_idx[block] is None or sb_idx[block][slot] is None:
              raise RuntimeError(f"Slot {slot} has full_attention with use_indexer=True but sb_idx[{block}][{slot}] is None.")
            slot_dict["selected_block_indices"] = sb_idx[block][slot]
          layer_aux_list.append(slot_dict)
    if self.suffix_layers:
      hidden_streams, lb_stats = _run_layers(self.suffix_layers, hidden_streams, lb_stats)
    last_block_act = hidden_streams
    normed = self.norm(self.hc_head(hidden_streams))
    logits = self.lm_head(normed) if compute_logits else None
    num_moe_layers = max(0, cfg.num_hidden_layers - cfg.first_k_dense_replace)
    if keep_router_logits and len(router_logits_list) != num_moe_layers:
      raise ValueError(f"Expected {num_moe_layers} MoE router_logits entries, got {len(router_logits_list)}.")
    moe_lb_loss = (jnp.zeros((), dtype=jnp.float32) if num_moe_layers == 0
                   else finalize_moe_lb_loss(lb_stats, cfg.n_routed_experts))
    aux: dict[str, Any] = {"router_logits": tuple(router_logits_list) if cfg.enable_debug_stats else (),
                           "moe_lb_loss": moe_lb_loss, "lb_stats": lb_stats}
    if return_aux:
      aux["layer_aux"] = tuple(layer_aux_list)
      if router_topk_list:
        aux["router_indices"] = jnp.stack(router_topk_list, axis=0)
      if routing_weights_list:
        aux["routing_weights"] = jnp.stack(routing_weights_list, axis=0)
      if router_probs_list:
        aux["router_probs"] = jnp.stack(router_probs_list, axis=0)
      aux["selected_block_indices"] = (jnp.stack(dsa_topk_indices_list, axis=0)
                                       if cfg.use_indexer and dsa_topk_indices_list else None)
    if keep_debug:
      if layer_0_act is None:
        raise RuntimeError("enable_debug_stats=True requires layer 0 activation, but layer_0_act was not captured.")
      l0_f32, last_f32 = layer_0_act.astype(jnp.float32), last_block_act.astype(jnp.float32)
      aux["debug_stats"] = {"first_block_mean": jnp.mean(l0_f32), "first_block_var": jnp.var(l0_f32),
                            "last_block_mean": jnp.mean(last_f32), "last_block_var": jnp.var(last_f32),
                            "hidden_norm_mean": jnp.mean(jnp.linalg.norm(normed, axis=-1)),
                            "logits_max": jnp.max(logits) if logits is not None else jnp.zeros((), dtype=normed.dtype),
                            "logits_min": jnp.min(logits) if logits is not None else jnp.zeros((), dtype=normed.dtype),
                            "router_topk_indices": tuple(router_topk_list),
                            "dsa_topk_indices": tuple(dsa_topk_indices_list)}
      aux["router_logits"] = tuple(router_logits_list)
      aux["router_topk_indices"] = tuple(router_topk_list)
      aux["dsa_topk_indices"] = tuple(dsa_topk_indices_list)
    return Glm5Output(logits=logits, hidden_states=normed, aux=aux)

  def __call__(self, decoder_input_tokens: jax.Array, decoder_positions: jax.Array | None = None,
               decoder_segment_ids: jax.Array | None = None, *, return_aux: bool = False, **kwargs: Any):
    out = self.forward_with_aux(decoder_input_tokens, decoder_positions, decoder_segment_ids,
                                return_aux=return_aux, **kwargs)
    if out.logits is None:
      raise RuntimeError("out.logits is None in Glm5Model.__call__.")
    return (out.logits, out.aux) if return_aux else out.logits

  def logits_from_hidden_states_for_vocab_tiling(self, hidden_states: jax.Array, deterministic: bool = True,
                                                  **kwargs: Any) -> jax.Array:
    del deterministic, kwargs
    return self.lm_head(hidden_states)

  @classmethod
  def from_maxtext_config(cls, config: Any, mesh: Mesh | None = None, *, rngs: nnx.Rngs,
                          quant_mode_str: str | None = None, place_on_devices: bool, **kwargs: Any) -> "Glm5Model":
    if not isinstance(place_on_devices, bool):
      raise TypeError("place_on_devices must be a bool, got" f" {type(place_on_devices).__name__}")
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    if not isinstance(config, glm5_config.Glm5Config) and config.quantization not in (None, ""):
      raise ValueError("Quantization is unsupported for GLM-5.3-Flash:" f" {config.quantization!r}.")
    if quant_mode_str not in (None, "", "train", "serve", "convert"):
      raise ValueError("Runtime quantization is unsupported for GLM-5.3-Flash:" f" {quant_mode_str!r}.")
    if kwargs:
      raise TypeError("Unexpected keyword arguments in from_maxtext_config:" f" {sorted(kwargs)!r}.")
    return cls(config=config, mesh=mesh, rngs=rngs, place_on_devices=place_on_devices)


Glm5ForCausalLM = Glm5Model


def create_glm5_3_flash_model(config: Any, mesh: Mesh | None = None, *, rngs: nnx.Rngs, **kwargs: Any) -> Glm5Model:
  if "place_on_devices" in kwargs:
    raise TypeError("place_on_devices is fixed to False in create_glm5_3_flash_model;"
                    " pass place_on_devices to Glm5Model or Glm5Model.from_maxtext_config directly instead.")
  if "remat_policy" in kwargs:
    raise TypeError("remat_policy must be configured via config.remat_policy, not as a keyword argument.")
  if not hasattr(config, "remat_policy"):
    raise AttributeError("config must define 'remat_policy'.")
  remat_policy = config.remat_policy
  if not isinstance(remat_policy, str):
    raise TypeError(f"remat_policy must be a str, got {type(remat_policy).__name__}.")
  if remat_policy == "none":
    policy = None
  elif remat_policy == "full":
    policy = jax.checkpoint_policies.nothing_saveable
  else:
    raise ValueError(f"Unsupported remat_policy for GLM-5.3-Flash: {remat_policy!r}. Supported policies: ('none', 'full').")
  model = Glm5Model.from_maxtext_config(config, mesh, rngs=rngs, place_on_devices=False, **kwargs)
  return apply_glm5_remat(model, policy=policy)


def compute_glm5_cross_entropy_loss(
    logits: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Compute masked token cross-entropy in float32."""
  logits_f32 = logits.astype(jnp.float32)
  targets_arr = jnp.asarray(decoder_target_tokens)
  mask_f32 = (jnp.asarray(decoder_target_mask) != 0).astype(jnp.float32)
  token_losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_f32, labels=targets_arr)
  masked_losses = token_losses * mask_f32
  xent_sum = jnp.sum(masked_losses).astype(jnp.float32)
  total_weights = jnp.sum(mask_f32).astype(jnp.float32)
  ce_loss = (xent_sum / jnp.maximum(total_weights, 1.0)).astype(jnp.float32)
  return ce_loss, xent_sum, total_weights


def _compute_tiled_cross_entropy(model: Glm5Model, hidden_states: jax.Array, decoder_target_tokens: jax.Array,
                                 decoder_target_mask: jax.Array, num_vocab_tiling: int):
  if num_vocab_tiling <= 1:
    logits = model.logits_from_hidden_states_for_vocab_tiling(hidden_states)
    return compute_glm5_cross_entropy_loss(logits, decoder_target_tokens, decoder_target_mask)
  batch, sequence, emb_dim = hidden_states.shape
  total_tokens = batch * sequence
  if total_tokens % num_vocab_tiling != 0:
    raise ValueError(f"Batch tokens ({batch} * {sequence} = {total_tokens}) must be divisible by"
                     f" num_vocab_tiling={num_vocab_tiling}.")
  tile_size = total_tokens // num_vocab_tiling
  hidden_tiles = hidden_states.reshape((num_vocab_tiling, tile_size, emb_dim))
  targets_tiles = jnp.asarray(decoder_target_tokens).reshape((num_vocab_tiling, tile_size))
  mask_tiles = (jnp.asarray(decoder_target_mask) != 0).astype(jnp.float32).reshape((num_vocab_tiling, tile_size))
  total_weights = jnp.sum(mask_tiles).astype(jnp.float32)

  @jax.checkpoint
  def _tile_xent_sum(h_tile, t_tile, m_tile):
    chunk_logits = model.logits_from_hidden_states_for_vocab_tiling(h_tile).astype(jnp.float32)
    chunk_loss = optax.softmax_cross_entropy_with_integer_labels(logits=chunk_logits, labels=t_tile)
    return jnp.sum(chunk_loss * m_tile).astype(jnp.float32)

  def _scan_step(acc, tile_inputs):
    h_tile, t_tile, m_tile = tile_inputs
    return acc + _tile_xent_sum(h_tile, t_tile, m_tile), None

  xent_sum, _ = jax.lax.scan(_scan_step, jnp.zeros((), dtype=jnp.float32),
                              (hidden_tiles, targets_tiles, mask_tiles))
  ce_loss = (xent_sum / jnp.maximum(total_weights, 1.0)).astype(jnp.float32)
  return ce_loss, xent_sum, total_weights


def compute_glm5_loss(model: Glm5Model, tokens: jax.Array, decoder_target_tokens: jax.Array,
                      decoder_target_mask: jax.Array, decoder_segment_ids: jax.Array | None = None,
                      decoder_positions: jax.Array | None = None, *, is_train: bool):
  if not isinstance(is_train, bool):
    raise TypeError(f"is_train must be a bool, got {type(is_train).__name__}.")
  if tokens is None:
    raise ValueError("tokens must be provided as a non-None array.")
  if decoder_target_tokens is None:
    raise ValueError("decoder_target_tokens must be provided as a non-None array.")
  if decoder_target_mask is None:
    raise ValueError("decoder_target_mask must be provided as a non-None array.")
  tokens_arr = jnp.asarray(tokens)
  target_tokens_arr = jnp.asarray(decoder_target_tokens)
  target_mask_arr = jnp.asarray(decoder_target_mask)
  if target_tokens_arr.shape != tokens_arr.shape:
    raise ValueError(f"decoder_target_tokens shape {target_tokens_arr.shape} != tokens shape {tokens_arr.shape}.")
  if target_mask_arr.shape != tokens_arr.shape:
    raise ValueError(f"decoder_target_mask shape {target_mask_arr.shape} != tokens shape {tokens_arr.shape}.")
  seg_ids = jnp.asarray(decoder_segment_ids) if decoder_segment_ids is not None else None
  positions = jnp.asarray(decoder_positions) if decoder_positions is not None else None
  num_vocab_tiling = model.config.num_vocab_tiling
  if num_vocab_tiling > 1:
    out = model.forward_with_aux(tokens_arr, decoder_positions=positions, decoder_segment_ids=seg_ids,
                                 compute_logits=False)
    ce_loss, xent_sum, total_weights = _compute_tiled_cross_entropy(
        model, out.hidden_states, target_tokens_arr, target_mask_arr, num_vocab_tiling)
  else:
    out = model.forward_with_aux(tokens_arr, decoder_positions=positions, decoder_segment_ids=seg_ids,
                                 compute_logits=True)
    if out.logits is None:
      raise RuntimeError("out.logits is None but compute_logits=True was requested.")
    ce_loss, xent_sum, total_weights = compute_glm5_cross_entropy_loss(out.logits, target_tokens_arr, target_mask_arr)
  moe_lb_loss = out.aux["moe_lb_loss"].astype(jnp.float32)
  mtp_loss = jnp.zeros((), dtype=jnp.float32)
  mtp_moe_lb_loss = jnp.zeros((), dtype=jnp.float32)
  if is_train and model.config.num_nextn_predict_layers > 0:
    if model.mtp is None:
      raise ValueError("model.mtp is None but model.config.num_nextn_predict_layers > 0")
    mtp_targets = jnp.roll(target_tokens_arr, -1, axis=-1)
    valid_mask = target_mask_arr != 0
    mtp_valid_mask = valid_mask & jnp.concatenate(
        [valid_mask[:, 1:], jnp.zeros((valid_mask.shape[0], 1), dtype=jnp.bool_)], axis=-1)
    if num_vocab_tiling > 1:
      mtp_normed, mtp_router_logits, _ = model.mtp(out.hidden_states, target_tokens_arr,
          embed_tokens=model.embed_tokens, lm_head=None, decoder_segment_ids=seg_ids)
      mtp_loss, _, _ = _compute_tiled_cross_entropy(model, mtp_normed, mtp_targets, mtp_valid_mask, num_vocab_tiling)
    else:
      mtp_logits, mtp_router_logits, _ = model.mtp(out.hidden_states, target_tokens_arr,
          embed_tokens=model.embed_tokens, lm_head=model.lm_head, decoder_segment_ids=seg_ids)
      mtp_loss, _, _ = compute_glm5_cross_entropy_loss(mtp_logits, mtp_targets, mtp_valid_mask)
    mtp_lb_stats = accumulate_moe_lb_stats(init_moe_lb_stats(model.config.n_routed_experts), mtp_router_logits,
        model.config.n_routed_experts, model.config.num_experts_per_tok,
        attention_mask=(seg_ids != 0) if seg_ids is not None else None)
    mtp_moe_lb_loss = finalize_moe_lb_loss(mtp_lb_stats, model.config.n_routed_experts).astype(jnp.float32)
    bb_stats = out.aux["lb_stats"]
    pooled_lb_stats = (bb_stats[0] + mtp_lb_stats[0], bb_stats[1] + mtp_lb_stats[1], bb_stats[2] + mtp_lb_stats[2])
    moe_lb_loss = finalize_moe_lb_loss(pooled_lb_stats, model.config.n_routed_experts).astype(jnp.float32)
    total_loss = (ce_loss + jnp.float32(model.config.moe_lb_loss_weight) * moe_lb_loss
                  + jnp.float32(model.config.mtp_loss_scaling_factor) * mtp_loss).astype(jnp.float32)
  else:
    total_loss = (ce_loss + jnp.float32(model.config.moe_lb_loss_weight) * moe_lb_loss).astype(jnp.float32)
  aux_metrics = {"ce_loss": ce_loss, "moe_lb_loss": moe_lb_loss, "mtp_loss": mtp_loss,
                 "mtp_moe_lb_loss": mtp_moe_lb_loss, "xent_sum": xent_sum, "total_weights": total_weights}
  return total_loss, aux_metrics


def glm5_loss_fn(model: Glm5Model, *, inputs: jax.Array, targets: jax.Array, targets_segmentation: jax.Array,
                 is_train: bool, inputs_segmentation: jax.Array | None = None,
                 inputs_position: jax.Array | None = None, targets_position: jax.Array | None = None):
  if not isinstance(is_train, bool):
    raise TypeError(f"is_train must be a bool, got {type(is_train).__name__}.")
  del targets_position
  return compute_glm5_loss(model=model, tokens=inputs, decoder_target_tokens=targets,
                           decoder_target_mask=targets_segmentation, decoder_segment_ids=inputs_segmentation,
                           decoder_positions=inputs_position, is_train=is_train)


def trainable_mask(model_or_params: Any) -> Any:
  tree = nnx.state(model_or_params, (nnx.Param, Glm5RouterBias)) if isinstance(model_or_params, nnx.Module) else model_or_params

  def _is_trainable(path: tuple[Any, ...], leaf: Any) -> bool:
    del leaf
    path_str = jax.tree_util.keystr(path)
    if "indexer" in path_str:
      return False
    if "e_score_correction_bias" in path_str:
      return False
    return True

  return jax.tree_util.tree_map_with_path(
      _is_trainable, tree, is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array)))
