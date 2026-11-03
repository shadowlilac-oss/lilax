"""Dot-product attention: logits, masking and the unnormalized local softmax-weighted values.

The layer passes its own `qk_product`, `wv_product`, `is_partition_in_decode` and `reverse_transepose`
methods as callables, so overrides of those methods keep taking effect here.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import (
    Array,
    AxisIdxes,
    BATCH_ATTN,
    Config,
    D_KV,
    DECODE_BATCH,
    DECODE_LENGTH,
    HEAD,
    KV_LENGTH,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_LENGTH,
)
from lilax._backend.inference.kvcache import KVQuant, KVTensor
from lilax._backend.layers.attention_masks import dense_masks
from lilax._backend.distributed.sharding import with_logical_constraint


def validate_compute_axis_order(s: AxisIdxes) -> None:
  valid_compute_axis_order = ((0, 1, 2, 3), (0, 2, 1, 3))
  if s not in valid_compute_axis_order:  # currently supported compute_axis_order
    raise ValueError(
        "Invalid compute_axis_order was passed. Valid options ",
        valid_compute_axis_order,
    )


def attention_logits(
    query: Array,
    key: Array | KVTensor,
    value: Array | KVTensor,
    model_mode: str,
    qk_product_einsum: Callable[..., Array],
    *,
    compute_axis_order: AxisIdxes,
    float32_qk_product: bool,
    float32_logits: bool,
    attn_logits_soft_cap: float | None,
    is_absorbed_mqa: bool,
    kv_quant: KVQuant | None,
    qk_product: Callable[..., Array],
    is_partition_in_decode: Callable[[int], bool],
) -> tuple[Array, Array | KVTensor, Array | KVTensor, Array]:
  """Returns the sharding-constrained `(query, key, value)` and their soft-capped `[b, n_kv, g, t, s]` logits."""
  validate_compute_axis_order(compute_axis_order)
  if float32_qk_product:
    if isinstance(key, KVTensor):
      key = key.dequant()
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  q_seq_len = query.shape[1]
  if is_absorbed_mqa:
    prefill_q_sharding = (BATCH_ATTN, PREFILL_LENGTH, HEAD, None)
    decode_q_sharding = (DECODE_BATCH, DECODE_LENGTH, HEAD, None)
    prefill_kv_sharding = (BATCH_ATTN, PREFILL_LENGTH, None, None)
    decode_kv_sharding = (DECODE_BATCH, DECODE_LENGTH, None, None)
    weights_decode_shd = (KV_LENGTH, None, HEAD, None, None)
    weights_prefill_shd = (BATCH_ATTN, None, HEAD, PREFILL_LENGTH, KV_LENGTH)
  else:
    prefill_q_sharding = (BATCH_ATTN, PREFILL_LENGTH, HEAD, D_KV)
    decode_q_sharding = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV)
    prefill_kv_sharding = prefill_q_sharding
    decode_kv_sharding = decode_q_sharding
    weights_decode_shd = (KV_LENGTH, HEAD, None, None, None)
    weights_prefill_shd = (BATCH_ATTN, HEAD, None, PREFILL_LENGTH, KV_LENGTH)

  if is_partition_in_decode(q_seq_len):
    query = with_logical_constraint(query, decode_q_sharding)
    if kv_quant and isinstance(key, KVTensor) and isinstance(value, KVTensor):
      key.qvalue = with_logical_constraint(key.qvalue, decode_kv_sharding)
      value.qvalue = with_logical_constraint(value.qvalue, decode_kv_sharding)
    else:
      key = with_logical_constraint(key, decode_kv_sharding)
      value = with_logical_constraint(value, decode_kv_sharding)
  elif model_mode == MODEL_MODE_PREFILL:
    query = with_logical_constraint(query, prefill_q_sharding)
    if kv_quant and isinstance(key, KVTensor) and isinstance(value, KVTensor):
      key.qvalue = with_logical_constraint(key.qvalue, prefill_kv_sharding)
      value.qvalue = with_logical_constraint(value.qvalue, prefill_kv_sharding)
    else:
      key = with_logical_constraint(key, prefill_kv_sharding)
      value = with_logical_constraint(value, prefill_kv_sharding)

  attn_weights = qk_product(query, key, q_seq_len, model_mode, qk_product_einsum)
  if is_partition_in_decode(q_seq_len):
    attn_weights = with_logical_constraint(attn_weights, weights_decode_shd)
  elif model_mode == MODEL_MODE_PREFILL:
    attn_weights = with_logical_constraint(attn_weights, weights_prefill_shd)

  if attn_logits_soft_cap:
    attn_weights = jnp.tanh(attn_weights / attn_logits_soft_cap)
    attn_weights = attn_weights * attn_logits_soft_cap
  if float32_logits:
    attn_weights = attn_weights.astype(jnp.float32)
  return query, key, value, attn_weights


def apply_attention_masks(
    attn_weights: Array,
    attn_mask: Array | None,
    indexer_mask: Array | None,
    q_seq_len: int,
    model_mode: str,
    *,
    is_partition_in_decode: Callable[[int], bool],
) -> Array:
  """Applies the sparse-indexer mask and then the layer's dense attention mask to the logits."""
  if indexer_mask is not None:
    indexer_mask = indexer_mask[:, None, None, :, :]
    attn_weights = dense_masks.apply_mask_to_logits(attn_weights, indexer_mask)
  if is_partition_in_decode(q_seq_len):
    attn_mask = with_logical_constraint(attn_mask, (KV_LENGTH, HEAD, None, None, None))
  elif model_mode == MODEL_MODE_PREFILL:
    attn_mask = with_logical_constraint(attn_mask, (BATCH_ATTN, HEAD, None, PREFILL_LENGTH, KV_LENGTH))
  if attn_mask is not None:
    attn_weights = dense_masks.apply_mask_to_logits(attn_weights, attn_mask)
  return attn_weights


def max_logits_per_head(attn_weights: Array) -> Array:
  """Reduces `[b, n_kv, g, t, s]` logits to the per-query-head maximum `[b, n_kv * g]` recorded for QK-Clip."""
  max_logits_per_group = jnp.max(attn_weights, axis=(-2, -1))
  b, n_kv, g = max_logits_per_group.shape
  return max_logits_per_group.reshape(b, n_kv * g)


def compute_local_attention(
    attn_weights: Array,
    value: Array | KVTensor,
    q_seq_len: int,
    model_mode: str,
    wv_product_einsum: Callable[..., Array],
    sinks: Array | None = None,
    *,
    reshape_q: bool,
    wv_product: Callable[..., Array],
    is_partition_in_decode: Callable[[int], bool],
) -> tuple[Array, Array, Array]:
  """Computes the attention of a local subset of the kv cache."""
  b, n_kv, g, t, s = attn_weights.shape
  n_q = n_kv * g
  logits = jnp.reshape(attn_weights, (b, n_q, t, s))
  if sinks is not None:
    sinks_param = sinks.astype(attn_weights.dtype)
    sinks_logits = sinks_param[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    sinks_logits = jnp.broadcast_to(sinks_logits, (b, n_q, t, 1))
    logits = jnp.concatenate([logits, sinks_logits], axis=-1)

  local_max = jnp.max(logits, axis=-1, keepdims=True)
  local_exps_combined = jnp.exp(logits - local_max)
  local_sum = jnp.sum(local_exps_combined, axis=-1, keepdims=True)
  local_exps = local_exps_combined[..., :s]
  local_exps = jnp.reshape(local_exps, (b, n_kv, g, t, s))
  local_max = jnp.transpose(local_max, (0, 2, 1, 3))
  local_sum = jnp.transpose(local_sum, (0, 2, 1, 3))

  local_out = wv_product(local_exps, value, model_mode, wv_product_einsum)
  if model_mode == MODEL_MODE_AUTOREGRESSIVE and is_partition_in_decode(q_seq_len):
    local_out = with_logical_constraint(local_out, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
  elif model_mode == MODEL_MODE_PREFILL:
    local_out = with_logical_constraint(local_out, (BATCH_ATTN, KV_LENGTH, HEAD, D_KV))

  if reshape_q and q_seq_len == 1:
    local_max = local_max[:, 0:1, :, :]
    local_sum = local_sum[:, 0:1, :, :]
    local_out = local_out[:, 0:1, :, :]

  if model_mode == MODEL_MODE_AUTOREGRESSIVE and is_partition_in_decode(q_seq_len):
    local_max = with_logical_constraint(local_max, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
    local_sum = with_logical_constraint(local_sum, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
    local_out = with_logical_constraint(local_out, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))

  return local_out, local_max, local_sum


def qk_product(
    query: Array,
    key: Array | KVTensor,
    q_seq_len: int,
    model_mode: str,
    einsum: Callable[..., Array],
    *,
    config: Config,
    num_kv_heads: int,
    compute_axis_order: AxisIdxes,
    reshape_q: bool,
) -> Array:
  """Query-Key product."""
  b, t, n, d = query.shape
  n_kv = key.shape[-2]
  assert n_kv == num_kv_heads and n % n_kv == 0
  precision_kwargs = {"precision": config.matmul_precision} if einsum is jnp.einsum else {}
  if model_mode == MODEL_MODE_TRAIN or compute_axis_order == (0, 1, 2, 3):
    query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
    if reshape_q and q_seq_len == 1:
      query = jnp.broadcast_to(query, (b, 2, n_kv, n // n_kv, d))
    result = einsum("btkgd,bskd->bkgts", query, key, **precision_kwargs)
  elif compute_axis_order == (0, 2, 1, 3):
    query = jnp.transpose(query, axes=compute_axis_order)
    key = jax.tree.map(lambda x: jnp.transpose(x, axes=compute_axis_order), key)
    query = jnp.reshape(query, (b, n_kv, n // n_kv, t, d))
    if reshape_q and q_seq_len == 1:
      query = jnp.broadcast_to(query, (b, n_kv, n // n_kv, 2, d))
    result = einsum("bkgtd,bksd->bkgts", query, key, **precision_kwargs)
  else:
    raise NotImplementedError(compute_axis_order)
  return result


def wv_product(
    attn_weights: Array,
    value: Array | KVTensor,
    model_mode: str,
    einsum: Callable[..., Array],
    *,
    config: Config,
    kv_quant: KVQuant | None,
    compute_axis_order: AxisIdxes,
    reverse_transepose: Callable[[Array, AxisIdxes], Array],
) -> Array:
  """weighted value product."""
  precision_kwargs = {"precision": config.matmul_precision} if einsum is jnp.einsum else {}
  if kv_quant:
    if isinstance(value, KVTensor) and kv_quant.dtype == jnp.float8_e4m3fn:
      value.qvalue = value.qvalue.astype(jnp.bfloat16)
  if model_mode == MODEL_MODE_TRAIN or compute_axis_order == (0, 1, 2, 3):
    out = einsum("bkgts,bskd->btkgd", attn_weights, value, **precision_kwargs)
    b, t, n_kv, g, d = out.shape
    result = jnp.reshape(out, (b, t, n_kv * g, d))
  elif compute_axis_order == (0, 2, 1, 3):
    value = jax.tree.map(lambda x: jnp.transpose(x, axes=compute_axis_order), value)
    out = einsum("bkgts,bksd->bkgtd", attn_weights, value, **precision_kwargs)
    b, n_kv, g, t, d = out.shape
    result = jnp.reshape(out, (b, n_kv * g, t, d))
    result = reverse_transepose(result, compute_axis_order)
  return result


def reverse_transepose(transposed_array, transpose_axis_order):
  return jax.numpy.moveaxis(transposed_array, (0, 1, 2, 3), transpose_axis_order)
