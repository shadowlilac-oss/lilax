"""Runs cuDNN fused flash attention through the JAX SDPA API."""

import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from lilax._backend.common.common_types import Array, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN


def align_qkv_for_cudnn_flash(
    query: Array,
    key: Array,
    value: Array,
) -> tuple[Array, Array, Array]:
  """Broadcasts key/value batch dim to match query batch dim for cuDNN flash attention."""
  if query.shape[0] != key.shape[0]:
    if key.shape[0] == 1 and query.shape[0] > 1:
      key = jnp.broadcast_to(key, (query.shape[0], *key.shape[1:]))
      value = jnp.broadcast_to(value, (query.shape[0], *value.shape[1:]))
    else:
      raise ValueError(
          "Query and key/value batch sizes must match for cuDNN flash attention: "
          f"{query.shape=}, {key.shape=}, {value.shape=}"
      )

  return query, key, value


def cudnn_jax_flash_attention(
    query: Array,
    key: Array,
    value: Array,
    decoder_segment_ids: Array | None,
    model_mode: str = MODEL_MODE_TRAIN,
    *,
    dropout_rate: float,
) -> tuple[Array, Array]:
  """CUDNN Flash Attention with JAX SDPA API."""
  from jax._src.cudnn.fused_attention_stablehlo import (
      dot_product_attention,
      MaskType,
  )

  _, _, _, head_dim = query.shape
  if model_mode == MODEL_MODE_AUTOREGRESSIVE:
    lengths = jnp.sum(decoder_segment_ids, axis=-1)
    if lengths.shape[0] == 1 and query.shape[0] > 1:
      lengths = jnp.broadcast_to(lengths, (query.shape[0],))

    output, lse = dot_product_attention(
        query,
        key,
        value,
        q_seqlen=lengths,
        kv_seqlen=lengths,
        mask_type=MaskType.PADDING,
        scale=1.0,
        dropout_rate=dropout_rate,
        qkv_layout="BTNH",
        return_residual=True,
    )
  else:
    output, lse = dot_product_attention(
        query,
        key,
        value,
        mask_type=MaskType.CAUSAL,
        scale=1.0,
        dropout_rate=dropout_rate,
        qkv_layout="BTNH",
        return_residual=True,
    )
  output = checkpoint_name(output, "context")
  lse = checkpoint_name(lse, "context")
  return output, lse
