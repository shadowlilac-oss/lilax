"""Decoder output projections for tied and untied embeddings."""

from flax import nnx
import jax.numpy as jnp

from lilax._backend.layers.embeddings import attend_on_embedding
from lilax._backend.autodiff import cotangents


def project_tied(y, shared_embedding, config, out_sharding):
  """Project through the shared token embedding."""
  embedding_table = shared_embedding.embedding[...]
  attend_dtype = jnp.float32 if config.logits_dot_in_fp32 else config.dtype
  # Release the head's input and embedding cotangents together, so the backward pass does not keep
  # the [batch, length, vocab] dlogits alive until the embedding-gradient matmul (see cotangents.tie_cotangents).
  y_head, head_table = y, embedding_table
  if config.tie_lm_head_cotangents:
    y_head, head_table = cotangents.tie_cotangents(y, embedding_table)
  logits = attend_on_embedding(y_head, head_table, attend_dtype, config, out_sharding)

  if config.normalize_embedding_logits:
    # Correctly normalize pre-softmax logits for this shared case.
    logits = logits / jnp.sqrt(y.shape[-1])
  if config.final_logits_soft_cap:
    logits = logits / config.final_logits_soft_cap
    logits = jnp.tanh(logits) * config.final_logits_soft_cap
  return logits


def project_dense_with_tied_cotangents(y, logits_dense, out_sharding):
  """Project through a separate output kernel with tied cotangents."""
  # Release the head's input and weight cotangents together, so the backward pass does not keep
  # the [batch, length, vocab] dlogits alive until the weight-gradient matmul (see cotangents.tie_cotangents).
  head_graphdef, head_state = nnx.split(logits_dense)
  y_head, head_state = cotangents.tie_cotangents(y, head_state)
  return nnx.merge(head_graphdef, head_state)(y_head, out_sharding=out_sharding)
