"""MLA query and key/value projections for the DeepSeek FP8 batch-split schedule."""

import math

import jax
import jax.numpy as jnp
from lilax._backend.models.deepseek_batchsplit_fp8.attention.norm import rms_norm
from lilax._backend.models.deepseek_batchsplit_fp8.attention.rope import yarn
from lilax._backend.models.deepseek_batchsplit_fp8.dot import dot


def query_projection(
    inputs_q,
    inputs_positions,
    wq_a_weights,
    wq_b_weights,
    q_norm_scale_weights,
    *,
    epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    dtype,
    mscale,
    quant,
):
  """Performs query projection."""
  # Set softmax scaling.
  qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
  softmax_scale = qk_head_dim**-0.5
  if max_position_embeddings > original_max_position_embeddings:
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    softmax_scale = softmax_scale * m * m

  # LoRA path
  low_rank_q = dot(inputs_q, wq_a_weights, quant=quant)
  low_rank_q = rms_norm(
      low_rank_q,
      q_norm_scale_weights,
      epsilon=epsilon,
      dtype=dtype,
  )
  low_rank_q = jax.ad_checkpoint.checkpoint_name(low_rank_q, "mla_q")
  q = dot(low_rank_q, wq_b_weights, quant=quant)

  # Split into non-positional and rotary parts.
  q_nope, q_pe = jnp.split(q, [qk_nope_head_dim], axis=-1)
  q_pe = yarn(
      q_pe,
      inputs_positions,
      embedding_dims=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      fprop_dtype=dtype,
  )
  query = jnp.concatenate([q_nope, q_pe], axis=-1) * softmax_scale
  return query


def kv_projection(
    inputs,
    inputs_positions,
    wkv_a_weights,
    wkv_b_weights,
    kv_norm_scale_weights,
    *,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_rope_head_dim,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    dtype,
    qk_nope_head_dim,
    num_query_heads,
    quant,
):
  """Performs KV projection."""
  low_rank = dot(inputs, wkv_a_weights, quant=quant)
  low_rank_main, low_rank_rope = jnp.split(low_rank, [kv_lora_rank], axis=-1)
  low_rank_main = rms_norm(
      low_rank_main,
      kv_norm_scale_weights,
      epsilon=kv_norm_epsilon,
      dtype=dtype,
  )
  low_rank_main = jax.ad_checkpoint.checkpoint_name(low_rank_main, "mla_kv")
  key_rope = jnp.expand_dims(low_rank_rope, axis=2)
  key_rope = yarn(
      key_rope,
      inputs_positions,
      embedding_dims=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      fprop_dtype=dtype,
  )

  return get_key_value(
      low_rank_main,
      key_rope,
      wkv_b_weights,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
      quant=quant,
  )


def get_key_value(low_rank_main, key_rope, wkv_b_weights, *, qk_nope_head_dim, num_query_heads, quant):
  """Gets key and value from compressed KV latent vector and key rope."""
  kv_out = dot(low_rank_main, wkv_b_weights, quant=quant)

  # Split kv_out into key_nope and value parts.
  key_nope, value = jnp.split(kv_out, [qk_nope_head_dim], axis=-1)
  key_rope = jnp.broadcast_to(
      key_rope,
      (
          key_nope.shape[0],
          key_nope.shape[1],
          num_query_heads,
          key_rope.shape[3],
      ),
  )

  key = jnp.concatenate([key_nope, key_rope], axis=-1)

  return key, value
