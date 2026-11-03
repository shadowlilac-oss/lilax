"""Multi-head latent attention forward, remat and backward for the DeepSeek BF16 batch-split schedule."""

import functools

import jax

from lilax._backend.models.deepseek_batchsplit.attention.projections import kv_projection, query_projection
from lilax._backend.models.deepseek_batchsplit.attention.splash import tpu_flash_attention, tpu_flash_attention_bwd
from lilax._backend.models.deepseek_batchsplit.dot import dot


def mla(
    inputs,
    yarn_freqs,
    weights,
    *,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    config,
    splash_kernel,
    pairwise_swap_and_negate_mask,
    dtype,
    mesh,
    activation_pspec,
):
  """Performs MLA."""
  (
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      out_weights,
  ) = weights
  query = query_projection(
      inputs,
      yarn_freqs,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      epsilon=epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      mscale=mscale,
      config=config,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  key, value = kv_projection(
      inputs,
      yarn_freqs,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      kv_lora_rank=kv_lora_rank,
      kv_norm_epsilon=kv_norm_epsilon,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
      config=config,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  attn_out, lse = tpu_flash_attention(
      query,
      key,
      value,
      mesh=mesh,
      splash_kernel=splash_kernel,
      activation_pspec=activation_pspec,
  )
  out = dot(attn_out, out_weights, axes=2)
  return out, {"attn_out": attn_out, "lse": lse}


def mla_remat(
    residuals,
    yarn_freqs,
    weights,
    *,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    config,
    splash_kernel,
    pairwise_swap_and_negate_mask,
    dtype,
    mesh,
    activation_pspec,
):
  """Performs remat for the mla function."""
  inputs, attn_out, lse = residuals
  (
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      out_weights,
  ) = weights
  query, query_projection_bwd = jax.vjp(
      functools.partial(
          query_projection,
          epsilon=epsilon,
          qk_rope_head_dim=qk_rope_head_dim,
          max_position_embeddings=max_position_embeddings,
          original_max_position_embeddings=original_max_position_embeddings,
          rope_factor=rope_factor,
          pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
          dtype=dtype,
          qk_nope_head_dim=qk_nope_head_dim,
          mscale=mscale,
          config=config,
          mesh=mesh,
          activation_pspec=activation_pspec,
      ),
      inputs,
      yarn_freqs,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
  )
  (key, value), kv_projection_bwd = jax.vjp(
      functools.partial(
          kv_projection,
          kv_lora_rank=kv_lora_rank,
          kv_norm_epsilon=kv_norm_epsilon,
          pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
          dtype=dtype,
          qk_nope_head_dim=qk_nope_head_dim,
          num_query_heads=num_query_heads,
          config=config,
          mesh=mesh,
          activation_pspec=activation_pspec,
      ),
      inputs,
      yarn_freqs,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
  )
  out, out_projection_bwd = jax.vjp(
      functools.partial(
          dot,
          axes=2,
      ),
      attn_out,
      out_weights,
  )
  return out, (
      query_projection_bwd,
      kv_projection_bwd,
      functools.partial(
          tpu_flash_attention_bwd,
          query=query,
          key=key,
          value=value,
          attention_output=attn_out,
          logsumexp=lse,
          mesh=mesh,
          splash_kernel=splash_kernel,
          activation_pspec=activation_pspec,
      ),
      out_projection_bwd,
  )


def mla_bwd(
    out_grad,
    bwds,
):
  """Performs the backward pass for the mla function."""
  query_projection_bwd, kv_projection_bwd, attn_op_bwd, out_projection_bwd = bwds
  attn_out_grad, out_weights_grad = out_projection_bwd(out_grad)
  query_grad, key_grad, value_grad = attn_op_bwd(attn_out_grad)
  inputs_grad_from_kv, _, wkv_a_weights_grad, wkv_b_weights_grad, kv_norm_scale_weights_grad = kv_projection_bwd(
      (key_grad, value_grad)
  )
  inputs_grad_from_q, _, wq_a_weights_grad, wq_b_weights_grad, q_norm_scale_weights_grad = query_projection_bwd(
      query_grad
  )
  return inputs_grad_from_kv + inputs_grad_from_q, (
      wq_a_weights_grad,
      wq_b_weights_grad,
      q_norm_scale_weights_grad,
      wkv_a_weights_grad,
      wkv_b_weights_grad,
      kv_norm_scale_weights_grad,
      out_weights_grad,
  )
