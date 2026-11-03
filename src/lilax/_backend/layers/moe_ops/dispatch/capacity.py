"""Capacity-limited dispatch and combine masks for dense (einsum) expert dispatch."""

import math

import jax
import jax.numpy as jnp
from lilax._backend.utils import max_logging


def subgroup_capacity_masks(
    top_k_indices, softmax_probs, num_experts, num_experts_per_tok, capacity_factor, cp, sub_seq, shard_with_logical
):
  """Subgroup mask generation for inference only."""
  batch_size, _, _ = top_k_indices.shape
  top_k_indices = jnp.reshape(top_k_indices, (batch_size, cp, sub_seq, top_k_indices.shape[2]))
  tokens_per_batch = sub_seq * num_experts_per_tok
  expert_capacity_per_batch = int(
      max(
          math.ceil(tokens_per_batch / num_experts) * capacity_factor,
          capacity_factor,
      )
  )
  max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")
  expert_mask = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=jnp.int32)
  expert_mask_fused = jnp.reshape(
      expert_mask,
      (batch_size, cp, sub_seq * num_experts_per_tok, num_experts),
  )
  expert_mask_fused = shard_with_logical(expert_mask_fused, ("activation_batch", None, None, None))
  expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=2)
  expert_token_count = jnp.reshape(
      expert_token_count_fused,
      ((batch_size, cp, sub_seq, num_experts_per_tok, num_experts)),
  )
  expert_token_count = shard_with_logical(
      expert_token_count,
      ("activation_batch", "activation_norm_length", None, None, None),
  )
  trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
  combined_expert_mask = jnp.sum(trunc_expert_mask, axis=3)
  softmax_probs = jnp.reshape(
      softmax_probs,
      ((batch_size, cp, sub_seq, num_experts)),
  )
  softmax_probs *= combined_expert_mask
  expert_token_position_fused = expert_mask_fused * expert_token_count_fused
  expert_token_position = jnp.reshape(
      expert_token_position_fused,
      (batch_size, cp, sub_seq, num_experts_per_tok, num_experts),
  )
  combined_expert_token_position = jnp.sum(expert_token_position, axis=3) * combined_expert_mask
  expert_token_position_in_capacity = jax.nn.one_hot(
      combined_expert_token_position,
      num_classes=expert_capacity_per_batch + 1,
      dtype=jnp.int32,
  )
  combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
  combine_mask = combine_mask[..., 1:]
  dispatch_mask = combine_mask.astype(bool)
  dispatch_mask = jnp.reshape(
      dispatch_mask,
      (batch_size, cp, sub_seq, num_experts, expert_capacity_per_batch),
  )
  combine_mask = jnp.reshape(
      combine_mask,
      (batch_size, cp, sub_seq, num_experts, expert_capacity_per_batch),
  )
  return dispatch_mask, combine_mask


def capacity_masks(top_k_indices, softmax_probs, num_experts, num_experts_per_tok, capacity_factor, shard_with_logical):
  """Generate masks."""
  batch_size, seq_len, _ = top_k_indices.shape
  tokens_per_batch = seq_len * num_experts_per_tok
  expert_capacity_per_batch = int(
      max(
          math.ceil(tokens_per_batch / num_experts) * capacity_factor,
          capacity_factor,
      )
  )
  max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")
  expert_mask = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=jnp.int32)
  expert_mask_fused = jnp.reshape(
      expert_mask,
      (batch_size, seq_len * num_experts_per_tok, num_experts),
  )
  expert_mask_fused = shard_with_logical(expert_mask_fused, ("activation_batch_moe", None, None))
  expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
  expert_token_count = jnp.reshape(
      expert_token_count_fused,
      ((batch_size, seq_len, num_experts_per_tok, num_experts)),
  )
  expert_token_count = shard_with_logical(
      expert_token_count,
      ("activation_batch", "activation_norm_length", None, None),
  )
  trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
  combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)
  softmax_probs *= combined_expert_mask
  expert_token_position_fused = expert_mask_fused * expert_token_count_fused
  expert_token_position = jnp.reshape(
      expert_token_position_fused,
      (batch_size, seq_len, num_experts_per_tok, num_experts),
  )
  combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
  expert_token_position_in_capacity = jax.nn.one_hot(
      combined_expert_token_position,
      num_classes=expert_capacity_per_batch + 1,
      dtype=jnp.int32,
  )
  combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
  combine_mask = combine_mask[..., 1:]
  dispatch_mask = combine_mask.astype(bool)
  return dispatch_mask, combine_mask
