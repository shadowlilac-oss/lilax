"""Up projection chunked along the embedding dimension, overlapping each chunk's token all-gather with its GMM."""

import jax
import jax.numpy as jnp


def emb_chunked_up_projection(
    x,
    logits,
    pre_bias_logits,
    w0,
    w1,
    w0_bias,
    w1_bias,
    wo_bias,
    sharded_input_ids,
    rngs,
    embed_dim,
    forced_routed_experts=None,
    per_expert_scale=None,
    *,
    config,
    expert_parallelism_size,
    expert_parallelism_name,
    tensor_parallelism_size,
    ring_dispatch,
    expert_gmm,
    lookup_transform_bias,
):
  """Overlap token all-gather and GMM computation along embedding dimension.

  Layer-owned values are zero-arg callables and `lookup_transform_bias` returns the layer's method; both are
  resolved at the point of use.
  """
  num_ep = expert_parallelism_size()
  expert_shard_id = jax.lax.axis_index(expert_parallelism_name()) if num_ep > 1 else 0

  chunk_dim = embed_dim // config().num_moe_emb_chunks
  chunk_rngs_key = (
      rngs.params() if rngs is not None and hasattr(rngs, "params") and callable(getattr(rngs, "params")) else None
  )

  first_x_unrouted = jax.lax.dynamic_slice_in_dim(x, 0, chunk_dim, axis=2)
  cur_x_chunk, routing, route_metadata = ring_dispatch(
      first_x_unrouted,
      logits,
      pre_bias_logits,
      num_ep,
      expert_shard_id,
      chunk_rngs_key,
      input_ids=sharded_input_ids,
      forced_routed_experts=forced_routed_experts,
      per_expert_scale=per_expert_scale,
  )

  if config().mlp_bias:
    if not config().use_te_grouped_gemm:
      w0_bias, w1_bias, wo_bias = lookup_transform_bias()(routing.selected_experts, w0_bias, w1_bias, wo_bias)
    elif tensor_parallelism_size() > 1:
      # TE fuses per-expert wi bias in the GEMM. wo bias is added after
      # psum_scatter, so index it per token like the non-TE path.
      (wo_bias,) = lookup_transform_bias()(routing.selected_experts, wo_bias)

  partial_sum0 = jnp.zeros((cur_x_chunk.shape[0], w0.shape[-1]), dtype=cur_x_chunk.dtype)
  partial_sum1 = jnp.zeros((cur_x_chunk.shape[0], w1.shape[-1]), dtype=cur_x_chunk.dtype)

  def scan_fn(carry, _):
    cur_x, ps0, ps1, chunk_idx = carry

    cur_w0 = jax.lax.dynamic_slice_in_dim(w0, chunk_idx * chunk_dim, chunk_dim, axis=1)
    cur_w1 = jax.lax.dynamic_slice_in_dim(w1, chunk_idx * chunk_dim, chunk_dim, axis=1)
    gmm_fn = expert_gmm.for_local_experts(cur_x, routing, route_metadata)
    next_ps0, next_ps1 = expert_gmm.up(
        cur_x,
        cur_w0,
        cur_w1,
        None,  # Only add biases once at the end of the loop
        None,  # Only add biases once at the end of the loop
        gmm_fn,
        partial_accum0=ps0,
        partial_accum1=ps1,
    )
    next_x_unrouted = jax.lax.dynamic_slice_in_dim(x, (chunk_idx + 1) * chunk_dim, chunk_dim, axis=2)
    next_x, _, _ = ring_dispatch(
        next_x_unrouted,
        logits,
        pre_bias_logits,
        num_ep,
        expert_shard_id,
        chunk_rngs_key,
        input_ids=sharded_input_ids,
        forced_routed_experts=forced_routed_experts,
        per_expert_scale=per_expert_scale,
    )
    return (next_x, next_ps0, next_ps1, chunk_idx + 1), None

  (last_x_chunk, ps0, ps1, _), _ = jax.lax.scan(
      scan_fn,
      (cur_x_chunk, partial_sum0, partial_sum1, 0),
      None,
      length=config().num_moe_emb_chunks - 1,
  )
  # gmm the last chunk
  last_w0 = jax.lax.dynamic_slice_in_dim(w0, (config().num_moe_emb_chunks - 1) * chunk_dim, chunk_dim, axis=1)
  last_w1 = jax.lax.dynamic_slice_in_dim(w1, (config().num_moe_emb_chunks - 1) * chunk_dim, chunk_dim, axis=1)
  gmm_fn = expert_gmm.for_local_experts(last_x_chunk, routing, route_metadata)
  mask = (
      jnp.arange(last_x_chunk.shape[0]) < expert_gmm.valid_token_count(last_x_chunk, routing, route_metadata)
      if config().mlp_bias
      else None
  )
  output0, output1 = expert_gmm.up(
      last_x_chunk,
      last_w0,
      last_w1,
      w0_bias,
      w1_bias,
      gmm_fn,
      partial_accum0=ps0,
      partial_accum1=ps1,
      mask=mask,
  )
  return output0, output1, gmm_fn, routing, route_metadata, wo_bias
