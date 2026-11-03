"""Single-layer custom VJP for the DeepSeek BF16 batch-split schedule."""

import functools

import jax

from lilax._backend.models.deepseek_batchsplit.attention.rope import initialize_yarn_freqs, initialize_yarn_mask
from lilax._backend.models.deepseek_batchsplit.attention.splash import init_splash_kernel
from lilax._backend.models.deepseek_batchsplit.microbatch import merge, split
from lilax._backend.models.deepseek_batchsplit.moe.unroute_stage import (
    unroute_ubatch_remat_and_bwd_shard_mapped,
    unroute_ubatch_shard_mapped,
)
from lilax._backend.models.deepseek_batchsplit.schedule.decoder_layer import (
    batch_split_schedule,
    batch_split_schedule_bwd,
)
from lilax._backend.models.deepseek_batchsplit.weights.fetch import fetch_weights
from lilax._backend.models.deepseek_batchsplit.weights.gather import gather_weights
from lilax._backend.models.deepseek_batchsplit.weights.grad_reduction import (
    all_reduce_ws_grad_dcn,
    reduce_scatter_ws_grad,
)


def batch_split_layer(
    inputs,
    params,
    positions,
    *,
    mesh,
    cfg,
):
  """Processes a single layer with batch-split schedule."""
  all_weights = fetch_weights(params, cfg.dtype)
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "expert"),
      None,
      None,
  )
  # The data mesh axis can be size 1, but we still want to keep it in the
  # partition spec because the code supports multi-slice runs and thus
  # expects the data axis to be present.
  inputs = jax.reshard(inputs, jax.sharding.NamedSharding(mesh, activation_pspec))
  yarn_freqs = initialize_yarn_freqs(
      positions=positions,
      embedding_dims=cfg.qk_rope_head_dim,
      rope_theta=cfg.rope_max_timescale,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      beta_fast=cfg.beta_fast,
      beta_slow=cfg.beta_slow,
      rope_factor=cfg.rope_factor,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  yarn_mask = initialize_yarn_mask(cfg.qk_rope_head_dim)
  splash_kernel = init_splash_kernel(cfg)

  @jax.custom_vjp
  def process_layer(inputs, all_weights, yarn_freqs):
    return process_layer_fwd(inputs, all_weights, yarn_freqs)[0]

  def process_layer_fwd(inputs, weights, yarn_freqs):
    ws = gather_weights(weights, mesh)
    outputs, res = batch_split_schedule(
        inputs,
        ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    moe_inputs, routed_expert_out, shared_expert_out, selected_experts = outputs[1]
    outputs[1], unroute_res = unroute_ubatch_shard_mapped(
        moe_inputs,
        routed_expert_out,
        shared_expert_out,
        selected_experts,
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        target_length=cfg.max_target_length,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    return outputs, (
        res,
        unroute_res,
        ws,
        yarn_freqs,
    )

  def process_layer_bwd(res, g):
    (
        res,
        unroute_res,
        ws,
        yarn_freqs,
    ) = res

    g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
        unroute_res["selected_experts"],
        g[1],
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    g, ws_grad = batch_split_schedule_bwd(
        res,
        g,
        ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    weights_grad = reduce_scatter_ws_grad(ws_grad, mesh)
    weights_grad = all_reduce_ws_grad_dcn(weights_grad, mesh)
    return g, weights_grad, None

  process_layer.defvjp(process_layer_fwd, process_layer_bwd)

  inputs = jax.shard_map(
      functools.partial(split, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=activation_pspec,
      out_specs=[activation_pspec] * cfg.batch_split_factor,
  )(inputs)
  yarn_freqs = split(yarn_freqs, split_factor=cfg.batch_split_factor)
  outputs = process_layer(inputs, all_weights, yarn_freqs)
  outputs = jax.shard_map(
      functools.partial(merge, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=([activation_pspec] * cfg.batch_split_factor,),
      out_specs=activation_pspec,
  )(outputs)
  return outputs
