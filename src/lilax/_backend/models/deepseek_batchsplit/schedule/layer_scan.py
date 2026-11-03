"""Pipelined layer-scan custom VJP for the DeepSeek BF16 batch-split schedule."""

import functools

import jax
import jax.numpy as jnp

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
from lilax._backend.models.deepseek_batchsplit.scheduling import scheduling_group
from lilax._backend.models.deepseek_batchsplit.weights.fetch import fetch_weights
from lilax._backend.models.deepseek_batchsplit.weights.gather import gather_weights
from lilax._backend.models.deepseek_batchsplit.weights.grad_reduction import (
    all_reduce_ws_grad_dcn,
    reduce_scatter_ws_grad,
)
from lilax._backend.models.deepseek_batchsplit.weights.layer_slices import extract_layer_weights, insert_layer_ws_grad


def scan_batch_split_layers(
    inputs,
    params,
    positions,
    *,
    mesh,
    cfg,
    num_layers,
):
  """Scans the layers with batch-split schedule."""
  assert num_layers >= 4, "num_layers must be at least 4 for batch-split schedule."

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
  def process_all_layers(inputs, all_weights, yarn_freqs):
    return process_all_layers_fwd(inputs, all_weights, yarn_freqs)[0]

  def process_all_layers_fwd(inputs, all_weights, yarn_freqs):
    def process_layer_scannable(carry, layer_idx, group_id):
      inputs, ws = carry
      # Prefetch weights for next layer.
      with scheduling_group(group_id=group_id):
        next_ws = gather_weights(extract_layer_weights(all_weights, layer_idx + 1, cfg.param_scan_axis), mesh)
      # Combine for previous layer's second microbatch.
      moe_inputs, routed_expert_out, shared_expert_out, selected_experts = inputs[1]
      inputs[1], unroute_res = unroute_ubatch_shard_mapped(
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
      # Current layer computation.
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
      # Offload to host memory.
      for residual_name in ("mlpwi_0", "mlpwi_1", "attn_out", "layer_inputs"):
        r = res.pop(residual_name)
        r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("pinned_host")), r)
        res[residual_name] = r
      return (outputs, next_ws), (res, unroute_res)

    # Prologue: do first two layers and prefetch weights for third layer.
    with scheduling_group(group_id=40):
      first_ws = gather_weights(extract_layer_weights(all_weights, 0, cfg.param_scan_axis), mesh)
    with scheduling_group(group_id=41):
      second_ws = gather_weights(extract_layer_weights(all_weights, 1, cfg.param_scan_axis), mesh)
    second_inputs, first_res = batch_split_schedule(
        inputs,
        first_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    # Offload first layer residuals to host memory.
    for residual_name in ("mlpwi_0", "mlpwi_1", "attn_out", "layer_inputs"):
      r = first_res.pop(residual_name)
      r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("pinned_host")), r)
      first_res[residual_name] = r
    third_carry, (second_res, second_unroute_res) = process_layer_scannable((second_inputs, second_ws), 1, group_id=42)
    # Scan middle layers.
    last_last_carry, (middle_res, middle_unroute_res) = jax.lax.scan(
        functools.partial(process_layer_scannable, group_id=43),
        third_carry,
        jnp.arange(2, num_layers - 2),
    )
    # Epilogue: do last two layers without prefetching weights and finish second microbatch.
    (last_inputs, last_ws), (last_last_res, last_last_unroute_res) = process_layer_scannable(
        last_last_carry, num_layers - 2, group_id=44
    )
    moe_inputs, routed_expert_out, shared_expert_out, selected_experts = last_inputs[1]
    last_inputs[1], last_unroute_res = unroute_ubatch_shard_mapped(
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
    outputs, last_res = batch_split_schedule(
        last_inputs,
        last_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    moe_inputs, routed_expert_out, shared_expert_out, selected_experts = outputs[1]
    outputs[1], epilogue_unroute_res = unroute_ubatch_shard_mapped(
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
        first_res,
        second_res,
        middle_res,
        last_last_res,
        last_res,
        second_unroute_res,
        middle_unroute_res,
        last_last_unroute_res,
        last_unroute_res,
        epilogue_unroute_res,
        last_ws,
        all_weights,
        yarn_freqs,
    )

  def process_all_layers_bwd(res, g):
    (
        first_res,
        second_res,
        middle_res,
        last_last_res,
        last_res,
        second_unroute_res,
        middle_unroute_res,
        last_last_unroute_res,
        last_unroute_res,
        epilogue_unroute_res,
        last_ws,
        all_weights,
        yarn_freqs,
    ) = res
    initial_ws_grad = jax.tree.map(jnp.zeros_like, all_weights)

    def process_layer_bwd_scannable(carry, res_and_layer_idx, group_id):
      g, ws, next_next_ws_grad, next_ws_grad, all_layer_ws_grad = carry
      res, unroute_res, layer_idx = res_and_layer_idx
      # Prefetch weights and post-scatter weight grads.
      with scheduling_group(group_id=group_id):
        prev_ws = gather_weights(extract_layer_weights(all_weights, layer_idx - 1, cfg.param_scan_axis), mesh)
        next_ws_grad = reduce_scatter_ws_grad(next_ws_grad, mesh)
      next_next_ws_grad = all_reduce_ws_grad_dcn(next_next_ws_grad, mesh)
      all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, next_next_ws_grad, layer_idx + 2, cfg.param_scan_axis)
      # Get residuals from host.
      for residual_name in ("mlpwi_0", "mlpwi_1", "attn_out", "layer_inputs"):
        r = res.pop(residual_name)
        r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device")), r)
        res[residual_name] = r

      # Current layer computation.
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
      g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
          unroute_res["selected_experts"],
          g[1],
          expert_axis_name="expert",
          use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
          mesh=mesh,
          activation_pspec=activation_pspec,
      )
      return (g, prev_ws, next_ws_grad, ws_grad, all_layer_ws_grad), None

    # Prologue: do computation for last two layers and prefetch weights.
    with scheduling_group(group_id=50):
      prev_ws = gather_weights(extract_layer_weights(all_weights, num_layers - 2, cfg.param_scan_axis), mesh)
    g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
        epilogue_unroute_res["selected_experts"],
        g[1],
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    g, ws_grad = batch_split_schedule_bwd(
        last_res,
        g,
        last_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
        last_unroute_res["selected_experts"],
        g[1],
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    with scheduling_group(group_id=51):
      prev_prev_ws = gather_weights(extract_layer_weights(all_weights, num_layers - 3, cfg.param_scan_axis), mesh)
      ws_grad = reduce_scatter_ws_grad(ws_grad, mesh)
    # Get residuals from host.
    for residual_name in ("mlpwi_0", "mlpwi_1", "attn_out", "layer_inputs"):
      r = last_last_res.pop(residual_name)
      r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device")), r)
      last_last_res[residual_name] = r
    g, prev_ws_grad = batch_split_schedule_bwd(
        last_last_res,
        g,
        prev_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
        last_last_unroute_res["selected_experts"],
        g[1],
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    # Scan middle layers.
    (g, second_ws, fourth_ws_grad, third_ws_grad, all_layer_ws_grad), _ = jax.lax.scan(
        functools.partial(process_layer_bwd_scannable, group_id=52),
        (g, prev_prev_ws, ws_grad, prev_ws_grad, initial_ws_grad),
        (middle_res, middle_unroute_res, jnp.arange(2, num_layers - 2)),
        reverse=True,
    )
    # Epilogue: do third and second layer computation and post-scatter weight grads.
    (g, first_ws, third_ws_grad, second_ws_grad, all_layer_ws_grad), _ = process_layer_bwd_scannable(
        (g, second_ws, fourth_ws_grad, third_ws_grad, all_layer_ws_grad),
        (second_res, second_unroute_res, 1),
        group_id=53,
    )
    with scheduling_group(group_id=54):
      second_ws_grad = reduce_scatter_ws_grad(second_ws_grad, mesh)
    third_ws_grad = all_reduce_ws_grad_dcn(third_ws_grad, mesh)
    all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, third_ws_grad, 2, cfg.param_scan_axis)
    # Get residuals from host.
    for residual_name in ("mlpwi_0", "mlpwi_1", "attn_out", "layer_inputs"):
      r = first_res.pop(residual_name)
      r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device")), r)
      first_res[residual_name] = r
    g, ws_grad = batch_split_schedule_bwd(
        first_res,
        g,
        first_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    second_ws_grad = all_reduce_ws_grad_dcn(second_ws_grad, mesh)
    all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, second_ws_grad, 1, cfg.param_scan_axis)
    with scheduling_group(group_id=55):
      ws_grad = reduce_scatter_ws_grad(ws_grad, mesh)
    ws_grad = all_reduce_ws_grad_dcn(ws_grad, mesh)
    all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, ws_grad, 0, cfg.param_scan_axis)
    return g, all_layer_ws_grad, None

  process_all_layers.defvjp(process_all_layers_fwd, process_all_layers_bwd)

  inputs = jax.shard_map(
      functools.partial(split, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=activation_pspec,
      out_specs=[activation_pspec] * cfg.batch_split_factor,
  )(inputs)
  yarn_freqs = split(yarn_freqs, split_factor=cfg.batch_split_factor)
  outputs = process_all_layers(inputs, all_weights, yarn_freqs)
  outputs = jax.shard_map(
      functools.partial(merge, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=([activation_pspec] * cfg.batch_split_factor,),
      out_specs=activation_pspec,
  )(outputs)
  return outputs
