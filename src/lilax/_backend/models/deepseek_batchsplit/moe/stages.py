"""Staggered route/compute/unroute stages of the DeepSeek BF16 batch-split MoE block on each shard."""

import functools

import jax
import jax.numpy as jnp

from lilax._backend.models.deepseek_batchsplit.gradients import sum_grads
from lilax._backend.models.deepseek_batchsplit.moe.experts import compute_gating, compute_linear
from lilax._backend.models.deepseek_batchsplit.moe.route_stage import shared_expert_and_route
from lilax._backend.models.deepseek_batchsplit.moe.unroute_stage import (
    unroute_ubatch_fn,
    unroute_ubatch_fn_bwd,
    unroute_ubatch_fn_remat,
)
from lilax._backend.models.deepseek_batchsplit.staggering import staggered_call


def route_compute_unroute(
    xs,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    num_expert_shards,
    capacity_factor,
    use_gather_mosaic_kernel,
    normalization_layer_epsilon,
    dtype,
    config,
):
  """Routes, processes, and unroutes activations."""
  target_length = xs[0].shape[1]
  (
      post_attn_scale,
      (gate_kernel, gate_bias),
      (routed_w0, routed_w1, routed_wo),
      (shared_w0, shared_w1, shared_wo),
  ) = weights

  def route_fn(inputs):
    x, y, selected_experts, weights, group_sizes = shared_expert_and_route(
        inputs,
        post_attn_scale,
        shared_w0,
        shared_w1,
        shared_wo,
        gate_kernel,
        gate_bias,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_expert_shards=num_expert_shards,
        n_routing_groups=n_routing_groups,
        topk_routing_group=topk_routing_group,
        top_k_in_group=top_k_in_group,
        routed_scaling_factor=routed_scaling_factor,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
        config=config,
        normalization_layer_epsilon=normalization_layer_epsilon,
        dtype=dtype,
    )
    return (inputs, x, y, selected_experts, weights, group_sizes), {}

  def compute_fn(inputs):
    moe_inputs, x, y, selected_experts, weights, group_sizes = inputs
    capacity = int(x.shape[0] / num_expert_shards * capacity_factor)
    layer_w0, layer_w1 = compute_gating_fn((x, group_sizes))
    x = compute_linear_fn((layer_w0, layer_w1, weights, group_sizes))
    return (moe_inputs, x, y, selected_experts), {"mlpwi_0": layer_w0[:capacity], "mlpwi_1": layer_w1[:capacity]}

  def compute_gating_fn(inputs):
    x, group_sizes = inputs
    layer_w0, layer_w1 = compute_gating(
        x,
        routed_w0,
        routed_w1,
        group_sizes,
        dtype=dtype,
        config=config,
    )
    return layer_w0, layer_w1

  def compute_linear_fn(inputs):
    layer_w0, layer_w1, weights, group_sizes = inputs
    x = compute_linear(
        layer_w0,
        layer_w1,
        routed_wo,
        group_sizes,
        weights,
        dtype=dtype,
        config=config,
    )
    return x

  def unroute_fn(inputs):
    moe_inputs, x, y, selected_experts = inputs
    return unroute_ubatch_fn(
        moe_inputs,
        x,
        y,
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
        target_length=target_length,
    )

  xs, _ = staggered_call(route_fn, xs)
  xs, res = staggered_call(compute_fn, xs)
  # We don't need the residuals from unroute for the first microbatch since they are calculated earlier in the layer.
  xs[0], _ = unroute_fn(xs[0])
  return xs, res


def route_compute_unroute_bwd(
    residuals,
    outputs_grad,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    num_expert_shards,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    use_gather_mosaic_kernel,
    normalization_layer_epsilon,
    dtype,
    config,
):
  """Performs the backward pass for route_compute_unroute."""
  xs = residuals.pop("mla_out")
  (
      post_attn_scale,
      (gate_kernel, gate_bias),
      (routed_w0, routed_w1, routed_wo),
      (shared_w0, shared_w1, shared_wo),
  ) = weights

  def route_fn_remat(inputs):
    (x, y, selected_experts, weights, group_sizes), shared_expert_and_route_bwd = jax.vjp(
        functools.partial(
            shared_expert_and_route,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_expert_shards=num_expert_shards,
            routed_scaling_factor=routed_scaling_factor,
            n_routing_groups=n_routing_groups,
            topk_routing_group=topk_routing_group,
            top_k_in_group=top_k_in_group,
            expert_axis_name=expert_axis_name,
            use_gather_mosaic_kernel=use_gather_mosaic_kernel,
            config=config,
            normalization_layer_epsilon=normalization_layer_epsilon,
            dtype=dtype,
        ),
        inputs,
        post_attn_scale,
        shared_w0,
        shared_w1,
        shared_wo,
        gate_kernel,
        gate_bias,
    )
    return (inputs, x, y, selected_experts, weights, group_sizes), shared_expert_and_route_bwd

  def route_fn_bwd(inputs):
    (
        inputs_grad1,
        x_grad,
        y_grad,
        selected_experts_grad,
        weights_grad,
        group_sizes_grad,
    ), shared_expert_and_route_bwd = inputs
    (
        inputs_grad2,
        post_attn_scale_grad,
        shared_w0_grad,
        shared_w1_grad,
        shared_wo_grad,
        gate_kernel_grad,
        gate_bias_grad,
    ) = shared_expert_and_route_bwd((x_grad, y_grad, selected_experts_grad, weights_grad, group_sizes_grad))
    return inputs_grad1 + inputs_grad2, (
        post_attn_scale_grad,
        shared_w0_grad,
        shared_w1_grad,
        shared_wo_grad,
        gate_kernel_grad,
        gate_bias_grad,
    )

  def compute_fn_remat(inputs):
    (moe_inputs, x, y, selected_experts, weights, group_sizes), layer_w0_res, layer_w1_res = inputs
    # I think this can be moved into the cond fns but it'd require caching using jit or some other method.
    # This shouldn't result in any duplicate computation though.
    _, compute_gating_bwd = compute_gating_fn_remat((x, group_sizes))

    def remat():
      return compute_gating(x, routed_w0, routed_w1, group_sizes, dtype=dtype, config=config)

    def use_res():
      # Pad to full token buffer size.
      layer_w0 = jnp.pad(layer_w0_res, [(0, x.shape[0] - layer_w0_res.shape[0]), (0, 0)])
      layer_w1 = jnp.pad(layer_w1_res, [(0, x.shape[0] - layer_w1_res.shape[0]), (0, 0)])
      return layer_w0, layer_w1

    tokens_in_capacity = jnp.sum(group_sizes) <= layer_w0_res.shape[0]
    layer_w0, layer_w1 = jax.lax.cond(tokens_in_capacity, use_res, remat)
    x, compute_linear_bwd = compute_linear_fn_remat((layer_w0, layer_w1, weights, group_sizes))
    return (moe_inputs, x, y, selected_experts), (compute_gating_bwd, compute_linear_bwd)

  def compute_fn_bwd(inputs):
    (moe_inputs_grad, x_grad, y_grad, selected_experts_grad), (compute_gating_bwd, compute_linear_bwd) = inputs
    (layer_w0_grad, layer_w1_grad, routed_wo_grad, _, weights_grad) = compute_linear_bwd(x_grad)
    (x_grad, routed_w0_grad, routed_w1_grad, group_sizes_grad) = compute_gating_bwd((layer_w0_grad, layer_w1_grad))
    return (moe_inputs_grad, x_grad, y_grad, selected_experts_grad, weights_grad, group_sizes_grad), (
        routed_w0_grad,
        routed_w1_grad,
        routed_wo_grad,
    )

  def compute_gating_fn_remat(inputs):
    x, group_sizes = inputs
    (layer_w0, layer_w1), compute_gating_bwd = jax.vjp(
        functools.partial(
            compute_gating,
            dtype=dtype,
            config=config,
        ),
        x,
        routed_w0,
        routed_w1,
        group_sizes,
    )
    return (layer_w0, layer_w1), compute_gating_bwd

  def compute_linear_fn_remat(inputs):
    layer_w0, layer_w1, weights, group_sizes = inputs
    x, compute_linear_bwd = jax.vjp(
        functools.partial(
            compute_linear,
            dtype=dtype,
            config=config,
        ),
        layer_w0,
        layer_w1,
        routed_wo,
        group_sizes,
        weights,
    )
    return x, compute_linear_bwd

  def unroute_fn_remat(inputs):
    _, _, _, selected_experts = inputs
    return unroute_ubatch_fn_remat(
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )

  def unroute_fn_bwd(inputs):
    outputs_grad, bwds = inputs
    return unroute_ubatch_fn_bwd(outputs_grad, bwds)

  route_bwds = [None] * len(xs)
  for i, x in enumerate(xs):
    xs[i], route_bwds[i] = route_fn_remat(x)
  compute_bwds = [None] * len(xs)
  for i, x in enumerate(zip(xs, residuals.pop("mlpwi_0"), residuals.pop("mlpwi_1"))):
    xs[i], compute_bwds[i] = compute_fn_remat(x)
  unroute_bwd = unroute_fn_remat(xs[0])
  outputs_grad[0] = unroute_fn_bwd((outputs_grad[0], unroute_bwd))
  compute_ws_grad = []
  for i, g in enumerate(outputs_grad):
    g, ws_grad = compute_fn_bwd((g, compute_bwds[i]))
    outputs_grad[i] = g
    compute_ws_grad.append(ws_grad)
  route_ws_grad = []
  for i, g in enumerate(outputs_grad):
    g, ws_grad = route_fn_bwd((g, route_bwds[i]))
    outputs_grad[i] = g
    route_ws_grad.append(ws_grad)
  (routed_w0_grad, routed_w1_grad, routed_wo_grad) = sum_grads(compute_ws_grad)
  (post_attn_scale_grad, shared_w0_grad, shared_w1_grad, shared_wo_grad, gate_kernel_grad, gate_bias_grad) = sum_grads(
      route_ws_grad
  )
  return outputs_grad, (
      post_attn_scale_grad,
      (gate_kernel_grad, gate_bias_grad),
      (routed_w0_grad, routed_w1_grad, routed_wo_grad),
      (shared_w0_grad, shared_w1_grad, shared_wo_grad),
  )
