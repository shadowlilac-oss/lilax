"""Dropless MoE layer for the DeepSeek FP8 batch-split schedule."""

import functools

import jax
import jax.numpy as jnp
from lilax._backend.layers import moe as moe_lib
from lilax._backend.models.deepseek_batchsplit_fp8.moe.dispatch import route, unroute
from lilax._backend.models.deepseek_batchsplit_fp8.moe.experts import compute, shared_expert
from lilax._backend.models.deepseek_batchsplit_fp8.moe.gating import expert_selection
from lilax._backend.models.deepseek_batchsplit_fp8.sharding import with_data_parallel_constraint
from lilax._backend.models.deepseek_batchsplit_fp8.staggering import staggered_call
from lilax._backend.distributed.sharding import logical_to_spec


def moe(
    inputs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    quant,
):
  """Performs dropless MoE with tensor/expert parallelism."""
  xs, ys = list(zip(*inputs))
  ys = with_data_parallel_constraint(
      process_activations(
          ys,
          weights,
          mesh=mesh,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          config=config,
          quant=quant,
      ),
      mesh,
  )
  return [x + y for x, y in zip(xs, ys)]


def route_compute_unroute(
    xs,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    mesh,
    quant,
):
  """Routes, processes, and unroutes activations."""
  orig_shape = xs[0].shape
  (
      (gate_kernel, gate_bias),
      (routed_w0, routed_w1, routed_wo),
      (shared_w0, shared_w1, shared_wo),
  ) = weights

  def route_fn(inputs):
    # Shared expert.
    y = shared_expert(inputs, shared_w0, shared_w1, shared_wo, quant=quant)

    inputs = jnp.reshape(inputs, (-1, inputs.shape[-1]))
    selected_experts, weights, group_sizes = expert_selection(
        inputs,
        gate_kernel,
        gate_bias,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        quant=quant,
    )
    x, selected_experts, weights, group_sizes = route(
        inputs,
        selected_experts,
        weights,
        group_sizes,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )
    return x, y, selected_experts, weights, group_sizes

  def compute_fn(inputs):
    x, y, selected_experts, weights, group_sizes = inputs
    x = compute(
        x,
        routed_w0,
        routed_w1,
        routed_wo,
        group_sizes,
        weights,
        config=config,
        mesh=mesh,
    )
    return x, y, selected_experts

  def unroute_fn(inputs):
    x, y, selected_experts = inputs
    x = unroute(
        x,
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )
    return jnp.reshape(x, orig_shape) + y

  xs = staggered_call(route_fn, xs)
  xs = staggered_call(compute_fn, xs)
  xs = staggered_call(unroute_fn, xs)
  return xs


def process_activations(
    xs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    quant,
):
  """Processes activations, which are fully sharded on the batch axis, with tensor/expert sharded weights."""
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  if config.quantization and config.use_qwix_quantization:
    gating_pspec, linear_pspec = moe_lib.get_batchsplit_init_kernel_axes()
    gating_pspec = logical_to_spec(gating_pspec)
    linear_pspec = logical_to_spec(linear_pspec)
  else:
    gating_pspec = jax.sharding.PartitionSpec(None, None, expert_axis_name)
    linear_pspec = jax.sharding.PartitionSpec(None, expert_axis_name, None)
  return jax.shard_map(
      functools.partial(
          route_compute_unroute,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          config=config,
          mesh=mesh,
          quant=quant,
      ),
      mesh=mesh,
      in_specs=(
          [activation_pspec] * len(xs),
          (
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  gating_pspec,
                  gating_pspec,
                  linear_pspec,
              ),
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
              ),
          ),
      ),
      out_specs=activation_pspec,
      check_vma=False,
  )([x.astype(config.dtype) for x in xs], weights)
