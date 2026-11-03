"""Per-microbatch unroute and combine stage of the DeepSeek BF16 batch-split MoE block."""

import functools

import jax
import jax.numpy as jnp

from lilax._backend.models.deepseek_batchsplit.moe.dispatch import route_impl_fwd, unroute


def unroute_ubatch_shard_mapped(
    moe_inputs,
    routed_expert_out,
    shared_expert_out,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    target_length,
    mesh,
    activation_pspec,
):
  """Performs the unroute operation for a single microbatch in a shard map."""
  return jax.shard_map(
      functools.partial(
          unroute_ubatch_fn,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          target_length=target_length,
      ),
      mesh=mesh,
      in_specs=(
          activation_pspec,
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(activation_pspec[0], None),
      ),
      out_specs=(
          activation_pspec,
          {
              "selected_experts": jax.sharding.PartitionSpec(activation_pspec[0], None),
          },
      ),
      check_vma=True,
  )(
      moe_inputs,
      routed_expert_out,
      shared_expert_out,
      selected_experts,
  )


def unroute_ubatch_fn(
    moe_inputs,
    routed_expert_out,
    shared_expert_out,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    target_length,
):
  """Performs the unroute operation for a single microbatch."""
  routed_expert_out = unroute(
      routed_expert_out,
      selected_experts,
      expert_axis_name=expert_axis_name,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )
  expert_out = jnp.reshape(routed_expert_out, (-1, target_length, routed_expert_out.shape[-1])) + shared_expert_out
  return moe_inputs + expert_out, {"selected_experts": selected_experts}


def unroute_ubatch_remat_and_bwd_shard_mapped(
    selected_experts,
    outputs_grad,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    mesh,
    activation_pspec,
):
  """Performs remat and backward pass for unroute_ubatch in a shard map."""

  def unroute_ubatch_remat_and_bwd_fn(
      selected_experts,
      outputs_grad,
      *,
      expert_axis_name,
      use_gather_mosaic_kernel,
  ):
    unroute_bwd = unroute_ubatch_fn_remat(
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )
    return unroute_ubatch_fn_bwd(outputs_grad, unroute_bwd)

  return jax.shard_map(
      functools.partial(
          unroute_ubatch_remat_and_bwd_fn,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
      ),
      mesh=mesh,
      in_specs=(
          jax.sharding.PartitionSpec(activation_pspec[0], None),
          activation_pspec,
      ),
      out_specs=(
          activation_pspec,
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(activation_pspec[0], None),
      ),
      check_vma=True,
  )(
      selected_experts,
      outputs_grad,
  )


def unroute_ubatch_fn_remat(
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  # Since we never need the outputs of unroute in the backward pass, we can just the backward function of unroute directly.
  return functools.partial(
      route_impl_fwd,
      selected_experts=selected_experts,
      expert_axis_name=expert_axis_name,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )


def unroute_ubatch_fn_bwd(
    outputs_grad,
    unroute_bwd,
):
  moe_inputs_grad, expert_out_grad = outputs_grad, outputs_grad
  routed_expert_out_grad, shared_expert_out_grad = expert_out_grad, expert_out_grad
  routed_expert_out_grad = jnp.reshape(routed_expert_out_grad, (-1, routed_expert_out_grad.shape[-1]))
  routed_expert_out_grad, selected_experts_grad = unroute_bwd(routed_expert_out_grad)
  return moe_inputs_grad, routed_expert_out_grad, shared_expert_out_grad, selected_experts_grad
