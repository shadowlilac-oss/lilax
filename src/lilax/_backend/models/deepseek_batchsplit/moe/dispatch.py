"""Expert-parallel token dispatch for the DeepSeek BF16 batch-split schedule."""

import functools

import jax
import jax.numpy as jnp

from lilax._backend.kernels import sort_activations


def route(
    x,
    selected_experts,
    weights,
    group_sizes,
    *,
    expert_axis_name,
    num_experts,
    num_expert_shards,
    use_gather_mosaic_kernel,
):
  """All-gather tokens and then perform local routing."""
  # Communicate local results across the expert axis.
  weights = jax.lax.all_gather(weights, axis_name=expert_axis_name, tiled=True)
  selected_experts = jax.lax.all_gather(selected_experts, axis_name=expert_axis_name, tiled=True)
  group_sizes = jax.lax.psum(group_sizes, axis_name=expert_axis_name)

  expert_id = jax.lax.axis_index(expert_axis_name)
  expert_shift = expert_id * (num_experts / num_expert_shards)
  selected_experts = (selected_experts - expert_shift) % num_experts

  weights = jnp.ravel(weights)[jnp.argsort(jnp.ravel(selected_experts))]
  x = route_impl(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
  )
  group_sizes = jnp.reshape(group_sizes, (num_expert_shards, -1))[expert_id]
  return x, selected_experts, weights, group_sizes


def unroute(
    x,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  """Undo `route()`."""
  return unroute_impl(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def route_impl(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  return route_impl_fwd(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
  )[0]


def route_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  """Routes the activations and all-gathers across the expert axis."""
  x, selected_experts = jax.lax.optimization_barrier((x, selected_experts))
  x = jax.lax.all_gather(x, axis_name=expert_axis_name, tiled=True)
  x = sort_activations.route(
      x,
      selected_experts,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )
  x = jax.lax.optimization_barrier(x)
  return x, selected_experts


def route_impl_bwd(expert_axis_name, use_gather_mosaic_kernel, res, grad):
  selected_experts = res
  return (
      unroute_impl_fwd(
          grad, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
      )[0],
      None,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def unroute_impl(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  return unroute_impl_fwd(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
  )[0]


def unroute_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  """Unroutes the activations and reduce-scatters across the expert axis."""
  x, selected_experts = jax.lax.optimization_barrier((x, selected_experts))
  x = sort_activations.unroute(
      x,
      selected_experts,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )

  # Sum across expert shards.
  x = jax.lax.psum_scatter(x, expert_axis_name, scatter_dimension=0, tiled=True)
  x = jax.lax.optimization_barrier(x)
  return x, selected_experts


def unroute_impl_bwd(expert_axis_name, use_gather_mosaic_kernel, res, grad):
  selected_experts = res
  return (
      route_impl_fwd(
          grad, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
      )[0],
      None,
  )


route_impl.defvjp(route_impl_fwd, route_impl_bwd)
unroute_impl.defvjp(unroute_impl_fwd, unroute_impl_bwd)
