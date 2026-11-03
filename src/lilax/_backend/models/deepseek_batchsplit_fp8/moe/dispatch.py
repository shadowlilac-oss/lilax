"""Expert-parallel token dispatch for the DeepSeek FP8 batch-split schedule."""

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
    use_gather_mosaic_kernel,
):
  """All-gather tokens and then perform local routing."""
  # Communicate local results across the expert axis.
  x = jax.lax.all_gather(x, axis_name=expert_axis_name, tiled=True)
  weights = jax.lax.all_gather(weights, axis_name=expert_axis_name, tiled=True)
  selected_experts = jax.lax.all_gather(selected_experts, axis_name=expert_axis_name, tiled=True)
  group_sizes = jax.lax.psum(group_sizes, axis_name=expert_axis_name)

  # Sort the gathered tokens and weights.
  weights = jnp.ravel(weights)[jnp.argsort(jnp.ravel(selected_experts))]
  x = sort_activations.route(
      x,
      selected_experts,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )

  return x, selected_experts, weights, group_sizes


def unroute(
    x,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  """Undo `route()`."""
  # Unsort the output.
  x = sort_activations.unroute(
      x,
      selected_experts,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )

  # Sum across expert shards.
  return jax.lax.psum_scatter(x, expert_axis_name, scatter_dimension=0, tiled=True)
