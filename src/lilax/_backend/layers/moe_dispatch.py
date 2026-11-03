"""MoE token dispatch operations."""

import jax
import jax.numpy as jnp

from lilax._backend.kernels import sort_activations


def _truncate_matrix(all_shards_group_sizes: jax.Array, buffer_size: int) -> jax.Array:
  """Truncates the traffic matrix to fit in buffer_size on receiver side.

  When ragged_buffer_factor > 0, the receiver buffer has a fixed capacity
  (buffer_size). Due to routing imbalance, some shards might receive more tokens
  than this capacity. We use a prefix sum to deterministically truncate the
  received tokens on all shards, ensuring we don't write out of bounds.
  """
  cumsum = jnp.cumsum(all_shards_group_sizes, axis=0)
  clamped_cumsum = jnp.minimum(cumsum, buffer_size)
  clamped_cumsum_extended = jnp.concatenate(
      [
          jnp.zeros((1, all_shards_group_sizes.shape[1]), dtype=clamped_cumsum.dtype),
          clamped_cumsum,
      ],
      axis=0,
  )
  return jnp.diff(clamped_cumsum_extended, axis=0)


def _sort_activations(
    inputs: jax.Array,
    sort_indices: jax.Array,
    use_custom_vjp: bool,
) -> jax.Array:
  """Sort activations by `sort_indices`.

  If `use_custom_vjp=True`, then we use a custom backward pass that
  reverses the sort order. Specifically, this unsort operation is simply a sort
  with `jnp.argsort(sort_indices)` as the sort indices. This is only needed in
  the case where the compiler generates a less efficient backward pass op.

  Note that `use_custom_vjp=True` assumes that `sort_indices` is a permutation
  of `jnp.arange(inputs.shape[0])`.

  Args:
    inputs: `(tokens, ...)`-shaped array of input activations to sort.
    sort_indices: `(tokens,)`-shaped array containing the sort order.
    use_custom_vjp: Whether to use the explicit backward pass.

  Returns:
    `(tokens, ...)`-shaped array of input activations sorted by `sort_indices`.
  """
  assert inputs.shape[0] == sort_indices.shape[0]

  with jax.named_scope("sort_activations"):
    if use_custom_vjp:
      return _sort_activations_custom(inputs, sort_indices)
    return inputs[sort_indices, ...]


def _route_activations(inputs: jax.Array, selected_experts: jax.Array) -> jax.Array:
  """Groups token activations by expert without materializing Top-K copies."""
  selected_experts = selected_experts.reshape((inputs.shape[0], -1))
  # When use_gather_mosaic_kernel=False, use the general JAX backward, which
  # gathers and sums gradients from all expert copies of each token. The
  # optimized Mosaic gather-reduce kernel currently requires exactly 8 selected
  # experts per token and is enabled only by the specialized batch-split path.
  return sort_activations.route(inputs, selected_experts, use_gather_mosaic_kernel=False)


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
  """Sort functions with custom vjp."""
  return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Forward pass of the custom vjp for `_sort_activations()`."""
  return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
  """Backward pass of the custom vjp for `_sort_activations()`."""
  sort_indices = residuals
  return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)
