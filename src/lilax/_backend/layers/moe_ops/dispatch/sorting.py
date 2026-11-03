"""Sorting of token activations into expert order and back."""

import jax
import jax.numpy as jnp

from lilax._backend.common import common_types as ctypes
from lilax._backend.kernels import sort_activations


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


def _unsort_and_weight_sum(
    intermediate,
    sorted_selected_experts,
    weights,
    num_experts_per_tok,
    use_custom_sort_vjp,
    matmul_precision,
    decoder_block,
    float32_weight_sum,
):
  """Restores token order of the expert outputs and sums each token's top-k outputs by their routing weights."""
  unsort_intermediate = _sort_activations(intermediate, jnp.argsort(sorted_selected_experts), use_custom_sort_vjp)
  reshaped_weights = jnp.reshape(weights, (-1, num_experts_per_tok))
  reshaped_intermediate = jnp.reshape(unsort_intermediate, (reshaped_weights.shape[0], num_experts_per_tok, -1))
  with jax.named_scope("weight_sum"):
    matmul_precision = jax.lax.Precision(matmul_precision)
    if decoder_block == ctypes.DecoderBlockType.LLAMA4:
      reshaped_weights = jnp.ones_like(reshaped_weights)
    if float32_weight_sum:
      reshaped_intermediate = reshaped_intermediate.astype(jnp.float32)
      reshaped_weights = reshaped_weights.astype(jnp.float32)
    return jnp.einsum("BKE,BK -> BE", reshaped_intermediate, reshaped_weights, precision=matmul_precision)
