"""Per-layer slicing of stacked weights for the DeepSeek BF16 batch-split schedule."""

import jax


def extract_layer_weights(all_weights, layer_idx, layer_axis):
  """Extracts the weights for given layer."""
  return jax.tree.map(
      lambda x: jax.lax.dynamic_index_in_dim(x, layer_idx, axis=layer_axis, keepdims=False),
      all_weights,
  )


def insert_layer_ws_grad(all_ws_grad, ws_grad, layer_idx, layer_axis):
  """Inserts the weight gradients for given layer."""
  return jax.tree.map(
      lambda x, y: jax.lax.dynamic_update_index_in_dim(x, y, layer_idx, axis=layer_axis),
      all_ws_grad,
      ws_grad,
  )
