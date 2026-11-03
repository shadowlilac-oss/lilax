"""Weight preparation for the DeepSeek BF16 batch-split schedule."""

from typing import Sequence

import jax
import jax.numpy as jnp


def fetch_weights(params, dtype):
  """Fetches weights from params in the proper format for batch-split schedule."""
  return jax.tree.map(
      # If x is a LogicallyPartitioned array, then x.value is the underlying
      # array. If not, use the array directly.
      lambda x: jnp.asarray(getattr(x, "value", x)[...], dtype),
      (
          (
              params["pre_self_attention_layer_norm"]["scale"],
              (
                  params["self_attention"]["wq_a"]["kernel"],
                  params["self_attention"]["wq_b"]["kernel"],
                  params["self_attention"]["q_norm"]["scale"],
                  params["self_attention"]["wkv_a"]["kernel"],
                  params["self_attention"]["wkv_b"]["kernel"],
                  params["self_attention"]["kv_norm"]["scale"],
                  params["self_attention"]["out"]["kernel"],
              ),
          ),
          (
              params["post_self_attention_layer_norm"]["scale"],
              (
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["kernel"],
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["bias"],
              ),
              (
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"],
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_1"],
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wo"],
              ),
              (
                  params["DeepSeekMoeBlock_0"]["shared_experts"]["wi_0"]["kernel"],
                  params["DeepSeekMoeBlock_0"]["shared_experts"]["wi_1"]["kernel"],
                  params["DeepSeekMoeBlock_0"]["shared_experts"]["wo"]["kernel"],
              ),
          ),
      ),
      is_leaf=lambda x: not isinstance(x, Sequence),
  )


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
