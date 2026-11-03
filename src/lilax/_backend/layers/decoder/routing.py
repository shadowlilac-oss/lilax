"""Per-layer routing and attention metadata for decoder execution."""

from typing import Any

import jax
import jax.numpy as jnp


def forced_routed_experts_per_layer(forced_routed_experts: jax.Array, num_layers: int) -> jax.Array:
  """Normalizes a batch's forced_routed_experts to a per-decoder-layer stack.

  Every supported architecture is homogeneous-MoE, so the layer axis is simply
  the decoder layer index.

  Args:
    forced_routed_experts: `[batch, seq, num_layers, top_k]` (per-layer) or
      `[batch, seq, top_k]` (broadcast to every layer).
    num_layers: Total decoder layers.

  Returns:
    Array shaped `[num_layers, batch, seq, top_k]`.

  Raises:
    ValueError: If forced_routed_experts is not 3D or 4D, or its layer axis
      does not match `num_layers`.
  """
  fre = forced_routed_experts
  if fre.ndim not in (3, 4):
    raise ValueError(
        "forced_routed_experts must be [batch, seq, top_k] (3D, broadcast to"
        " every layer) or [batch, seq, num_layers, top_k] (4D, per-layer); got"
        f" ndim={fre.ndim} with shape {fre.shape}."
    )
  if fre.ndim == 4:
    if fre.shape[2] != num_layers:
      raise ValueError(
          "forced_routed_experts layer axis must equal the number of decoder"
          f" layers ({num_layers}); got {fre.shape[2]} (shape {fre.shape})."
      )
    # [batch, seq, num_layers, top_k] -> [num_layers, batch, seq, top_k]
    return jnp.moveaxis(fre, 2, 0)
  # [batch, seq, top_k]: same forced routing for every layer.
  return jnp.broadcast_to(fre[None], (num_layers,) + fre.shape)


def reshape_forced_routed_experts_for_scan(
    forced_routed_experts: jax.Array,
    num_layers: int,
    scan_length: int,
    layers_per_cycle: int,
) -> jax.Array:
  """Reshapes a batch's forced_routed_experts into jax.lax.scan's xs layout.

  The outer scan runs `scan_length` iterations, each covering a cycle of
  `layers_per_cycle` decoder layers.

  Args:
    forced_routed_experts: `[batch, seq, num_layers, top_k]` (per-layer) or
      `[batch, seq, top_k]` (broadcast to every layer).
    num_layers: Total decoder layers (`scan_length * layers_per_cycle`).
    scan_length: Number of outer scan iterations.
    layers_per_cycle: Decoder layers per cycle.

  Returns:
    Array shaped `[scan_length, layers_per_cycle, batch, seq, top_k]`.

  Raises:
    ValueError: If forced_routed_experts is not 3D or 4D.
  """
  fre = forced_routed_experts_per_layer(forced_routed_experts, num_layers)
  return jnp.reshape(fre, (scan_length, layers_per_cycle) + fre.shape[1:])


def layer_attention_metadata(attention_metadata: dict[Any, Any], layer_idx: int) -> Any:
  """Returns decoder layer `layer_idx`'s entry of per-layer attention metadata.

  vLLM hands hybrid models (e.g. Qwen3-Next / Qwen3.5) a mapping from layer
  name (`layer.{i}`) to the metadata of that layer's KV-cache group; integer
  keys are accepted too. A layer with no entry gets None.

  Args:
    attention_metadata: The per-layer metadata mapping.
    layer_idx: Global index of the decoder layer.

  Returns:
    The layer's metadata, or None when the mapping has no entry for it.
  """
  return attention_metadata.get(f"layer.{layer_idx}", attention_metadata.get(layer_idx))
