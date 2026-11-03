"""Unscanning and rescanning of decoder layer groups in train state params."""

import functools

import jax
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp


def unscan_train_state_params(params, sharding, mesh, scan_axis, layer_groups):
  """
  Unrolls scanned parameter groups into per-layer entries.

  Args:
    train_state: training state with scanned `params`
    mesh: the mesh to use for sharding output
    scan_axis: axis along which scanning was applied (usually 0)
    layer_groups: list of tuples like:
      [("dense_layers", 4), ("moe_layers", 12)]
  """
  params_copy = params.unfreeze() if hasattr(params, "unfreeze") else params
  decoder = params_copy["params"]["decoder"]
  sharding_decoder = sharding["params"]["decoder"]

  # Helper function to remove the scan axis from a PartitionSpec
  def strip_scan_axis(pspec: P) -> P:
    """Removes the element at `scan_axis` from a PartitionSpec tuple."""
    spec_tuple = tuple(pspec)
    return P(*(spec_tuple[:scan_axis] + spec_tuple[scan_axis + 1 :]))

  for layer_name, num_layers in layer_groups:
    scanned_layers = decoder[layer_name]
    scanned_sharding = sharding_decoder[layer_name]

    # 1. Compute the target sharding for a single, unscanned layer.
    # This is done once per layer group, with no expensive compilation.
    unscanned_sharding_spec = jax.tree_util.tree_map(
        strip_scan_axis, jax.tree_util.tree_map(lambda x: x.spec, scanned_sharding)
    )
    unscanned_sharding = jax.tree_util.tree_map(lambda ps: jax.sharding.NamedSharding(mesh, ps), unscanned_sharding_spec)

    # 2. Create a list of PyTrees, one for each layer, by slicing the original.
    # This is more direct than repeatedly calling a JIT'd function.
    layer_pytrees = [
        jax.tree_util.tree_map(functools.partial(jnp.take, indices=i, axis=scan_axis), scanned_layers)
        for i in range(num_layers)
    ]

    # 3. Reshard each layer's PyTree and assign it to the new key.
    for i, layer_params in enumerate(layer_pytrees):
      resharded_params = jax.device_put(layer_params, unscanned_sharding)
      decoder[f"{layer_name}_{i}"] = resharded_params

    # 4. Clean up the original scanned parameter to free up memory.
    del decoder[layer_name]


def rescan_train_state_params(params, source_shardings, scan_axis, layer_groups):
  decoder = params["params"]["decoder"]
  sharding = source_shardings["params"]["decoder"]

  for layer_name, num_layers in layer_groups:

    def stack_layers(*layers):
      return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=scan_axis), *layers)

    compiled_stack = jax.jit(stack_layers, out_shardings=sharding[layer_name])
    layer_list = [decoder.pop(f"{layer_name}_{i}") for i in range(num_layers)]
    scanned = compiled_stack(*layer_list)
    decoder[layer_name] = scanned
