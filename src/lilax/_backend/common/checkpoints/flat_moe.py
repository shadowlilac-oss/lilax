"""Restore and reshape flat MoE checkpoint weights."""

import jax
import jax.numpy as jnp

_FLAT_MOE_LEAVES = ("wi_0", "wi_1", "wo")


def _flat_moe_restore_plan(want, stored_weights):
  plan = {}

  def _walk(want_node, stored_node, path):
    if isinstance(want_node, dict):
      for key, value in want_node.items():
        _walk(value, stored_node.get(key) if isinstance(stored_node, dict) else None, path + (key,))
      return
    if not path or path[-1] not in _FLAT_MOE_LEAVES or stored_node is None:
      return
    want_shape = tuple(getattr(want_node, "shape", ()) or ())
    stored_shape = tuple(getattr(stored_node, "shape", ()) or ())
    if not want_shape or not stored_shape or want_shape == stored_shape or len(want_shape) != len(stored_shape) - 1:
      return
    if len(stored_shape) == 4:
      experts, layers, rows, cols = stored_shape
      flat_shape = (experts * rows, layers, cols)
    elif len(stored_shape) == 3:
      experts, rows, cols = stored_shape
      flat_shape = (experts * rows, cols)
    else:
      return
    if want_shape != flat_shape:
      return
    sharding = getattr(want_node, "sharding", None)
    mesh = getattr(sharding, "mesh", None)
    spec = getattr(sharding, "spec", None)
    if mesh is None or spec is None:
      return
    dim0 = spec[0] if len(spec) > 0 else None
    dim0_axes = () if dim0 is None else ((dim0,) if isinstance(dim0, str) else tuple(dim0))
    expert_axis = dim0_axes[0] if len(dim0_axes) > 1 else None
    rest = dim0_axes[1:] if len(dim0_axes) > 1 else dim0_axes
    rest_spec = None if not rest else (rest[0] if len(rest) == 1 else rest)
    tail = tuple(spec)[1:] + (None,) * max(0, len(flat_shape) - len(spec))
    if len(stored_shape) == 4:
      stored_spec = jax.sharding.PartitionSpec(expert_axis, tail[0], rest_spec, tail[1])
    else:
      stored_spec = jax.sharding.PartitionSpec(expert_axis, rest_spec, tail[0])
    stored_sds = jax.ShapeDtypeStruct(stored_shape, want_node.dtype, sharding=jax.sharding.NamedSharding(mesh, stored_spec))
    plan[path] = (want_node, stored_sds)

  _walk(want, stored_weights, ())
  return plan


def _flat_moe_convert(x, target):
  out_sharding = target.sharding
  mesh = out_sharding.mesh
  spec = tuple(out_sharding.spec) + (None,) * (len(target.shape) - len(out_sharding.spec))
  if x.ndim == 4:
    experts, layer_count, rows, cols = x.shape
    layer_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(spec[0], spec[2]))

    def _layer_fn(array, index):
      return jax.lax.dynamic_index_in_dim(array, index, axis=1, keepdims=False).reshape(experts * rows, cols).astype(target.dtype)

    layer_fn = jax.jit(_layer_fn, out_shardings=layer_sharding)
    layers = [layer_fn(x, jnp.int32(index)) for index in range(layer_count)]
    x.delete()
    stacked = jax.jit(lambda *arrays: jnp.stack(arrays, axis=1), out_shardings=out_sharding)(*layers)
    for layer in layers:
      layer.delete()
    return stacked
  experts, rows, cols = x.shape
  result = jax.jit(lambda array: array.reshape(experts * rows, cols).astype(target.dtype), out_shardings=out_sharding)(x)
  x.delete()
  return result
