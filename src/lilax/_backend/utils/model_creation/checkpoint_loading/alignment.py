"""Align restored checkpoint arrays to the model's shapes and sharding."""

import jax
import jax.numpy as jnp
import numpy as np

from lilax._backend.integration.vllm.convert_utils import _partition_size

_VLLM_REPEAT_AXES = frozenset({"kv_heads", ("expert", "model")})
_VLLM_ZERO_PAD_AXES = frozenset({"mlp_moe", "activation_mlp", ("attn_dp", "model")})


def _normalize_logical_axes(axes):
  """Convert logical partition axes to a tuple when available."""
  if axes is None:
    return None
  if isinstance(axes, jax.sharding.PartitionSpec):
    return tuple(axes)
  if isinstance(axes, (tuple, list)):
    return tuple(axes)
  return None


def _zero_pad_axis(arr, axis, extra):
  """Append zeros to an axis, padding local shards when possible."""
  if extra == 0:
    return arr

  arr_sharding = getattr(arr, "sharding", None)
  pad_width = [(0, 0)] * arr.ndim
  if isinstance(arr_sharding, jax.sharding.NamedSharding):
    spec = arr_sharding.spec
    partition = spec[axis] if axis < len(spec) else None
    shards_along_axis = _partition_size(partition, arr_sharding.mesh)
    if shards_along_axis > 1:
      if extra % shards_along_axis != 0:
        raise ValueError(
            f"Cannot per-shard zero-pad axis {axis}: extra={extra} not divisible by "
            f"shards_along_axis={shards_along_axis} (sharding spec entry {partition!r})."
        )
      pad_width[axis] = (0, extra // shards_along_axis)

      def _pad_local(value):
        return jnp.pad(value, pad_width)

      return jax.shard_map(
          _pad_local, mesh=arr_sharding.mesh, in_specs=spec, out_specs=spec, check_vma=False
      )(arr)

  pad_width[axis] = (0, extra)
  return jnp.pad(arr, pad_width)


def _align_checkpoint_to_model_shapes(ckpt_arr, model_arr, logical_axes=None):
  """Align a checkpoint array to the model shape and sharding."""
  ckpt_shape = ckpt_arr.shape
  model_shape = model_arr.shape
  if ckpt_shape == model_shape:
    return jax.device_put(ckpt_arr, model_arr.sharding)
  if len(ckpt_shape) != len(model_shape):
    raise ValueError(
        f"Checkpoint and model arrays have different ranks: {ckpt_shape} vs {model_shape}. "
        "If the checkpoint was saved with scan_layers=True (stacked layers), convert it to "
        "unscanned format before loading with vLLM (vllm.yml sets scan_layers=False)."
    )
  if len(ckpt_shape) == 2 and ckpt_shape == model_shape[::-1]:
    return jax.device_put(jnp.transpose(ckpt_arr), model_arr.sharding)

  axes = _normalize_logical_axes(logical_axes)
  if axes is None or len(axes) != len(model_shape):
    axes = (None,) * len(model_shape)

  result = ckpt_arr
  for axis, (ckpt_dim, model_dim, axis_name) in enumerate(zip(ckpt_shape, model_shape, axes)):
    if model_dim == ckpt_dim:
      continue
    if axis_name in _VLLM_ZERO_PAD_AXES:
      if model_dim < ckpt_dim:
        raise ValueError(
            f"axis {axis} (logical={axis_name!r}): model_dim={model_dim} smaller than "
            f"ckpt_dim={ckpt_dim}; shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = _zero_pad_axis(result, axis, model_dim - ckpt_dim)
    elif axis_name in _VLLM_REPEAT_AXES:
      if model_dim % ckpt_dim != 0:
        raise ValueError(
            f"axis {axis} (logical={axis_name!r}): model_dim={model_dim} not divisible by "
            f"ckpt_dim={ckpt_dim}; shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = jnp.repeat(result, model_dim // ckpt_dim, axis=axis)
    else:
      if model_dim % ckpt_dim != 0:
        raise ValueError(
            f"Cannot align axis {axis} (logical={axis_name!r}): model_dim={model_dim} not "
            f"divisible by ckpt_dim={ckpt_dim}, and axis is not registered in _VLLM_REPEAT_AXES "
            f"({sorted(str(x) for x in _VLLM_REPEAT_AXES)}) or "
            f"_VLLM_ZERO_PAD_AXES ({sorted(str(x) for x in _VLLM_ZERO_PAD_AXES)}). "
            f"Full shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = jnp.repeat(result, model_dim // ckpt_dim, axis=axis)
  return jax.device_put(result, model_arr.sharding)


def filter_to_model_keys(ckpt, model_tree):
  if not hasattr(ckpt, "items") or not hasattr(model_tree, "items"):
    return ckpt
  return {key: filter_to_model_keys(ckpt[key], model_tree[key]) for key in model_tree if key in ckpt}


def walk_align(ckpt, model_array, axes):
  if isinstance(ckpt, dict):
    return {key: walk_align(value, model_array[key], axes.get(key) if isinstance(axes, dict) else None)
            for key, value in ckpt.items()}
  if not isinstance(ckpt, (jax.Array, jax.ShapeDtypeStruct, np.ndarray)):
    return ckpt
  return _align_checkpoint_to_model_shapes(ckpt, model_array, axes)


def align_checkpoint_tree(checkpoint, model_arrays, logical_axes):
  """Drop checkpoint entries absent from the model and align the rest to the model arrays."""
  return walk_align(filter_to_model_keys(checkpoint, model_arrays), model_arrays, logical_axes)
