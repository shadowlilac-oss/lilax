"""Convert hybrid Qwen projection layouts around the family hooks."""

from functools import partial

import jax.numpy as jnp
import numpy as np

from lilax._backend.common.common_types import is_fp8_dtype
from lilax._backend.checkpoint_conversion.projection_primitives import mlp_projections


def _scan_axes(path, config):
  if "-pipeline_module-layers-" in path:
    scans = int("-scanned_layers-" in path) + int("-local_layers-" in path)
  elif "-layers_outside_pipeline-" in path:
    scans = int(config.scan_layers) + int("-local_layers-" in path)
  elif config.scan_layers:
    scans = 2 if "-layers-local_layers-" in path else 1 if (
        "-layers-global_layer-" in path or "-layers_remainder-local_layers-" in path
    ) else 0
  else:
    scans = 0
  axes = (config.param_scan_axis,) * scans
  if "-pipeline_module-layers-" in path:
    axes += (0,) * (1 + int(config.num_pipeline_repeats > 1))
  return axes


def _projection_axis(path, config, axis):
  for scan_axis in _scan_axes(path, config):
    axis += int(scan_axis <= axis)
  return axis


def _validate_qkv_scales(projections, path, config):
  block_size = getattr(config, "weight_block_size", None)
  block_sizes = (block_size,) * 3 if isinstance(block_size, int) else block_size
  if block_sizes and len(block_sizes) != 3:
    block_sizes = (block_sizes[0],) * 3
  if (not block_sizes or any(not size or size < 0 for size in block_sizes) or config.head_dim % block_sizes[2]
      or 2 * (config.num_query_heads + config.num_kv_heads) >= block_sizes[1]):
    raise NotImplementedError("Hybrid Qwen fused FP8 QKV requires exact, uncompressed block scales.")
  axes = [_projection_axis(path, config, axis) for axis in range(3)]
  for projection in projections:
    kernel, scale = projection["kernel"], projection["kernel_scale"]
    if kernel is None or scale is None or not is_fp8_dtype(kernel.dtype):
      raise NotImplementedError("Hybrid Qwen fused QKV requires FP8 kernels and block scale grids.")
    shape = list(kernel.shape)
    for axis, size in zip(axes, block_sizes):
      dim = shape[axis]
      if dim >= size and dim % size:
        raise NotImplementedError("Hybrid Qwen fused FP8 QKV requires divisible block dimensions.")
      shape[axis] = dim if dim < size else dim // size
    if tuple(shape) != scale.shape:
      raise NotImplementedError("Hybrid Qwen fused FP8 QKV requires matching block scale grids.")


def _fuse_qkv(values, path, config, array_module=np):
  head_axis = _projection_axis(path, config, 1)
  dim_axis = _projection_axis(path, config, 2)
  query, key, value = (array_module.asarray(weight) for weight in values)
  if path.endswith("-kernel"):
    fp8 = [is_fp8_dtype(weight.dtype) for weight in (query, key, value)]
    if (is_fp8_dtype(config.weight_dtype) and not all(fp8)) or (any(fp8) and not is_fp8_dtype(config.weight_dtype)):
      raise NotImplementedError("Hybrid Qwen fused QKV requires matching FP8 kernels and block scales.")
  query, gate = array_module.split(query, 2, axis=dim_axis)
  return array_module.concatenate((query, gate, key, value), axis=head_axis)


def _split_qkv(weight, path, config):
  head_axis = _projection_axis(path, config, 1)
  dim_axis = _projection_axis(path, config, 2)
  query_heads = config.num_query_heads
  kv_heads = config.num_kv_heads
  if weight.shape[head_axis] != 2 * query_heads + 2 * kv_heads:
    raise ValueError(f"Unexpected fused QKV shape: {weight.shape}")
  query, gate, key, value = jnp.split(weight, [query_heads, 2 * query_heads, 2 * query_heads + kv_heads], axis=head_axis)
  return jnp.concatenate((query, gate), axis=dim_axis), key, value


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  for key, (_, shape) in target.items():
    if key.endswith(("-attention-attention-qkv_proj-kernel", "-attention-attention-qkv_proj-kernel_scale")):
      prefix, kind = key.rsplit("qkv_proj-", 1)
      names = ("query", "key", "value")
      if is_fp8_dtype(config.weight_dtype):
        if (prefix + "qkv_proj-kernel_scale" not in target
            or any(prefix + name + "-" + leaf not in weights for name in names for leaf in ("kernel", "kernel_scale"))):
          raise NotImplementedError("Hybrid Qwen fused FP8 QKV requires all three block scale grids.")
        _validate_qkv_scales([{leaf: weights[prefix + name + "-" + leaf] for leaf in ("kernel", "kernel_scale")}
                              for name in names], key, config)
      elif any(prefix + name + "-kernel_scale" in weights for name in names):
        raise NotImplementedError("Hybrid Qwen fused QKV does not support unexpected scale grids.")
      values = [weights[prefix + name + "-" + kind] for name in names]
      load = partial(_fuse_qkv, values, key, config)
    elif key.endswith(("-mlp-wi-kernel", "-mlp-shared_expert-wi-kernel")):
      prefix = key.removesuffix("wi-kernel")
      values = [weights.pop(prefix + name + "-kernel") for name in ("wi_0", "wi_1")]
      if is_fp8_dtype(config.weight_dtype) or any(is_fp8_dtype(value.dtype) for value in values):
        raise NotImplementedError("Fused hybrid Qwen MLP conversion does not support FP8.")
      load = partial(mlp_projections, values, _scan_axes(key, config))
    else:
      continue
    weights[key] = lazy_tensor(load, shape, values[0].dtype, name=key) if lazy_tensor else load()
    if weights[key].shape != shape:
      raise ValueError(f"Unexpected fused shape for {key}: {weights[key].shape}, expected {shape}")
  for key in target:
    if key.endswith(("-attention-attention-qkv_proj-kernel", "-attention-attention-qkv_proj-kernel_scale")):
      prefix, kind = key.rsplit("qkv_proj-", 1)
      for name in ("query", "key", "value"):
        weights.pop(prefix + name + "-" + kind)
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  for key in list(weights):
    if key.endswith("-attention-attention-qkv_proj-kernel"):
      if is_fp8_dtype(config.weight_dtype) or is_fp8_dtype(weights[key].dtype):
        raise NotImplementedError("Hybrid Qwen fused QKV export does not support FP8 block scales.")
      prefix = key.removesuffix("qkv_proj-kernel")
      for name, value in zip(("query", "key", "value"), _split_qkv(weights.pop(key), key, config)):
        weights[prefix + name + "-kernel"] = value
    elif key.endswith(("-mlp-wi-kernel", "-mlp-shared_expert-wi-kernel")):
      if is_fp8_dtype(config.weight_dtype) or is_fp8_dtype(weights[key].dtype):
        raise NotImplementedError("Fused hybrid Qwen MLP conversion does not support FP8.")
      prefix = key.removesuffix("wi-kernel")
      for name, value in zip(("wi_0", "wi_1"), mlp_projections(
          [weights.pop(key)], _scan_axes(key, config), split=True, array_module=jnp
      )):
        weights[prefix + name + "-kernel"] = value
  return weights


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  def convert(value, path):
    if not isinstance(value, dict):
      return value
    result = {key: convert(child, path + (key,)) for key, child in value.items()}
    if path[-2:] == ("attention", "attention"):
      projection_path = "-" + "-".join(path) + "-"
      if "qkv_proj" in result and not fused_qkv:
        projection = result.pop("qkv_proj")
        fp8 = is_fp8_dtype(config.weight_dtype) or is_fp8_dtype(projection["kernel"].dtype)
        kinds = {"kernel", "kernel_scale"} if fp8 else {"kernel"}
        if set(projection) != kinds:
          raise NotImplementedError("Fused hybrid Qwen QKV conversion supports unadapted kernels only.")
        if fp8:
          _validate_qkv_scales([projection], projection_path, config)
        split = {kind: _split_qkv(projection[kind], projection_path, config) for kind in kinds}
        for index, name in enumerate(("query", "key", "value")):
          result[name] = {kind: values[index] for kind, values in split.items()}
      elif fused_qkv and "query" in result:
        projections = [result.pop(name) for name in ("query", "key", "value")]
        fp8 = is_fp8_dtype(config.weight_dtype) or any(is_fp8_dtype(projection["kernel"].dtype) for projection in projections)
        kinds = {"kernel", "kernel_scale"} if fp8 else {"kernel"}
        if any(set(projection) != kinds for projection in projections):
          raise NotImplementedError("Fused hybrid Qwen QKV conversion supports unadapted kernels only.")
        if fp8:
          _validate_qkv_scales(projections, projection_path, config)
        result["qkv_proj"] = {kind: _fuse_qkv(
            [projection[kind] for projection in projections], projection_path, config, array_module=jnp
        ) for kind in kinds}
    if path[-1:] not in (("mlp",), ("shared_expert",)):
      return result
    scan_axes = _scan_axes("-" + "-".join(path) + "-", config)
    if ("wi" in result and not fused_mlp) or (fused_mlp and "wi_0" in result):
      kernels = [result[name]["kernel"] for name in ("wi",) if name in result] if not fused_mlp else [
          result[name]["kernel"] for name in ("wi_0", "wi_1")
      ]
      if is_fp8_dtype(config.weight_dtype) or any(is_fp8_dtype(kernel.dtype) for kernel in kernels):
        raise NotImplementedError("Fused hybrid Qwen MLP conversion does not support FP8.")
    if "wi" in result and not fused_mlp:
      projection = result.pop("wi")
      if set(projection) != {"kernel"}:
        raise NotImplementedError("Fused hybrid Qwen MLP conversion supports unadapted kernels only.")
      for name, weight in zip(("wi_0", "wi_1"), mlp_projections(
          [projection["kernel"]], scan_axes, split=True, array_module=jnp
      )):
        result[name] = {"kernel": weight}
    elif fused_mlp and "wi_0" in result:
      projections = [result.pop(name) for name in ("wi_0", "wi_1")]
      if any(set(projection) != {"kernel"} for projection in projections):
        raise NotImplementedError("Fused hybrid Qwen MLP conversion supports unadapted kernels only.")
      result["wi"] = {"kernel": mlp_projections(
          [projection["kernel"] for projection in projections], scan_axes, array_module=jnp
      )}
    return result

  return convert(state, ())
