"""Convert DeepSeek dense and shared-expert MLP projection layouts."""

from functools import partial

import jax.numpy as jnp
import numpy as np

from lilax._backend.checkpoint_conversion.projection_primitives import fuse_projections


def _axis(weight, config, *, fused):
  unscanned_rank = 3 if fused else 2
  if len(weight.shape) == unscanned_rank:
    return 1
  if config.scan_layers and len(weight.shape) == unscanned_rank + 1:
    if config.param_scan_axis not in (0, 1, 2):
      raise NotImplementedError(f"Unsupported DeepSeek scan axis: {config.param_scan_axis}")
    return 1 + int(config.param_scan_axis <= 1)
  raise NotImplementedError(f"Unsupported DeepSeek MLP kernel shape: {weight.shape}")


def _split(weight, config):
  axis = _axis(weight, config, fused=True)
  if weight.shape[axis] != 2:
    raise ValueError(f"Unexpected fused DeepSeek MLP shape: {weight.shape}")
  values = [jnp.take(weight, index, axis=axis) for index in range(2)]
  if len(weight.shape) == 4 and config.param_scan_axis == 2:
    values = [jnp.moveaxis(value, 1, 2) for value in values]
  return values


def _fuse(values, config, array_module=np):
  axis = _axis(values[0], config, fused=False)
  result = fuse_projections(values, axis, mlp=True, array_module=array_module)
  if len(values[0].shape) == 3 and config.param_scan_axis == 2:
    result = array_module.moveaxis(result, 3, 2)
  return result


def _check_scale_grids(kernel_shape, scale_shapes, config, *, fused, scales_fused=None, target_shape=None):
  block_size = config.weight_block_size
  if block_size is None or (not isinstance(block_size, int) and not block_size):
    raise NotImplementedError("DeepSeek fused MLP requires non-scalar matching block-scale grids.")
  kernel_shape = list(kernel_shape)
  unscanned_rank = 3 if fused else 2
  scan_size = None
  if config.scan_layers and len(kernel_shape) == unscanned_rank + 1:
    if config.param_scan_axis not in (0, 1, 2):
      raise NotImplementedError(f"Unsupported DeepSeek scan axis: {config.param_scan_axis}")
    scan_size = kernel_shape.pop(config.param_scan_axis)
  if len(kernel_shape) != unscanned_rank:
    raise NotImplementedError(f"Unsupported DeepSeek MLP kernel shape: {kernel_shape}")
  if fused:
    if kernel_shape[1] != 2:
      raise ValueError(f"Unexpected fused DeepSeek MLP shape: {kernel_shape}")
    unfused_shape = (kernel_shape[0], kernel_shape[2])
    fused_shape = tuple(kernel_shape)
  else:
    unfused_shape = tuple(kernel_shape)
    fused_shape = (kernel_shape[0], 2, kernel_shape[1])

  def scale_shape(shape):
    if isinstance(block_size, int):
      blocks = (block_size,) * len(shape)
    elif len(block_size) == len(shape):
      blocks = tuple(block_size)
    else:
      blocks = (block_size[0],) * len(shape)
    if any(block <= 0 for block in blocks):
      raise NotImplementedError("DeepSeek fused MLP requires positive block sizes.")
    return tuple(dim if dim < block else dim // block for dim, block in zip(shape, blocks))

  unfused_scale = scale_shape(unfused_shape)
  fused_scale = scale_shape(fused_shape)
  if fused_scale != (unfused_scale[0], 2, unfused_scale[1]) or not unfused_scale:
    raise NotImplementedError("DeepSeek fused MLP requires non-scalar matching block-scale grids.")
  if scan_size is not None:
    unfused_scale = list(unfused_scale)
    fused_scale = list(fused_scale)
    unfused_scale.insert(config.param_scan_axis, scan_size)
    fused_scale.insert(config.param_scan_axis, scan_size)
    unfused_scale, fused_scale = tuple(unfused_scale), tuple(fused_scale)
  expected = fused_scale if (fused if scales_fused is None else scales_fused) else unfused_scale
  if any(tuple(shape) != expected or not shape for shape in scale_shapes):
    raise NotImplementedError("DeepSeek fused MLP requires non-scalar matching block-scale grids.")
  if target_shape is not None and tuple(target_shape) != fused_scale:
    raise NotImplementedError("DeepSeek fused MLP requires non-scalar matching block-scale grids.")


def _check_qkv(config, fused_qkv=False):
  if config.fused_qkv or fused_qkv:
    raise ValueError("Fused QKV conversion is not supported for DeepSeek.")


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  _check_qkv(config)
  for key, (_, shape) in target.items():
    if not key.endswith(("-mlp-wi-kernel", "-shared_experts-wi-kernel",
                         "-mlp-wi-kernel_scale", "-shared_experts-wi-kernel_scale")):
      continue
    kind = "kernel_scale" if key.endswith("-kernel_scale") else "kernel"
    prefix = key.removesuffix("wi-" + kind)
    values = [weights.pop(prefix + name + "-" + kind) for name in ("wi_0", "wi_1")]
    if kind == "kernel_scale":
      _check_scale_grids(target[prefix + "wi-kernel"][1], [value.shape for value in values], config,
                         fused=True, scales_fused=False, target_shape=shape)
    load = partial(_fuse, values, config)
    weights[key] = lazy_tensor(load, shape, values[0].dtype, name=key) if lazy_tensor else load()
    if weights[key].shape != shape:
      raise ValueError(f"Unexpected fused shape for {key}: {weights[key].shape}, expected {shape}")
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  _check_qkv(config)
  if any(key.endswith("-kernel_scale") for key in weights):
    raise NotImplementedError("DeepSeek FP8 Hugging Face export is not supported.")
  for key in list(weights):
    if not key.endswith(("-mlp-wi-kernel", "-shared_experts-wi-kernel")):
      continue
    prefix = key.removesuffix("wi-kernel")
    for name, value in zip(("wi_0", "wi_1"), _split(weights.pop(key), config)):
      weights[prefix + name + "-kernel"] = value
  return weights


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  _check_qkv(config, fused_qkv)
  if not isinstance(state, dict):
    return state
  result = {key: convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
            for key, value in state.items()}
  for name in ("mlp", "shared_experts"):
    mlp = result.get(name)
    if not isinstance(mlp, dict):
      continue
    if config.fused_mlp and not fused_mlp and "wi" in mlp:
      projection = mlp.pop("wi")
      if set(projection) not in ({"kernel"}, {"kernel", "kernel_scale"}):
        raise NotImplementedError("DeepSeek MLP conversion supports unadapted kernels only.")
      if "kernel_scale" in projection:
        _check_scale_grids(projection["kernel"].shape, [projection["kernel_scale"].shape], config, fused=True)
      mlp["wi_0"], mlp["wi_1"] = {}, {}
      for kind, weight in projection.items():
        for projection_name, value in zip(("wi_0", "wi_1"), _split(weight, config)):
          mlp[projection_name][kind] = value
    elif not config.fused_mlp and fused_mlp and "wi_0" in mlp:
      projections = [mlp.pop(projection_name) for projection_name in ("wi_0", "wi_1")]
      if set(projections[0]) not in ({"kernel"}, {"kernel", "kernel_scale"}) or set(projections[0]) != set(projections[1]):
        raise NotImplementedError("DeepSeek MLP conversion supports unadapted kernels only.")
      if "kernel_scale" in projections[0]:
        if projections[0]["kernel"].shape != projections[1]["kernel"].shape:
          raise ValueError("DeepSeek MLP kernels must have matching shapes.")
        _check_scale_grids(projections[0]["kernel"].shape, [projection["kernel_scale"].shape for projection in projections],
                           config, fused=False)
      mlp["wi"] = {kind: _fuse([projection[kind] for projection in projections], config, array_module=jnp)
                   for kind in projections[0]}
  return result


def unfuse_native_state(state, config):
  return convert_native_state(state, config, fused_qkv=False, fused_mlp=False)
