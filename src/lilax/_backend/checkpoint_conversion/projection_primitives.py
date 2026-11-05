"""Shared projection layout primitives."""

import jax.numpy as jnp
import numpy as np


def projection_axis(config):
  return 1 + int(config.scan_layers and config.param_scan_axis <= 1)


def fuse_projections(values, axis, *, mlp=False, array_module=np):
  values = [array_module.asarray(value) for value in values]
  return array_module.stack(values, axis=axis) if mlp else array_module.concatenate(values, axis=axis)


def split_qkv(weight, config):
  axis = projection_axis(config)
  if weight.shape[axis] != config.num_query_heads + 2 * config.num_kv_heads:
    raise ValueError(f"Unexpected fused QKV shape: {weight.shape}")
  return jnp.split(weight, [config.num_query_heads, config.num_query_heads + config.num_kv_heads], axis=axis)


def mlp_projections(values, scan_axes, *, split=False, bias=False, count=2, array_module=np):
  values = [array_module.asarray(value) for value in values]
  axis = 0 if bias else 1
  for index, scan_axis in enumerate(reversed(scan_axes)):
    rank = values[0].ndim - index
    position = scan_axis if scan_axis < rank else 0
    values = [array_module.moveaxis(value, position, rank - 1) for value in values]
  if split:
    if values[0].shape[axis] != count:
      raise ValueError(f"Unexpected fused MLP shape: {values[0].shape}")
    values = [array_module.take(values[0], index, axis=axis) for index in range(count)]
  else:
    values = [array_module.stack(values, axis=axis)]
  rank = values[0].ndim - len(scan_axes)
  for index, scan_axis in enumerate(scan_axes):
    position = scan_axis if scan_axis < rank + index + 1 else 0
    values = [array_module.moveaxis(value, rank + index, position) for value in values]
  return values if split else values[0]


def split_mlp(weight, config):
  return mlp_projections([weight], (config.param_scan_axis,) if config.scan_layers else (), split=True, array_module=jnp)
