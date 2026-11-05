"""Convert GPT-OSS attention projection layouts around the family mapping hooks."""

from functools import partial

import jax.numpy as jnp

from lilax._backend.checkpoint_conversion.projection_primitives import fuse_projections, projection_axis, split_qkv


def _axis(config, kind):
  return int(config.scan_layers and config.param_scan_axis == 0) if kind == "bias" else projection_axis(config)


def _split(weight, config, kind):
  if kind == "kernel":
    return split_qkv(weight, config)
  axis = _axis(config, kind)
  if weight.shape[axis] != config.num_query_heads + 2 * config.num_kv_heads:
    raise ValueError(f"Unexpected fused QKV bias shape: {weight.shape}")
  return jnp.split(weight, [config.num_query_heads, config.num_query_heads + config.num_kv_heads], axis=axis)


def fuse_moe_weights(wi_0, wi_1, *, axis, num_shards):
  gate_chunks = jnp.split(wi_0, num_shards, axis=axis)
  up_chunks = jnp.split(wi_1, num_shards, axis=axis)
  return jnp.concatenate(
      [jnp.concatenate([gate, up], axis=axis) for gate, up in zip(gate_chunks, up_chunks)], axis=axis
  )


def split_moe_weights(wi, *, axis, num_shards):
  chunks = jnp.split(wi, num_shards, axis=axis)
  pairs = [jnp.split(chunk, 2, axis=axis) for chunk in chunks]
  return tuple(jnp.concatenate([pair[index] for pair in pairs], axis=axis) for index in range(2))


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  for key, (_, shape) in target.items():
    if not key.endswith(("-GptOssAttention-qkv_proj-kernel", "-GptOssAttention-qkv_proj-bias")):
      continue
    kind = key.rsplit("-", 1)[-1]
    prefix = key.removesuffix(f"qkv_proj-{kind}")
    values = [weights.pop(prefix + name + f"-{kind}") for name in ("query", "key", "value")]
    load = partial(fuse_projections, values, _axis(config, kind))
    weights[key] = lazy_tensor(load, shape, values[0].dtype, name=key) if lazy_tensor else load()
    if weights[key].shape != shape:
      raise ValueError(f"Unexpected fused shape for {key}: {weights[key].shape}, expected {shape}")
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  for key in list(weights):
    if key.endswith(("-GptOssAttention-qkv_proj-kernel", "-GptOssAttention-qkv_proj-bias")):
      kind = key.rsplit("-", 1)[-1]
      prefix = key.removesuffix(f"qkv_proj-{kind}")
      for name, value in zip(("query", "key", "value"), _split(weights.pop(key), config, kind)):
        weights[prefix + name + f"-{kind}"] = value
  return weights


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  if not isinstance(state, dict):
    return state
  result = {key: convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
            for key, value in state.items()}
  attention = result.get("GptOssAttention")
  if isinstance(attention, dict) and "qkv_proj" in attention and not fused_qkv:
    projection = attention.pop("qkv_proj")
    if not set(projection) <= {"kernel", "bias"} or "kernel" not in projection:
      raise NotImplementedError("Fused GPT-OSS QKV conversion supports unadapted kernels and biases only.")
    projections = {name: {} for name in ("query", "key", "value")}
    for kind, weight in projection.items():
      for name, value in zip(projections, _split(weight, config, kind)):
        projections[name][kind] = value
    attention.update(projections)
  elif isinstance(attention, dict) and fused_qkv and "query" in attention:
    projections = [attention.pop(name) for name in ("query", "key", "value")]
    kinds = set(projections[0])
    if not kinds <= {"kernel", "bias"} or "kernel" not in kinds or any(set(item) != kinds for item in projections):
      raise NotImplementedError("Fused GPT-OSS QKV conversion supports unadapted kernels and biases only.")
    attention["qkv_proj"] = {
        kind: fuse_projections([item[kind] for item in projections], _axis(config, kind), array_module=jnp)
        for kind in projections[0]
    }
  return result
