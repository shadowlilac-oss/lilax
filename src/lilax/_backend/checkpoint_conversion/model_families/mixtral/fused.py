"""Convert Mixtral attention projection layouts around the family mapping hooks."""

from functools import partial

import jax.numpy as jnp

from lilax._backend.checkpoint_conversion.projection_primitives import fuse_projections, projection_axis, split_qkv


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  for key, (_, shape) in target.items():
    if not key.endswith("-self_attention-qkv_proj-kernel"):
      continue
    prefix = key.removesuffix("qkv_proj-kernel")
    values = [weights.pop(prefix + name + "-kernel") for name in ("query", "key", "value")]
    load = partial(fuse_projections, values, projection_axis(config))
    weights[key] = lazy_tensor(load, shape, values[0].dtype, name=key) if lazy_tensor else load()
    if weights[key].shape != shape:
      raise ValueError(f"Unexpected fused shape for {key}: {weights[key].shape}, expected {shape}")
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  for key in list(weights):
    if key.endswith("-self_attention-qkv_proj-kernel"):
      prefix = key.removesuffix("qkv_proj-kernel")
      for name, value in zip(("query", "key", "value"), split_qkv(weights.pop(key), config)):
        weights[prefix + name + "-kernel"] = value
  return weights


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  if not isinstance(state, dict):
    return state
  result = {key: convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
            for key, value in state.items()}
  attention = result.get("self_attention")
  if isinstance(attention, dict) and "qkv_proj" in attention and not fused_qkv:
    projection = attention.pop("qkv_proj")
    if set(projection) != {"kernel"}:
      raise NotImplementedError("Fused Mixtral QKV conversion supports unadapted kernels only.")
    for name, value in zip(("query", "key", "value"), split_qkv(projection["kernel"], config)):
      attention[name] = {"kernel": value}
  elif isinstance(attention, dict) and fused_qkv and "query" in attention:
    projections = [attention.pop(name) for name in ("query", "key", "value")]
    if any(set(projection) != {"kernel"} for projection in projections):
      raise NotImplementedError("Fused Mixtral QKV conversion supports unadapted kernels only.")
    attention["qkv_proj"] = {"kernel": fuse_projections(
        [projection["kernel"] for projection in projections], projection_axis(config), array_module=jnp
    )}
  return result
