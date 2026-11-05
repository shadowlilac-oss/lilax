"""Convert Llama 3.1 projection layouts."""

from functools import partial

import jax.numpy as jnp

from lilax._backend.checkpoint_conversion.projection_primitives import (
    fuse_projections, mlp_projections, projection_axis, split_mlp, split_qkv,
)


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  """Fuse transformed flat weights to the target abstract model's layout."""
  axis = projection_axis(config)
  for key, (_, shape) in target.items():
    if key.endswith("-self_attention-qkv_proj-kernel"):
      prefix = key.removesuffix("qkv_proj-kernel")
      sources = [prefix + name + "-kernel" for name in ("query", "key", "value")]
      mlp = False
    elif key.endswith("-mlp-wi-kernel"):
      prefix = key.removesuffix("wi-kernel")
      sources = [prefix + "wi_0-kernel", prefix + "wi_1-kernel"]
      mlp = True
    else:
      continue
    values = [weights.pop(source) for source in sources]
    load = (partial(mlp_projections, values, (config.param_scan_axis,) if config.scan_layers else ()) if mlp
            else partial(fuse_projections, values, axis))
    weights[key] = lazy_tensor(load, shape, values[0].dtype, name=key) if lazy_tensor else load()
    if weights[key].shape != shape:
      raise ValueError(f"Unexpected fused shape for {key}: {weights[key].shape}, expected {shape}")
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  """Split fused decoder kernels before applying the Llama 3.1 export hooks."""
  for key in list(weights):
    if key.endswith("-self_attention-qkv_proj-kernel"):
      query, key_weight, value = split_qkv(weights.pop(key), config)
      prefix = key.removesuffix("qkv_proj-kernel")
      weights[prefix + "query-kernel"] = query
      weights[prefix + "key-kernel"] = key_weight
      weights[prefix + "value-kernel"] = value
    elif key.endswith("-mlp-wi-kernel"):
      first, second = split_mlp(weights.pop(key), config)
      prefix = key.removesuffix("wi-kernel")
      weights[prefix + "wi_0-kernel"] = first
      weights[prefix + "wi_1-kernel"] = second
  return weights


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  """Convert the projection layout of a native model's parameter tree."""
  if not isinstance(state, dict):
    return state
  result = {key: convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
            for key, value in state.items()}
  attention = result.get("self_attention")
  if isinstance(attention, dict) and "qkv_proj" in attention and not fused_qkv:
    projection = attention.pop("qkv_proj")
    if set(projection) != {"kernel"}:
      raise NotImplementedError("Fused Llama QKV export supports unadapted kernels only.")
    for name, value in zip(("query", "key", "value"), split_qkv(projection["kernel"], config)):
      attention[name] = {"kernel": value}
  elif isinstance(attention, dict) and fused_qkv and "query" in attention:
    projections = [attention.pop(name) for name in ("query", "key", "value")]
    if any(set(projection) != {"kernel"} for projection in projections):
      raise NotImplementedError("Fused Llama QKV conversion supports unadapted kernels only.")
    attention["qkv_proj"] = {"kernel": fuse_projections(
        [projection["kernel"] for projection in projections], projection_axis(config), array_module=jnp
    )}
  mlp = result.get("mlp")
  if isinstance(mlp, dict) and "wi" in mlp and not fused_mlp:
    projection = mlp.pop("wi")
    if set(projection) != {"kernel"}:
      raise NotImplementedError("Fused Llama MLP export supports unadapted kernels only.")
    for name, value in zip(("wi_0", "wi_1"), split_mlp(projection["kernel"], config)):
      mlp[name] = {"kernel": value}
  elif isinstance(mlp, dict) and fused_mlp and "wi_0" in mlp:
    projections = [mlp.pop(name) for name in ("wi_0", "wi_1")]
    if any(set(projection) != {"kernel"} for projection in projections):
      raise NotImplementedError("Fused Llama MLP conversion supports unadapted kernels only.")
    mlp["wi"] = {"kernel": mlp_projections(
        [projection["kernel"] for projection in projections],
        (config.param_scan_axis,) if config.scan_layers else (), array_module=jnp
    )}
  return result


def unfuse_native_state(state, config):
  """Return an unfused copy of a native model's parameter tree."""
  return convert_native_state(state, config, fused_qkv=False, fused_mlp=False)
