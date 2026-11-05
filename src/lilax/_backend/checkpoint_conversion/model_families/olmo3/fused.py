"""Convert OLMo 3 projection layouts after applying the family hooks."""

from functools import partial

import jax.numpy as jnp

from lilax._backend.checkpoint_conversion.projection_primitives import (
    fuse_projections, mlp_projections, projection_axis, split_mlp, split_qkv,
)


def _bias_axis(config):
  return int(config.scan_layers and config.param_scan_axis == 0)


def _split_bias(bias, config):
  axis = _bias_axis(config)
  if bias.shape[axis] != config.num_query_heads + 2 * config.num_kv_heads:
    raise ValueError(f"Unexpected fused QKV bias shape: {bias.shape}")
  return jnp.split(bias, [config.num_query_heads, config.num_query_heads + config.num_kv_heads], axis=axis)


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  """Fuse transformed projections to the target abstract model's layout."""
  for key, (_, shape) in target.items():
    if key.endswith(("-attention-qkv_proj-kernel", "-attention-qkv_proj-bias")):
      kind = key.rsplit("-", 1)[-1]
      prefix = key.removesuffix(f"qkv_proj-{kind}")
      sources = [prefix + name + f"-{kind}" for name in ("query", "key", "value")]
      axis = _bias_axis(config) if kind == "bias" else projection_axis(config)
      mlp = False
    elif key.endswith("-mlp-wi-kernel"):
      prefix = key.removesuffix("wi-kernel")
      sources = [prefix + "wi_0-kernel", prefix + "wi_1-kernel"]
      axis = projection_axis(config)
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
  """Split fused projections before applying the OLMo 3 export hooks."""
  for key in list(weights):
    if key.endswith(("-attention-qkv_proj-kernel", "-attention-qkv_proj-bias")):
      kind = key.rsplit("-", 1)[-1]
      prefix = key.removesuffix(f"qkv_proj-{kind}")
      values = _split_bias(weights.pop(key), config) if kind == "bias" else split_qkv(weights.pop(key), config)
      for name, value in zip(("query", "key", "value"), values):
        weights[prefix + name + f"-{kind}"] = value
    elif key.endswith("-mlp-wi-kernel"):
      prefix = key.removesuffix("wi-kernel")
      for name, value in zip(("wi_0", "wi_1"), split_mlp(weights.pop(key), config)):
        weights[prefix + name + "-kernel"] = value
  return weights


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  """Convert the projection layout of an OLMo 3 native parameter tree."""
  if not isinstance(state, dict):
    return state
  result = {key: convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
            for key, value in state.items()}
  attention = result.get("attention")
  if isinstance(attention, dict) and "qkv_proj" in attention and not fused_qkv:
    projection = attention.pop("qkv_proj")
    if not set(projection) <= {"kernel", "bias"} or "kernel" not in projection:
      raise NotImplementedError("Fused OLMo 3 QKV conversion supports unadapted kernels and biases only.")
    projections = {name: {} for name in ("query", "key", "value")}
    for kind, weight in projection.items():
      values = _split_bias(weight, config) if kind == "bias" else split_qkv(weight, config)
      for name, value in zip(projections, values):
        projections[name][kind] = value
    attention.update(projections)
  elif isinstance(attention, dict) and fused_qkv and "query" in attention:
    projections = [attention.pop(name) for name in ("query", "key", "value")]
    kinds = set(projections[0])
    if not kinds <= {"kernel", "bias"} or "kernel" not in kinds or any(set(item) != kinds for item in projections):
      raise NotImplementedError("Fused OLMo 3 QKV conversion supports unadapted kernels and biases only.")
    attention["qkv_proj"] = {
        kind: fuse_projections([item[kind] for item in projections],
                               _bias_axis(config) if kind == "bias" else projection_axis(config), array_module=jnp)
        for kind in projections[0]
    }
  mlp = result.get("mlp")
  if isinstance(mlp, dict) and "wi" in mlp and not fused_mlp:
    projection = mlp.pop("wi")
    if set(projection) != {"kernel"}:
      raise NotImplementedError("Fused OLMo 3 MLP conversion supports unadapted kernels only.")
    for name, value in zip(("wi_0", "wi_1"), split_mlp(projection["kernel"], config)):
      mlp[name] = {"kernel": value}
  elif isinstance(mlp, dict) and fused_mlp and "wi_0" in mlp:
    projections = [mlp.pop(name) for name in ("wi_0", "wi_1")]
    if any(set(projection) != {"kernel"} for projection in projections):
      raise NotImplementedError("Fused OLMo 3 MLP conversion supports unadapted kernels only.")
    mlp["wi"] = {"kernel": mlp_projections(
        [projection["kernel"] for projection in projections],
        (config.param_scan_axis,) if config.scan_layers else (), array_module=jnp
    )}
  return result
