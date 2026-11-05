"""Convert native GPT-3 projection layouts."""

import jax.numpy as jnp

from lilax._backend.checkpoint_conversion.projection_primitives import fuse_projections, mlp_projections, projection_axis


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  raise NotImplementedError("Hugging Face import of GPT-3 is not supported.")


def unfuse_for_huggingface(weights, config):
  raise NotImplementedError("Hugging Face export of GPT-3 is not supported.")


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  if not isinstance(state, dict):
    return state
  result = {key: convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
            for key, value in state.items()}
  attention = result.get("self_attention")
  if isinstance(attention, dict) and "qkv_proj" in attention and not fused_qkv:
    projection = attention.pop("qkv_proj")
    if not set(projection) <= {"kernel", "bias"} or "kernel" not in projection:
      raise NotImplementedError("GPT-3 QKV conversion supports unadapted kernels and biases only.")
    projections = {name: {} for name in ("query", "key", "value")}
    for kind, weight in projection.items():
      axis = projection_axis(config) if kind == "kernel" else int(config.scan_layers and config.param_scan_axis == 0)
      if weight.shape[axis] != 3:
        raise ValueError(f"Unexpected fused GPT-3 QKV shape: {weight.shape}")
      for index, name in enumerate(projections):
        projections[name][kind] = jnp.take(weight, index, axis=axis)
    attention.update(projections)
  elif isinstance(attention, dict) and fused_qkv and "query" in attention:
    projections = [attention.pop(name) for name in ("query", "key", "value")]
    kinds = set(projections[0])
    if not kinds <= {"kernel", "bias"} or "kernel" not in kinds or any(set(item) != kinds for item in projections):
      raise NotImplementedError("GPT-3 QKV conversion supports unadapted kernels and biases only.")
    attention["qkv_proj"] = {
        kind: fuse_projections([item[kind] for item in projections],
                               projection_axis(config) if kind == "kernel" else
                               int(config.scan_layers and config.param_scan_axis == 0), mlp=True, array_module=jnp)
        for kind in projections[0]
    }
  mlp = result.get("mlp")
  count = len(config.mlp_activations)
  if isinstance(mlp, dict) and config.fused_mlp != fused_mlp:
    names = ["wi"] if count == 1 else [f"wi_{index}" for index in range(count)]
    if config.fused_mlp:
      projection = mlp.pop("wi")
      if not set(projection) <= {"kernel", "bias"} or "kernel" not in projection:
        raise NotImplementedError("GPT-3 MLP conversion supports unadapted kernels and biases only.")
      projections = {name: {} for name in names}
      for kind, weight in projection.items():
        values = mlp_projections([weight], (config.param_scan_axis,) if config.scan_layers else (),
                                 split=True, bias=kind == "bias", count=count, array_module=jnp)
        for name, value in zip(names, values):
          projections[name][kind] = value
      mlp.update(projections)
    else:
      projections = [mlp.pop(name) for name in names]
      kinds = set(projections[0])
      if not kinds <= {"kernel", "bias"} or "kernel" not in kinds or any(set(item) != kinds for item in projections):
        raise NotImplementedError("GPT-3 MLP conversion supports unadapted kernels and biases only.")
      mlp["wi"] = {
          kind: mlp_projections([item[kind] for item in projections],
                                (config.param_scan_axis,) if config.scan_layers else (),
                                bias=kind == "bias", array_module=jnp)
          for kind in projections[0]
      }
  return result
