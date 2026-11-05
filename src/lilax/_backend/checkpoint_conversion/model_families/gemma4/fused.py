"""Convert Gemma decoder projection layouts without changing family weight hooks."""

from functools import partial

import jax.numpy as jnp

from lilax._backend.checkpoint_conversion.projection_primitives import fuse_projections, mlp_projections


def _scan_axes(path, config):
  if "-vision_encoder-" in path:
    return ()
  decoder = getattr(config.decoder_block, "value", config.decoder_block)
  if not config.scan_layers:
    scans = 0
  elif "-pipeline_module-layers-" in path:
    scans = int("-scanned_layers-" in path)
  elif "-layers_outside_pipeline-" in path:
    scans = 1
  elif decoder in ("gemma", "gemma2"):
    scans = int("-decoder-layers-" in path)
  elif decoder == "gemma3":
    scans = int("-decoder-layers-layers_" in path)
  elif decoder == "gemma4":
    scans = int("-decoder-scanned_blocks-" in path) + int("-local_layers-" in path)
  else:
    scans = 0
  axes = (config.param_scan_axis,) * scans
  if "-pipeline_module-layers-" in path:
    axes += (0,) * (1 + int(config.num_pipeline_repeats > 1))
  return axes


def _projection_axis(path, config, *, bias=False):
  axis = 0 if bias else 1
  for scan_axis in _scan_axes(path, config):
    axis += int(scan_axis <= axis)
  return axis


def _split(weight, axis, config, *, vision=False):
  query_heads = config.num_attention_heads_for_vit if vision else config.num_query_heads
  kv_heads, remainder = divmod(weight.shape[axis] - query_heads, 2)
  allowed = {query_heads} if vision else {config.num_kv_heads, getattr(config, "global_num_kv_heads", 0)}
  if remainder or kv_heads not in allowed or kv_heads <= 0:
    raise ValueError(f"Unexpected fused QKV shape: {weight.shape}")
  return jnp.split(weight, [query_heads, query_heads + kv_heads], axis=axis)


def _projection(key):
  if key.startswith("params-vision_encoder-Gemma3VisionEncoderLayer_0-"):
    for parameter in ("kernel", "bias"):
      if key.endswith(f"-MultiHeadDotProductAttention_0-qkv_proj-{parameter}"):
        return ("query", "key", "value"), f"qkv_proj-{parameter}", False, parameter
    return None
  if key.startswith("params-vision_encoder-Gemma4VisionEncoderLayer_0-"):
    for parameter in ("kernel", "bias"):
      if key.endswith(f"-attention-qkv_proj-{parameter}"):
        return ("query", "key", "value"), f"qkv_proj-{parameter}", False, parameter
      if key.endswith(f"-mlp-wi-{parameter}"):
        return ("wi_0", "wi_1"), f"wi-{parameter}", True, parameter
    return None
  if not key.startswith("params-decoder-"):
    return None
  for name in ("self_attention", "self_attention_local", "self_attention_global"):
    for parameter in ("kernel", "bias"):
      if key.endswith(f"-{name}-qkv_proj-{parameter}"):
        return ("query", "key", "value"), f"qkv_proj-{parameter}", False, parameter
  for name in ("mlp", "mlp_local", "mlp_global", "shared_experts"):
    for parameter in ("kernel", "bias"):
      if key.endswith(f"-{name}-wi-{parameter}"):
        return ("wi_0", "wi_1"), f"wi-{parameter}", True, parameter
  return None


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  """Fuse decoder projections after the family-specific import hooks."""
  for key, (_, shape) in target.items():
    projection = _projection(key)
    if projection is None:
      continue
    names, suffix, mlp, parameter = projection
    prefix = key.removesuffix(suffix)
    values = [weights.pop(prefix + name + "-" + parameter) for name in names]
    load = (partial(mlp_projections, values, _scan_axes(key, config), bias=parameter == "bias") if mlp else
            partial(fuse_projections, values, _projection_axis(key, config, bias=parameter == "bias")))
    weights[key] = lazy_tensor(load, shape, values[0].dtype, name=key) if lazy_tensor else load()
    if weights[key].shape != shape:
      raise ValueError(f"Unexpected fused shape for {key}: {weights[key].shape}, expected {shape}")
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  """Split decoder projections before the family-specific export hooks."""
  for key in list(weights):
    projection = _projection(key)
    if projection is None:
      continue
    names, suffix, mlp, parameter = projection
    prefix = key.removesuffix(suffix)
    value = weights.pop(key)
    splits = (mlp_projections([value], _scan_axes(key, config), split=True, bias=parameter == "bias", array_module=jnp)
              if mlp else _split(value, _projection_axis(key, config, bias=parameter == "bias"), config,
                                 vision=key.startswith("params-vision_encoder-")))
    for name, value in zip(names, splits):
      weights[prefix + name + "-" + parameter] = value
  return weights


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  """Convert decoder projection modules in a native parameter tree."""
  if fused_qkv and (getattr(config, "share_kv_projections", False) or getattr(config, "num_kv_shared_layers", 0)):
    raise ValueError("Shared KV projections or layers are incompatible with fused QKV.")

  def convert(node, path):
    if not isinstance(node, dict):
      return node
    result = {name: convert(value, path + (name,)) for name, value in node.items()}
    if not path:
      return result
    module = path[-1]
    vision = "vision_encoder" in path
    if vision and "Gemma3VisionEncoderLayer_0" in path and module == "MultiHeadDotProductAttention_0":
      names, fused, desired, mlp = ("query", "key", "value"), "qkv_proj", fused_qkv, False
    elif vision and "Gemma4VisionEncoderLayer_0" in path and module == "attention":
      names, fused, desired, mlp = ("query", "key", "value"), "qkv_proj", fused_qkv, False
    elif vision and "Gemma4VisionEncoderLayer_0" in path and module == "mlp":
      names, fused, desired, mlp = ("wi_0", "wi_1"), "wi", fused_mlp, True
    elif "decoder" not in path:
      return result
    elif module in ("self_attention", "self_attention_local", "self_attention_global"):
      names, fused, desired, mlp = ("query", "key", "value"), "qkv_proj", fused_qkv, False
    elif module in ("mlp", "mlp_local", "mlp_global", "shared_experts"):
      names, fused, desired, mlp = ("wi_0", "wi_1"), "wi", fused_mlp, True
    else:
      return result
    projection_path = "params-" + "-".join(path) + "-"
    if fused in result and not desired and isinstance(result[fused], dict):
      projection = result.pop(fused)
      if "kernel" not in projection or set(projection) - {"kernel", "bias"}:
        raise NotImplementedError("Fused Gemma conversion supports unadapted kernels only.")
      for name in names:
        result[name] = {}
      for parameter, value in projection.items():
        splits = (mlp_projections([value], _scan_axes(projection_path, config), split=True,
                                  bias=parameter == "bias", array_module=jnp) if mlp else
                  _split(value, _projection_axis(projection_path, config, bias=parameter == "bias"), config,
                         vision=vision))
        for name, split in zip(names, splits):
          result[name][parameter] = split
    elif desired and all(name in result and isinstance(result[name], dict) for name in names):
      projections = [result.pop(name) for name in names]
      parameters = set(projections[0])
      if "kernel" not in parameters or parameters - {"kernel", "bias"} or any(
          set(projection) != parameters for projection in projections
      ):
        raise NotImplementedError("Fused Gemma conversion supports unadapted kernels only.")
      result[fused] = {
          parameter: (mlp_projections([projection[parameter] for projection in projections],
                                      _scan_axes(projection_path, config), bias=parameter == "bias", array_module=jnp)
                      if mlp else fuse_projections([projection[parameter] for projection in projections],
                                                  _projection_axis(projection_path, config, bias=parameter == "bias"),
                                                  array_module=jnp)) for parameter in parameters
      }
    return result

  return convert(state, ())
