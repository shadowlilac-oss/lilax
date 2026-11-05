"""Convert Qwen3-VL projection layouts."""

from functools import partial

import jax.numpy as jnp
import numpy as np

from lilax._backend.checkpoint_conversion.model_families.qwen import fused as qwen_fused
from lilax._backend.checkpoint_conversion.projection_primitives import fuse_projections
from lilax._backend.common.common_types import is_fp8_dtype


def _validate_tower_projection(projection):
  if (not isinstance(projection, dict) or not set(projection) <= {"kernel", "bias"}
      or "kernel" not in projection):
    raise NotImplementedError("Tower projection conversion supports unadapted kernels and biases only.")
  if is_fp8_dtype(projection["kernel"].dtype):
    raise NotImplementedError("FP8 tower projection conversion is not supported.")


def _flat_projection(weights, prefix):
  return {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}


def validate_tower_fusion(weights, target, *, audio=False):
  for key in target:
    vision_qkv = key.startswith("params-vision_encoder-") and "-attn-attn-qkv_proj-" in key
    audio_qkv = audio and key.startswith("params-audio_encoder-") and "-self_attention_audio-qkv_proj-" in key
    if vision_qkv or audio_qkv:
      prefix = key.split("qkv_proj-", 1)[0]
      if set(_flat_projection(target, prefix + "qkv_proj-")) - {"kernel", "bias"}:
        raise NotImplementedError("Tower projection conversion supports unadapted kernels and biases only.")
      for name in ("query", "key", "value"):
        _validate_tower_projection(_flat_projection(weights, prefix + name + "-"))
    elif audio and key.startswith("params-audio_encoder-") and "-AudioMLP-wi-" in key:
      prefix = key.split("-AudioMLP-wi-", 1)[0] + "-AudioMLP-wi-"
      if set(_flat_projection(target, prefix)) - {"kernel", "bias"}:
        raise NotImplementedError("Tower projection conversion supports unadapted kernels and biases only.")
      _validate_tower_projection(_flat_projection(weights, prefix))


def fuse_tower_for_maxtext(weights, target, lazy_tensor=None, *, audio=False):
  validate_tower_fusion(weights, target, audio=audio)
  for key, (_, shape) in target.items():
    vision_qkv = key.startswith("params-vision_encoder-") and "-attn-attn-qkv_proj-" in key
    audio_qkv = audio and key.startswith("params-audio_encoder-") and "-self_attention_audio-qkv_proj-" in key
    if vision_qkv or audio_qkv:
      kind = key.rsplit("-", 1)[-1]
      if kind not in ("kernel", "bias"):
        continue
      prefix = key.removesuffix(f"qkv_proj-{kind}")
      values = [weights.pop(prefix + name + f"-{kind}") for name in ("query", "key", "value")]
      load = partial(fuse_projections, values, 1 if kind == "kernel" else 0)
    elif (audio and key.startswith("params-audio_encoder-") and "-AudioMLP-wi-" in key
          and key.rsplit("-", 1)[-1] in ("kernel", "bias") and len(weights[key].shape) != len(shape)):
      value = weights.pop(key)
      values = [value]
      load = partial(np.expand_dims, value, 1 if key.endswith("-kernel") else 0)
    else:
      continue
    weights[key] = lazy_tensor(load, shape, values[0].dtype, name=key) if lazy_tensor else load()
    if weights[key].shape != shape:
      raise ValueError(f"Unexpected fused shape for {key}: {weights[key].shape}, expected {shape}")


def validate_tower_unfusion(weights):
  for key in weights:
    if key.startswith("params-vision_encoder-") and "-attn-attn-qkv_proj-" in key:
      prefix = key.split("qkv_proj-", 1)[0] + "qkv_proj-"
      _validate_tower_projection(_flat_projection(weights, prefix))


def unfuse_tower_for_huggingface(weights):
  validate_tower_unfusion(weights)
  for key in list(weights):
    if key.startswith("params-vision_encoder-") and "-attn-attn-qkv_proj-" in key:
      kind = key.rsplit("-", 1)[-1]
      if kind not in ("kernel", "bias"):
        continue
      prefix = key.removesuffix(f"qkv_proj-{kind}")
      for name, value in zip(("query", "key", "value"), jnp.split(weights.pop(key), 3, axis=1 if kind == "kernel" else 0)):
        weights[prefix + name + f"-{kind}"] = value
  return weights


def convert_tower_state(state, *, fused_qkv, fused_mlp, audio=False):
  if not isinstance(state, dict):
    return state
  result = {key: convert_tower_state(value, fused_qkv=fused_qkv, fused_mlp=fused_mlp, audio=audio)
            for key, value in state.items()}
  for name in (("self_attention_audio",) if audio else ("attn",)):
    attention = result.get(name)
    if not isinstance(attention, dict):
      continue
    if "qkv_proj" in attention and not fused_qkv:
      projection = attention["qkv_proj"]
      _validate_tower_projection(projection)
      attention.pop("qkv_proj")
      projections = {part: {} for part in ("query", "key", "value")}
      for kind, weight in projection.items():
        for part, value in zip(projections, jnp.split(weight, 3, axis=1 if kind == "kernel" else 0)):
          projections[part][kind] = value
      attention.update(projections)
    elif fused_qkv and "query" in attention:
      projections = [attention[part] for part in ("query", "key", "value")]
      for projection in projections:
        _validate_tower_projection(projection)
      kinds = set(projections[0])
      if any(set(item) != kinds for item in projections):
        raise NotImplementedError("Tower QKV conversion supports unadapted kernels and biases only.")
      for part in ("query", "key", "value"):
        attention.pop(part)
      attention["qkv_proj"] = {
          kind: fuse_projections([item[kind] for item in projections], 1 if kind == "kernel" else 0, array_module=jnp)
          for kind in projections[0]
      }
  mlp = result.get("AudioMLP") if audio else None
  if isinstance(mlp, dict) and "wi" in mlp:
    projection = mlp["wi"]
    _validate_tower_projection(projection)
    for kind, weight in projection.items():
      axis = 1 if kind == "kernel" else 0
      unfused_ndim = 2 if kind == "kernel" else 1
      if fused_mlp and weight.ndim == unfused_ndim:
        projection[kind] = jnp.expand_dims(weight, axis)
      elif not fused_mlp and weight.ndim == unfused_ndim + 1:
        projection[kind] = jnp.squeeze(weight, axis=axis)
  return result


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  validate_tower_fusion(weights, target)
  text_target = {key: value for key, value in target.items() if key.startswith("params-decoder-")}
  qwen_fused.fuse_for_maxtext(weights, text_target, config, lazy_tensor)
  fuse_tower_for_maxtext(weights, target, lazy_tensor)
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  validate_tower_unfusion(weights)
  text_weights = {key: value for key, value in weights.items() if key.startswith("params-decoder-")}
  for key in text_weights:
    del weights[key]
  weights.update(qwen_fused.unfuse_for_huggingface(text_weights, config))
  return unfuse_tower_for_huggingface(weights)


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  if not isinstance(state, dict):
    return state
  return {
      key: qwen_fused.convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
      if key == "decoder" else convert_tower_state(value, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
      if key == "vision_encoder" else convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
      for key, value in state.items()
  }
