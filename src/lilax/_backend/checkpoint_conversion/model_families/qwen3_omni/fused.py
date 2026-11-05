"""Convert Qwen3-Omni text decoder projection layouts."""

from lilax._backend.checkpoint_conversion.model_families.qwen3_vl.fused import (
    convert_tower_state, fuse_tower_for_maxtext, validate_tower_fusion,
)
from lilax._backend.checkpoint_conversion.model_families.qwen import fused as qwen_fused


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  validate_tower_fusion(weights, target, audio=True)
  text_target = {key: value for key, value in target.items() if key.startswith("params-decoder-")}
  qwen_fused.fuse_for_maxtext(weights, text_target, config, lazy_tensor)
  fuse_tower_for_maxtext(weights, target, lazy_tensor, audio=True)
  return [weights[key] for key in target]


def _convert_native_towers(state, *, fused_qkv, fused_mlp):
  if not isinstance(state, dict):
    return state
  return {
      key: convert_tower_state(value, fused_qkv=fused_qkv, fused_mlp=fused_mlp, audio=key == "audio_encoder")
      if key in ("vision_encoder", "audio_encoder") else value if key == "decoder" else
      _convert_native_towers(value, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
      for key, value in state.items()
  }


def _convert_native_decoder(state, config, *, fused_qkv, fused_mlp):
  if not isinstance(state, dict):
    return state
  return {
      key: qwen_fused.convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
      if key == "decoder" else value if key in ("vision_encoder", "audio_encoder") else
      _convert_native_decoder(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
      for key, value in state.items()
  }


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  state = _convert_native_towers(state, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
  return _convert_native_decoder(state, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)


def unfuse_for_huggingface(weights, config):
  raise NotImplementedError("Hugging Face export of Qwen3-Omni is not supported.")
