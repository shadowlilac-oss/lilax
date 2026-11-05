"""Convert Qwen3.5 decoder and vision projection layouts."""

from lilax._backend.checkpoint_conversion.model_families.qwen3_next import fused as decoder_fused
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl import fused as tower_fused


def fuse_for_maxtext(weights, target, config, lazy_tensor=None):
  tower_fused.validate_tower_fusion(weights, target)
  text_target = {key: value for key, value in target.items() if key.startswith("params-decoder-")}
  decoder_fused.fuse_for_maxtext(weights, text_target, config, lazy_tensor)
  tower_fused.fuse_tower_for_maxtext(weights, target, lazy_tensor)
  return [weights[key] for key in target]


def unfuse_for_huggingface(weights, config):
  if any(key.startswith("params-vision_encoder-") for key in weights):
    raise NotImplementedError("Hugging Face export of Qwen3.5 vision weights is not supported.")
  return decoder_fused.unfuse_for_huggingface(weights, config)


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  def convert_towers(node):
    if not isinstance(node, dict):
      return node
    return {
        key: tower_fused.convert_tower_state(value, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
        if key == "vision_encoder" else value if key == "decoder" else convert_towers(value)
        for key, value in node.items()
    }

  def convert_decoders(node):
    if not isinstance(node, dict):
      return node
    return {
        key: decoder_fused.convert_native_state(value, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
        if key == "decoder" else value if key == "vision_encoder" else convert_decoders(value)
        for key, value in node.items()
    }

  return convert_decoders(convert_towers(state))
