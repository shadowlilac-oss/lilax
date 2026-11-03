"""Family-specific layer layouts used during decoder construction."""

from lilax._backend.common.common_types import DecoderBlockType
from lilax._backend.models import gemma3, gemma4, gpt_oss, llama4, olmo3


def scanned_generic_layout(config):
  """Return the number of scanned cycles and their family-specific arguments."""
  cycle_interval = config.inhomogeneous_layer_cycle_interval
  if config.num_decoder_layers % cycle_interval:
    raise ValueError(
        f"num_decoder_layers ({config.num_decoder_layers}) must be a multiple of"
        f" inhomogeneous_layer_cycle_interval ({cycle_interval}) to scan {config.decoder_block.value} layers:"
        " each scan step builds one whole cycle, so the remaining layers would be dropped."
    )
  num_layers = config.num_decoder_layers // cycle_interval
  layer_kwargs = {}
  if config.decoder_block == DecoderBlockType.LLAMA4:
    layer_kwargs = {
        "nope_layer_interval": config.nope_layer_interval,
        "interleave_moe_layer_step": config.interleave_moe_layer_step,
    }
  if config.decoder_block == DecoderBlockType.ENVY:
    layer_kwargs = {"interleave_moe_layer_step": config.interleave_moe_layer_step}
  return num_layers, layer_kwargs


def sequential_layer_kwargs(config, lyr):
  """Return the family-specific arguments for one sequential layer."""
  layer_kwargs = {}
  if config.decoder_block == DecoderBlockType.GEMMA3:
    layer_kwargs = {"attention_type": gemma3.get_attention_type(layer_id=lyr)}
  elif config.decoder_block == DecoderBlockType.GEMMA4:
    layer_kwargs = {"attention_type": gemma4.get_attention_type(layer_id=lyr)}
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    layer_kwargs = {
        "is_nope_layer": llama4.determine_is_nope_layer(lyr, config.nope_layer_interval),
        "is_moe_layer": llama4.determine_is_moe_layer(lyr, config.interleave_moe_layer_step),
    }
  elif config.decoder_block == DecoderBlockType.ENVY:
    layer_kwargs = {"is_moe_layer": (lyr + 1) % config.interleave_moe_layer_step == 0}
  elif config.decoder_block in {
      DecoderBlockType.QWEN3_NEXT,
      DecoderBlockType.QWEN3_5,
      DecoderBlockType.DEEPSEEK4,
      DecoderBlockType.MING,
      DecoderBlockType.HY4,
  }:
    layer_kwargs = {"layer_idx": lyr}
  elif config.decoder_block == DecoderBlockType.GPT_OSS:
    layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}
  elif config.decoder_block == DecoderBlockType.OLMO3:
    layer_kwargs = {"attention_type": olmo3.get_attention_type(layer_id=lyr)}
  return layer_kwargs
