"""FLOPs accounting for dense and mixture-of-experts feed-forward layers."""

from lilax._backend.common.common_types import DecoderBlockType


def calculate_ffn_mamtul_tflops_per_device(config, mlp_dim, in_dim=None):
  if in_dim is None:
    in_dim = config.emb_dim
  ffn1_flops = (
      2 * config.per_device_batch_size * config.max_target_length * mlp_dim * in_dim * len(config.mlp_activations)
  )
  ffn2_flops = 2 * config.per_device_batch_size * config.max_target_length * mlp_dim * in_dim
  return ffn1_flops + ffn2_flops


def get_shared_expert_mlp_dim(config):
  return config.mlp_dim if config.decoder_block == DecoderBlockType.GEMMA4 else config.moe_mlp_dim


def get_dense_moe_layers(config):
  if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.MING):
    num_dense_layers = config.first_num_dense_layers
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    return num_dense_layers, num_moe_layers
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    num_moe_layers = config.num_decoder_layers // config.interleave_moe_layer_step
    num_dense_layers = config.num_decoder_layers - num_moe_layers
    return num_dense_layers, num_moe_layers
  elif config.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.DEEPSEEK4):
    return 0, config.num_decoder_layers
  elif config.decoder_block == DecoderBlockType.DEFAULT:
    raise ValueError("Unsupported decoder block for dense/MoE layer calculation")

  num_experts = getattr(config, "num_experts", 0)
  if num_experts > 1:
    num_moe_layers = config.num_decoder_layers
    num_dense_layers = 0
  else:
    num_moe_layers = 0
    num_dense_layers = config.num_decoder_layers
  return num_dense_layers, num_moe_layers


def calculate_routed_and_shared_ffn_tflops_per_device(config):
  gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
  num_dense_layers, num_moe_layers = get_dense_moe_layers(config)
  dense_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * num_dense_layers
  shared_experts_flops = (
      calculate_ffn_mamtul_tflops_per_device(config, get_shared_expert_mlp_dim(config)) * config.shared_experts
  )
  routed_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
  moe_ffn_flops = (gate_flops + shared_experts_flops + routed_experts_flops) * num_moe_layers
  return dense_ffn_flops + moe_ffn_flops
