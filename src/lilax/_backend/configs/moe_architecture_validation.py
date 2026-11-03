"""Validation and derivation of native mixture-of-experts layer architecture."""

from lilax._backend.common.common_types import DecoderBlockType
from lilax._backend.configs.model_scaling import get_individual_scales
from lilax._backend.configs.quantization_types import TEGroupedGemmQuantizationType


def validate_moe_architecture(config):
  """Validates MoE dimensions, decoder routing support, and the TE MoE block; aligns Qwen3 MoE MLP dims."""
  if config.moe_mlp_dim <= 0:
    raise ValueError("moe_mlp_dim must be positive for MoE models (num_experts > 1)")
  is_fully_moe = (
      config.interleave_moe_layer_step == 1
      and config.first_num_dense_layers == 0
      and config.inhomogeneous_layer_cycle_interval == 1
  )
  if is_fully_moe and config.base_mlp_dim != config.base_moe_mlp_dim:
    if config.decoder_block == DecoderBlockType.QWEN3_MOE:
      config.base_mlp_dim = config.base_moe_mlp_dim
      _, _, mlp_dim_scale, _ = get_individual_scales(config.global_parameter_scale)
      config.mlp_dim = (2**mlp_dim_scale) * config.base_mlp_dim
    elif config.decoder_block != DecoderBlockType.GEMMA4:
      # Allow Gemma 4 to keep distinct shared and routed MLP dimensions
      raise ValueError(
          "For a fully MoE model, base_mlp_dim must equal base_moe_mlp_dim. "
          f"Got base_mlp_dim={config.base_mlp_dim}, base_moe_mlp_dim={config.base_moe_mlp_dim}."
      )
  if config.decoder_block == DecoderBlockType.GPT_OSS and not config.sparse_matmul and config.capacity_factor != -1:
    raise ValueError("GPT-OSS MoE only supports dropless (capacity_factor=-1) with dense matmul.")
  if (
      config.routed_bias
      and config.routed_bias_update_rate > 0.0
      and config.decoder_block not in (DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4)
  ):
    raise ValueError("Loss-free load balancing is only supported for the DeepSeek decoder block.")
  if config.te_moe_block and not config.sparse_matmul:
    raise ValueError("te_moe_block=True requires sparse_matmul=True.")
  if config.te_moe_quantize_before_fsdp_all_gather:
    if not config.te_moe_block or config.te_gmm_quantization != TEGroupedGemmQuantizationType.TE_MXFP8:
      raise ValueError(
          "te_moe_quantize_before_fsdp_all_gather=True requires te_moe_block=True and te_gmm_quantization=te_mxfp8."
      )
    if config.use_2d_fsdp_sharding:
      raise ValueError("te_moe_quantize_before_fsdp_all_gather=True does not support 2D FSDP weight sharding.")
  if config.te_moe_block and not config.prefuse_moe_weights:
    raise ValueError("te_moe_block=True requires prefuse_moe_weights=True.")
  if config.te_moe_block and config.routed_bias_update_rate > 0.0:
    raise ValueError("te_moe_block=True does not currently support routed_bias_update_rate > 0.")
  if config.te_moe_block and config.norm_topk_prob:
    raise ValueError("te_moe_block=True does not currently support norm_topk_prob=True.")
  if config.te_moe_block and config.use_random_routing:
    raise ValueError("te_moe_block=True does not support use_random_routing=True.")
  if config.te_moe_block and config.decoder_block == DecoderBlockType.LLAMA4:
    raise ValueError("te_moe_block=True does not currently support Llama4 routing semantics.")
  if config.te_moe_block and not config.te_gmm_quantization:
    raise ValueError(
        "te_gmm_quantization must be specified when te_moe_block=True. "
        "te_gmm_quantization=te_no_quant is supported for BF16."
    )
  if config.model_name.startswith("deepseek4") and config.first_num_hash_layers > 0 and config.use_ring_of_experts:
    raise ValueError("DeepSeek V4 hash routing is currently not supported with ring of experts.")
