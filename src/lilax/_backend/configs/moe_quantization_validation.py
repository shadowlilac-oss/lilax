"""Validation for quantized mixture-of-experts communication."""


def validate_moe_quantize_token_all_gather(config):
  """Validates that moe_quantize_token_all_gather is used with supported settings."""
  if config.moe_quantize_token_all_gather:
    if not config.sparse_matmul:
      raise ValueError("moe_quantize_token_all_gather=True requires sparse_matmul=True.")
    if not config.use_ring_of_experts:
      raise ValueError("moe_quantize_token_all_gather=True requires use_ring_of_experts=True.")
    if not config.use_gmm_v2:
      raise ValueError("moe_quantize_token_all_gather=True requires use_gmm_v2=True.")
    if config.num_moe_emb_chunks > 0:
      raise ValueError("moe_quantize_token_all_gather=True does not support num_moe_emb_chunks > 0.")
    if config.quantization == "" or not config.act_quantization_calibration_method.lower().startswith("fixed"):
      raise ValueError(
          "moe_quantize_token_all_gather=True requires quantization to be"
          " specified and act_quantization_calibration_method to be fixed"
          " (static scaling mode)."
      )


def validate_moe_quantize_combine_bwd_method(config):
  """Validates that moe_quantize_combine_bwd_method is only used with ring of experts."""
  if config.moe_quantize_combine_bwd_method:
    if not config.use_ring_of_experts:
      raise ValueError(
          f"moe_quantize_combine_bwd_method='{config.moe_quantize_combine_bwd_method}' requires use_ring_of_experts=True."
      )
    method = config.moe_quantize_combine_bwd_method
    if not (method == "rowwise" or method.startswith("fixed")):
      raise ValueError(
          f"Unsupported moe_quantize_combine_bwd_method: '{method}'. "
          "Supported options: '', 'rowwise', 'fixed,<bound>'."
      )
