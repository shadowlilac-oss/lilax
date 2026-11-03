"""Map precision arguments to configuration overrides."""

PRECISION_FIELDS = (
    "quantization", "use_qwix_quantization", "use_manual_quantization", "jax_mx_gradient_mode",
    "quant_cfg_path", "quantization_local_shard_count", "qwix_bwd_weight_grad_tile_count",
    "weight_quantization_calibration_method", "act_quantization_calibration_method",
    "bwd_quantization_calibration_method", "drhs_grad_quantization_calibration_method",
    "quantize_mtp", "quantize_router_proj", "quantize_logits_proj", "logits_proj_quant_calibration_method",
    "logits_proj_bwd_quant_calibration_method", "sa_qproj_bwd_bf16", "replicate_quant_scale",
)


def precision_overrides(args):
  return {name: getattr(args, name) for name in PRECISION_FIELDS if getattr(args, name) is not None}
