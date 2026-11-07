"""Precision validation for shared trainers."""

from ..arguments.training_args import PRECISION_FIELDS


def check_precision_overrides(graph_config, overrides):
  changed = [
      name for name in PRECISION_FIELDS
      if name in overrides and overrides[name] != getattr(graph_config, name)
  ]
  for name in ("dtype", "grad_dtype", "weight_dtype", "matmul_precision", "remat_policy", "remat_policy_for_vit",
               "parameter_memory_host_offload", "optimizer_memory_host_offload"):
    if name in overrides:
      requested, current = overrides[name], getattr(graph_config, name)
      if str(getattr(requested, "value", requested)) != str(getattr(current, "value", current)):
        changed.append(name)
  if changed:
    raise ValueError(f"Cannot change precision settings on a materialized model: {', '.join(changed)}.")


def check_precision_options(config, overrides):
  quant = config.quantization
  qwix = config.use_qwix_quantization and not config.use_batch_split_schedule
  qwix_modes = ("int4", "int8", "fp4", "fp8", "fp8_e5m2", "fp8_e4m3", "fp8_full", "fp8_gpu", "fp8_nanoo")
  if qwix and quant and quant not in qwix_modes:
    raise ValueError(f"Qwix has no provider for quantization={quant!r}.")
  if overrides.get("use_qwix_quantization") and not quant:
    raise ValueError("use_qwix_quantization requires a quantization recipe.")
  full = quant == "fp8_full" and (qwix or (config.use_batch_split_schedule and not config.use_manual_quantization))
  applicable = {
      "jax_mx_gradient_mode": quant in ("jax_mxfp4", "jax_mxfp8"),
      "quant_cfg_path": quant == "intmp" and not qwix,
      "quantization_local_shard_count": quant == "int8" and not qwix and not config.use_batch_split_schedule,
      "replicate_quant_scale": quant in ("int8", "intmp") and not qwix and not config.use_batch_split_schedule,
      "qwix_bwd_weight_grad_tile_count": qwix and quant in qwix_modes and quant != "fp8_full",
      "weight_quantization_calibration_method": full,
      "act_quantization_calibration_method": full or quant == "serve_fp8_weight",
      "bwd_quantization_calibration_method": full,
      "drhs_grad_quantization_calibration_method": qwix and quant == "fp8_full",
      "quantize_mtp": qwix and quant == "fp8_full",
      "quantize_router_proj": qwix and quant in qwix_modes,
      "quantize_logits_proj": qwix and quant == "fp8_full",
      "logits_proj_quant_calibration_method": qwix and quant == "fp8_full" and config.quantize_logits_proj,
      "logits_proj_bwd_quant_calibration_method": qwix and quant == "fp8_full" and config.quantize_logits_proj,
      "sa_qproj_bwd_bf16": qwix and quant == "fp8_full",
      "use_manual_quantization": config.use_batch_split_schedule,
  }
  unused = [name for name, supported in applicable.items() if name in overrides and not supported]
  if unused:
    raise ValueError(f"Precision settings do not apply to the selected recipe: {', '.join(unused)}.")
