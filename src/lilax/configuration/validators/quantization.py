from lilax.configuration.domain_types import DType, QuantizationType
from lilax.configuration.domains.execution_sharding import ShardMode


def validate_quantization_options(self):
  if self.quantize_kvcache and not self.kv_quant_axis:
    raise ValueError("`kv_quant_axis` cannot be empty when quantize_kvcache is True.")
  if self.mtp_reuse_input_embedding and self.use_multimodal:
    raise ValueError("`mtp_reuse_input_embedding` does not support multimodal inputs.")
  if self.mtp_norm_main_hidden_state:
    if self.num_vocab_tiling > 1:
      raise ValueError("`mtp_norm_main_hidden_state` is not supported with `num_vocab_tiling > 1`.")
    if self.mhc_expansion_rate > 1:
      raise ValueError("`mtp_norm_main_hidden_state` is not supported with `mhc_expansion_rate > 1`.")
  if self.quantize_mtp:
    if self.mtp_num_layers <= 0:
      raise ValueError("`quantize_mtp` can only be enabled when `mtp_num_layers > 0`.")
    if self.quantization != "fp8_full":
      raise ValueError("`quantize_mtp` can only be enabled when `quantization='fp8_full'`.")
  if self.quantize_logits_proj:
    if self.logits_via_embedding:
      raise ValueError("`quantize_logits_proj` cannot be enabled when `logits_via_embedding=True`.")
    if self.quantization != "fp8_full":
      raise ValueError("`quantize_logits_proj` can only be enabled when `quantization='fp8_full'`.")
    if self.num_vocab_tiling > 1:
      raise ValueError("`quantize_logits_proj` is not supported with `num_vocab_tiling > 1`.")
    if self.logits_dot_in_fp32:
      raise ValueError("`logits_dot_in_fp32=True` is rejected with `quantize_logits_proj=True`.")
  if self.quantization and self.use_qwix_quantization and self.quantize_router_proj and self.float32_gate_logits:
    raise ValueError("`float32_gate_logits=True` is rejected with `quantize_router_proj=True`.")
  if self.quantization in ("fp8", "nanoo_fp8", "fp8_gpu", "te_fp8_delayedscaling") and (
      self.gradient_accumulation_steps > 1
  ):
    raise ValueError("FP8 quantization is not compatible with gradient accumulation.")
  if self.use_batch_split_schedule and self.quantization and self.quantization != "fp8_full":
    raise ValueError("Batch split quantization only supports `quantization=fp8_full`")
  if self.use_sliced_mla_proj and self.quantization and (
      not self.use_qwix_quantization or self.use_batch_split_schedule
  ):
    raise ValueError(
        "`use_sliced_mla_proj` with quantization is only supported via Qwix interception "
        "(use_qwix_quantization=True, use_batch_split_schedule=False)."
    )
  if self.use_manual_quantization and not self.use_batch_split_schedule:
    raise ValueError("manual quantization is only used when `use_batch_split_schedule=True`.")
  if self.quantization == QuantizationType.SERVE_FP8_WEIGHT:
    if self.weight_dtype not in (DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2):
      raise ValueError(
          "quantization='serve_fp8_weight' requires weight_dtype to be an FP8 dtype "
          f"('float8_e4m3fn' or 'float8_e5m2'), got weight_dtype={self.weight_dtype!r}."
      )
    if self.use_qwix_quantization:
      raise ValueError("quantization='serve_fp8_weight' is not supported with use_qwix_quantization=True.")
  if self.quantization in (QuantizationType.JAX_MXFP8, QuantizationType.JAX_MXFP4):
    name = QuantizationType(self.quantization).value
    if self.use_qwix_quantization:
      raise ValueError(f"quantization='{name}' does not support use_qwix_quantization=True.")
    if self.shard_mode == ShardMode.EXPLICIT:
      raise ValueError(f"quantization='{name}' does not support shard_mode='explicit'.")
    if self.num_experts > 1:
      raise ValueError(f"quantization='{name}' does not support MoE models (num_experts > 1).")
    if self.weight_dtype in (DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2):
      raise ValueError(
          f"quantization='{name}' quantizes the weights itself; weight_dtype must not be fp8, got"
          f" '{DType(self.weight_dtype).value}'."
      )
  return self
