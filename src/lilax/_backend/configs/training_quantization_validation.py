"""Validation of native quantized training projections and accumulation."""


def validate_training_quantization(config):
  if config.quantize_mtp:
    if config.mtp_num_layers <= 0:
      raise ValueError("`quantize_mtp` can only be enabled when `mtp_num_layers > 0`.")
    if config.quantization != "fp8_full":
      raise ValueError("`quantize_mtp` can only be enabled when `quantization='fp8_full'`.")
  if config.quantize_logits_proj:
    if config.logits_via_embedding:
      raise ValueError(
          "`quantize_logits_proj` cannot be enabled when input and output embeddings are tied"
          " (`logits_via_embedding=True`)."
      )
    if config.quantization != "fp8_full":
      raise ValueError("`quantize_logits_proj` can only be enabled when `quantization='fp8_full'`.")
    if config.num_vocab_tiling > 1:
      raise ValueError(
          "`quantize_logits_proj` is not supported with `num_vocab_tiling > 1`: under vocab tiling the"
          " decoder skips `apply_output_head` in train mode, so `logits_dense` is absent from the forward"
          " pass that `qwix.quantize_model` traces, and the projection runs later inside"
          " `vocab_tiling_nnx_loss` on a merged model copy. Whether interception survives that path is"
          " unverified; this combination is rejected until it is tested."
      )
    if config.logits_dot_in_fp32:
      raise ValueError(
          "`logits_dot_in_fp32=True` is rejected with `quantize_logits_proj=True`: the fp32 cast on"
          " the logits operands is undone by requantization at the projection matmul, so the projection"
          " remains quantized while believing you configured fp32. Set `quantize_logits_proj=False` to keep"
          " the projection in fp32."
      )
  if config.quantization and config.use_qwix_quantization and config.quantize_router_proj and config.float32_gate_logits:
    raise ValueError(
        "`float32_gate_logits=True` is rejected with `quantize_router_proj=True`: the fp32 cast on"
        " the gate operands is undone by requantization at the projection matmul, so the projection"
        " remains quantized while believing you configured fp32. The flag's remaining effects (bias"
        " add, score function, gemma4 router norm) are second-order next to the quantization error."
        " Set `quantize_router_proj=False` to keep the projection in fp32."
    )
  if (
      config.quantization in ("fp8", "nanoo_fp8", "fp8_gpu", "te_fp8_delayedscaling")
      and config.gradient_accumulation_steps > 1
  ):
    raise ValueError("FP8 quantization is not compatible with gradient accumulation.")
