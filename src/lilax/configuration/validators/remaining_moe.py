from lilax.configuration.domain_types import DecoderBlockType, DType


def validate_moe_options(self) -> "MaxTextConfig":
  if self.moe_use_direct_token_gather and self.use_gather_mosaic_kernel:
    raise ValueError("`moe_use_direct_token_gather=True` currently requires `use_gather_mosaic_kernel=False`.")
  if self.te_moe_disable_cudnn_gmm_fusion and not self.te_moe_block:
    raise ValueError("te_moe_disable_cudnn_gmm_fusion=True requires te_moe_block=True.")
  if self.use_gmm_v2:
    if not self.use_tokamax_gmm:
      raise ValueError("GMM v2 requires `use_tokamax_gmm=True`.")
    if self.use_batch_split_schedule:
      raise ValueError("GMM v2 is not supported with a batch split schedule.")
  if self.use_gmm_v2_heuristic_tiling and not self.use_gmm_v2:
    raise ValueError("`use_gmm_v2_heuristic_tiling=True` requires `use_gmm_v2=True`.")
  if self.use_te_grouped_gemm:
    if not self.sparse_matmul:
      raise ValueError(
          "use_te_grouped_gemm=True requires sparse_matmul=True; the TransformerEngine "
          "grouped GEMM backend is only reachable from the sparse MoE path, so it would "
          "otherwise be silently ignored. Note that some GPU model configs "
          "(e.g. gpu/models/gpt-oss-20b.yml) set sparse_matmul=false."
      )
    if self.megablox and self.hardware != "cpu":
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with megablox=True. "
          "Set megablox=False to use the TransformerEngine grouped GEMM backend."
      )
    if self.use_tokamax_gmm:
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with use_tokamax_gmm=True. "
          "Disable Tokamax GMM to use the TransformerEngine grouped GEMM backend."
      )
    if self.quantization:
      raise ValueError(
          "use_te_grouped_gemm=True does not support quantization yet "
          f"(got quantization={self.quantization!r}). Set quantization=''."
      )
    if self.moe_tc_ragged_sort:
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with moe_tc_ragged_sort=True: the TensorCore ragged sort "
          "applies routing weights per buffer row, which TE's per-expert grouped GEMM bias cannot carry."
      )
    if self.moe_flat_fsdp_weights:
      raise ValueError(
          "use_te_grouped_gemm=True is not compatible with moe_flat_fsdp_weights=True: the TransformerEngine "
          "grouped GEMM takes 3D [experts, rows, cols] expert weights, not the flat 2D layout."
      )
  if self.num_moe_token_chunks > 1 and self.max_target_length % self.num_moe_token_chunks != 0:
    raise ValueError(
        f"num_moe_token_chunks={self.num_moe_token_chunks} must evenly divide "
        f"max_target_length={self.max_target_length}."
    )
  if self.moe_accumulate_chunk_wgrad and not (
      self.use_tokamax_gmm and self.use_gmm_v2 and self.num_moe_emb_chunks == 0 and not self.prefuse_moe_weights
  ):
    raise ValueError(
        "moe_accumulate_chunk_wgrad=True requires use_tokamax_gmm=True, use_gmm_v2=True, num_moe_emb_chunks=0"
        " and prefuse_moe_weights=False."
    )
  if self.moe_accumulate_wi_dlhs and not (self.use_tokamax_gmm and self.use_gmm_v2 and not self.prefuse_moe_weights):
    raise ValueError(
        "moe_accumulate_wi_dlhs=True requires use_tokamax_gmm=True, use_gmm_v2=True and prefuse_moe_weights=False."
    )
  if self.moe_bwd_prequant_before_unsort and not (
      self.moe_tc_ragged_3d_dispatch and self.moe_tc_ragged_weights_on_activation
  ):
    raise ValueError(
        "moe_bwd_prequant_before_unsort=True requires moe_tc_ragged_3d_dispatch=True and"
        " moe_tc_ragged_weights_on_activation=True."
    )
  if self.moe_combine_bwd_direct_qarray and not self.moe_bwd_prequant_before_unsort:
    raise ValueError("moe_combine_bwd_direct_qarray=True requires moe_bwd_prequant_before_unsort=True.")
  if self.moe_bwd_prequant_before_unsort and self.quantization:
    dynamic = [
        f"{name}_quantization_calibration_method={method!r}"
        for name in ("weight", "act", "bwd")
        if not (method := getattr(self, f"{name}_quantization_calibration_method")).lower().startswith("fixed")
    ]
    if dynamic:
      raise ValueError(
          "moe_bwd_prequant_before_unsort=True with quantization requires fixed weight, act and bwd calibration,"
          f" got {', '.join(dynamic)}."
      )
  if self.moe_combine_bwd_direct_qarray and self.moe_quantize_combine_bwd_method:
    raise ValueError(
        f"moe_quantize_combine_bwd_method={self.moe_quantize_combine_bwd_method!r} has no effect with"
        " moe_combine_bwd_direct_qarray=True: the fused EP combine quantizes the cotangent with the gmm bwd rule."
        " Unset moe_quantize_combine_bwd_method."
    )
  if self.moe_3d_dlhs_partial_sum and not (self.moe_accumulate_wi_dlhs and self.moe_tc_ragged_3d_gmm):
    raise ValueError("moe_3d_dlhs_partial_sum=True requires moe_accumulate_wi_dlhs=True and moe_tc_ragged_3d_gmm=True.")
  if self.moe_flat_fsdp_weights:
    if not (
        self.sparse_matmul and (self.shard_embed_moe_on_fsdp or not self.quantization)
        and ((self.use_tokamax_gmm and self.use_gmm_v2) or self.prefetch_fsdp_weights)
        and not self.prefuse_moe_weights
    ):
      raise ValueError(
          "moe_flat_fsdp_weights=True requires sparse_matmul=True, (shard_embed_moe_on_fsdp=True or unquantized),"
          " ((use_tokamax_gmm=True and use_gmm_v2=True) or prefetch_fsdp_weights=True) and prefuse_moe_weights=False."
      )
    if (
        self.shard_exp_on_fsdp or self.use_2d_fsdp_sharding or self.use_batch_split_schedule
        or self.weight_dtype in (DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2)
        or self.decoder_block == DecoderBlockType.GEMMA4
    ):
      raise ValueError(
          "moe_flat_fsdp_weights=True is incompatible with shard_exp_on_fsdp, use_2d_fsdp_sharding,"
          " use_batch_split_schedule, fp8 weight_dtype and the gemma4 decoder block."
      )
  return self
