from lilax.configuration.domain_types import DatasetType


def validate_attention_kernel_options(self) -> "MaxTextConfig":
  if self.force_q_layout and not self.use_jax_splash:
    raise ValueError("`force_q_layout` can only be true if `use_jax_splash` is also true.")
  if self.use_qk_clip and self.attention_type != "mla":
    raise ValueError(
        f"QK-Clip is only supported when attention_type='mla', but found attention_type='{self.attention_type}'."
    )
  if self.use_qk_clip and self.attn_logits_soft_cap not in (None, 0.0):
    raise ValueError(
        "QK-Clip monitors raw dot products, but attn_logits_soft_cap is enabled. "
        "Recording pre-cap max_logits is not fully supported yet. "
        "Please disable attn_logits_soft_cap when using use_qk_clip."
    )
  if self.experimental_sa_quant_q_fp8 and self.attention_type != "mla":
    raise ValueError(
        "Q quantization is currently only supported with"
        f" attention_type='mla'. Found attention_type='{self.attention_type}'"
    )
  if self.experimental_sa_quant_k_fp8 and self.attention_type != "mla":
    raise ValueError(
        "K quantization is currently only supported with"
        f" attention_type='mla'. Found attention_type='{self.attention_type}'"
    )
  if self.sa_qproj_bwd_bf16 and not (
      self.quantization == "fp8_full" and self.use_qwix_quantization and not self.use_batch_split_schedule
  ):
    raise ValueError(
        "sa_qproj_bwd_bf16=True only takes effect with quantization='fp8_full', use_qwix_quantization=True and"
        " use_batch_split_schedule=False."
    )
  if self.sa_q_prescaled_log2e:
    if not (
        self.experimental_sa_quant_q_fp8 and self.sa_use_base2_exp
        and self.use_tokamax_splash and self.attention == "flash"
    ):
      raise ValueError(
          "sa_q_prescaled_log2e requires experimental_sa_quant_q_fp8=True, sa_use_base2_exp=True,"
          " use_tokamax_splash=True and attention='flash'."
      )
    if self.attention_type != "mla":
      raise ValueError(f"sa_q_prescaled_log2e only supports attention_type='mla'; got '{self.attention_type}'.")
    if self.use_mla_absorbed_mqa:
      raise ValueError("sa_q_prescaled_log2e is not supported with use_mla_absorbed_mqa=True.")
    if self.context_parallel_strategy.lower() in ("ring", "usp"):
      raise ValueError(
          "sa_q_prescaled_log2e is not supported with context_parallel_strategy"
          f" '{self.context_parallel_strategy}'; use 'all_gather' or 'ulysses'."
      )
    if self.sa_dkv_reuse_scaled_q:
      raise ValueError("sa_q_prescaled_log2e subsumes sa_dkv_reuse_scaled_q; set only one of them.")
  if self.sa_dkv_reuse_scaled_q:
    if not (self.sa_use_base2_exp and self.use_tokamax_splash and self.attention == "flash"):
      raise ValueError(
          "sa_dkv_reuse_scaled_q only takes effect with sa_use_base2_exp=True, use_tokamax_splash=True and"
          " attention='flash'."
      )
    if self.context_parallel_strategy.lower() in ("ring", "usp"):
      raise ValueError(
          "sa_dkv_reuse_scaled_q is not supported with context_parallel_strategy"
          f" '{self.context_parallel_strategy}'; use 'all_gather' or 'ulysses'."
      )
  if self.sa_ignore_segment_ids:
    train_stream_chunked = self.use_stream_chunking is True or (
        self.use_stream_chunking is None and not self.tokenize_train_data
    )
    eval_stream_chunked = self.use_stream_chunking is True or (
        self.use_stream_chunking is None and not self.tokenize_eval_data
    )
    if not (
        self.dataset_type == DatasetType.C4MLPERF and train_stream_chunked
        and (self.eval_interval <= 0 or eval_stream_chunked)
    ):
      raise ValueError(
          "sa_ignore_segment_ids requires dataset_type='c4_mlperf' with"
          " stream chunking for train and eval (use_stream_chunking=True, or"
          " None with pre-tokenized data); packed sequences need the segment mask."
      )
    if self.attention != "flash":
      raise ValueError("sa_ignore_segment_ids only applies to attention='flash'; got" f" attention='{self.attention}'.")
    if self.attention_type not in ("global", "mla", "local_sliding", "chunk"):
      raise ValueError(
          "sa_ignore_segment_ids supports attention_type in ('global',"
          " 'mla', 'local_sliding', 'chunk'); got"
          f" attention_type='{self.attention_type}'."
      )
    if self.use_lineage:
      raise ValueError("sa_ignore_segment_ids is not supported with use_lineage=True.")
  if self.share_kv_projections and self.fused_qkv:
    raise ValueError("`share_kv_projections` is not compatible with `fused_qkv`.")
  if self.share_kv_projections and self.attention_type == "mla":
    raise ValueError("`share_kv_projections` is not compatible with `attention_type='mla'`.")
  if self.num_kv_shared_layers > 0:
    if self.fused_qkv:
      raise ValueError("`num_kv_shared_layers > 0` is not compatible with `fused_qkv`.")
    if self.share_kv_projections:
      raise ValueError("`num_kv_shared_layers > 0` is not compatible with `share_kv_projections`.")
  for ratio in self.compress_ratios:
    if ratio != 0 and ratio < 4:
      raise ValueError(f"compress_ratio must be 0 (disabled) or >= 4, got {ratio}")
  if self.rope_direct_position_freqs and self.max_target_length > self.max_position_embeddings:
    raise ValueError(
        "rope_direct_position_freqs=True requires max_target_length <= max_position_embeddings, got "
        f"max_target_length={self.max_target_length}, max_position_embeddings={self.max_position_embeddings}."
    )
  return self
