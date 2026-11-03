"""Validation of native Splash attention kernel flags."""

from lilax._backend.configs.dataset_types import DatasetType


def validate_splash_attention_flags(config):
  """Validates Splash layout, QK-Clip, FP8 Q/K, base-2 exponent, and segment-id kernel flags."""
  if config.force_q_layout and not config.use_jax_splash:
    raise ValueError("`force_q_layout` can only be true if `use_jax_splash` is also true.")
  if config.use_qk_clip and config.attention_type != "mla":
    raise ValueError(
        f"QK-Clip is only supported when attention_type='mla', but found attention_type='{config.attention_type}'."
    )
  if config.use_qk_clip and config.attn_logits_soft_cap is not None:
    raise ValueError(
        "QK-Clip monitors raw dot products, but attn_logits_soft_cap is enabled. "
        "Recording pre-cap max_logits is not fully supported yet. "
        "Please disable attn_logits_soft_cap when using use_qk_clip."
    )
  if config.experimental_sa_quant_q_fp8 and config.attention_type != "mla":
    raise ValueError(
        "Q quantization is currently only supported with"
        f" attention_type='mla'. Found attention_type='{config.attention_type}'"
    )
  if config.experimental_sa_quant_k_fp8 and config.attention_type != "mla":
    raise ValueError(
        "K quantization is currently only supported with"
        f" attention_type='mla'. Found attention_type='{config.attention_type}'"
    )
  if config.sa_qproj_bwd_bf16 and not (
      config.quantization == "fp8_full" and config.use_qwix_quantization and not config.use_batch_split_schedule
  ):
    raise ValueError(
        "sa_qproj_bwd_bf16=True only takes effect with quantization='fp8_full', use_qwix_quantization=True and"
        " use_batch_split_schedule=False."
    )
  if config.sa_q_prescaled_log2e:
    if not (
        config.experimental_sa_quant_q_fp8
        and config.sa_use_base2_exp
        and config.use_tokamax_splash
        and config.attention == "flash"
    ):
      raise ValueError(
          "sa_q_prescaled_log2e requires experimental_sa_quant_q_fp8=True, sa_use_base2_exp=True,"
          " use_tokamax_splash=True and attention='flash'."
      )
    # The LOG2E fold is only in the MLA query path (attention_mla.py).
    if config.attention_type != "mla":
      raise ValueError(f"sa_q_prescaled_log2e only supports attention_type='mla'; got '{config.attention_type}'.")
    if config.use_mla_absorbed_mqa:
      raise ValueError("sa_q_prescaled_log2e is not supported with use_mla_absorbed_mqa=True.")
    # Ring and USP context parallelism use MaxText's vendored splash kernel (natural exp, no
    # q_prescaled_log2e), which would apply the LOG2E-scaled q as natural-exp logits.
    if config.context_parallel_strategy.lower() in ("ring", "usp"):
      raise ValueError(
          "sa_q_prescaled_log2e is not supported with context_parallel_strategy"
          f" '{config.context_parallel_strategy}'; use 'all_gather' or 'ulysses'."
      )
    if config.sa_dkv_reuse_scaled_q:
      raise ValueError("sa_q_prescaled_log2e subsumes sa_dkv_reuse_scaled_q; set only one of them.")
  if config.sa_dkv_reuse_scaled_q:
    if not (config.sa_use_base2_exp and config.use_tokamax_splash and config.attention == "flash"):
      raise ValueError(
          "sa_dkv_reuse_scaled_q only takes effect with sa_use_base2_exp=True, use_tokamax_splash=True and"
          " attention='flash'."
      )
    if config.context_parallel_strategy.lower() in ("ring", "usp"):
      raise ValueError(
          "sa_dkv_reuse_scaled_q is not supported with context_parallel_strategy"
          f" '{config.context_parallel_strategy}'; use 'all_gather' or 'ulysses'."
      )
  if config.sa_ignore_segment_ids:
    # Mirrors the use_stream_chunking=None resolution in tfds_data_processing_c4_mlperf.py: stream
    # chunking (uniform segment ids per row) is the default for pre-tokenized data.
    def _stream_chunked(tokenize_data: bool) -> bool:
      return config.use_stream_chunking is True or (config.use_stream_chunking is None and not tokenize_data)

    if not (
        config.dataset_type == DatasetType.C4MLPERF
        and _stream_chunked(config.tokenize_train_data)
        and (config.eval_interval <= 0 or _stream_chunked(config.tokenize_eval_data))
    ):
      raise ValueError(
          "sa_ignore_segment_ids requires dataset_type='c4_mlperf' with"
          " stream chunking for train and eval (use_stream_chunking=True, or"
          " None with pre-tokenized data); packed sequences need the segment"
          " mask."
      )
    if config.attention != "flash":
      raise ValueError("sa_ignore_segment_ids only applies to attention='flash'; got" f" attention='{config.attention}'.")
    # Types whose masks come from positions only. COMPRESSED also uses segment ids (-1) to mask block padding.
    if config.attention_type not in ("global", "mla", "local_sliding", "chunk"):
      raise ValueError(
          "sa_ignore_segment_ids supports attention_type in ('global',"
          " 'mla', 'local_sliding', 'chunk'); got"
          f" attention_type='{config.attention_type}'."
      )
    if config.use_lineage:
      raise ValueError("sa_ignore_segment_ids is not supported with use_lineage=True.")
