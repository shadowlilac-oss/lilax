from lilax.configuration.domain_types import QuantizationType, RematLocation


def validate_attention_indexer(self):
  if self.use_indexer:
    if self.attention_type not in ("mla", "compressed"):
      raise ValueError(
          "`use_indexer=True` requires `attention_type='mla'` or `attention_type='compressed'`, "
          "since only MLA and Compressed Attention indexers produce this mask."
      )
    if self.q_lora_rank == 0:
      raise NotImplementedError("Sparse indexer has not implemented for q_lora_rank = 0.")
    if not (self.attention == "dot_product" or (self.attention == "flash" and self.use_tokamax_splash)):
      raise ValueError(
          f"Sparse indexer with {self.attention_type} is only supported with dot_product attention or flash "
          "attention with tokamax splash."
      )
    if self.attention_type == "mla":
      if (
          self.attention == "flash" and self.context_parallel_strategy == "all_gather"
          and self.ici_context_parallelism * self.dcn_context_parallelism > 1 and self.attention_sink
      ):
        raise ValueError(
            "Sparse indexer with all-gather context parallelism for flash attention does not support attention sinks."
        )
      if self.indexer_loss_scaling_factor > 0.0 and self.indexer_topk >= self.max_target_length:
        raise ValueError(
            f"`indexer_topk` ({self.indexer_topk}) must be < `max_target_length` ({self.max_target_length}) "
            "when indexer loss is enabled (`indexer_loss_scaling_factor > 0.0`); otherwise the indexer "
            "short-circuits to select all tokens and no indexer loss is produced."
        )
    elif self.indexer_loss_scaling_factor > 0.0 and self.indexer_topk >= self.max_target_length // 4:
      max_blocks = self.max_target_length // 4
      raise ValueError(
          f"`indexer_topk` ({self.indexer_topk}) must be < total compressed blocks ({max_blocks}) "
          f"(max_target_length={self.max_target_length} // compress_rate=4) "
          "when indexer loss is enabled (`indexer_loss_scaling_factor > 0.0`); otherwise the indexer "
          "short-circuits to select all compressed blocks and no indexer loss is produced."
      )
  elif self.indexer_cutoff_threshold != RematLocation.REMAT:
    raise ValueError(
        f"Setting `indexer_cutoff_threshold='{self.indexer_cutoff_threshold}'` is only valid when "
        "`use_indexer=True` (DeepSeek Sparse Attention / MLA Indexer)."
    )
  if self.use_mla_absorbed_mqa:
    if self.attention_type != "mla":
      raise ValueError(
          "`use_mla_absorbed_mqa=True` requires `attention_type='mla'`, "
          f"but found attention_type='{self.attention_type}'."
      )
    if self.attention not in ("dot_product", "autoselected"):
      raise ValueError(
          "`use_mla_absorbed_mqa=True` requires `attention` to be 'dot_product' or 'autoselected', "
          f"but found attention='{self.attention}'."
      )
    if self.mla_naive_kvcache:
      raise ValueError("`use_mla_absorbed_mqa=True` is incompatible with `mla_naive_kvcache=True`.")
    if (
        self.quantize_kvcache or self.use_qwix_quantization or self.use_manual_quantization
        or self.experimental_sa_quant_q_fp8 or self.experimental_sa_quant_k_fp8
        or (self.quantization and self.quantization not in (QuantizationType.NONE, QuantizationType.TE_NO_QUANT))
    ):
      raise ValueError("`use_mla_absorbed_mqa=True` is incompatible with quantization or FP8 options.")
  if self.attention_type == "chunk" and (
      not isinstance(self.chunk_attn_window_size, int) or self.chunk_attn_window_size <= 0
  ):
    raise ValueError("`chunk_attn_window_size` must be an integer > 0 for 'chunk' attention.")
  if self.attention_type == "local_sliding" and (
      not isinstance(self.sliding_window_size, int) or self.sliding_window_size <= 0
  ):
    raise ValueError("`sliding_window_size` must be an integer > 0 for 'local_sliding' attention.")
  return self
