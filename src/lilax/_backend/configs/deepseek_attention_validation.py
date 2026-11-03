"""Validation of native MLA, compressed attention, and sparse indexer configuration."""

from lilax._backend.common.common_types import AttentionType
from lilax._backend.configs.quantization_types import QuantizationType
from lilax._backend.configs.remat_types import RematLocation


def validate_qk_head_chunking(config):
  if config.mla_qk_head_chunk_size > 0:
    if (
        config.mla_qk_head_chunk_size > config.num_query_heads
        or config.num_query_heads % config.mla_qk_head_chunk_size != 0
    ):
      raise ValueError(
          f"`mla_qk_head_chunk_size` ({config.mla_qk_head_chunk_size}) must cleanly divide exactly into "
          f"`num_query_heads` ({config.num_query_heads})."
      )
    if config.use_indexer and (
        config.mla_qk_head_chunk_size > config.indexer_n_heads
        or config.indexer_n_heads % config.mla_qk_head_chunk_size != 0
    ):
      raise ValueError(
          f"`mla_qk_head_chunk_size` ({config.mla_qk_head_chunk_size}) must cleanly divide exactly into "
          f"`indexer_n_heads` ({config.indexer_n_heads})."
      )
  if config.csa_qk_head_chunk_size > 0:
    if (
        config.csa_qk_head_chunk_size > config.num_query_heads
        or config.num_query_heads % config.csa_qk_head_chunk_size != 0
    ):
      raise ValueError(
          f"`csa_qk_head_chunk_size` ({config.csa_qk_head_chunk_size}) must cleanly divide exactly into "
          f"`num_query_heads` ({config.num_query_heads})."
      )
    if config.use_indexer and (
        config.csa_qk_head_chunk_size > config.indexer_n_heads
        or config.indexer_n_heads % config.csa_qk_head_chunk_size != 0
    ):
      raise ValueError(
          f"`csa_qk_head_chunk_size` ({config.csa_qk_head_chunk_size}) must cleanly divide exactly into "
          f"`indexer_n_heads` ({config.indexer_n_heads})."
      )


def validate_sparse_indexer(config):
  if config.use_indexer:
    if config.attention_type not in (AttentionType.MLA.value, AttentionType.COMPRESSED.value):
      raise ValueError(
          f"`use_indexer=True` requires `attention_type='{AttentionType.MLA.value}'` or "
          f"`attention_type='{AttentionType.COMPRESSED.value}'`, since only MLA and "
          "Compressed Attention indexers produce this mask."
      )
    if config.q_lora_rank == 0:
      raise NotImplementedError("Sparse indexer has not implemented for q_lora_rank = 0.")
    supports_dot_product = config.attention == "dot_product"
    supports_flash_splash = config.attention == "flash" and config.use_tokamax_splash
    if not (supports_dot_product or supports_flash_splash):
      raise ValueError(
          f"Sparse indexer with {config.attention_type} is only supported with dot_product attention or flash "
          "attention with tokamax splash."
      )
    if config.attention_type == AttentionType.MLA.value:
      if (
          config.attention == "flash"
          and config.context_parallel_strategy == "all_gather"
          and config.ici_context_parallelism * config.dcn_context_parallelism > 1
          and config.attention_sink
      ):
        raise ValueError(
            "Sparse indexer with all-gather context parallelism for flash attention does not support attention sinks."
        )
      if config.indexer_loss_scaling_factor > 0.0 and config.indexer_topk >= config.max_target_length:
        raise ValueError(
            f"`indexer_topk` ({config.indexer_topk}) must be < `max_target_length` ({config.max_target_length}) "
            "when indexer loss is enabled (`indexer_loss_scaling_factor > 0.0`); otherwise the indexer "
            "short-circuits to select all tokens and no indexer loss is produced."
        )
    elif config.attention_type == AttentionType.COMPRESSED.value:
      # DeepSeek-V4 CSA natively uses a compression rate of 4 for the indexer blocks.
      compress_rate = 4
      max_blocks = config.max_target_length // compress_rate
      if config.indexer_loss_scaling_factor > 0.0 and config.indexer_topk >= max_blocks:
        raise ValueError(
            f"`indexer_topk` ({config.indexer_topk}) must be < total compressed blocks ({max_blocks}) "
            f"(max_target_length={config.max_target_length} // compress_rate={compress_rate}) "
            "when indexer loss is enabled (`indexer_loss_scaling_factor > 0.0`); otherwise the indexer "
            "short-circuits to select all compressed blocks and no indexer loss is produced."
        )
  if not config.use_indexer and config.indexer_cutoff_threshold != RematLocation.REMAT:
    raise ValueError(
        f"Setting `indexer_cutoff_threshold='{config.indexer_cutoff_threshold}'` is only valid when "
        "`use_indexer=True` (DeepSeek Sparse Attention / MLA Indexer)."
    )


def validate_mla_absorbed_mqa(config):
  if config.use_mla_absorbed_mqa:
    if config.attention_type != AttentionType.MLA.value:
      raise ValueError(
          f"`use_mla_absorbed_mqa=True` requires `attention_type='{AttentionType.MLA.value}'`, "
          f"but found attention_type='{config.attention_type}'."
      )
    if config.attention not in ("dot_product", "autoselected"):
      raise ValueError(
          f"`use_mla_absorbed_mqa=True` requires `attention` to be 'dot_product' or 'autoselected', "
          f"but found attention='{config.attention}'."
      )
    if config.mla_naive_kvcache:
      raise ValueError("`use_mla_absorbed_mqa=True` is incompatible with `mla_naive_kvcache=True`.")
    has_quant = (
        config.quantize_kvcache
        or config.use_qwix_quantization
        or config.use_manual_quantization
        or config.experimental_sa_quant_q_fp8
        or config.experimental_sa_quant_k_fp8
        or (config.quantization and config.quantization not in (QuantizationType.NONE, QuantizationType.TE_NO_QUANT))
    )
    if has_quant:
      raise ValueError("`use_mla_absorbed_mqa=True` is incompatible with quantization or FP8 options.")
