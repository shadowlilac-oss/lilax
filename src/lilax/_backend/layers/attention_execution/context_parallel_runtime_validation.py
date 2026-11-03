"""Validates per-call constraints of context-parallel attention dispatch and TPU flash strategies."""

from typing import Any

from lilax._backend.common.common_types import Array, AttentionType, Config
from lilax._backend.kernels.attention import tokamax_ring_attention
from lilax._backend.kernels.attention import ulysses_attention
from lilax._backend.kernels.attention import usp_attention


def validate_context_parallel_dispatch(
    config: Config,
    attention_type: AttentionType,
    attention_kernel: str,
    target_hardware: str,
    cp_size: int,
) -> None:
  """Validates that the requested context-parallel strategy can run with the dispatched kernel and hardware."""
  if attention_type == AttentionType.COMPRESSED and cp_size > 1:
    raise ValueError(f"Context parallelism (cp_size={cp_size}) is not supported with AttentionType.COMPRESSED.")
  if (
      target_hardware == "tpu"
      and tokamax_ring_attention.is_context_parallel_ring_requested(config)
      and attention_kernel != "flash"
  ):
    raise ValueError("TPU Tokamax ring attention requires attention_kernel='flash'.")
  if ulysses_attention.is_context_parallel_ulysses_requested(config):
    if target_hardware != "tpu":
      raise ValueError("Ulysses context parallelism (context_parallel_strategy='ulysses') is only supported on TPU.")
    if attention_kernel != "flash":
      raise ValueError("TPU Ulysses attention requires attention_kernel='flash'.")
  if usp_attention.is_context_parallel_usp_requested(config):
    if target_hardware != "tpu":
      raise ValueError("USP context parallelism (context_parallel_strategy='usp') is only supported on TPU.")
    if attention_kernel != "flash":
      raise ValueError("TPU USP attention requires attention_kernel='flash'.")


def validate_tpu_tokamax_ring_runtime(
    config: Config,
    attention_type: AttentionType,
    *,
    model_mode: str,
    previous_chunk: Any = None,
    bidirectional_mask: Any = None,
    sinks: Array | None = None,
    indexer_mask: Array | None = None,
    use_ragged_attention: bool = False,
    record_max_logits: bool = False,
) -> None:
  """Validates runtime constraints for the TPU Tokamax ring path."""
  if attention_type == AttentionType.COMPRESSED:
    raise ValueError("TPU Tokamax ring attention does not support AttentionType.COMPRESSED (DeepSeek-V4 HCA/CSA).")
  if getattr(config, "use_indexer", False) and indexer_mask is None:
    raise ValueError(
        "`indexer_mask` cannot be None when `use_indexer` is True. "
        "Ring Attention drops its static causal mask when use_indexer is True, "
        "so passing None would result in no causal masking."
    )
  tokamax_ring_attention.validate_tokamax_ring_runtime(
      model_mode=model_mode,
      previous_chunk=previous_chunk,
      sinks=sinks,
      indexer_mask=indexer_mask,
      use_ragged_attention=use_ragged_attention,
      bidirectional_mask=bidirectional_mask,
      record_max_logits=record_max_logits,
  )


def validate_tpu_ulysses_runtime(
    attention_type: AttentionType,
    *,
    model_mode: str,
    previous_chunk: Any = None,
    bidirectional_mask: Any = None,
    sinks: Array | None = None,
    indexer_mask: Array | None = None,
    use_ragged_attention: bool = False,
    record_max_logits: bool = False,
) -> None:
  """Validates runtime constraints for the TPU Ulysses path."""
  if attention_type == AttentionType.COMPRESSED:
    raise ValueError(
        "TPU Ulysses attention does not support AttentionType.COMPRESSED (DeepSeek-V4 HCA/CSA) "
        "due to MQA (num_kv_heads=1), asymmetric Q/KV sequence lengths, and dynamic/ragged masking."
    )
  ulysses_attention.validate_ulysses_runtime(
      model_mode=model_mode,
      previous_chunk=previous_chunk,
      sinks=sinks,
      indexer_mask=indexer_mask,
      use_ragged_attention=use_ragged_attention,
      bidirectional_mask=bidirectional_mask,
      record_max_logits=record_max_logits,
  )


def validate_tpu_usp_runtime(
    attention_type: AttentionType,
    *,
    model_mode: str,
    previous_chunk: Any = None,
    bidirectional_mask: Any = None,
    sinks: Array | None = None,
    indexer_mask: Array | None = None,
    use_ragged_attention: bool = False,
    record_max_logits: bool = False,
) -> None:
  """Validates runtime constraints for the TPU USP path."""
  if attention_type == AttentionType.COMPRESSED:
    raise ValueError("TPU USP attention does not support AttentionType.COMPRESSED (DeepSeek-V4 HCA/CSA).")
  usp_attention.validate_usp_runtime(
      model_mode=model_mode,
      previous_chunk=previous_chunk,
      sinks=sinks,
      indexer_mask=indexer_mask,
      use_ragged_attention=use_ragged_attention,
      bidirectional_mask=bidirectional_mask,
      record_max_logits=record_max_logits,
  )
