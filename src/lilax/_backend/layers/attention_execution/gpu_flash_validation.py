"""Rejects features that the GPU flash attention kernels do not support."""

from lilax._backend.common.common_types import Array


def validate_gpu_flash_attention(
    sinks: Array | None,
    record_max_logits: bool,
    attention_kernel: str | None = None,
) -> None:
  """Helper function to check for unsupported features with flash attention on GPU."""
  if sinks is not None and attention_kernel not in ("cudnn_flash_te",):
    raise ValueError(
        "Attention sinks on GPU are only supported with attention=cudnn_flash_te. "
        f"Got attention_kernel={attention_kernel!r}."
    )
  if record_max_logits:
    raise NotImplementedError("record_max_logits (QK-Clip) is not supported for GPU flash attention kernels yet.")
