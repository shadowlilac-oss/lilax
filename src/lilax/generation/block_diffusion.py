"""Block-diffusion corruption and target alignment primitives."""

from lilax._backend.diffusion.block_diffusion.corruption import BlockDiffusionCorruptionResult, corrupt_tokens
from lilax._backend.diffusion.block_diffusion.target_alignment import align_logits_to_targets

__all__ = ["BlockDiffusionCorruptionResult", "align_logits_to_targets", "corrupt_tokens"]
