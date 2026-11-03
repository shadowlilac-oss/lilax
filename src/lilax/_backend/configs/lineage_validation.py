"""Validation of native lineage model requirements."""

from lilax._backend.common.common_types import DecoderBlockType
from lilax._backend.configs.positional_embedding import RopeType


def validate_lineage(config):
  if config.use_lineage:
    if not config.scan_layers:
      raise ValueError("use_lineage=True requires scan_layers=True.")
    if config.decoder_block != DecoderBlockType.DEEPSEEK:
      raise ValueError(
          f"use_lineage=True requires decoder_block='deepseek', got decoder_block={config.decoder_block!r}."
      )
    if config.attention_type != "mla":
      raise ValueError(f"use_lineage=True requires attention_type='mla', got attention_type={config.attention_type!r}.")
    if config.rope_type != RopeType.YARN:
      raise ValueError(f"use_lineage=True requires rope_type='yarn', got rope_type={config.rope_type!r}.")
    if config.capacity_factor <= 0:
      raise ValueError(f"use_lineage=True requires capacity_factor > 0, got capacity_factor={config.capacity_factor}.")
