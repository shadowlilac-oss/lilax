"""General attention configuration."""

from typing import Literal

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt


class Attention(BaseModel):
  """General configuration for the attention mechanism."""

  attention: str = Field(
      "autoselected",
      description="The attention algorithm to use (dot_product, flash, cudnn_flash_te, vllm_rpa, vllm_batched_rpa, etc).",
  )
  attention_type: Literal["global", "local_sliding", "chunk", "mla", "full", "compressed", "block_diffusion"] = Field(
      "global", description="The variant of attention to use."
  )
  share_kv_projections: bool = Field(False, description="If True, for global attention, Key and Value projections share the same weights.")
  global_num_kv_heads: int = Field(0, description="If greater than 0, sets the number of KV heads for global attention.")
  hidden_size_per_layer_input: NonNegativeInt = Field(
      0, description=("Per-Layer-Embedding (PLE) inner dim used by Gemma 4 small models. "
                      "When > 0, each decoder layer adds an extra PLE sub-block at the end.")
  )
  vocab_size_per_layer_input: NonNegativeInt = Field(
      0, description=("Vocab size of the per-layer input embedding table used by Gemma 4 small models. "
                      "Defaults to 0 (disabled); typically set equal to `vocab_size`.")
  )
  num_kv_shared_layers: NonNegativeInt = Field(
      0, description=("Number of trailing decoder layers that reuse K/V from the last non-shared layer "
                      "of the same attention type (sliding↔sliding, full↔full). Used by Gemma 4 small.")
  )
  use_double_wide_mlp: bool = Field(
      False, description=("When True, KV-shared layers double their MLP intermediate size to compensate for "
                          "the missing K/V projections. Used by Gemma 4 small.")
  )
  attention_sink: bool = Field(False, description="If True, enables attention sinks.")
  float32_qk_product: bool = Field(False, description="In dot-product attention, cast query-key product to fp32.")
  float32_logits: bool = Field(False, description="In dot-product attention, cast logits to fp32 before softmax.")
  sliding_window_size: NonNegativeInt = Field(0, description="The size of the sliding window for local attention.")
  chunk_attn_window_size: NonNegativeInt = Field(0, description="The window size for chunked attention.")
  causal_block_size: PositiveInt = Field(32, description="The number of token positions in each bidirectional block for block-causal attention.")
  attn_logits_soft_cap: None | NonNegativeFloat = Field(None, description="Soft-cap value for attention logits. None means no cap.")
  use_post_attn_norm: bool = Field(False, description="Apply LayerNorm after the attention block.")
  use_post_ffw_norm: bool = Field(False, description="Apply LayerNorm after the feed-forward block.")
  use_ragged_attention: bool = Field(False, description="Whether to use ragged attention kernels.")
  ragged_block_size: int = Field(256, description="Block size for ragged attention.")
  enable_padding_causal_mask: bool = Field(True, description="Temporary flag for Transformer Engine padding.")
  use_tokamax_splash: bool = Field(False, description="Whether to use tokamax splash attention.")
  use_jax_splash: bool = Field(False, description="Whether to use jax splash attention.")
  force_q_layout: bool = Field(False, description="Force the Q layout")
  use_qk_clip: bool = Field(False, description="Whether to use QK-Clip (MuonClip) for training stability.")
  qk_clip_threshold: float = Field(100.0, description="Threshold for QK-Clip (tau).")
  mla_qk_head_chunk_size: int = Field(
      0, ge=0, description=("Chunk size over heads dimension for QK attention dot product in mla.  "
                            "Default is 0 (no chunking). Reduces memory footprint at the cost of time.")
  )
  csa_qk_head_chunk_size: int = Field(
      0, ge=0, description=("Chunk size over heads dimension for QK attention dot product in CSA (DeepSeek-V4).  "
                            "Default is 0 (no chunking). Reduces memory footprint at the cost of time.")
  )
