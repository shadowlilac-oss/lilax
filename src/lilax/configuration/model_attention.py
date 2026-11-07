"""Attention semantics, position encoding and kernel selection."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .domains.data_position import (
    PositionalEmbedding as PositionalEmbeddingConfig, Rope as RopeConfig, YarnRope as YarnRopeConfig,
)
from .domains.model_attention import Attention
from .domains.model_attention_variants import (
    AttentionIndexer as AttentionIndexerConfig, CompressedAttention as CompressedAttentionConfig,
    Llama4Attention as Llama4AttentionConfig, MlaAttention as MlaAttentionConfig, MoBa as MoBaConfig,
)
from .domains.model_moe import MoEKernels as MoEKernelsConfig


class AttentionParameters(Attention):
  model_config = ConfigDict(extra="forbid")
  attention: ClassVar[str] = "autoselected"
  use_ragged_attention: ClassVar[bool] = False
  ragged_block_size: ClassVar[int] = 256
  enable_padding_causal_mask: ClassVar[bool] = True
  use_tokamax_splash: ClassVar[bool] = False
  use_jax_splash: ClassVar[bool] = False
  force_q_layout: ClassVar[bool] = False


class AttentionConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  general: AttentionParameters | None = None
  moba: MoBaConfig | None = None
  mla: MlaAttentionConfig | None = None
  compressed: CompressedAttentionConfig | None = None
  indexer: AttentionIndexerConfig | None = None
  llama4: Llama4AttentionConfig | None = None


class PositionConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  embedding: PositionalEmbeddingConfig | None = None
  rope: RopeConfig | None = None
  yarn: YarnRopeConfig | None = None
  use_mrope: bool = False
  mrope_section: list[int] = Field(default_factory=lambda: [24, 20, 20])
  position_id_per_seconds: int = 25


class SplashAttentionConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  experimental_sa_quant_q_fp8: bool = False
  experimental_sa_quant_k_fp8: bool = False
  sa_ignore_segment_ids: bool = False


class KernelConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  attention: str = "autoselected"
  use_ragged_attention: bool = False
  enable_padding_causal_mask: bool = True
  use_tokamax_splash: bool = False
  use_jax_splash: bool = False
  splash: SplashAttentionConfig | None = None
  moe: MoEKernelsConfig | None = None
