"""Attention variant configurations."""

from pydantic import BaseModel, Field, NonNegativeInt


class MoBa(BaseModel):
  """Configuration for Mixture of Block Attention (MoBA)."""

  moba: bool = Field(False, description="If True, enables Mixture of Block Attention.")
  moba_chunk_size: int = Field(1024, description="The chunk size for MoBA.")
  moba_topk: int = Field(8, description="The number of top-k chunks to select in MoBA.")


class MlaAttention(BaseModel):
  """Configuration for Multi-Layer Attention (MLA)."""

  mla_naive_kvcache: bool = Field(True, description="Whether to use naive kvcache for MLA attention.")
  q_lora_rank: NonNegativeInt = Field(0, description="Query LoRA rank for MLA.")
  kv_lora_rank: NonNegativeInt = Field(512, description="Key/Value LoRA rank for MLA.")
  qk_nope_head_dim: NonNegativeInt = Field(128, description="Dimension for non-RoPE part of QK heads in MLA.")
  qk_rope_head_dim: NonNegativeInt = Field(64, description="Dimension for RoPE part of QK heads in MLA.")
  v_head_dim: NonNegativeInt = Field(128, description="Dimension of V heads in MLA.")
  use_sliced_mla_proj: bool = Field(False, description="Whether to slice projection kernel weights before contraction in MLA"
                                    " instead of running full projection + jnp.split.")
  use_mla_absorbed_mqa: bool = Field(False, description="Whether to use Absorbed-Query Latent MQA mode in MLA. Trades higher prefill/training FLOPs "
                                     "for 57x smaller KV cache footprint and memory bandwidth during autoregressive decode.")


class CompressedAttention(BaseModel):
  """Configuration for Compressed Attention."""

  o_lora_rank: NonNegativeInt = Field(0, description="Output LoRA rank for Compressed Attention.")
  o_groups: NonNegativeInt = Field(0, description="Output groups for Compressed Attention.")
  compress_ratios: list[int] = Field(default_factory=list, description="Per-layer compression ratios (0, 4, 128, etc).")
  compressed_rope_max_timescale: int = Field(160000, description="If positive, used for Compressed Sparse/Heavy Attention.")
  use_csa_streamindex_kernel: bool = Field(False, description="Whether to use Pallas TPU kernel for CSA StreamIndex score computation.")


class AttentionIndexer(BaseModel):
  """Configuration for DeepSeek Sparse Attention (DSA): MLA or Compressed Attention with indexer."""

  use_indexer: bool = Field(False, description="Whether to use sparse indexer for MLA or Compressed Attention.")
  indexer_head_dim: NonNegativeInt = Field(128, description="Head dim for indexer query and key.")
  indexer_n_heads: NonNegativeInt = Field(64, description="Number of query heads in indexer.")
  indexer_topk: NonNegativeInt = Field(2048, description="Number of tokens selected by the query token in indexer.")
  indexer_sparse_training: bool = Field(False, description="Determines the training strategy for the indexer: Dense Warm-up or Sparse Training stage.")
  indexer_loss_scaling_factor: float = Field(0.0, description="Multiplier for the indexer KL divergence loss.")
  indexer_use_approx_top_k: bool = Field(False, description="Whether to use approximate top-k selection for the indexer on TPU.")
  indexer_approx_top_k_recall: float = Field(0.95, description="Recall target for approximate top-k selection.")
  indexer_mask_exact_topk: bool = Field(True, description="When True, enforces that exactly k elements are unmasked by the indexer under boundary ties."
                                       " When False, uses raw thresholding which is faster but may unmask > k elements"
                                       " during ties.")


class Llama4Attention(BaseModel):
  """Configuration specific to Llama4-style models."""

  use_qk_norm: bool = Field(False, description="Whether to apply L2 normalization to Query/Key vectors after RoPE.")
  temperature_tuning: bool = Field(False, description="Dynamically scale attention temperature based on sequence length.")
