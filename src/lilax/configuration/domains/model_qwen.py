"""Qwen model configurations."""

from typing import Literal

from pydantic import BaseModel, Field


class Qwen3Next(BaseModel):
  """Configuration specific to Qwen3-Next models with Gated Delta Net."""

  gdn_conv_kernel_dim: int = Field(4, description="Kernel size for the 1D convolution in the Gated Delta Net.")
  gdn_key_head_dim: int = Field(128, description="Head dimension for the key/query in the Gated Delta Net.")
  gdn_value_head_dim: int = Field(128, description="Head dimension for the value in the Gated Delta Net.")
  gdn_num_key_heads: int = Field(16, description="Number of key/query heads in the Gated Delta Net.")
  gdn_num_value_heads: int = Field(32, description="Number of value heads in the Gated Delta Net.")
  gdn_chunk_size: int = Field(64, description="Chunk size for the parallel scan algorithm in the Gated Delta Net.")
  use_qk_norm_in_gdn: bool = Field(
      True, description="Whether to apply L2 normalization to query and key tensors inside the Gated Delta Rule kernel."
  )
  gdn_mamba_block_size: int = Field(
      0,
      description=(
          "Tokens per mamba block when serving under vLLM with mamba prefix caching "
          '("align" mode). Set by the vLLM adapter from cache_config.mamba_block_size; '
          "0 means the recurrent state is addressed by a resident per-request slot "
          "instead of by block id."
      ),
  )
  partial_rotary_factor: float = Field(1.0, description="The ratio of dimension to apply ROPE on")


class Qwen3_8FlashNext(BaseModel):
  """Configuration specific to Qwen3.8-Flash-Next models."""

  hc_count: int = Field(4, description="Number of streams in hyper-connections.")
  hc_lowrank: int = Field(320, description="Bottleneck dimension in hyper-connections.")
  index_n_heads: int = Field(4, description="Number of query heads in QSA Indexer.")
  index_kv_heads: int = Field(1, description="Number of key/value heads in QSA Indexer.")
  index_head_dim: int = Field(128, description="Head dimension for QSA Indexer.")
  index_select_block_size: int = Field(4, description="Block compression size for QSA Indexer.")
  index_select_budget: int = Field(2048, description="Token budget for QSA Indexer.")
  gdn_output_gate_activation: str = Field("sigmoid", description="Output gate activation function for GDN linear attention.")
  gdn_chunk_inverse: Literal["solve", "block_doubling"] = Field(
      "block_doubling",
      description=(
          "Qwen3.8 GDN chunk-matrix inverse. 'block_doubling' (default) builds the exact inverse from MXU matmuls"
          " via bottom-up block doubling (TPU-friendly; stable in float32 even for correlated keys with weak decay);"
          " 'solve' uses jax.scipy.linalg.solve_triangular as the reference path."
      ),
  )
  qsa_indexer_mask: Literal["gather", "reshape"] = Field(
      "gather",
      description=(
          "Qwen3.8 QSA indexer mask construction: 'gather' uses per-token take_along_axis, 'reshape' uses"
          " broadcast/reshape expansion plus a gather-free one-hot contraction when segment ids are present."
      ),
  )
  qsa_attention_kernel: Literal["dense", "flash"] = Field(
      "dense",
      description=(
          "Qwen3.8 QSA attention kernel: 'dense' materializes the score matrix, 'flash' dispatches through"
          " Tokamax Splash Attention (requires max_target_length <= index_select_budget)."
      ),
  )
  use_ple_layer: bool = Field(False, description="Whether to use Per-Layer Embedding (PLE).")
  ple_layer_id: int = Field(1, description="0-indexed layer where PLE is applied (corresponds to HF 1-indexed ple_layer_ids=[2]).")
  ple_embed_dim: int = Field(2560, description="Embedding dimension of the PLE layer (HF ple_embed_dim; HF defaults it to hidden_size).")
  ple_conv_kernel_size: int = Field(4, description="Kernel size for PLE 1D convolution.")
  ngram_size: int = Field(3, description="N-gram window size for PLE.")
  heads_per_ngram: int = Field(8, description="Number of hash heads per n-gram in PLE.")
  ngram_vocab_size_base: int = Field(20_000_000, description="Base n-gram vocabulary size for PLE.")
  make_ngram_vocab_size_divisible_by: int = Field(128, description="Alignment divisor for per-head n-gram vocab sizes in PLE.")
  split_ngram_parts: int = Field(128, description="Number of embedding shards for PLE.")
  ple_seed: int = Field(1234, description="Hash seed for PLE n-gram hashing.")
  ple_eos_token_id: int = Field(248044, description="EOS token ID used for PLE n-gram boundary resets (HF eos_token_id).")
