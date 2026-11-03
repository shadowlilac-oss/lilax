"""Engram model configuration."""

from pydantic import BaseModel, Field


class Engram(BaseModel):
  """Configuration for DeepSeek Engram (https://www.arxiv.org/pdf/2601.07372)."""

  engram_layers: list[int] = Field(
      default_factory=list, description="Indices of transformer layers where Engram are integrated."
  )
  engram_num_heads: int = Field(8, description="Number of heads dedicated to the Engram.")
  engram_head_dim: int = Field(1280, description="Head dimension for heads.")
  engram_vocab_bases: list[int] = Field(
      default_factory=list, description="List of minimum head vocab sizes for each n-gram order."
  )
  engram_max_ngram_size: int = Field(3, description="The max 'n' in N-gram.")
  engram_kernel_size: int = Field(4, description="Temporal window size for Engram convolution.")
  engram_seed: int = Field(0, description="The seed for Engram hash mapping.")
