"""Configuration for positional embeddings and rotary scaling."""

from enum import Enum

from pydantic.fields import Field
from pydantic.functional_validators import model_validator
from pydantic.main import BaseModel


class RopeType(str, Enum):
  """Supported Rotary Positional Embedding (RoPE) implementations."""

  DEFAULT = "default"
  LLAMA3_1 = "llama3.1"
  YARN = "yarn"


class PositionalEmbedding(BaseModel):
  """General configuration for positional embeddings."""

  use_iota_embed: bool = Field(
      False,
      description="Use iota operator in Embed, an efficient way to represent positions.",
  )
  use_untrainable_positional_embedding: bool = Field(
      False, description="Use untrainable sinusoidal positional embeddings."
  )
  trainable_position_size: int = Field(
      -1,
      description="Enables GPT-3 style trainable positional embeddings if positive.",
  )
  nope_layer_interval: int = Field(-1, description="If positive, every N-th layer will NOT use RoPE (Llama4).")


class Rope(BaseModel):
  """Configuration for Rotary Positional Embedding (RoPE)."""

  rope_type: RopeType = Field(RopeType.DEFAULT, description="The type of RoPE to use.")
  rope_use_scale: bool = Field(True, description="Apply RoPE scaling for Llama3.1 style.")
  rope_min_timescale: int = Field(1, description="The minimum timescale for RoPE.")
  rope_max_timescale: int = Field(10_000, description="The maximum timescale for RoPE.")
  rope_linear_scaling_factor: float = Field(1.0, description="Linear scaling factor for 'default' RoPE implementation.")
  local_rope_max_timescale: int = Field(-1, description="If positive, used for local window attention RoPE.")
  global_rope_max_timescale: int = Field(-1, description="If positive, used for global attention RoPE.")
  global_rope_proportion: float = Field(0.25, description="Proportion of dimension to apply RoPE on in global layers.")
  local_rope_proportion: float = Field(1.0, description="Proportion of dimension to apply RoPE on in local layers.")


class YarnRope(BaseModel):
  """Configuration specific to YaRN (Yet another RoPE) scaling."""

  max_position_embeddings: int = Field(163840, description="The maximum position embeddings for YaRN scaling.")
  original_max_position_embeddings: int = Field(4096, description="The original max position embeddings before scaling.")
  rope_factor: int = Field(40, description="The scaling factor for YaRN.")
  beta_fast: int = Field(32, description="The 'beta_fast' parameter for YaRN.")
  beta_slow: int = Field(1, description="The 'beta_slow' parameter for YaRN.")
  mscale: float = Field(1.0, description="The 'mscale' parameter for YaRN.")
  rope_interleave: bool = Field(True, description="Whether RoPE sin/cos are interleaved vs concatenated.")
  rope_truncate: bool = Field(True, description="Whether to floor/ceil the correction range for YaRN.")
  rope_attention_scaling: bool = Field(
      False,
      description="Scale the rotary embedding output. Used by some models like gpt-oss.",
  )
  rope_pairwise: bool = Field(
      False,
      description=(
          "Keep rank-5 pair tensor intact "
          "[batch_size, sequence_length, num_heads, half_dim, 2] "
          "where half_dim represents the independent 2D rotation planes "
          "and 2 represents the real/imag coordinates, returning interleaved RoPE."
      ),
  )
  rope_direct_position_freqs: bool = Field(
      False,
      description=(
          "Compute YaRN RoPE angles directly from position * inv_freqs "
          "instead of indexing a [max_position_embeddings, half_dim] lookup table. "
          "Outputs differ for positions >= max_position_embeddings: the table lookup clamps them "
          "to the last row, while the direct computation uses the true angle."
      ),
  )
  rope_dot_pairwise: bool = Field(
      False,
      description=(
          "Keep rank-4 [batch, seq, heads, dim] tensor intact during pairwise RoPE "
          "via a block-skew matmul on the minor dimension instead of reshaping to rank-5."
      ),
  )

  @model_validator(mode="after")
  def validate_rope_pairwise(self) -> "YarnRope":
    if self.rope_pairwise and not self.rope_interleave:
      raise ValueError("rope_pairwise=True requires rope_interleave=True.")
    if self.rope_dot_pairwise and not self.rope_pairwise:
      raise ValueError("rope_dot_pairwise=True requires rope_pairwise=True.")
    return self
