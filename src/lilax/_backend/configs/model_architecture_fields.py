"""Native core decoder architecture configuration fields."""

from lilax._backend.common.common_types import DecoderBlockType
from pydantic.fields import Field
from pydantic.main import BaseModel


class ModelArchitecture(BaseModel):
  """Core model architecture parameters."""

  decoder_block: DecoderBlockType = Field(
      "llama2",
      description="The style of DecoderBlock to use (e.g., 'llama2', 'gemma').",
  )
  global_parameter_scale: int = Field(1, description="A global scaling factor for model dimensions.")
  base_emb_dim: int = Field(2048, description="Base embedding dimension.")
  base_num_query_heads: int = Field(16, description="Base number of query heads.")
  base_num_kv_heads: int = Field(16, description="Base number of key/value heads.")
  base_mlp_dim: int = Field(7168, description="Base dimension of the MLP layer.")
  dense_init_scale: float = Field(1.0, description="Initialization scale for dense layers")
  base_num_decoder_layers: int = Field(16, description="Base number of decoder layers.")
  head_dim: int = Field(
      128,
      description="Model query and key head dimension.",
  )
  attention_output_dim: int = Field(
      -1,
      description="Override output dimension for attention block if set to a positive value.",
  )
  global_head_dim: int = Field(
      0,
      description="Model query and key head dimension for global attention layers.",
  )
  mlp_activations: list[str] = Field(["silu", "linear"], description="Activation functions in the MLP layer.")
  mlp_activations_limit: float = Field(
      -1.0,
      description="Upper bound to clip the MLP activation values. -1.0 means no clipping.",
  )

  normalization_layer_epsilon: float = Field(1.0e-05, description="Epsilon value for normalization layers.")
  fused_qkv: bool = Field(False, description="If supported, fuse the Q, K, and V projections.")
  attention_bias: bool = Field(
      False,
      description="If True, adds a learnable bias to the query, key, and value projections.",
  )
  fused_mlp: bool = Field(False, description="If supported, fuse the MLP layers.")
  qk_norm_with_scale: bool = Field(
      True,
      description="Whether to apply scale on query and key normalizations (default True).",
  )
  v_norm_with_scale: bool = Field(
      True,
      description="Whether to apply scale on value normalization (default True).",
  )
