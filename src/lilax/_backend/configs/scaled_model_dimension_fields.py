"""Native effective model dimension fields scaled by the global parameter scale."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class ScaledModelDimensions(BaseModel):
  """Effective model dimensions derived from base dimensions and `global_parameter_scale`."""

  emb_dim: None | int = Field(
      None,
      description="Effective embedding dimension, scaled by `global_parameter_scale`.",
  )
  mlp_dim: None | int = Field(None, description="Effective MLP dimension, scaled by `global_parameter_scale`.")
  moe_mlp_dim: None | int = Field(
      None,
      description="Effective MLP dimension for MoE layers, scaled by `global_parameter_scale`.",
  )
  num_decoder_layers: None | int = Field(
      None,
      description="Effective number of decoder layers, scaled by `global_parameter_scale`.",
  )
  num_kv_heads: None | int = Field(
      None,
      description="Effective number of key/value heads, scaled by `global_parameter_scale`.",
  )
  num_query_heads: None | int = Field(
      None,
      description="Effective number of query heads, scaled by `global_parameter_scale`.",
  )
