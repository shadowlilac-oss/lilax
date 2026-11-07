"""Model architecture and explicit effective-dimension scaling."""

from pydantic import BaseModel, ConfigDict, Field

from .domain_types import DecoderBlockType


class ArchitectureConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  decoder_block: DecoderBlockType = DecoderBlockType.LLAMA2
  dense_init_scale: float = 1.0
  attention_output_dim: int = -1
  global_head_dim: int = 0
  mlp_activations: list[str] = Field(default_factory=lambda: ["silu", "linear"])
  mlp_activations_limit: float = -1.0
  normalization_layer_epsilon: float = 1e-5
  fused_qkv: bool = False
  attention_bias: bool = False
  fused_mlp: bool = False
  qk_norm_with_scale: bool = True
  v_norm_with_scale: bool = True


class ModelLayoutConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  inhomogeneous_layer_cycle_interval: int = 1


class ModelScalingConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  global_parameter_scale: int = 1
  base_emb_dim: int = 2048
  base_num_query_heads: int = 16
  base_num_kv_heads: int = 16
  base_mlp_dim: int = 7168
  base_num_decoder_layers: int = 16
  base_moe_mlp_dim: int = -1

  def apply(self, config):
    scale = self.global_parameter_scale
    if scale <= 0 or scale & (scale - 1):
      raise ValueError("global_parameter_scale must be a positive power of two")
    base_scale, remainder = divmod(scale.bit_length() - 1, 3)
    head_scale = base_scale + int(remainder > 0)
    updated = config.to_dict()
    updated.update({
        "hidden_size": 2 ** (base_scale + int(remainder > 1)) * self.base_emb_dim,
        "num_attention_heads": 2 ** head_scale * self.base_num_query_heads,
        "num_key_value_heads": 2 ** head_scale * self.base_num_kv_heads,
        "intermediate_size": 2 ** head_scale * self.base_mlp_dim,
        "num_hidden_layers": 2 ** base_scale * self.base_num_decoder_layers,
    })
    updated["moe_intermediate_size"] = 2 ** head_scale * self.base_moe_mlp_dim
    return type(config)(**updated)
