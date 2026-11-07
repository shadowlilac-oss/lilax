"""Hugging Face model configuration with focused model-domain composition."""

import copy
from enum import Enum

from pydantic import BaseModel, TypeAdapter
from transformers import PreTrainedConfig, __version__ as transformers_version
from lilax._backend.configs.inert import MaxTextConfig as InertConfig
from lilax._backend.checkpoint_conversion.model_families.hf_model_config_registry import HF_MODEL_CONFIGS

from .model_architecture import ArchitectureConfig, ModelLayoutConfig
from .model_attention import AttentionConfig, KernelConfig, PositionConfig
from .model_families import (
    AudioConfig, EngramConfig, HyperconnectionConfig, LogitsConfig, MingConfig, MoEConfig, MTPConfig,
    QwenConfig, VisionConfig, VisionProjectorConfig,
)
from .model_precision import PrecisionConfig
from .quantization import KVCacheQuantizationConfig, QuantizationRecipe, SparsityConfig, _DISPATCH
from .domains.data_qat import QwixQATConfig


_MODEL_DOMAINS = {
    "architecture_config": ArchitectureConfig, "attention_config": AttentionConfig, "position_config": PositionConfig,
    "moe_config": MoEConfig, "qwen_config": QwenConfig, "engram_config": EngramConfig, "ming_config": MingConfig,
    "hyperconnection_config": HyperconnectionConfig, "vision_config": VisionConfig,
    "vision_projector_config": VisionProjectorConfig, "audio_config": AudioConfig, "mtp_config": MTPConfig,
    "logits_config": LogitsConfig, "precision_config": PrecisionConfig,
    "kv_cache_quantization_config": KVCacheQuantizationConfig, "sparsity_config": SparsityConfig,
    "kernel_config": KernelConfig, "layout_config": ModelLayoutConfig,
}


def _json_value(value):
  if isinstance(value, BaseModel):
    return value.model_dump(mode="json")
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, dict):
    return {key: _json_value(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_json_value(item) for item in value]
  return value


class MaxTextConfig(PreTrainedConfig):
  """Effective model dimensions and model behavior, without whole-run execution arguments."""

  model_type = "lilax"
  has_no_defaults_at_init = True

  def __init__(
      self, *, model_name: str, vocab_size: int, hidden_size: int, intermediate_size: int,
      num_hidden_layers: int, num_attention_heads: int, num_key_value_heads: int, head_dim: int,
      max_position_embeddings: int = 2048, enable_dropout: bool = True, dropout_rate: float = 0.0,
      moe_intermediate_size: int | None = None,
      architecture_config: ArchitectureConfig | None = None, attention_config: AttentionConfig | None = None,
      position_config: PositionConfig | None = None, moe_config: MoEConfig | None = None,
      qwen_config: QwenConfig | None = None, engram_config: EngramConfig | None = None,
      ming_config: MingConfig | None = None, hyperconnection_config: HyperconnectionConfig | None = None,
      vision_config: VisionConfig | None = None, vision_projector_config: VisionProjectorConfig | None = None,
      audio_config: AudioConfig | None = None, mtp_config: MTPConfig | None = None,
      logits_config: LogitsConfig | None = None, precision_config: PrecisionConfig | None = None,
      quantization_config: QuantizationRecipe | None = None,
      kv_cache_quantization_config: KVCacheQuantizationConfig | None = None, sparsity_config: SparsityConfig | None = None,
      kernel_config: KernelConfig | None = None, layout_config: ModelLayoutConfig | None = None, **kwargs,
  ):
    removed = kwargs.keys() & (InertConfig.model_fields.keys() - {"dtype"})
    if removed:
      raise TypeError(f"Model configuration does not accept backend or run arguments: {sorted(removed)}")
    values = locals().copy()
    values.pop("self")
    values.pop("kwargs")
    values.pop("__class__", None)
    values.pop("removed", None)
    selected_model_type = kwargs.pop("model_type", "lilax")
    if selected_model_type == "lilax" and model_name in HF_MODEL_CONFIGS:
      selected_model_type = HF_MODEL_CONFIGS[model_name].model_type
    if isinstance(quantization_config, dict):
      kind = quantization_config.get("quant_method")
      recipes = {recipe.__name__: recipe for recipe in (*_DISPATCH, QwixQATConfig)}
      if kind not in recipes:
        raise ValueError(f"Unknown quantization recipe {kind!r}")
      quantization_config = recipes[kind].model_validate(
          {name: value for name, value in quantization_config.items() if name != "quant_method"}
      )
      values["quantization_config"] = quantization_config
    elif quantization_config is not None:
      TypeAdapter(QuantizationRecipe).validate_python(quantization_config)
    for name, domain in _MODEL_DOMAINS.items():
      if values[name] is not None:
        values[name] = TypeAdapter(domain).validate_python(values[name])
    super().__init__(**kwargs)
    for name, value in values.items():
      setattr(self, name, value)
    self.model_type = selected_model_type

  def to_diff_dict(self):
    """Retain focused recipe serialization while omitting unchanged HF metadata."""
    defaults = PreTrainedConfig().to_dict()
    return {name: value for name, value in self.to_dict().items()
            if name != "_name_or_path" and (
                name not in defaults or name == "transformers_version" or value != defaults[name]
            )}

  def to_dict(self):
    values = copy.deepcopy(self.__dict__)
    self._remove_keys_not_serialized(values)
    values = {name: _json_value(value) for name, value in values.items() if not name.startswith("_lilax_")}
    values["model_type"] = self.model_type
    values["transformers_version"] = transformers_version
    if self.quantization_config is not None:
      values["quantization_config"] = {
          "quant_method": type(self.quantization_config).__name__, **self.quantization_config.model_dump(mode="json"),
      }
    self.dict_dtype_to_str(values)
    return values
