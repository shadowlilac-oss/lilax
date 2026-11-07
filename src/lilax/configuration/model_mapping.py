"""Translate focused model configuration to the existing runtime model fields."""

import copy
from typing import get_args

from pydantic import BaseModel

from lilax._backend.configs.inert import MaxTextConfig as InertConfig
from lilax.launch._profile import active_profile
from .model_architecture import ArchitectureConfig, ModelLayoutConfig
from .model_attention import (
    AttentionConfig, AttentionIndexerConfig, AttentionParameters, CompressedAttentionConfig, KernelConfig,
    Llama4AttentionConfig, MlaAttentionConfig, MoBaConfig, MoEKernelsConfig, PositionConfig, PositionalEmbeddingConfig,
    RopeConfig, SplashAttentionConfig, YarnRopeConfig,
)
from .model_families import (
    AudioConfig, DeepSeekMoEConfig, EngramConfig, HyperconnectionConfig, LogitsConfig, MingConfig, MoEConfig,
    MoEParameters, MTPConfig, Qwen3NextConfig, Qwen3_8FlashNextConfig, VisionConfig, VisionProjectorConfig,
)
from .model_precision import PrecisionConfig
from .quantization import (
    AqtInt8Config, AqtMixedPrecisionConfig, BatchSplitManualConfig, BatchSplitQwixConfig, ComputeRecipeConfig,
    JaxMXConfig, KVCacheQuantizationConfig, QwixFP8FullConfig, QwixRecipeConfig, ServeFP8WeightConfig,
    SparsityConfig, recipe_source_values,
)
from .values import load_config_values
from .schema import MaxTextConfig, _MODEL_DOMAINS, _json_value


_DIMENSIONS = {
    "hidden_size": "base_emb_dim", "intermediate_size": "base_mlp_dim", "num_hidden_layers": "base_num_decoder_layers",
    "num_attention_heads": "base_num_query_heads", "num_key_value_heads": "base_num_kv_heads",
    "head_dim": "head_dim", "vocab_size": "vocab_size", "max_position_embeddings": "max_target_length",
    "enable_dropout": "enable_dropout", "dropout_rate": "dropout_rate",
}


def _domain_values(domain, *, explicit_only=False):
  if domain is None:
    return {}
  values = {}
  for name in type(domain).model_fields:
    value = getattr(domain, name)
    if isinstance(value, BaseModel):
      values.update(_domain_values(value, explicit_only=explicit_only))
    elif value is None and any(hasattr(item, "model_fields") for item in get_args(type(domain).model_fields[name].annotation)):
      continue
    elif not explicit_only or name in domain.model_fields_set:
      values[name] = value
  return values


def project_model_config(config):
  """Build one internal source config without turning resolved defaults into explicit inputs."""
  values = load_config_values(config.model_name)
  updates = {target: getattr(config, name) for name, target in _DIMENSIONS.items()}
  updates["global_parameter_scale"] = 1
  explicit = set(getattr(config, "_lilax_explicit_keys", updates.keys()))
  for name in _MODEL_DOMAINS:
    domain = getattr(config, name)
    updates.update(_domain_values(domain))
    explicit.update(_domain_values(domain, explicit_only=True))
  if config.moe_intermediate_size is not None:
    updates["base_moe_mlp_dim"] = config.moe_intermediate_size
  if config.quantization_config is not None:
    recipe = recipe_source_values(config.quantization_config)
    batch_split = updates.get("use_batch_split_schedule", values["use_batch_split_schedule"])
    if isinstance(config.quantization_config, (BatchSplitQwixConfig, BatchSplitManualConfig)) and not batch_split:
      raise ValueError("Batch-split quantization requires the separately selected supported batch-split schedule.")
    if (isinstance(config.quantization_config, (QwixRecipeConfig, QwixFP8FullConfig))
        and recipe["quantization"] and batch_split):
      raise ValueError("Qwix graph recipes require use_batch_split_schedule=False; select a batch-split recipe instead.")
    calibration = "act_quantization_calibration_method"
    moe = None if config.moe_config is None else config.moe_config.general
    if moe is not None and calibration in moe.model_fields_set and calibration in recipe:
      if (calibration in config.quantization_config.model_fields_set
          and recipe[calibration] != getattr(moe, calibration)):
        raise ValueError("MoE communication and the quantization recipe share act_quantization_calibration_method; "
                         "explicit values must agree.")
      recipe[calibration] = getattr(moe, calibration)
    updates.update(recipe)
    explicit.update(recipe)
  values.update(updates)
  profile = active_profile()
  if profile is not None:
    values.update({name: value for name, value in profile.source_values().items()
                   if name not in explicit or name == "hardware"})
  values.update(getattr(config, "_lilax_checkpoint_layout", {}))
  return InertConfig._from_resolved(values, {"model_name", *explicit})


def _read_domain(cls, values, explicit):
  result = cls.model_validate({name: values[name] for name in cls.model_fields if name in values})
  object.__setattr__(result, "__pydantic_fields_set__", set(cls.model_fields) & set(explicit))
  return result


def _read_recipe(source):
  if source.qwix_qat_config is not None:
    return source.qwix_qat_config
  if not source.quantization:
    return None
  if source.use_batch_split_schedule and source.quantization:
    cls = BatchSplitManualConfig if source.use_manual_quantization else BatchSplitQwixConfig
  elif source.use_qwix_quantization:
    cls = QwixFP8FullConfig if source.quantization == "fp8_full" else QwixRecipeConfig
  else:
    cls = {
        "int8": AqtInt8Config, "intmp": AqtMixedPrecisionConfig, "serve_fp8_weight": ServeFP8WeightConfig,
        "jax_mxfp8": JaxMXConfig, "jax_mxfp4": JaxMXConfig,
    }.get(source.quantization, ComputeRecipeConfig)
  return _read_domain(cls, source.get_source_values(), source._explicit_keys)


def model_config_from_effective(source, native_config, *, updates, metadata_config):
  """Persist model-owned runtime values without runtime dtype or enum identities."""
  reference = model_config_from_source(source)
  names = set(_DIMENSIONS.values()) | {"base_moe_mlp_dim", "quantization", "use_qwix_quantization",
                                       "use_manual_quantization"}
  for domain in _MODEL_DOMAINS:
    names.update(_domain_values(getattr(reference, domain)))
  if reference.quantization_config is not None:
    names.update(recipe_source_values(reference.quantization_config).keys() - {"qwix_qat_config"})
  values = source.get_source_values()
  for name in names:
    value = _json_value(getattr(native_config, name))
    values[name] = str(value) if name in ("dtype", "grad_dtype", "weight_dtype") else value
  values.update(updates)
  effective = InertConfig._from_resolved(values, source._explicit_keys | names | updates.keys())
  return model_config_from_source(effective, metadata_config=metadata_config)


def model_config_from_source(source, *, model_type="lilax", metadata_config=None):
  """Recover model fields while retaining explicit-field provenance separately from defaults."""
  values = source.get_source_values()
  explicit = source._explicit_keys
  read = lambda cls: _read_domain(cls, values, explicit)
  decoder = getattr(source.decoder_block, "value", source.decoder_block)
  qwen = read(Qwen3_8FlashNextConfig) if source.model_name in ("qwen3.8-flash-next", "qwen3.8-flash-next-mini") else (
      read(Qwen3NextConfig) if decoder in ("qwen3_next", "qwen3_5") else None
  )
  config = MaxTextConfig(
      model_name=source.model_name, model_type=model_type, vocab_size=source.vocab_size,
      hidden_size=source.emb_dim, intermediate_size=source.mlp_dim, num_hidden_layers=source.num_decoder_layers,
      num_attention_heads=source.num_query_heads, num_key_value_heads=source.num_kv_heads, head_dim=source.head_dim,
      max_position_embeddings=source.max_target_length, enable_dropout=source.enable_dropout,
      dropout_rate=source.dropout_rate, architecture_config=read(ArchitectureConfig),
      attention_config=AttentionConfig(
          general=read(AttentionParameters), moba=read(MoBaConfig), mla=read(MlaAttentionConfig),
          compressed=read(CompressedAttentionConfig), indexer=read(AttentionIndexerConfig), llama4=read(Llama4AttentionConfig),
      ),
      position_config=PositionConfig(
          embedding=read(PositionalEmbeddingConfig), rope=read(RopeConfig), yarn=read(YarnRopeConfig),
          use_mrope=source.use_mrope, mrope_section=source.mrope_section,
          position_id_per_seconds=source.position_id_per_seconds,
      ),
      moe_config=MoEConfig(general=read(MoEParameters), deepseek=read(DeepSeekMoEConfig)),
      qwen_config=qwen, engram_config=read(EngramConfig), ming_config=read(MingConfig),
      hyperconnection_config=read(HyperconnectionConfig), vision_config=read(VisionConfig),
      vision_projector_config=read(VisionProjectorConfig), audio_config=read(AudioConfig), mtp_config=read(MTPConfig),
      logits_config=read(LogitsConfig), precision_config=read(PrecisionConfig), quantization_config=_read_recipe(source),
      kv_cache_quantization_config=read(KVCacheQuantizationConfig), sparsity_config=read(SparsityConfig),
      kernel_config=KernelConfig(
          **{name: values[name] for name in ("attention", "use_ragged_attention", "enable_padding_causal_mask",
                                            "use_tokamax_splash", "use_jax_splash")},
          splash=read(SplashAttentionConfig), moe=read(MoEKernelsConfig),
      ),
      layout_config=read(ModelLayoutConfig), moe_intermediate_size=source.moe_mlp_dim,
  )
  config._lilax_explicit_keys = explicit
  config._lilax_checkpoint_layout = {
      name: getattr(source, name) for name in ("scan_layers", "param_scan_axis") if name in explicit
  }
  object.__setattr__(config.kernel_config, "__pydantic_fields_set__", set(config.kernel_config.model_fields) & set(explicit))
  object.__setattr__(config.position_config, "__pydantic_fields_set__",
                     set(config.position_config.model_fields) & set(explicit))
  if metadata_config is not None:
    effective, config = config, copy.deepcopy(metadata_config)
    for name in (*_DIMENSIONS, *_MODEL_DOMAINS, "model_name", "quantization_config", "moe_intermediate_size",
                 "_lilax_explicit_keys", "_lilax_checkpoint_layout"):
      setattr(config, name, getattr(effective, name))
  return config
