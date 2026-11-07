"""Resolve inert public configurations for the native runtime."""

from lilax._backend.configs import pyconfig, types
from lilax._backend.configs.inert import (
    CustomRule, DecoderBlockType, MaxTextConfig, ReorderStrategy, RLConfig, ShardMode, VisionEncoderBlockType,
)


def _native_enum_values(value):
  if isinstance(value, (CustomRule, DecoderBlockType, ReorderStrategy, ShardMode, VisionEncoderBlockType)):
    return value.value
  if isinstance(value, dict):
    return {key: _native_enum_values(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_native_enum_values(item) for item in value]
  if isinstance(value, tuple):
    return tuple(_native_enum_values(item) for item in value)
  return value


def resolve_runtime_config(
    public_config, *, overrides=None, train_dataset_provided=False, eval_dataset_provided=False,
) -> pyconfig.HyperParameters:
  if not isinstance(public_config, (MaxTextConfig, RLConfig)):
    raise TypeError("Runtime configuration must be an inert MaxTextConfig or RLConfig.")
  config_class = types.RLConfig if isinstance(public_config, RLConfig) else types.MaxTextConfig
  overrides = {} if overrides is None else overrides
  if config_class is types.MaxTextConfig:
    values = public_config.get_source_values()
    values.update(overrides)
    values.pop("qwix_qat_config", None)
  else:
    values = public_config.model_copy(update=overrides).get_keys()
  resolved = pyconfig._resolve_native_config(
      _native_enum_values(values), (public_config._explicit_keys | overrides.keys()) - {"qwix_qat_config"},
      config_class, resolved_values=True,
      train_dataset_provided=train_dataset_provided, eval_dataset_provided=eval_dataset_provided,
  )
  if "tensorboard_dir" in overrides:
    resolved = pyconfig.HyperParameters(resolved._pydantic_config.model_copy(
        update={"tensorboard_dir": overrides["tensorboard_dir"]},
    ))
  return resolved


def resolve_initialized_config(
    public_config, initialized_config, *, overrides=None, train_dataset_provided=False, eval_dataset_provided=False,
) -> pyconfig.HyperParameters:
  if not isinstance(public_config, MaxTextConfig):
    raise TypeError("Initialized configuration must use an inert MaxTextConfig.")
  overrides = {} if overrides is None else overrides
  values = public_config.get_source_values()
  for name in (
      "hardware", "run_name", "base_output_directory", "hf_access_token", "dump_hlo_xla_flags", "dump_hlo_gcs_dir",
      "dump_jaxpr_gcs_dir",
  ):
    if name not in overrides:
      values[name] = getattr(initialized_config, name)
  values.update(overrides)
  values.pop("qwix_qat_config", None)
  prepared = pyconfig._prepare_for_pydantic(_native_enum_values(values), config_class=types.MaxTextConfig)
  config = types.MaxTextConfig.model_validate(
      prepared, context={
          "hlo_defaults_prepared": True, "resolved_values": True,
          "train_dataset_provided": train_dataset_provided, "eval_dataset_provided": eval_dataset_provided,
      }
  )
  object.__setattr__(
      config, "__pydantic_fields_set__",
      (public_config._explicit_keys | overrides.keys()) - {"qwix_qat_config"} & types.MaxTextConfig.model_fields.keys(),
  )
  if "tensorboard_dir" in overrides:
    config = config.model_copy(update={"tensorboard_dir": overrides["tensorboard_dir"]})
  return pyconfig.HyperParameters(config)


def with_inference_sweep_overrides(
    config: pyconfig.HyperParameters, *, prefill_cache_axis_order: str, ar_cache_axis_order: str,
    tensorboard_dir: str, run_name: str,
) -> pyconfig.HyperParameters:
  """Copy the four native sweep settings without repeating runtime initialization."""
  overrides = {
      "prefill_cache_axis_order": prefill_cache_axis_order,
      "ar_cache_axis_order": ar_cache_axis_order,
      "tensorboard_dir": tensorboard_dir,
      "run_name": run_name,
  }
  config_copy = config._pydantic_config.model_copy(deep=True, update=overrides)
  object.__setattr__(config_copy, "__pydantic_fields_set__", config._pydantic_config.model_fields_set | overrides.keys())
  return pyconfig.HyperParameters(config_copy)
