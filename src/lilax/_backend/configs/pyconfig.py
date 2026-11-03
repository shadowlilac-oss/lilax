"""Native configuration preparation and read-only HyperParameters wrapper."""

import ast
import copy
import logging
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache
from omegaconf import OmegaConf

from lilax._backend.common.common_types import DecoderBlockType, ShardMode
from lilax._backend.configs import types
from lilax._backend.configs.multimodal_fields import Multimodal
from lilax._backend.inference.inference_utils import str2bool
from lilax._backend.utils import max_logging
from lilax._backend.distributed.jax_distributed import initialization
from lilax._backend.utils.globals import HF_IDS, MAXTEXT_ASSETS_ROOT, MAXTEXT_CONFIGS_DIR, MAXTEXT_PKG_DIR


logger = logging.getLogger(__name__)
KEYS_NO_LOGGING = ("hf_access_token",)
_CONFIG_FILE_MAPPING = {
    "trainers.pre_train.train": "base.yml",
    "trainers.pre_train.train_compile": "base.yml",
    "trainers.post_train.distillation.train_distill": "post_train/distillation.yml",
    "trainers.post_train.dpo.train_dpo": "post_train/dpo.yml",
    "trainers.post_train.rl.train_rl": "post_train/rl.yml",
    "trainers.post_train.sft.train_sft": "post_train/sft.yml",
    "trainers.post_train.sft.train_sft_native": "post_train/sft.yml",
    "inference.decode": "base.yml",
    "inference.decode_multi": "base.yml",
    "inference.inference_microbenchmark": "base.yml",
    "inference.inference_microbenchmark_sweep": "base.yml",
    "inference.maxengine.server": "base.yml",
    "inference.mlperf.microbenchmarks.benchmark_chunked_prefill": "base.yml",
    "inference.vllm_decode": "base.yml",
    "checkpoint_conversion.to_maxtext": "base.yml",
    "checkpoint_conversion.to_huggingface": "base.yml",
}


def resolve_config_path(path: str) -> str:
  if os.path.isfile(path):
    return path
  if path.startswith("src/maxtext/configs/"):
    candidate = os.path.join(MAXTEXT_CONFIGS_DIR, path[len("src/maxtext/configs/"):])
    if os.path.isfile(candidate):
      return candidate
  return os.path.join("src", path)


def _resolve_or_infer_config(argv: list[str] | None, overrides: dict[str, Any]) -> tuple[str, list[str]]:
  if overrides.get("base_config"):
    logger.info("Using config : %s", overrides["base_config"])
    return resolve_config_path(overrides["base_config"]), argv[1:] if argv else []
  if argv and len(argv) >= 2 and argv[1].lower().endswith((".yml", ".yaml")):
    return resolve_config_path(argv[1]), argv[2:]
  module = None
  if argv:
    real_path = os.path.realpath(argv[0])
    package_path = os.path.realpath(MAXTEXT_PKG_DIR)
    if real_path.startswith(package_path + os.sep):
      module = os.path.relpath(real_path, package_path).replace(os.sep, ".").removesuffix(".py")
  if module not in _CONFIG_FILE_MAPPING:
    logger.warning("No config file provided and no default config found for module '%s', using base.yml", module)
    return os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"), argv[1:] if argv else []
  config_path = os.path.join(MAXTEXT_CONFIGS_DIR, _CONFIG_FILE_MAPPING[module])
  logger.warning("No config file provided, using default config mapping: %s", config_path)
  return config_path, argv[1:]


def _load_config(config_name: str):
  config = OmegaConf.load(config_name)
  if "base_config" in config:
    base_path = config["base_config"]
    if not os.path.isabs(base_path):
      parent_path = os.path.join(os.path.dirname(config_name), base_path)
      if not os.path.isfile(parent_path):
        parent_path = os.path.join(MAXTEXT_CONFIGS_DIR, base_path)
    else:
      parent_path = base_path
    config = OmegaConf.merge(_load_config(parent_path), config)
  return config


def _load_raw_config(config_path: str, overrides: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
  base = _load_config(config_path)
  override_config = OmegaConf.create(overrides)
  combined = OmegaConf.merge(base, override_config)
  if not combined.get("base_output_directory"):
    max_logging.warning("base_output_directory is not provided; Using local directory called maxtext_output")
    override_config["base_output_directory"] = os.path.abspath("maxtext_output")
  if not combined.get("hf_access_token") and os.environ.get("HF_TOKEN"):
    override_config["hf_access_token"] = os.environ["HF_TOKEN"]
  combined = OmegaConf.merge(base, override_config)
  model_name = combined.get("model_name", "default").replace("-Instruct", "")
  model = OmegaConf.create({})
  if model_name != "default":
    paths = (
        os.path.join(os.path.dirname(config_path), "models", f"{model_name}.yml"),
        os.path.join(os.path.dirname(os.path.dirname(config_path)), "models", f"{model_name}.yml"),
        os.path.join(MAXTEXT_CONFIGS_DIR, "models", f"{model_name}.yml"),
    )
    model_path = next((path for path in paths if os.path.isfile(path)), paths[-1])
    if os.path.exists(model_path):
      loaded_model = OmegaConf.load(model_path)
      if combined.get("override_model_config"):
        model = OmegaConf.create({key: value for key, value in loaded_model.items() if key not in override_config})
      else:
        conflicts = []
        for key in loaded_model:
          if key not in override_config:
            continue
          try:
            model_value = OmegaConf.to_container(OmegaConf.create({key: loaded_model[key]}), resolve=True)[key]
          except Exception:
            model_value = loaded_model[key]
          try:
            override_value = OmegaConf.to_container(OmegaConf.create({key: override_config[key]}), resolve=True)[key]
          except Exception:
            override_value = override_config[key]
          if model_value != override_value:
            conflicts.append(key)
        if conflicts:
          raise ValueError(
              f"Keys {conflicts} are overridden by both model config and CLI/kwargs with different values."
              "This is not allowed, unless setting `override_model_config=True`."
          )
        model = loaded_model
    else:
      logger.warning("Model config for '%s' not found at %s", model_name, model_path)

  model_keys = set(model)
  rules = []
  for config in (base, model, override_config):
    config_rules = config.get("logical_axis_rules", [])
    config_rules = OmegaConf.to_container(config_rules, resolve=True) if config_rules else []
    if config.get("override_logical_axis_rules"):
      rules = config_rules
    elif config_rules:
      new_keys = {rule[0] for rule in config_rules}
      rules = [rule for rule in rules if rule[0] not in new_keys] + config_rules
    if "logical_axis_rules" in config:
      del config["logical_axis_rules"]
  merged = OmegaConf.merge(base, model, override_config)
  merged["logical_axis_rules"] = rules
  raw_keys = OmegaConf.to_container(merged, resolve=True)
  env_keys = set()
  for key in tuple(raw_keys):
    env_key = "M_" + key.upper()
    if env_key not in os.environ:
      continue
    if key in overrides:
      raise ValueError(
          f"Key '{key}' is overridden by both CLI/kwargs and environment variable '{env_key}'. This is not allowed."
      )
    if not combined.get("override_model_config") and key in model_keys:
      raise ValueError(
          f"Key '{key}' is overridden by both model config and environment variable '{env_key}'."
          "This is not allowed, unless setting `override_model_config=True`."
      )
    env_keys.add(key)
    original_value = raw_keys[key]
    parser = str2bool if isinstance(original_value, bool) else type(original_value)
    if not isinstance(original_value, (str, int, float, bool)):
      raise TypeError(f"Type {type(original_value)} for key '{key}' not supported for ENV override.")
    try:
      raw_keys[key] = parser(os.environ[env_key])
    except (ValueError, KeyError) as error:
      raise ValueError(f"Couldn't parse value from ENV '{os.environ[env_key]}' for key '{key}'") from error
  return raw_keys, set(overrides) | env_keys


def _tuples_to_lists(value: Any) -> Any:
  return [_tuples_to_lists(item) for item in value] if isinstance(value, (list, tuple)) else value


def _lists_to_tuples(value: Any) -> Any:
  return tuple(_lists_to_tuples(item) for item in value) if isinstance(value, list) else value


def _coerce_to_list(value: Any) -> Any:
  if isinstance(value, str):
    cleaned = value.strip()
    if (cleaned.startswith("[") and cleaned.endswith("]")) or (cleaned.startswith("(") and cleaned.endswith(")")):
      try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, (list, tuple)):
          return list(parsed)
        return [str(parsed)]
      except (ValueError, SyntaxError):
        return [value]
    return [value]
  if isinstance(value, tuple):
    return list(value)
  return value


def _check_for_invalid_keys(key: str, valid_fields) -> bool:
  if key in valid_fields:
    return True
  for field_info in valid_fields.values():
    annotation = getattr(field_info, "annotation", None)
    if hasattr(annotation, "model_fields") and _check_for_invalid_keys(key, annotation.model_fields):
      return True
  return False


def _prepare_for_pydantic(raw_keys: dict[str, Any], config_class: type[Any] = types.MaxTextConfig) -> dict[str, Any]:
  pydantic_kwargs = {}
  valid_fields = config_class.model_fields
  use_multimodal = raw_keys.get("use_multimodal", False)

  for key, value in raw_keys.items():
    if not _check_for_invalid_keys(key, valid_fields):
      if not use_multimodal and key in Multimodal.model_fields:
        logger.warning("Ignoring multimodal field %s because use_multimodal is False.", repr(key))
        continue
      logger.warning("Ignoring invalid/unsupported field from YAML/CLI: %s", repr(key))
      raise ValueError(f"{key!r} not in {', '.join(map(repr, valid_fields))}.")

    new_value = value
    if key != "vision_encoder_block" and isinstance(new_value, str) and new_value.lower() == "none":
      new_value = None
    if key in ("logical_axis_rules", "data_sharding"):
      if isinstance(new_value, tuple):
        new_value = _tuples_to_lists(new_value)
      if key == "data_sharding" and isinstance(new_value, list) and new_value and isinstance(new_value[0], str):
        new_value = [new_value]
    if key in ("train_data_columns", "eval_data_columns", "trainable_parameters_mask", "adamw_mask"):
      new_value = _coerce_to_list(new_value)
    if key in (
        "hf_train_files", "hf_eval_files", "hf_access_token", "hf_name", "hf_data_dir", "hf_eval_split", "tokenizer_path"
    ) and new_value == "":
      new_value = None
    if key in ("run_name", "dump_hlo_local_module_name", "dump_hlo_module_name") and new_value is None:
      new_value = ""
    if key == "tokenizer_path" and new_value is None:
      try:
        model_name = raw_keys.get("model_name", "default")
        new_value = HF_IDS[model_name]
        if model_name != "default":
          pydantic_kwargs["_auto_hf_tokenizer"] = True
      except KeyError:
        new_value = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers/tokenizer.llama2")
        max_logging.warning(
            "tokenizer_path not found in HF_IDS in maxtext/src/maxtext/utils/globals.py. "
            "Using the default src/maxtext/assets/tokenizers/tokenizer.llama2 instead. "
            "Please pass tokenizer_path in your command if this is not intended."
        )
    pydantic_kwargs[key] = new_value

  if pydantic_kwargs.pop("_auto_hf_tokenizer", False):
    pydantic_kwargs["tokenizer_type"] = "huggingface"
  return pydantic_kwargs


class HyperParameters:
  """Read-only native configuration with JAX-compatible values."""

  def __init__(self, pydantic_config: types.MaxTextConfig):
    object.__setattr__(self, "_pydantic_config", pydantic_config)
    final_dict = pydantic_config.model_dump()
    final_dict["dtype"] = jnp.dtype(final_dict["dtype"])
    final_dict["grad_dtype"] = jnp.dtype(final_dict["grad_dtype"])
    final_dict["weight_dtype"] = jnp.dtype(final_dict["weight_dtype"])
    final_dict["rollout_weight_dtype"] = (
        final_dict["weight_dtype"] if not final_dict["rollout_weight_dtype"] else jnp.dtype(final_dict["rollout_weight_dtype"])
    )
    final_dict["mu_dtype"] = final_dict["weight_dtype"] if not final_dict["mu_dtype"] else jnp.dtype(final_dict["mu_dtype"])
    final_dict["logical_axis_rules"] = _lists_to_tuples(final_dict["logical_axis_rules"])
    final_dict["logical_axis_rules_for_eval"] = _lists_to_tuples(final_dict["logical_axis_rules_for_eval"])
    final_dict["data_sharding"] = _lists_to_tuples(final_dict["data_sharding"])
    final_dict["decoder_block"] = DecoderBlockType(final_dict["decoder_block"])
    final_dict["shard_mode"] = ShardMode(final_dict["shard_mode"])
    object.__setattr__(self, "_flat_config", final_dict)

  def __deepcopy__(self, memo):
    return HyperParameters(copy.deepcopy(self._pydantic_config, memo))

  def tree_flatten(self):
    return (), self

  def __getattr__(self, attr: str) -> Any:
    flat_config = object.__getattribute__(self, "_flat_config")
    if attr in flat_config:
      value = flat_config[attr]
      if isinstance(value, dict) and attr in ("debug", "rl", "lora"):
        return getattr(object.__getattribute__(self, "_pydantic_config"), attr)
      return value
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

  def __setattr__(self, attr: str, value: Any) -> None:
    raise ValueError("Configuration is read-only and cannot be modified after initialization.")

  def get_keys(self) -> dict[str, Any]:
    return self._flat_config


def _resolve_native_config(
    raw_keys: dict[str, Any], explicit_keys: frozenset[str] | set[str], config_class: type[Any],
    *, resolved_values: bool = False, train_dataset_provided: bool = False, eval_dataset_provided: bool = False,
) -> HyperParameters:
  raw_keys = dict(raw_keys)
  if not raw_keys.get("base_output_directory"):
    max_logging.warning("base_output_directory is not provided; Using local directory called maxtext_output")
    raw_keys["base_output_directory"] = os.path.abspath("maxtext_output")
  if not raw_keys.get("hf_access_token") and os.environ.get("HF_TOKEN"):
    raw_keys["hf_access_token"] = os.environ["HF_TOKEN"]
  if "hardware" not in explicit_keys:
    if "M_HARDWARE" in os.environ:
      raw_keys["hardware"] = os.environ["M_HARDWARE"]
    elif os.environ.get("JAX_COORDINATOR_IP") is not None or os.path.exists("/dev/nvidiactl"):
      raw_keys["hardware"] = "gpu"
  prepared = _prepare_for_pydantic(raw_keys, config_class=config_class)
  if prepared.get("use_tokamax_splash") and prepared.get("use_jax_splash"):
    raise ValueError("At most one of `use_tokamax_splash` and `use_jax_splash` can be set to True.")
  hlo_defaults_prepared = issubclass(config_class, types.MaxTextConfig)
  if hlo_defaults_prepared:
    prepared = types.prepare_runtime_hlo_defaults(prepared, config_class)
  if prepared.get("jax_debug_log_modules"):
    jax.config.update("jax_debug_log_modules", prepared["jax_debug_log_modules"])
  if "pytest" not in sys.modules:
    initialization.maybe_initialize_jax_distributed_system(prepared)
  if prepared.get("jax_cache_dir"):
    if prepared.get("dump_hlo"):
      max_logging.warning("JAX compilation cache is disabled because dump_hlo is True. HLO dumping requires recompilation.")
    else:
      compilation_cache.set_cache_dir(os.path.expanduser(prepared["jax_cache_dir"]))
  pydantic_config = config_class.model_validate(
      prepared, context={
          "hlo_defaults_prepared": hlo_defaults_prepared, "resolved_values": resolved_values,
          "train_dataset_provided": train_dataset_provided, "eval_dataset_provided": eval_dataset_provided,
      }
  )
  object.__setattr__(pydantic_config, "__pydantic_fields_set__", set(explicit_keys) & config_class.model_fields.keys())
  config = HyperParameters(pydantic_config)
  if config.log_config:
    for key, value in sorted(config.get_keys().items()):
      if key not in KEYS_NO_LOGGING:
        logger.info("Config param %s: %s", key, value)
  return config


def initialize(argv: list[str] | None = None, config_class: type[Any] = types.MaxTextConfig, **kwargs) -> HyperParameters:
  config_path, cli_args = _resolve_or_infer_config(argv, kwargs)
  cli_config = OmegaConf.from_cli(cli_args)
  if "hf_access_token" in cli_config:
    logger.warning(
        "WARNING: Passing 'hf_access_token' via command-line arguments is deprecated and insecure because it makes "
        "your token visible in 'ps' and shell history. Please set the 'HF_TOKEN' environment variable instead."
    )
  overrides = OmegaConf.to_container(OmegaConf.merge(cli_config, OmegaConf.create(kwargs)), resolve=True)
  raw_keys, explicit_keys = _load_raw_config(config_path, overrides)
  return _resolve_native_config(raw_keys, explicit_keys, config_class)


def initialize_pydantic(argv: list[str] | None = None, config_class: type[Any] = types.MaxTextConfig, **kwargs) -> Any:
  return initialize(argv, config_class=config_class, **kwargs)._pydantic_config
