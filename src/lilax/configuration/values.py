"""Load inert MaxText configuration values from packaged YAML files."""

import ast
import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel

from lilax._backend.utils.globals import HF_IDS, MAXTEXT_ASSETS_ROOT, MAXTEXT_CONFIGS_DIR


logger = logging.getLogger(__name__)
_CONFIGS_DIR = Path(MAXTEXT_CONFIGS_DIR)


def _load_config(path: Path):
  config = OmegaConf.load(path)
  if "base_config" in config:
    base_path = Path(config.base_config)
    if not base_path.is_absolute():
      candidate = path.parent / base_path
      base_path = candidate if candidate.is_file() else _CONFIGS_DIR / base_path
    config = OmegaConf.merge(_load_config(base_path), config)
  return config


def _merge_rules(base_rules, new_rules, config):
  if config.get("override_logical_axis_rules"):
    return new_rules
  if not new_rules:
    return base_rules
  new_keys = {rule[0] for rule in new_rules}
  return [rule for rule in base_rules if rule[0] not in new_keys] + new_rules


def _coerce_to_list(value):
  if isinstance(value, str):
    cleaned = value.strip()
    if (cleaned.startswith("[") and cleaned.endswith("]")) or (cleaned.startswith("(") and cleaned.endswith(")")):
      try:
        parsed = ast.literal_eval(cleaned)
        return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
      except (ValueError, SyntaxError):
        return [value]
    return [value]
  return list(value) if isinstance(value, tuple) else value


def _tuples_to_lists(value):
  return [_tuples_to_lists(item) for item in value] if isinstance(value, (list, tuple)) else value


def _prepare_values(raw_values: dict[str, Any]) -> dict[str, Any]:
  values = {}
  auto_tokenizer = False
  for key, value in raw_values.items():
    if key != "vision_encoder_block" and isinstance(value, str) and value.lower() == "none":
      value = None
    if key in ("logical_axis_rules", "data_sharding") and isinstance(value, tuple):
      value = _tuples_to_lists(value)
    if key == "data_sharding" and isinstance(value, list) and value and isinstance(value[0], str):
      value = [value]
    if key in ("train_data_columns", "eval_data_columns", "trainable_parameters_mask", "adamw_mask"):
      value = _coerce_to_list(value)
    if key in ("hf_train_files", "hf_eval_files", "hf_access_token", "hf_name", "hf_data_dir", "hf_eval_split", "tokenizer_path") and value == "":
      value = None
    if key in ("run_name", "dump_hlo_local_module_name", "dump_hlo_module_name") and value is None:
      value = ""
    if key == "tokenizer_path" and value is None:
      model_name = raw_values.get("model_name", "default")
      if model_name in HF_IDS:
        value = HF_IDS[model_name]
        auto_tokenizer = model_name != "default"
      else:
        value = str(Path(MAXTEXT_ASSETS_ROOT) / "tokenizers" / "tokenizer.llama2")
        logger.warning("tokenizer_path not found in HF_IDS; using the default Llama 2 tokenizer")
    values[key] = value
  if auto_tokenizer:
    values["tokenizer_type"] = "huggingface"
  return values


def load_config_values(
    model_name: str | None = None, *, base_config: str | Path | None = None,
    _saved_values: dict[str, Any] | None = None, _apply_environment: bool = True, **overrides,
) -> dict[str, Any]:
  """Resolve base, model and explicit overrides without initializing a runtime."""
  config_path = Path(base_config) if base_config is not None else _CONFIGS_DIR / "base.yml"
  base = _load_config(config_path)
  overrides = {**({"model_name": model_name} if model_name is not None else {}), **overrides}
  overrides = {key: value.model_dump(mode="json") if isinstance(value, BaseModel) else value for key, value in overrides.items()}
  override_config = OmegaConf.create(overrides)
  combined = OmegaConf.merge(base, override_config)
  architecture = combined.get("model_name", "default").replace("-Instruct", "")
  model = OmegaConf.create({})
  if architecture != "default":
    candidates = (config_path.parent / "models" / f"{architecture}.yml",
                  config_path.parent.parent / "models" / f"{architecture}.yml",
                  _CONFIGS_DIR / "models" / f"{architecture}.yml")
    model_path = next((path for path in candidates if path.is_file()), candidates[-1])
    if model_path.exists():
      loaded = OmegaConf.load(model_path)
      if combined.get("override_model_config"):
        model = OmegaConf.create({key: value for key, value in loaded.items() if key not in override_config})
      else:
        conflicts = []
        for key in loaded:
          if key in override_config:
            try:
              model_value = OmegaConf.to_container(OmegaConf.create({key: loaded[key]}), resolve=True)[key]
            except Exception:
              model_value = loaded[key]
            try:
              override_value = OmegaConf.to_container(OmegaConf.create({key: override_config[key]}), resolve=True)[key]
            except Exception:
              override_value = override_config[key]
            if model_value != override_value:
              conflicts.append(key)
        if conflicts:
          raise ValueError(f"Keys {conflicts} are overridden by both model config and CLI/kwargs with different values."
                           "This is not allowed, unless setting `override_model_config=True`.")
        model = loaded
    else:
      logger.warning("Model config for '%s' not found at %s", architecture, model_path)

  model_keys = set(model)
  base_rules = base.get("logical_axis_rules", [])
  rules = OmegaConf.to_container(base_rules, resolve=True) if base_rules else []
  if "logical_axis_rules" in base:
    del base["logical_axis_rules"]
  saved_config = OmegaConf.create(_saved_values or {})
  for config in (model, saved_config, override_config):
    config_rules = config.get("logical_axis_rules", [])
    config_rules = OmegaConf.to_container(config_rules, resolve=True) if config_rules else []
    rules = _merge_rules(rules, config_rules, config)
    if "logical_axis_rules" in config:
      del config["logical_axis_rules"]
  merged = OmegaConf.merge(base, model, saved_config, override_config)
  merged["logical_axis_rules"] = rules
  raw_values = OmegaConf.to_container(merged, resolve=True)
  for key in tuple(raw_values):
    if not _apply_environment:
      break
    env_key = "M_" + key.upper()
    if env_key not in os.environ:
      continue
    if key in overrides:
      raise ValueError(f"Key '{key}' is overridden by both CLI/kwargs and environment variable '{env_key}'. This is not allowed.")
    if not combined.get("override_model_config") and key in model_keys:
      raise ValueError(
          f"Key '{key}' is overridden by both model config and environment variable '{env_key}'."
          "This is not allowed, unless setting `override_model_config=True`."
      )
    original_value = raw_values[key]
    if not isinstance(original_value, (str, int, float, bool)):
      raise TypeError(f"Type {type(original_value)} for key '{key}' not supported for ENV override.")
    try:
      if isinstance(original_value, bool):
        proposed = os.environ[env_key].lower()
        if proposed not in ("y", "yes", "t", "true", "1", "n", "no", "f", "false", "0"):
          raise ValueError(f"Invalid value '{proposed}'!")
        raw_values[key] = proposed in ("y", "yes", "t", "true", "1")
      else:
        raw_values[key] = type(original_value)(os.environ[env_key])
    except (ValueError, KeyError) as error:
      raise ValueError(f"Couldn't parse value from ENV '{os.environ[env_key]}' for key '{key}'") from error
  values = _prepare_values(raw_values)
  if values.get("use_tokamax_splash") and values.get("use_jax_splash"):
    raise ValueError("At most one of `use_tokamax_splash` and `use_jax_splash` can be set to True.")
  if _apply_environment and not values.get("hf_access_token") and os.environ.get("HF_TOKEN"):
    values["hf_access_token"] = os.environ["HF_TOKEN"]
  return values


