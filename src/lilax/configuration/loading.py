"""Load supported Hugging Face model configuration and checkpoint metadata."""

import json
import os
from pathlib import Path

from lilax.configuration.schema import MaxTextConfig
from lilax._backend.configs.inert import MaxTextConfig as InertConfig
from lilax._backend.common.checkpoints.model_path import MODEL_CHECKPOINT_DIRECTORY
from lilax.configuration.model_mapping import model_config_from_source
from lilax.configuration.values import load_config_values
from lilax.modeling.hub import resolve_model_name, resolve_model_snapshot


class AutoConfig:
  """Load model configuration without creating a runtime or model weights."""

  @staticmethod
  def from_pretrained(
      pretrained_model_name_or_path: str | Path,
      *,
      revision: str | None = None,
      token: str | bool | None = None,
      cache_dir: str | Path | None = None,
      local_files_only: bool = False,
      model_name: str | None = None,
  ) -> MaxTextConfig:
    if model_name is not None:
      resolve_model_name(pretrained_model_name_or_path, model_name=model_name)
    snapshot = resolve_model_snapshot(
        pretrained_model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=["config.json", "maxtext_easy_config_metadata.json"],
    )
    resolved_name = resolve_model_name(pretrained_model_name_or_path, model_name=model_name, snapshot=snapshot)
    saved_values = None
    saved_explicit_keys = []
    snapshot_config = {}
    snapshot_config_path = snapshot / "config.json"
    if snapshot_config_path.is_file():
      with snapshot_config_path.open(encoding="utf-8") as snapshot_config_file:
        snapshot_config = json.load(snapshot_config_file)
      if "model_name" in snapshot_config and "hidden_size" in snapshot_config:
        resolve_model_name(snapshot, model_name=resolved_name)
        config = MaxTextConfig(**snapshot_config)
        metadata_path = snapshot / "maxtext_easy_config_metadata.json"
        if metadata_path.is_file():
          with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
          if (not isinstance(metadata, dict)
              or set(metadata) - {"model_checkpoint_directory", "model_checkpoint_format"}
              != {"explicit_keys", "checkpoint_layout"}
              or metadata.get("model_checkpoint_directory", MODEL_CHECKPOINT_DIRECTORY) != MODEL_CHECKPOINT_DIRECTORY
              or metadata.get("model_checkpoint_format", "maxtext") not in ("maxtext", "huggingface")
              or not isinstance(metadata["explicit_keys"], list)
              or any(not isinstance(key, str) for key in metadata["explicit_keys"])):
            raise ValueError("Model checkpoint metadata must contain explicit_keys and checkpoint_layout")
          layout = metadata["checkpoint_layout"]
          if (not isinstance(layout, dict) or set(layout) != {"scan_layers", "param_scan_axis"}
              or type(layout["scan_layers"]) is not bool or type(layout["param_scan_axis"]) is not int):
            raise ValueError("Checkpoint layout must contain boolean scan_layers and integer param_scan_axis")
          config._lilax_checkpoint_layout = layout
          config._lilax_explicit_keys = frozenset(metadata["explicit_keys"]) | layout.keys()
        return config
      if "model_name" in snapshot_config:
        resolve_model_name(snapshot, model_name=resolved_name)
        saved_values = snapshot_config
        metadata_path = snapshot / "maxtext_easy_config_metadata.json"
        if metadata_path.is_file():
          with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
          if not isinstance(metadata, dict) or set(metadata) != {"explicit_keys"}:
            raise ValueError("Configuration metadata must contain only explicit_keys")
          saved_explicit_keys = metadata["explicit_keys"]
          if not isinstance(saved_explicit_keys, list) or any(
              not isinstance(key, str) or key not in saved_values for key in saved_explicit_keys
          ):
            raise ValueError("Configuration metadata explicit_keys must be strings present in config.json")
    values = load_config_values(resolved_name, _saved_values=saved_values)
    env_keys = {key for key in values if "M_" + key.upper() in os.environ}
    saved_hardware = {"hardware"} if saved_values is not None and "hardware" in saved_values else set()
    source = InertConfig._from_resolved(values, {"model_name", *saved_explicit_keys, *saved_hardware, *env_keys})
    config = model_config_from_source(source, model_type=snapshot_config.get("model_type", "lilax"))
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "architectures"):
      if name in snapshot_config:
        setattr(config, name, snapshot_config[name])
    return config
