"""Load inert MaxText configuration values from packaged YAML files."""

import json
import os
from pathlib import Path

from lilax.configuration.schema import MaxTextConfig
from lilax.configuration.values import load_config_values
from lilax.hub import resolve_model_name, resolve_model_snapshot


class AutoConfig:
  """Load a supported model's inert MaxText configuration."""

  @staticmethod
  def from_pretrained(
      pretrained_model_name_or_path: str | Path,
      *,
      revision: str | None = None,
      token: str | bool | None = None,
      cache_dir: str | Path | None = None,
      local_files_only: bool = False,
      model_name: str | None = None,
      **overrides,
  ) -> MaxTextConfig:
    resolved_name = resolve_model_name(pretrained_model_name_or_path, model_name=model_name)
    snapshot = resolve_model_snapshot(
        pretrained_model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    saved_values = None
    saved_explicit_keys = []
    snapshot_config_path = snapshot / "config.json"
    if snapshot_config_path.is_file():
      with snapshot_config_path.open(encoding="utf-8") as snapshot_config_file:
        snapshot_config = json.load(snapshot_config_file)
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
    values = load_config_values(resolved_name, _saved_values=saved_values, **overrides)
    env_keys = {key for key in values if "M_" + key.upper() in os.environ}
    saved_hardware = {"hardware"} if saved_values is not None and "hardware" in saved_values else set()
    return MaxTextConfig._from_resolved(values, {"model_name", *saved_explicit_keys, *saved_hardware, *overrides, *env_keys})
