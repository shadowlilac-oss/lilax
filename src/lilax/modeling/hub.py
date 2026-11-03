"""Resolve supported MaxText model identities and local or Hub snapshots."""

from dataclasses import dataclass
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from lilax._backend.checkpoint_conversion.capabilities import (
    HF_EXPORT_MODELS, HF_IMPORT_MODELS, HF_REPOS, MODEL_CONFIG_NAMES, NATIVE_MODELS,
)


@dataclass(frozen=True)
class ModelCapabilities:
  model_name: str
  native: bool
  hf_import: bool
  hf_export: bool
  hf_repo_id: str | None


def get_model_capabilities(model_name: str) -> ModelCapabilities:
  if model_name not in NATIVE_MODELS | HF_IMPORT_MODELS | HF_EXPORT_MODELS:
    raise ValueError(f"Unknown MaxText model name: {model_name!r}")
  return ModelCapabilities(
      model_name, model_name in NATIVE_MODELS, model_name in HF_IMPORT_MODELS,
      model_name in HF_EXPORT_MODELS, HF_REPOS.get(model_name),
  )


def resolve_model_name(
    pretrained_model_name_or_path: str | Path, *, model_name: str | None = None, snapshot: str | Path | None = None,
) -> str:
  if model_name is not None:
    get_model_capabilities(model_name)
    model_name = MODEL_CONFIG_NAMES.get(model_name, model_name)
  name = str(pretrained_model_name_or_path)
  path = Path(pretrained_model_name_or_path).expanduser()
  matches = {MODEL_CONFIG_NAMES.get(key, key) for key, repo_id in HF_REPOS.items() if repo_id == name}
  if name in NATIVE_MODELS | MODEL_CONFIG_NAMES.keys() and not path.is_dir():
    matches = {MODEL_CONFIG_NAMES.get(name, name)}
  if snapshot is not None:
    path = Path(snapshot).expanduser()
  if path.is_dir():
    config_path = path / "config.json"
    if config_path.is_file():
      with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
      snapshot_model_name = config.get("model_name") if isinstance(config, dict) else None
      if snapshot_model_name is not None:
        get_model_capabilities(snapshot_model_name)
        snapshot_matches = {MODEL_CONFIG_NAMES.get(snapshot_model_name, snapshot_model_name)}
      else:
        repo_name = config.get("_name_or_path") if isinstance(config, dict) else None
        snapshot_matches = {MODEL_CONFIG_NAMES.get(key, key) for key, repo_id in HF_REPOS.items() if repo_id == repo_name}
      if snapshot_matches:
        if matches and not matches.intersection(snapshot_matches):
          raise ValueError(f"Snapshot identity in {config_path} conflicts with {name!r}")
        matches = matches.intersection(snapshot_matches) if matches else snapshot_matches
  if model_name is not None:
    if matches and model_name not in matches:
      raise ValueError(f"Model {name!r} conflicts with {model_name!r}")
    return model_name
  if len(matches) != 1:
    raise ValueError(f"Unknown or ambiguous Hugging Face model: {name!r}; specify model_name")
  return matches.pop()


def resolve_model_snapshot(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
  path = Path(pretrained_model_name_or_path).expanduser()
  if path.is_dir():
    return path
  return Path(
      snapshot_download(
          repo_id=HF_REPOS.get(str(pretrained_model_name_or_path), str(pretrained_model_name_or_path)),
          revision=revision,
          token=token,
          cache_dir=cache_dir,
          local_files_only=local_files_only,
      )
  )
