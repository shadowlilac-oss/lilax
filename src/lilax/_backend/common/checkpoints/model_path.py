"""Locate owned native model payloads and select the active saved format."""

import json

from etils import epath


MODEL_CHECKPOINT_DIRECTORY = "_lilax_model"


def model_checkpoint_path(path):
  directory = epath.Path(path)
  metadata_path = directory / "maxtext_easy_config_metadata.json"
  if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_checkpoint_directory") == MODEL_CHECKPOINT_DIRECTORY:
      return directory / MODEL_CHECKPOINT_DIRECTORY
  return directory


def model_checkpoint_format(path):
  metadata_path = epath.Path(path) / "maxtext_easy_config_metadata.json"
  if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("model_checkpoint_format", "maxtext")
  return "maxtext"
