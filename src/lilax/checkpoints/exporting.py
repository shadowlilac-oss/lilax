"""Hugging Face snapshot selection for portable weight export."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class HuggingFaceExportConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  hf_model_path: str | Path | None = None
  revision: str | None = None
  token: str | bool | None = None
  cache_dir: str | Path | None = None
  local_files_only: bool = False
  override_model_architecture: bool = False
