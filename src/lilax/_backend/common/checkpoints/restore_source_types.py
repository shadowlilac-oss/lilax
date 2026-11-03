"""Checkpoint restore source and path normalization."""

import dataclasses
from typing import Any


def _normalize_checkpoint_root(path_str):
  path_str = str(path_str).rstrip("/")
  if path_str == "items":
    return "."
  return path_str.removesuffix("/items")


@dataclasses.dataclass(frozen=True)
class RestoreSource:
  kind: str
  path: str | None = None
  step: int | None = None
  layout: str | None = None
  manager: Any = dataclasses.field(default=None, repr=False, compare=False)
