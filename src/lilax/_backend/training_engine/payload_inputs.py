"""Turning a trainer payload into the loss function's inputs."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any


def prepare_batch(payload: Any, gen_model_input_fn: Callable[[Any], dict[str, Any]] | None) -> Any:
  if gen_model_input_fn is not None:
    if dataclasses.is_dataclass(payload) and "metadata" in payload.__dataclass_fields__:
      payload = dataclasses.replace(payload, metadata={})
    return gen_model_input_fn(payload)
  if dataclasses.is_dataclass(payload):
    return {
        key: getattr(payload, key)
        for key in payload.__dataclass_fields__
        if getattr(payload, key) is not None and key != "metadata"
    }
  return payload
