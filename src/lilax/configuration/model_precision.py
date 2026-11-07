"""Compute and stored-weight precision."""

from pydantic import ConfigDict, Field

from .domains.model_data_types import DataTypes


class PrecisionConfig(DataTypes):
  model_config = ConfigDict(extra="forbid")
  unquantized_modules: list[str] = Field(default_factory=list)
  weight_block_size: int | list[int] | None = None
