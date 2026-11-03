"""Native JAX runtime development and debugging configuration fields."""

from typing import Any

from lilax._backend.configs.path_types import PathStr
from pydantic.fields import Field
from pydantic.functional_validators import field_validator
from pydantic.main import BaseModel


class DevelopmentAndDebugging(BaseModel):
  """General settings for development and debugging."""

  constant_bound_config: list = Field([], description="Legacy configuration for constant bounds.")
  jax_cache_dir: PathStr | None = Field(
      "~/jax_cache",
      description="Directory for JAX compilation cache.",
  )
  jax_distributed_initialization_timeout: int = Field(300, description="Timeout for jax.distributed.initialize.")
  jax_debug_log_modules: str = Field("", description="Set to 'jax' for verbose JAX logging.")
  skip_jax_distributed_system: bool = Field(False, description="If True, do not initialize the jax distributed system.")
  enable_single_controller: bool = Field(False, description="Enable single-controller mode (Pathways).")
  subslice_shape: str = Field("", description="Subslice shape in the form of 'x,y,z' for Pathways.")
  max_checkify: bool = Field(
      False,
      description="If True, perform extra checks using jax.checkify, affecting performance.",
  )

  @classmethod
  def _clean_empty_string_for_list(cls, v: Any) -> Any:
    """Coerces an empty string from YAML into an empty list before validation."""
    if v == "":
      return []
    elif isinstance(v, str):
      return list(map(float, v.split(",")))
    return v

  _validate_config = field_validator("constant_bound_config", mode="before")(_clean_empty_string_for_list)
