import os
from tempfile import gettempdir
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DevelopmentAndDebugging(BaseModel):
  """General settings for development and debugging."""

  constant_bound_config: list = Field([], description="Legacy configuration for constant bounds.")
  jax_cache_dir: str | None = Field("~/jax_cache", description="Directory for JAX compilation cache.")
  jax_distributed_initialization_timeout: int = Field(300, description="Timeout for jax.distributed.initialize.")
  jax_debug_log_modules: str = Field("", description="Set to 'jax' for verbose JAX logging.")
  skip_jax_distributed_system: bool = Field(False, description="If True, do not initialize the jax distributed system.")
  enable_single_controller: bool = Field(False, description="Enable single-controller mode (Pathways).")
  subslice_shape: str = Field("", description="Subslice shape in the form of 'x,y,z' for Pathways.")
  max_checkify: bool = Field(False, description="If True, perform extra checks using jax.checkify, affecting performance.")

  @field_validator("constant_bound_config", mode="before")
  @classmethod
  def _clean_empty_string_for_list(cls, value: Any) -> Any:
    if value == "":
      return []
    if isinstance(value, str):
      return list(map(float, value.split(",")))
    return value


class HloDump(BaseModel):
  """Configuration for dumping HLO modules for debugging."""

  dump_hlo: bool = Field(False, description="Enable HLO dumping.")
  dump_step: int = Field(-1, description="Dump HLO at a specific step. -1 disables step-specific dump.")
  dump_hlo_local_dir: str = Field(os.path.join(gettempdir(), "xla_dump", ""), description="Local directory to dump HLO.")
  dump_hlo_delete_local_after: bool = Field(True, description="Delete local HLO dump after uploading to GCS.")
  dump_hlo_gcs_dir: str = Field("", description="GCS directory to upload HLO dumps.")
  dump_hlo_module_name: str = Field("jit_train_step", description="Filter modules to dump by this name.")
  dump_hlo_local_module_name: str = Field("jit_train_step", description="Filter modules to save locally by this name.")
  dump_hlo_xla_flags: str = Field("", description="Pass custom XLA flags for HLO dumping.")
  dump_hlo_upload_all: bool = Field(False, description="Upload HLO from all hosts.")
  dump_jaxpr: bool = Field(False, description="Enable jaxpr dumping.")
  dump_jaxpr_local_dir: str = Field(os.path.join(gettempdir(), "jaxpr_dump", ""), description="Local directory to dump jaxpr.")
  dump_jaxpr_delete_local_after: bool = Field(True, description="Delete local jaxpr dump after uploading to GCS.")
  dump_jaxpr_gcs_dir: str = Field("", description="GCS directory to upload jaxpr dumps.")
