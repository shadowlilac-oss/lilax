"""Native HLO and jaxpr dump configuration fields."""

import os
from tempfile import gettempdir

from lilax._backend.configs.path_types import PathStr
from pydantic.fields import Field
from pydantic.main import BaseModel


class HloDump(BaseModel):
  """Configuration for dumping HLO modules for debugging."""

  dump_hlo: bool = Field(False, description="Enable HLO dumping.")
  dump_step: int = Field(-1, description="Dump HLO at a specific step. -1 disables step-specific dump.")
  dump_hlo_local_dir: PathStr = Field(
      os.path.join(gettempdir(), "xla_dump", ""),
      description="Local directory to dump HLO.",
  )
  dump_hlo_delete_local_after: bool = Field(True, description="Delete local HLO dump after uploading to GCS.")
  dump_hlo_gcs_dir: PathStr = Field("", description="GCS directory to upload HLO dumps.")
  dump_hlo_module_name: str = Field("jit_train_step", description="Filter modules to dump by this name.")
  dump_hlo_local_module_name: str = Field("jit_train_step", description="Filter modules to save locally by this name.")
  dump_hlo_xla_flags: str = Field("", description="Pass custom XLA flags for HLO dumping.")
  dump_hlo_upload_all: bool = Field(False, description="Upload HLO from all hosts.")
  dump_jaxpr: bool = Field(False, description="Enable jaxpr dumping.")
  dump_jaxpr_local_dir: PathStr = Field(
      os.path.join(gettempdir(), "jaxpr_dump", ""),
      description="Local directory to dump jaxpr.",
  )
  dump_jaxpr_delete_local_after: bool = Field(True, description="Delete local jaxpr dump after uploading to GCS.")
  dump_jaxpr_gcs_dir: PathStr = Field("", description="GCS directory to upload jaxpr dumps.")
