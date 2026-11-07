"""Reusable execution profiles, independent of model and training identities."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

from lilax.configuration.domains.data_aot import AOT
from lilax.configuration.domains.data_debugging import DevelopmentAndDebugging, HloDump
from lilax.configuration.domains.data_diloco import DilocoParams
from lilax.configuration.domains.data_profiling import Profiling
from lilax.configuration.domains.execution_mesh import HardwareAndMesh
from lilax.configuration.domains.execution_monitoring import GcpMonitoring, Goodput, ManagedMLDiagnostics
from lilax.configuration.domains.execution_parallelism import DcnParallelism, IciParallelism, PipelineParallelism
from lilax.configuration.domains.execution_remat import RematAndOffload
from lilax.configuration.domains.execution_sharding import LayoutAndSharding
from lilax.configuration.domains.execution_training import ElasticTraining
from lilax.configuration.domains.model_checkpointing import EmergencyCheckpointing
from lilax.configuration.domains.model_data_types import DataTypes
from lilax.configuration.domains.model_splash import SplashAttention

from ._process import ProcessConfig
from ._tuning import (
    AttentionTuning, CheckpointStorage, ExecutionTuning, MetricsTuning, QuantizationTransport, RLProfilingPlacement,
    XProfServer,
)


class ExecutionProfile(BaseModel):
  """Sparse domain settings consumed before the model runtime is initialized."""

  model_config = ConfigDict(extra="forbid")
  version: Literal[1] = 1
  process: ProcessConfig = Field(default_factory=ProcessConfig)
  mesh: HardwareAndMesh | None = None
  ici: IciParallelism | None = None
  dcn: DcnParallelism | None = None
  pipeline: PipelineParallelism | None = None
  sharding: LayoutAndSharding | None = None
  precision: DataTypes | None = None
  remat: RematAndOffload | None = None
  diloco: DilocoParams | None = None
  debugging: DevelopmentAndDebugging | None = None
  hlo: HloDump | None = None
  compilation: AOT | None = None
  splash: SplashAttention | None = None
  kernels: AttentionTuning | None = None
  execution: ExecutionTuning | None = None
  transport: QuantizationTransport | None = None
  storage: CheckpointStorage | None = None
  recovery: ElasticTraining | None = None
  replication: EmergencyCheckpointing | None = None
  profiling: Profiling | None = None
  metrics: MetricsTuning | None = None
  goodput: Goodput | None = None
  diagnostics: ManagedMLDiagnostics | None = None
  monitoring: GcpMonitoring | None = None
  xprof: XProfServer | None = None
  rl_profiling: RLProfilingPlacement | None = None
  xla_flags: str | None = None

  @field_validator("*", mode="before")
  @classmethod
  def reject_unknown_domain_fields(cls, value, info):
    annotation = cls.model_fields[info.field_name].annotation
    domain = next((item for item in getattr(annotation, "__args__", ()) if hasattr(item, "model_fields")), None)
    if domain is not None and isinstance(value, dict):
      unknown = value.keys() - domain.model_fields.keys()
      if unknown:
        raise ValueError(f"Unknown {info.field_name} settings: {sorted(unknown)}")
    return value

  def source_values(self, *, rl=False):
    values = {}
    for name in type(self).model_fields:
      section = getattr(self, name)
      if name == "rl_profiling" and not rl:
        continue
      if name not in ("process", "version", "xla_flags", "xprof") and section is not None:
        selected = section.model_dump(mode="json", exclude_unset=True)
        if name in ("ici", "dcn"):
          degrees = list(selected.values())
          if any(degree == 0 or degree < -1 for degree in degrees):
            raise ValueError(f"{name} degrees must be positive integers or -1 for an unspecified axis")
          if degrees.count(-1) > 1:
            raise ValueError(f"At most one {name} axis can be unspecified (-1)")
        conflicts = values.keys() & selected.keys()
        if conflicts:
          raise ValueError(f"Profile settings have multiple owners: {sorted(conflicts)}")
        values.update(selected)
    if "hardware" in values and values["hardware"] != self.process.hardware:
      raise ValueError("mesh.hardware must match process.hardware")
    values["hardware"] = self.process.hardware
    if self.process.worker_arrangement == "single_controller":
      values["enable_single_controller"] = True
    return values


def load_profile(path):
  with Path(path).expanduser().open(encoding="utf-8") as stream:
    profile = ExecutionProfile.model_validate(yaml.safe_load(stream))
  profile.source_values()
  return profile


def save_profile(profile, path):
  """Write only selected settings; preserve an existing profile."""
  profile.source_values()
  with Path(path).expanduser().open("x", encoding="utf-8") as stream:
    yaml.safe_dump(profile.model_dump(mode="json", exclude_unset=True), stream, sort_keys=False)


def active_profile():
  payload = os.environ.get("LILAX_EXECUTION_PROFILE")
  return ExecutionProfile.model_validate_json(payload) if payload else None
