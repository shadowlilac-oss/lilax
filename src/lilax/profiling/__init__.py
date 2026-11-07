"""Common performance-profile settings."""

from pydantic import BaseModel, ConfigDict

from lilax.configuration.domain_types import ProfilerType
from .m3 import profile_m3


class ProfilingConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  profiler: ProfilerType = ProfilerType.NONE
  skip_first_n_steps_for_profiler: int = 1
  profiler_steps: int = 5
  profile_cleanly: bool = True
  profile_periodically_period: int = -1
  hide_profiler_step_metric: bool = False
  enable_continuous_profiling: bool = False


__all__ = ["ProfilingConfig", "profile_m3"]
