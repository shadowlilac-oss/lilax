"""Native profiling configuration fields."""

from lilax._backend.configs.profiling_types import ProfilerType, XProfTPUPowerTraceMode
from pydantic.fields import Field
from pydantic.main import BaseModel


class ProfilingCore(BaseModel):
  """Profiler backend selection shared by every trainer."""

  __module__ = "lilax._backend.configs.types"

  profiler: ProfilerType = Field(ProfilerType.NONE, description="Profiler to use ('xplane', 'nsys').")


class Profiling(ProfilingCore):
  """Configuration for step-window performance profiling (`common/profiler.py`)."""

  __module__ = "lilax._backend.configs.types"

  upload_all_profiler_results: bool = Field(False, description="Upload profiler results from all hosts.")
  skip_first_n_steps_for_profiler: int = Field(1, description="Number of initial steps to skip for profiling.")
  profiler_steps: int = Field(5, description="Number of steps to profile.")
  profile_cleanly: bool = Field(True, description="Add block_until_ready to align profile for each step.")
  profile_periodically_period: int = Field(-1, description="If positive, profile every N steps.")
  hide_profiler_step_metric: bool = Field(False, description="Whether to enable profiler step metric.")
  enable_continuous_profiling: bool = Field(False, description="If true, it will support saving profile > 2GB.")
  enable_jax_profiler: bool = Field(False, description="Enable the JAX live profiler.")
  jax_profiler_port: int = Field(9999, description="Port for the JAX profiler.")
  enable_tpu_profiling_options: bool = Field(False, description="Enable TPU advanced profiling options.")
  tpu_num_chips_to_profile_per_task: int = Field(1, description="Specifies the number of TPU chips to profile per task.")
  tpu_num_sparse_cores_to_trace: int = Field(2, description="Specifies the number of TPU chips to profile per task.")
  tpu_num_sparse_core_tiles_to_trace: int = Field(
      1,
      description="Specifies the number of tiles within each sparse core to trace on the TPU.",
  )
  xprof_tpu_power_trace_level: XProfTPUPowerTraceMode = Field(
      XProfTPUPowerTraceMode.POWER_TRACE_NONE,
      description=(
          "TPU power trace level. The value should be 0 (POWER_TRACE_NONE), 1"
          " (POWER_TRACE_NORMAL), or 2 (POWER_TRACE_SPI)"
      ),
  )
  xprof_e2e_enable_fw_throttle_event: bool = Field(False, description="Enable FW throttle event.")
  xprof_e2e_enable_fw_power_level_event: bool = Field(False, description="Enable FW power level event.")
  xprof_e2e_enable_fw_thermal_event: bool = Field(False, description="Enable FW thermal event.")
  profile_power_events: bool = Field(
      False,
      description="Enable TPU-specific power/thermal profiling events."
      " Defaults to False to avoid breaking GPU xplane tracing.",
  )
