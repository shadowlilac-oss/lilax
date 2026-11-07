"""Select existing native profilers from the Trainer callback list."""

from transformers import TrainerCallback

from lilax.profiling import ProfilingConfig


class ProfilerCallback(TrainerCallback):
  def __init__(self, config: ProfilingConfig):
    self.config = config

  def _runtime_overrides(self):
    return self.config.model_dump(mode="json")


class RLProfilerCallback(TrainerCallback):
  def __init__(self, *, start_invocation=2, num_invocations=1, timeout_secs=60.0):
    self.start_invocation = start_invocation
    self.num_invocations = num_invocations
    self.timeout_secs = timeout_secs

  def _runtime_overrides(self):
    return {"profiler": "xplane", "profiler_start_invocation": self.start_invocation,
            "profiler_num_invocations": self.num_invocations, "profiler_timeout_secs": self.timeout_secs}
