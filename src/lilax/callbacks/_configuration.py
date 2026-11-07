"""Install built-in callback settings into their existing backend consumers."""

from .mlperf import MLPerfCallback
from .profiling import ProfilerCallback, RLProfilerCallback
from .stopping import NonFiniteLossCallback


def callback_overrides(callbacks):
  values = {}
  for callback in callbacks or ():
    callback = callback() if isinstance(callback, type) else callback
    if isinstance(callback, (ProfilerCallback, RLProfilerCallback, MLPerfCallback, NonFiniteLossCallback)):
      values.update(callback._runtime_overrides())
  return values
