"""Timing utility functions."""

import datetime
import tempfile

import jax


def simple_timeit(f, *args, tries=10, task=None, enable_profile=False):
  """Simple utility to time a function for multiple runs."""
  assert task is not None

  trace_name = f"{task}"
  trace_dir = tempfile.mkdtemp(prefix=trace_name + "_")
  print(trace_dir)

  outcomes_ms = []
  jax.block_until_ready(f(*args))
  if enable_profile:
    jax.profiler.start_trace(trace_dir)
  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  if enable_profile:
    jax.profiler.stop_trace()
  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"Average time ms for mm for {task} is {round(average_time_ms, 3)}")
  return average_time_ms / 1000
