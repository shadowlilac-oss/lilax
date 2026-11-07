"""MLPerf run-clock start, preceded by optional program warmup and a cross-host barrier."""

import datetime
import time

import jax
from jax.experimental import multihost_utils

from lilax._backend.common.telemetry import mllog_utils
from lilax._backend.observability import max_logging


def start_run_clock(config, start_step, warmup_fn=None, barrier_fn=None, python_vars=None):
  mllog_utils.init_print(config)
  if warmup_fn is not None:
    start = time.perf_counter()
    warmup_fn()
    warmed = time.perf_counter()
    if barrier_fn is None:
      barrier_fn = lambda: multihost_utils.sync_global_devices("warmup_programs_in_init")
    barrier_fn()
    done = time.perf_counter()
    max_logging.log(
        f"warmup_programs_in_init: warmup {warmed - start:.3f} s, barrier wait {done - warmed:.3f} s "
        f"(process {jax.process_index()})"
    )
    if python_vars is not None:
      python_vars["last_step_completion"] = datetime.datetime.now()
  mllog_utils.init_stop()
  mllog_utils.run_start()
  mllog_utils.block_start(config, start_step)
