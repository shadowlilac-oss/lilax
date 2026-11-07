"""Memory usage reporting for checkpoint conversion."""

import resource

import psutil
from tqdm import tqdm

from lilax._backend.observability import max_logging


def print_ram_usage(stage=""):
  memory = psutil.virtual_memory()
  max_logging.log(
      f"[{stage}] RAM Usage: {memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f} GB ({memory.percent:.1f}%)"
  )


def print_peak_memory():
  peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  max_logging.log(f"Peak Memory: {peak_memory_kb / 1024**2:.2f} GB")


class MemoryMonitorTqdm(tqdm):
  """Progress bar displaying memory usage."""

  def format_meter(self, n, total, elapsed, postfix=None, **extra_kwargs):
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    memory_percent = memory.percent
    memory_info = f"RAM: {used_gb:.1f}/{total_gb:.1f}GB ({memory_percent:.1f}%)"
    if postfix:
      if isinstance(postfix, dict):
        postfix["memory"] = memory_info
      else:
        postfix = f"{postfix}, {memory_info}"
    else:
      postfix = memory_info
    return super().format_meter(n=n, total=total, elapsed=elapsed, postfix=postfix, **extra_kwargs)
