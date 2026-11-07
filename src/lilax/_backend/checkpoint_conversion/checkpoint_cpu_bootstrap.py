"""Run checkpoint operations in a fresh CPU interpreter."""

import json
import os
import subprocess
import sys

from lilax.configuration.serialization import serialize_config
from lilax._backend.configs import pyconfig, types


def run_on_cpu(operation, config=None, simulated_cpu_devices_count=None, **options):
  if isinstance(config, pyconfig.HyperParameters):
    native = config._pydantic_config
    config_payload = {
        "config_type": "native_rl" if isinstance(native, types.RLConfig) else "native_maxtext",
        "values": native.model_dump(mode="json"),
        "explicit_keys": sorted(native.model_fields_set),
    }
  else:
    config_payload = serialize_config(config) if config is not None else None
  payload = {
      "operation": operation, "config": config_payload, "simulated_cpu_devices_count": simulated_cpu_devices_count,
      **options,
  }
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  if simulated_cpu_devices_count is not None:
    environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
  child = subprocess.run(
      [sys.executable, "-c", (
          "import os; result_fd = os.dup(1); os.dup2(2, 1); "
          "from lilax._backend.checkpoint_conversion.checkpoint_cpu_worker import run; run(result_fd)"
      )],
      input=json.dumps(payload), text=True, stdout=subprocess.PIPE, env=environment, check=False,
  )
  if child.returncode:
    raise RuntimeError(f"CPU checkpoint subprocess exited with code {child.returncode}")
  result = json.loads(child.stdout)
  if "error" in result:
    raise RuntimeError(f"CPU checkpoint operation failed: {result['error']}")
  return result["value"]
