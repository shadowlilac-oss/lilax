"""Run Hugging Face checkpoint conversion in an isolated CPU interpreter."""

import base64
import json
import os
import pickle
import subprocess
import sys

import jax
import numpy as np
from huggingface_hub import get_token
from jax.experimental import multihost_utils

from lilax.configuration import MaxTextConfig, RLConfig, serialize_config


def convert_on_cpu(
    public_config, snapshot_path, output_root, *, lazy_load_tensors=True,
    eager_load_method="safetensors", revision=None, save_dtype="bfloat16", simulated_cpu_devices_count=16,
    scan_layers=None, conversion_mode="automatic",
):
  """Convert on process zero and broadcast the checkpoint path or failure to all JAX processes."""
  is_source = jax.process_index() == 0
  result_bytes = b""
  if is_source:
    try:
      native_config_data = None
      if not isinstance(public_config, (MaxTextConfig, RLConfig)):
        native_config = public_config
        credential = native_config.hf_access_token or get_token()
        if not credential:
          raise ValueError("hf_access_token or Hugging Face authentication is required for checkpoint conversion.")
        native_config_data = base64.b64encode(
            pickle.dumps((native_config._pydantic_config, native_config.get_keys()))
        ).decode("ascii")
      payload = {
          "config": serialize_config(public_config) if native_config_data is None else None,
          "native_config": native_config_data,
          "snapshot_path": os.fspath(snapshot_path) if snapshot_path is not None else None,
          "output_root": os.fspath(output_root),
          "lazy_load_tensors": lazy_load_tensors,
          "eager_load_method": eager_load_method,
          "revision": revision,
          "save_dtype": save_dtype,
          "simulated_cpu_devices_count": simulated_cpu_devices_count,
          "scan_layers": scan_layers,
          "conversion_mode": conversion_mode,
      }
      environment = os.environ.copy()
      if native_config_data is not None:
        environment["HF_TOKEN"] = credential
      environment["JAX_PLATFORMS"] = "cpu"
      environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
      child = subprocess.run(
          [sys.executable, "-c", (
              "import os; result_fd = os.dup(1); os.dup2(2, 1); "
              "from lilax._backend.checkpoint_conversion.cpu_worker import run; run(result_fd)"
          )],
          input=json.dumps(payload), text=True, stdout=subprocess.PIPE, env=environment, check=False,
      )
      if child.returncode:
        result = {"error": f"CPU conversion subprocess exited with code {child.returncode}"}
      else:
        result = json.loads(child.stdout)
    except Exception as error:
      result = {"error": f"{type(error).__name__}: {error}"}
    result_bytes = json.dumps(result).encode("utf-8")

  result_size = int(multihost_utils.broadcast_one_to_all(np.array(len(result_bytes), dtype=np.int32), is_source=is_source))
  result_buffer = np.frombuffer(result_bytes, dtype=np.uint8) if is_source else np.zeros(result_size, dtype=np.uint8)
  result = json.loads(multihost_utils.broadcast_one_to_all(result_buffer, is_source=is_source).tobytes())
  if "error" in result:
    raise RuntimeError(f"Checkpoint conversion failed: {result['error']}")
  multihost_utils.sync_global_devices("from_pretrained_convert_checkpoint")
  return result["checkpoint_path"]
