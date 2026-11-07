"""Child-side CPU checkpoint operation callable."""

import json
import os
import sys

import jax

from lilax._backend.checkpoint_conversion.compare_hf_ckpt import _compare_hf_ckpt
from lilax._backend.checkpoint_conversion.inspect_checkpoint import _inspect_maxtext
from lilax._backend.checkpoint_conversion.reshard_checkpoint import _reshard_checkpoint
from lilax._backend.checkpoint_conversion.standalone_scripts.convert_qwen3_next_unscanned import _convert
from lilax._backend.configs import pyconfig, types
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax.configuration.serialization import deserialize_config


def run(result_fd):
  try:
    payload = json.load(sys.stdin)
    operation = payload.pop("operation")
    config_payload = payload.pop("config")
    simulated_cpu_devices_count = payload.pop("simulated_cpu_devices_count")
    if operation == "inspect_maxtext":
      overrides = {"attention": "dot_product", "skip_jax_distributed_system": True}
      config_payload["values"].update(overrides)
      config_payload["explicit_keys"] = sorted(set(config_payload["explicit_keys"]) | overrides.keys())
    if operation == "reshard_checkpoint":
      jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    if config_payload is None:
      config = None
    elif config_payload["config_type"] in ("native_maxtext", "native_rl"):
      config_class = types.RLConfig if config_payload["config_type"] == "native_rl" else types.MaxTextConfig
      config = pyconfig._resolve_native_config(
          config_payload["values"], set(config_payload["explicit_keys"]), config_class, resolved_values=True
      )
    else:
      config = resolve_runtime_config(deserialize_config(config_payload))
    if operation == "inspect_maxtext":
      value = _inspect_maxtext(config, **payload)
    elif operation == "compare_hf_ckpt":
      value = _compare_hf_ckpt(config=config, **payload)
    elif operation == "reshard_checkpoint":
      value = _reshard_checkpoint(config)
    elif operation == "convert_qwen3_next_unscanned":
      value = _convert(simulated_cpu_devices_count=simulated_cpu_devices_count, **payload)
    else:
      raise ValueError(f"Unknown CPU checkpoint operation: {operation}")
    result = {"value": value}
  except Exception as error:
    result = {"error": f"{type(error).__name__}: {error}"}
  with os.fdopen(result_fd, "w", encoding="utf-8") as result_file:
    json.dump(result, result_file)
