"""Child-side CPU checkpoint conversion callable."""

import json
import os
import sys

import jax

from lilax._backend.checkpoint_conversion.to_maxtext import convert_to_maxtext
from lilax._backend.configs import pyconfig
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax.configuration.serialization import deserialize_config
from lilax.configuration.conversion import project_cpu_conversion_config


def run(result_fd):
  """Read a serialized conversion request and write its result to the supplied descriptor."""
  try:
    payload = json.load(sys.stdin)
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    if payload["native_config"] is not None:
      config = pyconfig.initialize(
          [""], **payload["native_config"], base_output_directory=payload["output_root"], use_multimodal=False,
          skip_jax_distributed_system=True,
      )
    else:
      public_config = deserialize_config(payload["config"])
      public_config = project_cpu_conversion_config(
          public_config, output_directory=payload["output_root"], conversion_mode=payload["conversion_mode"],
      )
      if payload["scan_layers"] is not None:
        public_config = public_config.model_copy(update={"scan_layers": payload["scan_layers"]})
      config = resolve_runtime_config(public_config)
    checkpoint_path = convert_to_maxtext(
        config, lazy_load_tensors=payload["lazy_load_tensors"], eager_load_method=payload["eager_load_method"],
        hf_model_path=payload["snapshot_path"], revision=payload["revision"], save_dtype=payload["save_dtype"],
        simulated_cpu_devices_count=payload["simulated_cpu_devices_count"],
    )
    result = {"checkpoint_path": os.fspath(checkpoint_path)}
  except Exception as error:
    result = {"error": f"{type(error).__name__}: {error}"}
  with os.fdopen(result_fd, "w", encoding="utf-8") as result_file:
    json.dump(result, result_file)
