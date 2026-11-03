"""Select benchmark models and construct healthscan workloads."""

import time
from copy import deepcopy

from lilax._backend.benchmarks.catalog import AVAILABLE_MODELS, C4_MODELS
from lilax._backend.benchmarks.cluster_runner import MaxTextModel, WorkloadConfig


def select_model(name: str) -> MaxTextModel:
  """Select a named benchmark model using the original catalog precedence."""
  for device_type in ("v6e", "v5litepod", "v5p"):
    if name in AVAILABLE_MODELS[device_type]:
      return deepcopy(AVAILABLE_MODELS[device_type][name])
  if name in C4_MODELS:
    return deepcopy(C4_MODELS[name])
  raise KeyError(name)


def healthscan_workload(
    device_type: str, base_output_directory: str, num_steps: int, model_selector, workload_type
) -> WorkloadConfig:
  """Construct the source healthscan workload for a supported TPU device type."""
  models = {
      "v5p-128": "llama2_70b_v5p_128", "v5p-256": "llama4_scout_dropless_v5p_256",
      "v5p-512": "deepseek_v3_ep_256_v5p_512", "v5p-1024": "deepseek_v3_ep_256_v5p_512",
      "v5p-2048": "deepseek_v3_ep_256_v5p_512", "v6e-8": "llama2_7b_4096",
      "v6e-16": "llama2_7b_4096", "v6e-32": "llama2_7b_4096", "v6e-64": "llama2_7b_4096",
      "v6e-128": "llama2_70b_4096", "v6e-256": "llama2_70b_4096",
  }
  model = model_selector(models[device_type])
  model.tuning_params["gcs_metrics"] = False
  return workload_type(
      model=model, base_output_directory=base_output_directory, device_type=device_type, num_steps=num_steps,
      run_name=f"{time.strftime('%Y%m%d')}-health-test",
  )
