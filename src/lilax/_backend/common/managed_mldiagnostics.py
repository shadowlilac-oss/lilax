"""Managed ML Diagnostics run lifecycle management."""

import json
from typing import Any

import google_cloud_mldiagnostics as mldiag

from lilax._backend.common.cloud import is_decoupled
from lilax._backend.configs.pyconfig import KEYS_NO_LOGGING


def mldiagnostics_available() -> bool:
  return not is_decoupled()


def _config_items(config: Any) -> dict[str, Any]:
  if hasattr(config, "get_keys"):
    return config.get_keys()
  return config.model_dump(mode="json")


class ManagedMLDiagnostics:
  """Centralized ML Diagnostics run manager."""

  _instance = None

  def __new__(cls, *args: Any, **kwargs: Any):
    if cls._instance is None:
      cls._instance = super(ManagedMLDiagnostics, cls).__new__(cls)
    return cls._instance

  def __init__(self, config: Any, sampler_config: Any | None = None):
    if hasattr(self, "_initialized"):
      return
    self._initialized = True
    if not config.managed_mldiagnostics or not mldiagnostics_available():
      return

    def should_log_key(key: str, value: Any) -> bool:
      if key in KEYS_NO_LOGGING:
        return False
      try:
        json.dumps(value, allow_nan=False)
      except (TypeError, ValueError):
        return False
      return True

    config_dict = {key: value for key, value in _config_items(config).items() if should_log_key(key, value)}
    if sampler_config is not None and sampler_config is not config:
      for key, value in _config_items(sampler_config).items():
        if should_log_key(key, value) and config_dict.get(key) != value:
          config_dict[f"sampler.{key}"] = value

    region = config.managed_mldiagnostics_region if config.managed_mldiagnostics_region else None
    mldiag.machinelearning_run(
        name=f"{config.run_name}",
        run_group=config.managed_mldiagnostics_run_group,
        configs=config_dict,
        gcs_path=config.managed_mldiagnostics_dir,
        on_demand_xprof=config.managed_mldiagnostics_on_demand_profiling,
        region=region,
        metric_only_run=config.managed_mldiagnostics_metric_only_run,
        application_framework="MAXTEXT",
    )
