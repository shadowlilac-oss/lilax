"""Calculate training TFLOPs per device."""

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.common.training_flops import calculate_tflops_training_per_device


def get_flops(config):
  """Return total training TFLOPs per device per step for the given config."""
  tflops, _, _ = calculate_tflops_training_per_device(resolve_runtime_config(config), log=False)
  return tflops
