"""Hugging Face Qwen3.8 text-only weight shapes for checkpoint conversion."""

from lilax._backend.checkpoint_conversion.model_families.qwen3_5.hf_shape import QWEN3_5_HF_WEIGHTS_TO_SHAPE


def QWEN3_8_HF_WEIGHTS_TO_SHAPE(config):
  """Returns HF weight shapes for the text-only Qwen3.8 checkpoint."""
  return QWEN3_5_HF_WEIGHTS_TO_SHAPE(config, hf_prefix="model")
