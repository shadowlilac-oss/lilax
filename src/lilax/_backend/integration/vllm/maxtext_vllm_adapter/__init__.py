"""MaxText vLLM adapter package."""

import os

from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model

from .adapter import MaxTextForCausalLM, patch_kv_cache_manager
from .multimodal import get_multimodal_handler


logger = init_logger(__name__)


def register(config=None):
  """Register the MaxText model with tpu-inference and vLLM."""
  logger.info("Registering MaxTextForCausalLM model with tpu_inference and vllm.")
  model_name = os.environ.get("MAXTEXT_MODEL_NAME")
  if config is not None:
    model_name = config.model_name
    os.environ["MAXTEXT_MODEL_NAME"] = model_name
  multimodal_handler = get_multimodal_handler(model_name)
  MaxTextForCausalLM.supports_multimodal = multimodal_handler is not None
  if multimodal_handler is not None:
    logger.info("Setting supports_multimodal = True for %s", model_name)
  register_model("MaxTextForCausalLM", MaxTextForCausalLM)
  patch_kv_cache_manager()
  if multimodal_handler is not None:
    multimodal_handler.register_processor(MaxTextForCausalLM)
  logger.info("Successfully registered MaxTextForCausalLM model.")
