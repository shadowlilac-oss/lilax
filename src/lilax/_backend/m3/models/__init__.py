"""Model constructors, FLOPs estimators, and engine configuration for m3."""

from collections.abc import Callable

from lilax._backend.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from lilax._backend.m3.models.qwen3_8_flash_next import qwen3_8_flops
from lilax._backend.m3.models.qwen3_8_flash_next.modeling_qwen3_8_flash_next import create_qwen3_8_flash_next_model
from lilax._backend.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model


def create_glm5_3_flash_model(config, mesh=None, *, rngs, **kwargs):
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  from lilax._backend.m3.models.glm5_3_flash import modeling_glm5_3_flash

  return modeling_glm5_3_flash.create_glm5_3_flash_model(config, mesh=mesh, rngs=rngs, **kwargs)


MODEL_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": create_glm5_3_flash_model,
    "qwen3-0.6b": create_qwen3_model,
    "qwen3-vl-2b": create_qwen3_vl_model,
    "qwen3.8-flash-next": create_qwen3_8_flash_next_model,
    "qwen3.8-flash-next-mini": create_qwen3_8_flash_next_model,
}


def create_model(config, mesh, **kwargs):
  if config.model_name not in MODEL_REGISTRY:
    raise ValueError(
        f"Model {config.model_name!r} does not support the m3 backend (use_m3_model=True). "
        f"Registered models: {', '.join(sorted(MODEL_REGISTRY))}."
    )
  return MODEL_REGISTRY[config.model_name](config, mesh, **kwargs)


def calculate_glm5_3_flash_tflops_per_device(config, log: bool = True):
  from lilax._backend.m3.models.glm5_3_flash import glm5_flops

  return glm5_flops.calculate_glm5_3_flash_tflops_per_device(config, log=log)


MODEL_FLOPS_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": calculate_glm5_3_flash_tflops_per_device,
    "qwen3.8-flash-next": qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device,
    "qwen3.8-flash-next-mini": qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device,
}


def calculate_tflops_per_device(config, log: bool = True):
  if config.model_name not in MODEL_FLOPS_REGISTRY:
    raise ValueError(
        "TFLOPs calculation not implemented for m3 model"
        f" {config.model_name!r}. Registered models:"
        f" {', '.join(sorted(MODEL_FLOPS_REGISTRY))}."
    )
  return MODEL_FLOPS_REGISTRY[config.model_name](config, log=log)


def configure_glm5_3_flash_engine(engine):
  from lilax._backend.m3.models.glm5_3_flash import glm5_train

  return glm5_train.configure_glm5_engine(engine)


MODEL_ENGINE_CONFIG_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": configure_glm5_3_flash_engine,
}


def configure_engine(engine):
  model_name = engine._config.model_name
  if model_name in MODEL_ENGINE_CONFIG_REGISTRY:
    return MODEL_ENGINE_CONFIG_REGISTRY[model_name](engine)
  return engine
