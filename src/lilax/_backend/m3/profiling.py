"""M3 profiling loop."""

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.m3.models.glm5_3_flash.glm5_train import (
    configure_glm5_engine,
    create_glm5_frozen_mask,
    create_glm5_optimizer,
    glm5_engine_loss_fn,
    train_step as glm5_train_step,
)
from lilax._backend.m3.train import loop
from lilax._backend.configs.inert import MaxTextConfig


def profile_training(config: MaxTextConfig) -> float:
  """Run the M3 profiling loop with synthetic token batches and return its last loss."""
  return loop.train_loop(resolve_runtime_config(config))
