from typing import override

import jax
import jax.numpy as jnp

from lilax._backend.trainers.post_train.hooks import BaseTrainingHooks, BaseDataHooks


class SFTTrainingHooks(BaseTrainingHooks):
  @override
  def get_total_weights(self, batch, *, is_training=True) -> jax.Array:
    loss_weights = batch.get("targets_loss_mask")
    if getattr(self.config, "training_objective", "causal_lm") != "block_diffusion":
      loss_weights = batch["targets_segmentation"]
      if is_training:
        batch_size = self.config.micro_batch_size_to_train_on
        if not self.config.use_tunix_gradient_accumulation:
          batch_size *= self.config.gradient_accumulation_steps
      else:
        batch_size = self.config.micro_batch_size_to_eval_on
      loss_weights = loss_weights[:batch_size]
    elif loss_weights is None:
      loss_weights = batch["targets_segmentation"]
    return jnp.sum(loss_weights != 0)

  @override
  def eval_loss_is_preaveraged(self) -> bool:
    return True


class SFTDataHooks(BaseDataHooks):
  pass
