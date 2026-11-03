from typing import override

import jax
import jax.numpy as jnp

from lilax._backend.trainers.post_train.hooks import BaseTrainingHooks, BaseDataHooks


class SFTTrainingHooks(BaseTrainingHooks):
  @override
  def get_total_weights(self, batch) -> jax.Array:
    loss_weights = batch.get("targets_loss_mask")
    if loss_weights is None:
      loss_weights = batch["targets_segmentation"]
    return jnp.sum(loss_weights != 0)

  @override
  def eval_loss_is_preaveraged(self) -> bool:
    return getattr(self.config, "training_objective", "causal_lm") == "block_diffusion"


class SFTDataHooks(BaseDataHooks):
  pass
