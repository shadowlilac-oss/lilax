from typing import override

import jax
import jax.numpy as jnp

from lilax._backend.trainers.post_train.hooks import BaseDataHooks, BaseTrainingHooks


class DPOTrainingHooks(BaseTrainingHooks):
  @override
  def eval_loss_is_preaveraged(self) -> bool:
    return True

  @override
  def get_total_weights(self, batch, *, is_training=True) -> jax.Array:
    return jnp.sum(batch["chosen_mask"] != 0) + jnp.sum(batch["rejected_mask"] != 0)


class DPODataHooks(BaseDataHooks):
  pass
