"""Training and optimizer helpers for GLM-5.3-Flash."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp
from lilax._backend.m3.models.glm5_3_flash import modeling_glm5_3_flash
import optax

if TYPE_CHECKING:
  from lilax._backend.training_engine import abstract_engine

_GLM5_LOSS_BATCH_KEYS = (
    "inputs", "targets", "targets_segmentation", "targets_position", "inputs_segmentation", "inputs_position",
)


def _is_router_bias(v: Any) -> bool:
  return isinstance(v, modeling_glm5_3_flash.Glm5RouterBias)


def _get_param_label(path: tuple[Any, ...], v: Any) -> str:
  if _is_router_bias(v):
    return "frozen"
  path_str = jax.tree_util.keystr(path)
  if "indexer" in path_str or "e_score_correction_bias" in path_str:
    return "frozen"
  return "train"


def glm5_engine_loss_fn(model: modeling_glm5_3_flash.Glm5Model, config: Any, data: Mapping[str, Any],
                        dropout_rng: Any = None, params: Any = None, *, is_train: bool
                        ) -> tuple[abstract_engine.WeightedMetric, dict[str, jax.Array]]:
  from lilax._backend.training_engine import abstract_engine

  del config, dropout_rng, params
  filtered_batch = {key: data[key] for key in _GLM5_LOSS_BATCH_KEYS if key in data}
  total_loss, aux = modeling_glm5_3_flash.glm5_loss_fn(model, is_train=is_train, **filtered_batch)
  return (abstract_engine.WeightedMetric(unreduced_sum=(total_loss * aux["total_weights"]).astype(jnp.float32),
                                         denominator=aux["total_weights"]), aux)


def create_glm5_frozen_mask(params: Any) -> Any:
  return jax.tree_util.tree_map(lambda is_train: not is_train, modeling_glm5_3_flash.trainable_mask(params))


def create_glm5_optimizer(*, learning_rate: float, weight_decay: float, **adamw_kwargs: Any
                          ) -> optax.GradientTransformation:
  if not isinstance(learning_rate, (float, int)):
    raise TypeError(f"learning_rate must be a float or int, got {type(learning_rate)}")
  if not isinstance(weight_decay, (float, int)):
    raise TypeError(f"weight_decay must be a float or int, got {type(weight_decay)}")
  base_tx = optax.adamw(learning_rate=float(learning_rate), weight_decay=float(weight_decay), **adamw_kwargs)
  return optax.multi_transform(
      {"train": base_tx, "frozen": optax.set_to_zero()},
      param_labels=lambda params: jax.tree_util.tree_map_with_path(
          _get_param_label, params, is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array))),
  )


def configure_glm5_engine(engine: Any) -> Any:
  engine.with_loss_fn(glm5_engine_loss_fn, has_aux=True)
  engine.with_gen_model_input_fn(None)
  base_tx = engine._optimizer.tx
  masked_tx = optax.multi_transform(
      {"train": base_tx, "frozen": optax.set_to_zero()},
      param_labels=lambda params: jax.tree_util.tree_map_with_path(
          _get_param_label, params, is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array))),
  )
  engine._optimizer = engine._build_optimizer(masked_tx)
  engine._tx = masked_tx
  return engine


def train_step(model: modeling_glm5_3_flash.Glm5Model, optimizer: nnx.Optimizer, tokens: jax.Array,
               decoder_target_tokens: jax.Array, decoder_target_mask: jax.Array,
               decoder_segment_ids: jax.Array | None = None, decoder_positions: jax.Array | None = None
               ) -> tuple[jax.Array, dict[str, jax.Array]]:
  def loss_fn(mod: modeling_glm5_3_flash.Glm5Model) -> tuple[jax.Array, dict[str, jax.Array]]:
    return modeling_glm5_3_flash.compute_glm5_loss(
        model=mod, tokens=tokens, decoder_target_tokens=decoder_target_tokens, decoder_target_mask=decoder_target_mask,
        decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions, is_train=True)

  (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
  optimizer.update(model, grads)
  return loss, aux
