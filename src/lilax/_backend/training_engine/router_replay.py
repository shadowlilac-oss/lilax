"""Router replay: adapting Tunix RL payloads to MaxText's loss with forced expert routing."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from lilax._backend.configs import pyconfig
from lilax._backend.trainers.pre_train.steps import loss as maxtext_loss
from lilax._backend.training_engine import abstract_engine
import numpy as np


def _row_and_mask(payload: abstract_engine.RLTrainerPayload) -> tuple[jax.Array, jax.Array]:
  parts = [jnp.asarray(payload.prompt_ids), jnp.asarray(payload.completion_ids)]
  masks = [
      jnp.ones_like(part) if mask is None else jnp.asarray(mask).astype(jnp.int32)
      for part, mask in zip(parts, (payload.prompt_mask, payload.completion_mask))
  ]
  return jnp.concatenate(parts, axis=-1), jnp.concatenate(masks, axis=-1).astype(jnp.int32)


def _aligned(field: Any, name: str, width: int) -> jax.Array | None:
  if field is None:
    return None
  array = jnp.asarray(field)
  if array.shape[-1] != width:
    raise ValueError(
        f"{name} has width {array.shape[-1]} but the model row is {width} wide. Tunix sets "
        f"{name} only on the packed path, where `prompt_ids` is zero-width and the two agree; "
        "a completion-width array cannot be aligned against a non-empty prompt."
    )
  return array


def router_replay_gen_model_input_fn(payload: abstract_engine.RLTrainerPayload) -> dict[str, Any]:
  token_ids, token_mask = _row_and_mask(payload)
  width = token_ids.shape[-1]
  segment_ids = _aligned(payload.segment_ids, "segment_ids", width)
  segment_positions = _aligned(payload.segment_positions, "segment_positions", width)
  if segment_ids is None:
    segment_ids = token_mask
  if segment_positions is not None:
    positions = segment_positions.astype(jnp.int32)
  else:
    real = (token_mask != 0).astype(jnp.int32)
    preceding = jnp.cumsum(real, axis=-1) - real
    starts = jnp.concatenate(
        [jnp.ones((segment_ids.shape[0], 1), dtype=bool), segment_ids[:, 1:] != segment_ids[:, :-1]], axis=-1
    )
    segment_start_count = jax.lax.cummax(jnp.where(starts, preceding, -1), axis=preceding.ndim - 1)
    positions = (preceding - segment_start_count).astype(jnp.int32)
  targets_segmentation = jnp.roll(token_mask, -1, axis=-1)
  targets_segmentation = targets_segmentation.at[:, -1].set(0)
  same_segment = segment_ids[:, :-1] == segment_ids[:, 1:]
  targets_segmentation = targets_segmentation.at[:, :-1].multiply(same_segment.astype(token_mask.dtype))
  kwargs = {
      "inputs": token_ids,
      "inputs_position": positions,
      "inputs_segmentation": segment_ids,
      "targets": jnp.roll(token_ids, -1, axis=-1),
      "targets_segmentation": targets_segmentation,
  }
  routed_experts = getattr(payload, "routed_experts", None)
  if routed_experts is not None:
    kwargs["forced_routed_experts"] = jnp.asarray(routed_experts)
  return kwargs


def expand_dummy_routed_experts(config: pyconfig.HyperParameters, data: Any) -> Any:
  """Returns `data` with an empty `routed_experts` expanded to the `[batch, seq, layers, top_k]` shape compile needs."""
  if not isinstance(data, abstract_engine.RLTrainerPayload):
    return data
  routed_experts = data.routed_experts
  if routed_experts is None or routed_experts.size != 0:
    return data
  num_layers = config.num_decoder_layers
  top_k = config.num_experts_per_tok
  batch_size, seq_len = routed_experts.shape[:2]
  expanded = np.zeros((batch_size, seq_len, num_layers, top_k), dtype=routed_experts.dtype)
  return dataclasses.replace(data, routed_experts=expanded)


def make_router_replay_loss_fn(
    config: pyconfig.HyperParameters, dropout_rng: jax.Array | None = None
) -> Callable[..., Any]:
  if config.enable_dropout and dropout_rng is None:
    raise ValueError(
        "make_router_replay_loss_fn requires a dropout_rng when"
        " config.enable_dropout is set; pass one, or set enable_dropout=False"
        " for router replay."
    )

  def router_replay_loss_fn(
      model, inputs, inputs_position, inputs_segmentation, targets, targets_segmentation, forced_routed_experts=None
  ):
    data = {
        "inputs": inputs,
        "inputs_position": inputs_position,
        "inputs_segmentation": inputs_segmentation,
        "targets": targets,
        "targets_segmentation": targets_segmentation,
    }
    if forced_routed_experts is not None:
      data["forced_routed_experts"] = forced_routed_experts
    return maxtext_loss.loss_fn(model, config, data, dropout_rng=dropout_rng, params=None, is_train=True)

  return router_replay_loss_fn
