"""Utilities for QK-Clip (Muon Clip)."""

import jax
import jax.numpy as jnp
from flax import nnx


def _get_key_name(k):
  if hasattr(k, "key"):
    return k.key
  if hasattr(k, "idx"):
    return k.idx
  return k


def calculate_max_logit_metric(intermediate_outputs):
  all_max_logits = []

  def extract_logits(path, val):
    if not path:
      return
    last_key = _get_key_name(path[-1])
    parent_key = _get_key_name(path[-2]) if len(path) >= 2 else None
    if last_key == "max_logits" or parent_key == "max_logits":
      all_max_logits.append(val)

  jax.tree_util.tree_map_with_path(extract_logits, intermediate_outputs)
  if not all_max_logits:
    return None
  return jnp.max(jnp.stack([jnp.max(x) for x in all_max_logits]))


def _check_attention_type(config):
  if getattr(config, "attention_type", None) != "mla":
    raise ValueError(
        f"QK-Clip is only supported for MLA attention (attention_type='mla'). "
        f"Current configuration: {getattr(config, 'attention_type', 'None')}"
    )


def _max_logits_at(curr):
  if not isinstance(curr, dict):
    return None
  ml = curr.get("max_logits")
  if ml is None and "attention_op" in curr and isinstance(curr["attention_op"], dict):
    ml = curr["attention_op"].get("max_logits")
  if ml is None:
    return None
  if isinstance(ml, (tuple, list)):
    return ml[0] if ml else None
  return ml


def _scale_from_max_logits(max_logits_batch, tau):
  axes = tuple(range(max_logits_batch.ndim - 1))
  s_max = jnp.max(max_logits_batch, axis=axes)
  return jnp.minimum(1.0, tau / (s_max + 1e-6))


def _clip_mla_weight(layer_name, param, scale, qk_nope):
  scale_b = jnp.expand_dims(scale, axis=-1)
  head = param[..., :qk_nope]
  tail = param[..., qk_nope:]
  head_new = head * jnp.sqrt(scale_b)
  if layer_name == "wq_b":
    tail_new = tail * scale_b
  else:
    tail_new = tail
  return jnp.concatenate([head_new, tail_new], axis=-1)


def apply_qk_clip(state, intermediate_outputs, config):
  _check_attention_type(config)
  tau = float(config.qk_clip_threshold)

  def clip_mla_weights(path, param):
    if len(path) < 2:
      return param
    layer_name = _get_key_name(path[-2])
    if layer_name not in ("wq_b", "wkv_b"):
      return param
    curr = intermediate_outputs.get("intermediates", intermediate_outputs)
    for node in path[:-2]:
      key = _get_key_name(node)
      if isinstance(curr, dict) and key in curr:
        curr = curr[key]
      else:
        return param
    max_logits_batch = _max_logits_at(curr)
    if max_logits_batch is None:
      return param
    scale = _scale_from_max_logits(max_logits_batch, tau)
    return _clip_mla_weight(layer_name, param, scale, config.qk_nope_head_dim)

  new_params = jax.tree_util.tree_map_with_path(clip_mla_weights, state.params)
  return state.replace(params=new_params)


def apply_qk_clip_nnx(state, intermediate_outputs, config):
  _check_attention_type(config)
  tau = float(config.qk_clip_threshold)
  _, params_state, _ = nnx.split(state.model, nnx.Param, ...)
  params_dict = params_state.to_pure_dict()

  def clip_mla_weights(path, param):
    if len(path) < 2:
      return param
    layer_name = _get_key_name(path[-2])
    if layer_name not in ("wq_b", "wkv_b"):
      return param
    curr = intermediate_outputs
    for node in path[:-2]:
      key = _get_key_name(node)
      if isinstance(curr, dict) and key in curr:
        curr = curr[key]
      else:
        return param
    max_logits_batch = _max_logits_at(curr)
    if max_logits_batch is None:
      return param
    scale = _scale_from_max_logits(max_logits_batch, tau)
    return _clip_mla_weight(layer_name, param, scale, config.qk_nope_head_dim)

  new_params_dict = jax.tree_util.tree_map_with_path(clip_mla_weights, params_dict)
  nnx.replace_by_pure_dict(params_state, new_params_dict)
  nnx.update(state.model, params_state)
  return state
