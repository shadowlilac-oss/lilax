"""Auxiliary-loss-free MoE routing bias updates for the native pretraining step."""

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp

from lilax._backend.layers.moe import load_balance_updates_from_counts
from lilax._backend.observability import max_logging


def collect_moe_bias_updates(config, intermediate_outputs):
  moe_bias_updates = None
  mtp_moe_bias_updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    for path, value in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
      keys = tuple(key.key for key in path if hasattr(key, "key"))
      if not keys or keys[-1] != "moe_bias_updates":
        continue
      if "decoder" in keys:
        moe_bias_updates = (value,)
      elif "mtp_block" in keys:
        if mtp_moe_bias_updates is None:
          mtp_moe_bias_updates = []
        mtp_moe_bias_updates.append(value)
  return moe_bias_updates, mtp_moe_bias_updates


def _find_gate_bias(module: nnx.Module | None) -> nnx.Variable | None:
  if module is None:
    return None
  for _, node in nnx.iter_graph(module):
    if type(node).__name__ == "GateLogit" and hasattr(node, "bias") and node.bias is not None:
      return node.bias
  return None


def _decode_moe_bias_update(update, config):
  update = jnp.asarray(update[0] if isinstance(update, (tuple, list)) else update)
  if not getattr(config, "using_pipeline_parallelism", False):
    return update
  counts = update.astype(jnp.float32)
  return jnp.sign(jnp.mean(counts, axis=-1, keepdims=True) - counts) * config.routed_bias_update_rate


def _apply_moe_bias_updates_by_path(root: nnx.Module, intermediates: dict, config):
  diag_bias_values, diag_bias_updates = [], []
  for path, update in traverse_util.flatten_dict(intermediates).items():
    if path[-1] != "moe_bias_updates":
      continue
    target = root
    for key in path[:-1]:
      target = getattr(target, key)
    bias = _find_gate_bias(target)
    if bias is None:
      continue
    update = _decode_moe_bias_update(update, config)
    if jnp.issubdtype(update.dtype, jnp.integer):
      update = load_balance_updates_from_counts(update, config.num_experts, config.routed_bias_update_rate)
    scan_axis = bias.get_metadata().get("param_scan_axis", 0)
    if scan_axis != 0 and not getattr(config, "using_pipeline_parallelism", False):
      update = jnp.moveaxis(update, 0, scan_axis)
    if update.shape != bias.value.shape:
      raise ValueError(
          f"moe_bias_updates at {'/'.join(map(str, path))} has shape {update.shape}, "
          f"but the gate bias it updates has shape {bias.value.shape}."
      )
    if getattr(config, "using_pipeline_parallelism", False):
      update = update.astype(bias.value.dtype)
    bias.value = bias.value + update
    diag_bias_values.append(bias.value)
    diag_bias_updates.append(update)
  return diag_bias_values, diag_bias_updates


def _apply_deepseek4_bias_updates(model, intermediate_outputs, config):
  bias_metrics, diag_bias_values, diag_bias_updates = {}, [], []
  max_logging.log("DeepSeek V4: Applying auxiliary-loss-free routing bias via pure NNX MoEBiasVar.")
  flat_intermediates = traverse_util.flatten_dict(intermediate_outputs)
  for path, update in flat_intermediates.items():
    if path[-1] != "moe_bias_updates":
      continue
    target = model
    prefix = path[1:-1] if path[0] == "intermediates" else path[:-1]
    for key in prefix:
      if hasattr(target, key):
        target = getattr(target, key)
      elif isinstance(target, dict) and key in target:
        target = target[key]
      else:
        target = None
        break
    if target is None:
      continue
    for _, node in nnx.iter_graph(target):
      if type(node).__name__ == "GateLogit" and hasattr(node, "bias") and node.bias is not None:
        update_val = _decode_moe_bias_update(update, config)
        name_prefix = "-".join(map(str, prefix))
        if getattr(config, "log_moe_bias_norms", False):
          bias_metrics[f"learning/moe_bias_before_norm_{name_prefix}"] = jnp.linalg.norm(node.bias.value)
        node.bias.value = node.bias.value + jnp.array(update_val)
        diag_bias_values.append(node.bias.value)
        diag_bias_updates.append(update_val)
        if getattr(config, "log_moe_bias_norms", False):
          bias_metrics[f"learning/moe_bias_update_norm_{name_prefix}"] = jnp.linalg.norm(jnp.array(update_val))
  return bias_metrics, diag_bias_values, diag_bias_updates


def _apply_decoder_and_mtp_bias_updates(model, aux, config):
  diag_bias_values, diag_bias_updates = [], []
  mtp_moe_bias_updates = aux.get("mtp_moe_bias_updates")
  if getattr(config, "using_pipeline_parallelism", False):
    _apply_moe_bias_updates_by_path(model.decoder, aux["intermediate_outputs"].get("decoder", {}), config)
  else:
    diag_bias_values, diag_bias_updates = _apply_moe_bias_updates_by_path(
        model.decoder, aux["intermediate_outputs"].get("decoder", {}), config
    )
  if mtp_moe_bias_updates is not None and hasattr(model, "mtp_block"):
    for index, update in enumerate(mtp_moe_bias_updates):
      mtp_layer = getattr(model.mtp_block, f"mtp_layer_{index + 1}", None)
      mtp_bias = _find_gate_bias(mtp_layer)
      if mtp_bias is not None:
        mtp_update = _decode_moe_bias_update(update, config)
        mtp_bias.value = mtp_bias.value + mtp_update
        diag_bias_values.append(mtp_bias.value)
        diag_bias_updates.append(mtp_update)
  return {}, diag_bias_values, diag_bias_updates


def apply_routed_bias_updates(model, aux, config):
  if getattr(config, "model_name", "").startswith("deepseek4"):
    return _apply_deepseek4_bias_updates(model, aux.get("intermediate_outputs", {}), config)
  return _apply_decoder_and_mtp_bias_updates(model, aux, config)
