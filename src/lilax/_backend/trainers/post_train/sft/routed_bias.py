"""Optimizer-window accumulation of ordinary MaxText routed-bias counts."""

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp

from lilax._backend.layers.moe import GateLogit, load_balance_updates_from_counts
from lilax._backend.trainers.pre_train.moe.bias import (
    _find_gate_bias, apply_routed_bias_updates, collect_moe_bias_updates,
)


class RoutedBiasAccumulator(nnx.Module):
  def __init__(self, model):
    self.counts = nnx.Dict({
        "/".join(map(str, path)): nnx.Variable(jnp.zeros_like(
            jnp.moveaxis(node.bias.value, node.bias.get_metadata().get("param_scan_axis", 0), 0), dtype=jnp.int32
        ))
        for path, node in nnx.iter_graph(model) if isinstance(node, GateLogit) and node.bias is not None
    })

  def add(self, model, aux):
    buffers = {
        id(node.bias): self.counts["/".join(map(str, path))]
        for path, node in nnx.iter_graph(model) if isinstance(node, GateLogit) and node.bias is not None
    }
    intermediates = traverse_util.flatten_dict(aux["intermediate_outputs"])
    for path, update in intermediates.items():
      if path[-1] != "moe_bias_updates":
        continue
      target = model
      prefix = path[1:-1] if path[0] == "intermediates" else path[:-1]
      for key in prefix:
        target = target[key] if isinstance(target, dict) else getattr(target, key)
      bias = _find_gate_bias(target)
      if bias is None:
        continue
      counts = jnp.asarray(update[0] if isinstance(update, (tuple, list)) else update)
      buffer = buffers[id(bias)]
      buffer.value = buffer.value + counts
      intermediates[path] = (buffer.value,) if isinstance(update, (tuple, list)) else buffer.value
    return dict(aux, intermediate_outputs=traverse_util.unflatten_dict(intermediates))

  def apply(self, model, aux, config):
    def finalize_counts(value):
      if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.integer):
        return load_balance_updates_from_counts(value, config.num_experts, config.routed_bias_update_rate)
      return value

    aux = dict(aux)
    if not config.using_pipeline_parallelism:
      intermediates = traverse_util.flatten_dict(aux["intermediate_outputs"])
      for path, update in intermediates.items():
        if path[-1] == "moe_bias_updates":
          intermediates[path] = jax.tree.map(finalize_counts, update)
      aux["intermediate_outputs"] = traverse_util.unflatten_dict(intermediates)
    aux["moe_bias_updates"], aux["mtp_moe_bias_updates"] = collect_moe_bias_updates(config, aux["intermediate_outputs"])
    apply_routed_bias_updates(model, aux, config)
    for buffer in self.counts.values():
      buffer.value = jnp.zeros_like(buffer.value)
