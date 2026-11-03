"""Partition specs of the sparse MoE shard_map: activations, expert weights and outputs.

Layer-owned values (including `config`) are zero-arg callables and `lookup_*` return the layer's bound method; both
are resolved at the point of use, so each layer read happens where, and as often as, the layer performs it.
"""

import dataclasses
from typing import Any

from aqt.jax.v2 import aqt_tensor as aqt
import jax
from jax.sharding import PartitionSpec as P
from lilax._backend.distributed.sharding import (
    CompoundLogicalAxis,
    mesh_axes_for_dim,
    remove_incompatible_mesh_axes_from_partition_spec,
    remove_mesh_axes_from_partition_spec,
)
import qwix.pallas as qpl


def reduced_axes(w) -> frozenset:
  """Mesh axes a weight is typed `reduced` over (all-gathered by fsdp_prefetch); empty for non-JAX values.

  Expert kernels may be quantized containers (e.g. qwix QArrays) rather than arrays, which have no such type.
  """
  if not isinstance(w, jax.Array):  # Includes tracers.
    return frozenset()
  spec = getattr(getattr(jax.typeof(w), "sharding", None), "spec", None)
  return frozenset(spec.reduced) if spec is not None else frozenset()


def is_batch_sharded_by_ep(input_activation):
  # The batch is sharded by expert, except during inference decoding (where batch size == 1).
  # In the decoding case, the expert axis is instead replicated along the tensor's batch dimension.
  return input_activation.shape[0] > 1


def explicitly_weight_ag(config):
  if not (config().shard_exp_on_fsdp or config().shard_embed_moe_on_fsdp):
    return False
  quantization_rule = qpl.get_current_rule("gmm")
  if quantization_rule and quantization_rule.weight_calibration_method.startswith("fixed"):
    return True
  return False


@dataclasses.dataclass(frozen=True)
class RoutedMoEShardings:
  """Partition specs of the sparse MoE shard_map operands."""

  batch_logical_axis: str
  input: Any
  gate_logits: Any
  pre_bias_logits: Any
  w0: Any
  w1: Any
  wo: Any
  w0_bias: Any
  w1_bias: Any
  wo_bias: Any
  decoder_tokens: Any


def get_routed_moe_shardings(
    config, lookup_logical_to_mesh_axes, is_batch_sharded_by_expert, has_input_ids, wi_kernel_axes, wo_kernel_axes
):
  if is_batch_sharded_by_expert:
    batch_logical_axis = "activation_batch"
  else:
    batch_logical_axis = "decode_batch_moe"

  input_partition_pspec = lookup_logical_to_mesh_axes()((batch_logical_axis, "activation_norm_length", None))
  w0_bias_pspec = lookup_logical_to_mesh_axes()(("exp", "activation_mlp"))
  w1_bias_pspec = lookup_logical_to_mesh_axes()(("exp", "activation_mlp"))
  wo_bias_pspec = lookup_logical_to_mesh_axes()(("exp", "activation_embed"))

  gate_logits_pspec = lookup_logical_to_mesh_axes()((batch_logical_axis, "activation_norm_length", None))
  # NOTE: deepseek2 has a different pattern
  if config().model_name.startswith(("deepseek3", "deepseek4", "kimi-k2", "ming", "hy4")):
    pre_bias_logits_pspec = lookup_logical_to_mesh_axes()((batch_logical_axis, "activation_norm_length", None))
  else:
    # pre_bias_logits is None for non-deepseek3/4 models, including deepseek2
    pre_bias_logits_pspec = None

  if has_input_ids:
    decoder_tokens_pspec = lookup_logical_to_mesh_axes()((batch_logical_axis, "activation_norm_length"))
  else:
    decoder_tokens_pspec = None

  # w0, w1, wo needs to be un sharded on fsdp / fsdp_transpose axis, so use
  # mlp_no_fsdp axis
  if config().shard_exp_on_fsdp:
    if explicitly_weight_ag(config):
      # special sharding when using static scaling for weights in quantization with shard_exp_on_fsdp
      w0_pspec = lookup_logical_to_mesh_axes()(wi_kernel_axes())
      w1_pspec = lookup_logical_to_mesh_axes()(wi_kernel_axes())
      wo_pspec = lookup_logical_to_mesh_axes()(wo_kernel_axes())
    else:
      # special sharding for dsv3 to remove overhead between gmm/AG
      w0_pspec = lookup_logical_to_mesh_axes()((None, None, "mlp_no_fsdp"))
      w1_pspec = lookup_logical_to_mesh_axes()((None, None, "mlp_no_fsdp"))
      wo_pspec = lookup_logical_to_mesh_axes()((None, "mlp_no_fsdp", None))
  elif config().use_2d_fsdp_sharding:
    w0_pspec = lookup_logical_to_mesh_axes()((None, "mlp_no_fsdp", None))
    w1_pspec = lookup_logical_to_mesh_axes()((None, "mlp_no_fsdp", None))
    wo_pspec = lookup_logical_to_mesh_axes()((None, "mlp_no_fsdp", None))
  elif config().shard_embed_moe_on_fsdp and explicitly_weight_ag(config):
    # Keep embed_moe sharded so we can manually QAG it over FSDP
    if config().moe_flat_fsdp_weights:
      # Flat [E * rows, cols] weights: keep the (expert, fsdp) row sharding; gmm QAGs dim 0 over FSDP.
      w0_pspec = lookup_logical_to_mesh_axes()((CompoundLogicalAxis(("exp", "embed_moe")), "mlp_no_fsdp"))
      w1_pspec = lookup_logical_to_mesh_axes()((CompoundLogicalAxis(("exp", "embed_moe")), "mlp_no_fsdp"))
      wo_pspec = lookup_logical_to_mesh_axes()((CompoundLogicalAxis(("exp", "embed_moe")), None))
    else:
      w0_pspec = lookup_logical_to_mesh_axes()(("exp", "embed_moe", "mlp_no_fsdp"))
      w1_pspec = lookup_logical_to_mesh_axes()(("exp", "embed_moe", "mlp_no_fsdp"))
      wo_pspec = lookup_logical_to_mesh_axes()(("exp", "mlp_no_fsdp", "embed_moe"))
  else:
    # Tell XLA to automatically gather the D dimension (e.g. for dynamic absmax scaling)
    if config().moe_flat_fsdp_weights:
      w0_pspec = lookup_logical_to_mesh_axes()((CompoundLogicalAxis(("exp", "embed_moe")), "mlp_no_fsdp"))
      w1_pspec = lookup_logical_to_mesh_axes()((CompoundLogicalAxis(("exp", "embed_moe")), "mlp_no_fsdp"))
      wo_pspec = lookup_logical_to_mesh_axes()((CompoundLogicalAxis(("exp", "embed_moe")), None))
    else:
      w0_pspec = lookup_logical_to_mesh_axes()(("exp", None, "mlp_no_fsdp"))
      w1_pspec = lookup_logical_to_mesh_axes()(("exp", None, "mlp_no_fsdp"))
      wo_pspec = lookup_logical_to_mesh_axes()(("exp", "mlp_no_fsdp", None))
    # Update kernel pspec for FSDP AG
    w0_pspec = remove_mesh_axes_from_partition_spec(w0_pspec, ("fsdp",))
    w1_pspec = remove_mesh_axes_from_partition_spec(w1_pspec, ("fsdp",))
    wo_pspec = remove_mesh_axes_from_partition_spec(wo_pspec, ("fsdp",))
  return RoutedMoEShardings(
      batch_logical_axis=batch_logical_axis,
      input=input_partition_pspec,
      gate_logits=gate_logits_pspec,
      pre_bias_logits=pre_bias_logits_pspec,
      w0=w0_pspec,
      w1=w1_pspec,
      wo=wo_pspec,
      w0_bias=w0_bias_pspec,
      w1_bias=w1_bias_pspec,
      wo_bias=wo_bias_pspec,
      decoder_tokens=decoder_tokens_pspec,
  )


def with_aqt_partition(shardings, w0_kernel, w1_kernel, wo_kernel):
  w0_pspec, w1_pspec, wo_pspec = shardings.w0, shardings.w1, shardings.wo
  if isinstance(w0_kernel, aqt.QTensor):
    w0_pspec = aqt.partition_spec(w0_pspec, (1,), w0_kernel.dtype, use_bias=False)
  if isinstance(w1_kernel, aqt.QTensor):
    w1_pspec = aqt.partition_spec(w1_pspec, (1,), w1_kernel.dtype, use_bias=False)
  if isinstance(wo_kernel, aqt.QTensor):
    wo_pspec = aqt.partition_spec(wo_pspec, (1,), wo_kernel.dtype, use_bias=False)
  return dataclasses.replace(shardings, w0=w0_pspec, w1=w1_pspec, wo=wo_pspec)


def _maybe_replicate_incompatible_batch(pspec, value, allow_batch_replication, mesh):
  """Replicate an incompatible batch only when routing is not expert-parallel."""
  if not allow_batch_replication or pspec is None or value is None:
    return pspec
  return remove_incompatible_mesh_axes_from_partition_spec(
      pspec,
      value.shape,
      mesh(),
      dims=(0,),
      allow_remove_axes=True,
  )


def with_replicated_incompatible_batches(
    shardings, output_pspec, inputs, gate_logits, pre_bias_logits, input_ids, allow_batch_replication, mesh
):
  """Returns (shardings, output_pspec) with incompatible batch axes replicated, in input/gate/pre-bias/tokens/output
  order."""

  def replicate(pspec, value):
    return _maybe_replicate_incompatible_batch(pspec, value, allow_batch_replication, mesh)

  shardings = dataclasses.replace(
      shardings,
      input=replicate(shardings.input, inputs),
      gate_logits=replicate(shardings.gate_logits, gate_logits),
      pre_bias_logits=replicate(shardings.pre_bias_logits, pre_bias_logits),
      decoder_tokens=replicate(shardings.decoder_tokens, input_ids),
  )
  return shardings, replicate(output_pspec, inputs)


def _keep_reduced(pspec, w):
  reduced = reduced_axes(w)
  if pspec is None or not reduced or not isinstance(pspec, P):
    return pspec
  return P(*pspec, unreduced=pspec.unreduced, reduced=reduced, unreduced_kind=pspec.unreduced_kind)


def shard_map_in_specs(shardings, w0_kernel, w1_kernel, wo_kernel, has_forced_routed_experts, has_per_expert_scale):
  return (
      shardings.input,
      shardings.gate_logits,
      shardings.pre_bias_logits,
      _keep_reduced(shardings.w0, w0_kernel),
      _keep_reduced(shardings.w1, w1_kernel),
      _keep_reduced(shardings.wo, wo_kernel),
      shardings.w0_bias,
      shardings.w1_bias,
      shardings.wo_bias,
      shardings.decoder_tokens,
      P(),
      shardings.gate_logits if has_forced_routed_experts else None,
      P() if has_per_expert_scale else None,
  )


def overflow_mesh_axes(in_specs, mesh_axis_names):
  input_mesh_axes = {
      axis
      for spec in in_specs
      if spec is not None
      for dim in (spec.partitions if isinstance(spec, P) else spec)
      for axis in mesh_axes_for_dim(dim)
  }
  return tuple(axis for axis in mesh_axis_names() if axis in input_mesh_axes)


def shard_map_out_specs(output_pspec, roe_bias_axis_names, defer_small_ars, overflow_axes, log_max_load_ratio):
  return (
      output_pspec,
      P(),
      P(roe_bias_axis_names, None, None) if defer_small_ars else P(),
      P(overflow_axes),
      P(),
      P(),
  ) + ((P(),) if log_max_load_ratio else ())
