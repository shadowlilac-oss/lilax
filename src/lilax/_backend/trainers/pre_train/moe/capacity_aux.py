"""MoE capacity, overflow, and load statistics collected from loss intermediates and reported per step."""

import jax.numpy as jnp

from lilax._backend.utils import maxtext_utils


def update_moe_capacity_aux(aux, config, intermediate_outputs):
  te_moe_block = getattr(config, "te_moe_block", False)
  if te_moe_block:
    overflow_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_capacity_overflow")
    total_recv_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_total_recv_tokens")
    capacity_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_recv_capacity_per_rank")
    if not overflow_values or not total_recv_values or not capacity_values:
      raise ValueError("te_moe_block=True did not produce TE MoE receive-capacity intermediates.")
    aux.update({
        "te_moe_capacity_overflow": jnp.any(jnp.concatenate(overflow_values)),
        "te_moe_max_total_recv_tokens": jnp.max(jnp.concatenate(total_recv_values)),
        "te_moe_recv_capacity_per_rank": jnp.min(jnp.concatenate(capacity_values)),
    })
  has_moe_overflow = jnp.bool_(False)
  if config.moe_dropless_fallback == "step":
    moe_overflow_flags = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_has_overflow", ravel=False)
    if moe_overflow_flags:
      has_moe_overflow = jnp.any(jnp.stack([jnp.any(value) for value in moe_overflow_flags]))
  aux["has_moe_overflow"] = has_moe_overflow
  if config.moe_dropless_fallback == "layer":
    fallback_flags = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_dropless_fallback")
    aux["moe_dropless_fallback_layers"] = sum(
        (jnp.sum(jnp.asarray(value).astype(jnp.int32)) for value in fallback_flags), jnp.int32(0)
    )
  if getattr(config, "log_required_ragged_buffer_factor", False):
    required_rbf_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_required_rbf")
    if required_rbf_values:
      aux["moe_required_rbf"] = jnp.concatenate([value.astype(jnp.float32) for value in required_rbf_values])
  if getattr(config, "moe_log_max_load_ratio", False):
    load_ratios = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_max_load_ratio")
    if load_ratios:
      load_ratios = jnp.concatenate(load_ratios)
      aux["moe_max_load_ratio"] = jnp.max(load_ratios)
      aux["moe_max_load_ratio_mean"] = jnp.mean(load_ratios)


def moe_capacity_scalar_metrics(config, aux):
  scalar_metrics = {}
  if getattr(config, "te_moe_block", False):
    scalar_metrics.update({
        "learning/te_moe_capacity_overflow": aux["te_moe_capacity_overflow"].astype(jnp.int32),
        "learning/te_moe_max_total_recv_tokens": aux["te_moe_max_total_recv_tokens"],
        "learning/te_moe_recv_capacity_per_rank": aux["te_moe_recv_capacity_per_rank"],
    })
  if config.moe_dropless_fallback == "layer":
    scalar_metrics["learning/moe_dropless_fallback_layers"] = aux["moe_dropless_fallback_layers"]
  if "moe_max_load_ratio" in aux:
    scalar_metrics["learning/moe_max_load_ratio"] = aux["moe_max_load_ratio"]
    scalar_metrics["learning/moe_max_load_ratio_mean"] = aux["moe_max_load_ratio_mean"]
  return scalar_metrics
