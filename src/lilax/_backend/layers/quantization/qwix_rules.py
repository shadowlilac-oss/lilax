"""Qwix quantization rules selecting which modules and ops each recipe quantizes."""

import qwix
from qwix._src.core import sparsity

import jax.numpy as jnp

from lilax._backend.common.common_types import Config


def _get_router_proj_unquantized_rule() -> qwix.QtRule:
  """Returns a Qwix rule that keeps the MoE router (gate) projection unquantized.

  Setting all qtypes to None bypasses quantization, falling back to standard
  unquantized jax.lax.dot_general for both forward and backward passes.

  Note: Because Qwix rules are evaluated in a first-match way, this rule must be
  placed before broader layer catch-all rules (e.g., 'decoder/.*layers.*').
  """
  return qwix.QtRule(
      module_path=r".*/gate$",
      weight_qtype=None,
      act_qtype=None,
      bwd_qtype=None,
      op_names=("dot_general",),
  )


def _drhs_grad_calibration_override(config: Config) -> dict:
  """Qwix DotGeneralQtConfig override for the weight-gradient arm's cotangent calibration (see types.py)."""
  m = getattr(config, "drhs_grad_quantization_calibration_method", None)
  return {"drhs_grad_calibration_method": m} if m else {}


def get_fp8_full_qwix_rule_w_sparsity(config: Config):
  """Returns Qwix quantization rules for fp8_full with optional weight sparsity."""
  sparsity_rule = None
  if config.weight_sparsity_n and config.weight_sparsity_m:
    sparsity_rule = sparsity.SparsityRule(
        weight_sparsity_n=config.weight_sparsity_n,
        weight_sparsity_m=config.weight_sparsity_m,
        weight_sparsity_update_step=config.weight_sparsity_update_step,
        weight_sparsity_start_step=config.weight_sparsity_start_step,
    )

  rules = []
  if not config.quantize_router_proj:
    rules.append(_get_router_proj_unquantized_rule())

  if config.quantize_logits_proj:
    logits_calib = config.logits_proj_quant_calibration_method or None
    logits_bwd_calib = config.logits_proj_bwd_quant_calibration_method or None
    # A logits-specific bwd calibration applies to both gradient arms and takes precedence over the
    # global drhs_grad_quantization_calibration_method.
    if logits_bwd_calib:
      logits_drhs_override = {"drhs_grad_calibration_method": logits_bwd_calib}
    else:
      logits_drhs_override = _drhs_grad_calibration_override(config)
    rules.append(
        qwix.QtRule(
            module_path="decoder/logits_dense.*",
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            weight_calibration_method=logits_calib or config.weight_quantization_calibration_method,
            act_calibration_method=logits_calib or config.act_quantization_calibration_method,
            bwd_calibration_method=logits_bwd_calib or config.bwd_quantization_calibration_method,
            additional_qt_config=logits_drhs_override or None,
            op_names=("dot_general",),
        )
    )

  paths = ["decoder/.*layers.*"]  # Main transformer layers
  if config.quantize_mtp:
    paths.append("mtp_block/.*")  # Multi-token prediction block
  # Disjunct regex paths, e.g. "(path1|path2|...)"
  module_path = f"({'|'.join(paths)})" if len(paths) > 1 else paths[0]

  if config.sa_qproj_bwd_bf16:
    rules.append(
        qwix.QtRule(
            module_path=(rf"{module_path}/(self_)?attention\w*/(query|wq_a|wq_b)"),
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=None,
            weight_calibration_method=config.weight_quantization_calibration_method,
            act_calibration_method=config.act_quantization_calibration_method,
            bwd_calibration_method=config.bwd_quantization_calibration_method,
            additional_qt_config={"sparsity_rule": sparsity_rule},
            op_names=("dot_general",),
        )
    )

  rules.append(
      qwix.QtRule(
          module_path=module_path,
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e5m2,
          weight_calibration_method=config.weight_quantization_calibration_method,
          act_calibration_method=config.act_quantization_calibration_method,
          bwd_calibration_method=config.bwd_quantization_calibration_method,
          additional_qt_config={"sparsity_rule": sparsity_rule, **_drhs_grad_calibration_override(config)},
          op_names=("dot_general", "gmm", "ragged_dot"),
      )
  )
  return rules


def get_quantization_rule(config: Config):
  """Returns a list of qwix.QtRule from `dtype`."""

  def make_qt_rule(dtype) -> list[qwix.QtRule]:
    rules = []
    if not config.quantize_router_proj:
      rules.append(_get_router_proj_unquantized_rule())
    rules.append(
        qwix.QtRule(
            module_path="decoder/.*layers.*",
            weight_qtype=dtype,
            act_qtype=dtype,
            bwd_qtype=dtype,
            bwd_weight_grad_tile_size=(
                1 / config.qwix_bwd_weight_grad_tile_count if config.qwix_bwd_weight_grad_tile_count > 1 else None
            ),
            disable_channelwise_axes=False,
            op_names=("dot_general",),
        )
    )
    return rules

  match config.quantization:
    case "int4":
      return make_qt_rule(jnp.int4)

    case "int8":
      return make_qt_rule(jnp.int8)

    case "fp4" | "fp4_e2m1":
      return make_qt_rule(jnp.float4_e2m1fn)

    case "fp8_e5m2":
      return make_qt_rule(jnp.float8_e5m2)

    case "fp8" | "fp8_e4m3" | "fp8_gpu" | "fp8_nanoo":
      return make_qt_rule(jnp.float8_e4m3fn)

    case "fp8_full":
      return get_fp8_full_qwix_rule_w_sparsity(config)
    case "fp8_gpu":
      return make_qt_rule(jnp.float8_e4m3fn)
    case "":
      return None
