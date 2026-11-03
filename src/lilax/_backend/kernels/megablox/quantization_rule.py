"""Quantization rule selection for grouped matrix multiplication."""

import qwix
import qwix.pallas as qpl


def select_rule(use_qwix_quantization, qwix_rule, lhs_quantize_dtype, rhs_quantize_dtype):
  """Select the intercepted or manually specified GMM quantization rule."""
  quantization_rule = None
  if use_qwix_quantization:
    quantization_rule = qwix_rule if qwix_rule else qpl.get_current_rule("gmm")
    if quantization_rule and not isinstance(quantization_rule, qwix.QtRule):
      raise ValueError("Expect a QtRule for quantized training.")
  elif lhs_quantize_dtype or rhs_quantize_dtype:
    quantization_rule = qwix.QtRule(
        weight_qtype=rhs_quantize_dtype,
        weight_calibration_method="absmax",
        act_qtype=lhs_quantize_dtype,
        act_calibration_method="absmax",
    )
  return quantization_rule
