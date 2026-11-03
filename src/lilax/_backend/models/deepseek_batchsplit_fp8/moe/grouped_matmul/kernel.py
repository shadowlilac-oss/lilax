"""Grouped matmul dispatch for the DeepSeek FP8 batch-split routed experts."""

import jax
from lilax._backend.kernels import megablox
from lilax._backend.layers.quantization.qwix_rules import get_fp8_full_qwix_rule_w_sparsity
import tokamax


def gmm(
    inputs,
    kernel,
    tiling,
    group_sizes,
    preferred_element_type,
    weight_gather_axes,
    *,
    config,
):
  """Runs the qwix megablox grouped matmul when qwix-quantized, else the tokamax ragged dot."""
  if config.quantization and config.use_qwix_quantization:
    output = megablox.gmm(
        lhs=inputs,
        rhs=kernel,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=tiling,
        use_qwix_quantization=config.use_qwix_quantization,
        use_tokamax_backend=config.use_tokamax_gmm,
        weight_gather_axes=weight_gather_axes,
        qwix_rule=get_fp8_full_qwix_rule_w_sparsity(config)[-1],
    )
  else:
    output = tokamax.ragged_dot(
        lhs=inputs,
        rhs=kernel,
        group_sizes=tokamax.RaggedDotGroupSizes(group_sizes, len(inputs)),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        implementation="mosaic",
    )
  return output
