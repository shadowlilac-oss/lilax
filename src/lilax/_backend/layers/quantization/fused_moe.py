"""Expert weight quantization and scale layout for tpu-inference's fused MoE kernel (gmm_v2)."""

from typing import Tuple, Callable

import qwix.pallas as qpl

import qwix
from qwix._src import interception as qwix_interception

import jax
import jax.numpy as jnp

from lilax._backend.utils import max_logging


def prepare_fused_gmm_scale(scale: jnp.ndarray, kernel_shape: Tuple[int, ...]) -> jnp.ndarray:
  """Expands a per-expert weight scale to gmm_v2's rhs_scale
  layout: (num_experts, num_k_blocks, 1, n_dim). This kernel needs a scale per
  output (N) channel, unlike gmm v2 elsewhere -- N-axis groups that
  shared one value in the checkpoint are expanded by repeating it (lossless).
  """
  num_experts, k_dim, n_dim = kernel_shape
  if scale.ndim == 1:
    scale = scale.reshape(num_experts, 1, 1)
  if scale.ndim != 3 or scale.shape[0] != num_experts:
    raise ValueError(f"Unexpected weight scale shape {scale.shape} for kernel shape {kernel_shape}.")
  _, k_blocks, n_blocks = scale.shape
  if k_dim % k_blocks != 0 or n_dim % n_blocks != 0:
    raise ValueError(f"Weight scale shape {scale.shape} does not evenly divide kernel shape {kernel_shape}.")
  n_repeat = n_dim // n_blocks
  scale = jnp.repeat(scale.astype(jnp.float32), n_repeat, axis=2)  # (E, k_blocks, n_dim)
  return jnp.expand_dims(scale, axis=2)  # (E, k_blocks, 1, n_dim)


# Quantized weight dtypes that tpu-inference's fused MoE kernel (gmm_v2) takes with
# per-block scales and dequantizes in-kernel. Any other qtype keeps the expert weights
# unquantized on the fused path.
FUSED_MOE_KERNEL_WEIGHT_QTYPES = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.int8))


def get_fused_moe_rule() -> qwix.QuantizationRule | None:
  """Returns the qwix rule that governs the fused MoE grouped matmul, if any.

  The fused kernel is the same grouped matmul that MaxText's own megablox / ragged_dot
  paths implement, so it takes its rule from the "gmm" op exactly like they do. Rules
  that only list "dot_general" (the plain fp8 / int8 recipes) leave the experts
  unquantized on every MoE path, including this one.
  """
  return qpl.get_current_rule("gmm")


def quantize_weight_for_fused_moe(
    kernel: jax.Array, rule: qwix.QuantizationRule | None
) -> tuple[jax.Array, jax.Array | None]:
  """Quantizes an [E, K, N] expert weight for tpu-inference's fused MoE kernel.

  The kernel contracts over K, dequantizes after the matmul with scales laid out as
  [E, num_blocks, 1, N] (blocks along K), and quantizes the activations itself with a
  32-bit accumulator. Scale granularity follows the qwix rule: channelwise over (E, N),
  plus blockwise along K when the rule sets tile_size.

  Returns the (possibly unchanged) weight and its scale, or None when the rule does not
  ask for a weight dtype the kernel supports.
  """

  weight_qtype = getattr(rule, "weight_qtype", None) if rule is not None else None
  if weight_qtype is None:
    return kernel, None
  qtype = jnp.dtype(weight_qtype)
  if qtype not in FUSED_MOE_KERNEL_WEIGHT_QTYPES:
    max_logging.log(f"fused MoE kernel does not take {qtype} weights; keeping expert weights in {kernel.dtype}.")
    return kernel, None
  if kernel.ndim != 3:
    raise ValueError(f"fused MoE expert weights must be [E, K, N], got {kernel.shape}")

  tile_size = getattr(rule, "tile_size", None)
  tiled_axes = {1: tile_size} if tile_size else {}
  cal_method = getattr(rule, "weight_calibration_method", None)
  quant_kwargs = {"calibration_method": cal_method} if cal_method is not None else {}

  quantized = qpl.quantize(
      kernel,
      qtype,
      channelwise_axes=(0, 2),
      tiled_axes=tiled_axes,
      scale_dtype=jnp.float32,
      **quant_kwargs,
  )
  scale = jnp.expand_dims(quantized.scale, axis=2)
  return quantized.qvalue, scale


def without_qwix_interception(fn: Callable) -> Callable:
  """Returns `fn` wrapped so that qwix's op interception is off while it runs.

  For ops that do their own quantization (tpu-inference's fused MoE), the qwix rule is
  applied once up front to their inputs and the op itself is opaque to qwix: neither
  the routing math around the kernel nor anything traced inside it gets rewritten.
  """
  return qwix_interception.disable_interceptions(fn)
