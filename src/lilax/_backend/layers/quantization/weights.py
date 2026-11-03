"""Weight-only quantization configuration and dynamic weight dequantization."""

from dataclasses import dataclass

import jax.numpy as jnp

from lilax._backend.common.common_types import Array, DType, Config, is_fp8_dtype, get_weight_dtype


@dataclass(frozen=True)
class WeightQuantConfig:
  """Configuration for weight-only quantization and dynamic dequantization."""

  quant_type: str = "none"
  weight_dtype: DType = jnp.float8_e4m3fn
  scale_dtype: DType = jnp.float32
  block_size: int | tuple[int, ...] | None = None


def get_weight_quant_config(config: Config, module_name: str) -> WeightQuantConfig | None:
  """Constructs a WeightQuantConfig for module_name if quantized, else returns None."""
  if not is_fp8_dtype(config.weight_dtype):
    return None
  resolved_weight_dtype = get_weight_dtype(config, module_name)
  if not is_fp8_dtype(resolved_weight_dtype):
    return None
  return WeightQuantConfig(
      quant_type="fp8",
      weight_dtype=resolved_weight_dtype,
      scale_dtype=jnp.float32,
      block_size=getattr(config, "weight_block_size", None),
  )


def dequantize_weight(
    w: Array,
    scale: Array | None,
    compute_dtype: DType = jnp.bfloat16,
) -> Array:
  """Dequantizes weight tensor `w` dynamically to `compute_dtype` using `scale`.

  Supports:
    1. No scale (scale is None): casts w to compute_dtype.
    2. Per-tensor scalar scale: scalar broadcast multiplication.
    3. Block-wise scale: 2D/3D block broadcast multiplication.
    4. General broadcastable scale: standard JAX broadcasting.
  """
  if scale is None:
    return w.astype(compute_dtype)

  w_c = w.astype(compute_dtype)
  scale_c = jnp.asarray(scale, compute_dtype)

  # Per-tensor scalar scale or matching shape
  if scale_c.ndim == 0 or scale_c.shape == w.shape or scale_c.size == 1:
    return w_c * scale_c.reshape(()) if scale_c.size == 1 else w_c * scale_c

  # Block-wise scale (e.g. 2D for Dense, 3D for MoE)
  if scale_c.ndim == w.ndim and any(s > 1 and s != d for s, d in zip(scale_c.shape, w.shape)):
    if not all(d % s == 0 for d, s in zip(w.shape, scale_c.shape)):
      raise ValueError(
          f"Block scaling requires weight dimensions {w.shape} to be divisible by scale dimensions {scale_c.shape}."
      )
    interleaved_shape = tuple(dim for s, w_d in zip(scale_c.shape, w.shape) for dim in (s, w_d // s))
    scale_shape = tuple(dim for s in scale_c.shape for dim in (s, 1))
    return (w_c.reshape(interleaved_shape) * scale_c.reshape(scale_shape)).reshape(w.shape)

  # Leading-dimension scale (e.g. per-expert (num_experts,) on (num_experts, in_dim, out_dim))
  if scale_c.ndim < w.ndim and w.ndim >= 3 and w.shape[: scale_c.ndim] == scale_c.shape:
    scale_c = scale_c.reshape(scale_c.shape + (1,) * (w.ndim - scale_c.ndim))

  # Standard JAX broadcasting handles per-channel or broadcastable shapes
  return w_c * scale_c
