"""Power-of-two FP8 token scales and lane-packed FP8/bfloat16 conversions on SparseCore."""

import inspect

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
_BITCAST_HAS_DIM = "dim" in inspect.signature(pltpu.bitcast).parameters


def pow2_exponent(a):
  """Returns the biased exponent of the power-of-two scale that maps `a` into FP8 range."""
  bits = jax.lax.bitcast_convert_type(a * (1.0 / FP8_MAX), jnp.int32)
  exp_biased = jnp.right_shift(bits, 23) & 0xFF
  mant = bits & 0x7FFFFF
  kb = jnp.clip(jnp.where(mant != 0, exp_biased + 1, exp_biased), 1, 253)
  return jnp.where(exp_biased > 0, kb, 127)


def pow2_from_exponent(e):
  return jax.lax.bitcast_convert_type(jnp.left_shift(e, 23), jnp.float32)


def to_compressed_bf16(v):
  if not _BITCAST_HAS_DIM:
    return v.astype(jnp.bfloat16)
  u32 = pltpu.bitcast(v, jnp.uint32, dim=-2)
  return pltpu.bitcast(u32, jnp.float8_e4m3fn, dim=-1).astype(jnp.bfloat16)


def from_compressed_bf16(acc):
  acc = jnp.clip(acc, -FP8_MAX, FP8_MAX)
  if not _BITCAST_HAS_DIM:
    return acc.astype(jnp.float8_e4m3fn)
  u32 = pltpu.bitcast(acc.astype(jnp.float8_e4m3fn), jnp.uint32, dim=-1)
  return pltpu.bitcast(u32, jnp.float8_e4m3fn, dim=-2)
