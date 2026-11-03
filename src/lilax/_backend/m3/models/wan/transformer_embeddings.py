"""Wan transformer sinusoidal and rotary embedding calculations."""

import math
from typing import Any, Union

import jax.numpy as jnp

from lilax._backend.common import common_types

Array = common_types.Array


def apply_rope(xq: Array, xk: Array, freqs_cis: Any) -> tuple[Array, Array]:
  if isinstance(freqs_cis, (tuple, list)):
    cos, sin = freqs_cis
    if cos.ndim == 2:
      seq_len = cos.shape[0]
      if xq.ndim == 4 and xq.shape[2] == seq_len:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
      else:
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    elif cos.ndim == 3 and cos.shape[0] == 1:
      seq_len = cos.shape[1]
      if xq.ndim == 4 and xq.shape[2] == seq_len:
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
      else:
        cos, sin = cos[:, :, None, :], sin[:, :, None, :]

    def _rotate(x):
      x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
      x_real = x_reshaped[..., 0]
      x_imag = x_reshaped[..., 1]
      return jnp.stack([-x_imag, x_real], axis=-1).reshape(*x.shape)

    xq_out = xq * cos + _rotate(xq) * sin
    xk_out = xk * cos + _rotate(xk) * sin
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)
  xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape(*xq.shape).astype(xq.dtype), xk_out.reshape(*xk.shape).astype(xk.dtype)


def get_sinusoidal_embeddings(timesteps: jnp.ndarray, embedding_dim: int, freq_shift: float = 1,
                              min_timescale: float = 1, max_timescale: float = 1.0e4,
                              flip_sin_to_cos: bool = False, scale: float = 1.0) -> jnp.ndarray:
  assert timesteps.ndim <= 2, "Timesteps should be a 1d or 2d-array"
  assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
  num_timescales = float(embedding_dim // 2)
  log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
  inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
  emb = jnp.expand_dims(timesteps, -1) * inv_timescales
  scaled_time = scale * emb
  if flip_sin_to_cos:
    return jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=-1)
  return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)


def get_1d_rotary_pos_embed(dim: int, pos: Union[jnp.array, int], theta: float = 10000.0,
                            linear_factor=1.0, ntk_factor=1.0, freqs_dtype=jnp.float32):
  assert dim % 2 == 0
  if isinstance(pos, int):
    pos = jnp.arange(pos)
  theta = theta * ntk_factor
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)) / linear_factor
  freqs = jnp.outer(pos, freqs)
  return jnp.exp(1j * freqs)
