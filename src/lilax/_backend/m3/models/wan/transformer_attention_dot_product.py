"""Wan transformer XLA dot-product attention kernel."""

import jax.numpy as jnp
import tokamax

from lilax._backend.common import common_types

Array = common_types.Array


def _dot_product_attention(query: Array, key: Array, value: Array, heads: int, scale: float, dtype: jnp.dtype,
                           use_memory_efficient_attention: bool, attention_mask: Array | None) -> Array:
  def to_bshd(x):
    return x.reshape(*x.shape[:2], heads, x.shape[2] // heads) if x.ndim == 3 else jnp.transpose(x, (0, 2, 1, 3))

  out_dtype = jnp.result_type(dtype, value.dtype)
  query, key, value = to_bshd(query), to_bshd(key), to_bshd(value)
  if key.shape[1] == 0:
    return jnp.zeros((*query.shape[:2], heads * value.shape[-1]), out_dtype)
  out = tokamax.dot_product_attention(query, key, value,
                                       mask=None if attention_mask is None else attention_mask[:, None, None, :],
                                       scale=scale, implementation="xla_chunked" if use_memory_efficient_attention else "xla")
  if attention_mask is not None:
    out = jnp.where(jnp.any(attention_mask, axis=-1)[:, None, None, None], out, 0)
  return out.reshape(*out.shape[:2], -1).astype(out_dtype)
