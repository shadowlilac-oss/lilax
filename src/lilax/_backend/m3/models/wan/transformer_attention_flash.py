"""Wan transformer flash attention kernel and padding."""

import jax.numpy as jnp

from lilax._backend.common import common_types
from .utils_wan import BlockSizes

Array = common_types.Array


def unflatten_heads(tensor: Array, heads: int) -> Array:
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
  return jnp.transpose(tensor, (0, 2, 1, 3))


def _pad_axis(x: Array, axis: int, multiple: int) -> Array:
  pad = -x.shape[axis] % multiple
  if pad == 0:
    return x
  widths = [(0, 0)] * x.ndim
  widths[axis] = (0, pad)
  return jnp.pad(x, widths)


def _splash_block_sizes(block_sizes: BlockSizes | None, q_len: int, kv_len: int, dtype: jnp.dtype) -> BlockSizes:
  if block_sizes is not None and q_len == kv_len:
    return block_sizes
  q_default = 1024 if dtype == jnp.bfloat16 else 512
  block_q = block_sizes.block_q if block_sizes is not None else q_default
  block_kv = q_default if q_len == kv_len else -kv_len % 128 + kv_len
  return BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv, block_q_dkv=block_q,
                    block_kv_dkv=block_kv, block_kv_dkv_compute=block_kv)
