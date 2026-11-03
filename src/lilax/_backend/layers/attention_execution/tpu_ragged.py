"""Runs TPU ragged decode attention over per-sequence KV lengths."""

import functools
from typing import Callable

import jax
from jax.sharding import Mesh

from lilax._backend.common.common_types import Array, AxisNames
from lilax._backend.inference.kvcache import KVTensor
from lilax._backend.kernels.attention.ragged_attention import ragged_gqa
from lilax._backend.kernels.attention.ragged_attention import ragged_mha


def tpu_ragged_attention(
    query: Array,
    key: Array | KVTensor,
    value: Array | KVTensor,
    lengths: Array,
    block_size: int,
    *,
    mesh: Mesh,
    ragged_lengths_names: AxisNames,
    cache_logical_axis_names: AxisNames,
    logical_to_mesh_axes: Callable,
) -> tuple[Array, Array, Array]:
  """Ragged Attention."""
  if isinstance(query, KVTensor):
    raise TypeError("Ragged attention does not currently support quantized tensors.")
  b = logical_to_mesh_axes(ragged_lengths_names)
  bsnd = logical_to_mesh_axes(cache_logical_axis_names)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          bsnd,
          bsnd,
          bsnd,
          b,
          None,
      ),
      out_specs=bsnd,
      check_vma=False,
  )
  def wrap_ragged_attention(query, key, value, lengths, block_size):
    if query.shape[-2] == key.shape[-2]:
      return ragged_mha(query, key, value, lengths, block_size=block_size)
    else:
      return ragged_gqa(query, key, value, lengths, block_size=block_size)

  return wrap_ragged_attention(query, key, value, lengths, block_size)
