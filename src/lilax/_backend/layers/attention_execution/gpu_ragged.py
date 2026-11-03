"""Runs GPU ragged decode attention through the Tokamax Pallas Triton kernel."""

import functools
from typing import Callable, Tuple

import jax
from jax.sharding import Mesh

from lilax._backend.common.common_types import Array, AxisNames, CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE
from lilax._backend.inference.kvcache import KVTensor
from tokamax._src.ops.attention import base as tokamax_attention_base
from tokamax._src.ops.attention import pallas_triton as tokamax_pallas_triton


def gpu_ragged_attention(
    q: Array,
    k: Array | KVTensor,
    v: Array | KVTensor,
    lengths: Array,
    block_size: int,
    *,
    mesh: Mesh,
    ragged_lengths_names: AxisNames,
    cache_logical_axis_names: AxisNames,
    logical_to_mesh_axes: Callable,
):
  """gpu ragged attention"""
  batch_size, q_length, q_heads, head_dim = q.shape

  # Define logical axis names - clearer and avoids repeated calls.
  b = logical_to_mesh_axes(ragged_lengths_names)
  bsnd = logical_to_mesh_axes(cache_logical_axis_names)
  bht = logical_to_mesh_axes((CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE))

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(bsnd, bsnd, bsnd, b, None),
      out_specs=(bsnd, bht, bht),
      check_vma=False,
  )
  def wrap_ragged_attention(
      q: Array, k: Array, v: Array, lengths: Array, block_size: int
  ) -> Tuple[Array, Array, Array]:
    """Wraps the tokamax attention with appropriate sharding.

    Args:
        q: Query tensor [batch, 1, num_heads, head_dim].
        k: Key tensor [batch, kv_seq_len, num_kv_heads, head_dim].
        v: Value tensor [batch, kv_seq_len, num_kv_heads, head_dim].
        lengths: Sequence lengths.
        block_size: Block size for attention.

    Returns:
        A tuple containing the output, max, and sum tensors.
    """
    mask = tokamax_attention_base.Mask(
        k_end=lengths[..., None],
    )
    gpu_flash_attn = tokamax_pallas_triton.PallasTritonFlashAttention()
    local_out, (local_max, local_sum) = gpu_flash_attn(
        q,
        k,
        v,
        logits_scale=1.0,
        mask=mask,
        return_residuals=True,
        normalize_output=False,
    )
    return local_out, local_max, local_sum

  local_out, local_max, local_sum = wrap_ragged_attention(q, k, v, lengths, block_size)

  # Reshape local_out, local_max and local_sum to match Maxtext requirements
  local_out = local_out.reshape(batch_size, q_length, q_heads, head_dim)
  local_max = local_max.reshape(batch_size, q_length, q_heads, 1)
  local_sum = local_sum.reshape(batch_size, q_length, q_heads, 1)
  return local_out, local_max, local_sum
