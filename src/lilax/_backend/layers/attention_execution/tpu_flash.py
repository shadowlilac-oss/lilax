"""Runs TPU Splash flash attention, including the ring, Ulysses and USP context-parallel strategies."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from lilax._backend.common.common_types import (
    Array,
    AttentionType,
    AxisNames,
    BATCH_ATTN,
    Config,
    HEAD,
    KV_LENGTH,
    Q_LENGTH,
)
from lilax._backend.kernels.attention import tokamax_ring_attention
from lilax._backend.kernels.attention import ulysses_attention
from lilax._backend.kernels.attention import usp_attention
from lilax._backend.layers.attention_execution import tpu_flash_shard
from lilax._backend.layers.attention_execution import tpu_splash_kernels
from lilax._backend.layers.attention_execution.tpu_splash_tuning import SplashBlockSizes, SplashTuning


def prepare_flash_inputs(
    config: Config,
    query: Array,
    key: Array,
    value: Array,
    decoder_segment_ids: Array | None,
    decoder_segment_ids_kv: Array | None,
):
  """Returns `(query, key, value, decoder_segment_ids, decoder_segment_ids_kv_in, orig_q_len, orig_kv_len)`.

  Q/K/V are transposed to ('batch', 'heads', 'length', 'kv'); KV segment ids default to the query's.
  """
  # Transpose to ('batch', 'heads', 'length', 'kv')
  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))

  orig_q_len = query.shape[2]
  orig_kv_len = key.shape[2]
  decoder_segment_ids_kv_in = decoder_segment_ids_kv if decoder_segment_ids_kv is not None else decoder_segment_ids
  if getattr(config, "sa_ignore_segment_ids", False):
    # Every row is one segment (stream chunking), so the segment mask is a no-op; causal masking still applies.
    decoder_segment_ids = None
    decoder_segment_ids_kv_in = None
  return query, key, value, decoder_segment_ids, decoder_segment_ids_kv_in, orig_q_len, orig_kv_len


def run_flash_attention(
    query: Array,
    key: Array,
    value: Array,
    decoder_segment_ids: Array | None,
    decoder_segment_ids_kv_in: Array | None,
    orig_q_len: int,
    orig_kv_len: int,
    attn_logits_soft_cap: float | None,
    sinks: Array | None,
    indexer_mask: Array | None,
    record_max_logits: bool,
    pad_kv_total: int,
    compress_ratio: int,
    *,
    config: Config,
    mesh: Mesh,
    attention_type: AttentionType,
    num_kv_heads: int,
    splash_tuning: SplashTuning,
    blocks: SplashBlockSizes,
    sliding_window_size: int | None,
    chunk_attn_window_size: int | None,
    causal_block_size: int | None,
    flash_axis_names_q: AxisNames,
    flash_axis_names_kv: AxisNames,
    flash_axis_names_splash_kernel: AxisNames,
    use_tokamax_ring: bool,
    use_ulysses: bool,
    use_usp: bool,
    cp_size: int,
    load_balanced_context_parallel: bool,
    logical_to_mesh_axes: Callable,
    maybe_shard_with_pspec: Callable,
) -> tuple[Array, Array]:
  """Pads, shards and runs the prepared Q/K/V through the selected Splash strategy inside a `shard_map`."""
  pad_q = 0
  pad_kv = 0
  # Pad sequences to block-sized boundaries upfront for AttentionType.COMPRESSED
  if attention_type == AttentionType.COMPRESSED:
    if query.shape[2] % blocks.block_q != 0:
      pad_q = (blocks.block_q - (query.shape[2] % blocks.block_q)) % blocks.block_q
      query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_q), (0, 0)))
      if decoder_segment_ids is not None:
        decoder_segment_ids = jnp.pad(decoder_segment_ids, ((0, 0), (0, pad_q)), constant_values=-1)

    if key.shape[2] % blocks.block_kv != 0:
      pad_kv = (blocks.block_kv - (key.shape[2] % blocks.block_kv)) % blocks.block_kv
      key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_kv), (0, 0)))
      value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_kv), (0, 0)))
      if decoder_segment_ids_kv_in is not None:
        decoder_segment_ids_kv_in = jnp.pad(decoder_segment_ids_kv_in, ((0, 0), (0, pad_kv)), constant_values=-1)

  segment_axis_names_q = None
  segment_axis_names_kv = None
  sink_axis_names = logical_to_mesh_axes((HEAD,))
  if decoder_segment_ids is not None:
    segment_axis_names_q = logical_to_mesh_axes((BATCH_ATTN, Q_LENGTH))
    segment_axis_names_kv = logical_to_mesh_axes((BATCH_ATTN, KV_LENGTH))

  axis_names_splash_kernel = logical_to_mesh_axes(flash_axis_names_splash_kernel)
  axis_names_q = logical_to_mesh_axes(flash_axis_names_q)
  axis_names_kv = logical_to_mesh_axes(flash_axis_names_kv)
  indexer_mask_axis_names = logical_to_mesh_axes((BATCH_ATTN, Q_LENGTH, KV_LENGTH))
  # Each strategy branch below sets the mesh axis its shard-local exchange runs over.
  context_axis = None
  ulysses_axis = None
  if use_tokamax_ring:
    context_axis = config.context_sharding
    segment_axis_names_q = tokamax_ring_attention.with_sequence_axis(
        segment_axis_names_q,
        context_axis,
        sequence_dim=1,
    )
    axis_names_kv = tokamax_ring_attention.with_sequence_axis(
        axis_names_kv,
        context_axis,
        sequence_dim=2,
    )
    segment_axis_names_kv = tokamax_ring_attention.with_sequence_axis(
        segment_axis_names_kv,
        context_axis,
        sequence_dim=1,
    )
  elif use_ulysses:
    context_axis = config.context_sharding
    segment_axis_names_q = ulysses_attention.with_sequence_axis(
        segment_axis_names_q,
        context_axis,
        sequence_dim=1,
    )
    axis_names_kv = ulysses_attention.with_sequence_axis(
        axis_names_kv,
        context_axis,
        sequence_dim=2,
    )
    segment_axis_names_kv = ulysses_attention.with_sequence_axis(
        segment_axis_names_kv,
        context_axis,
        sequence_dim=1,
    )
  elif use_usp:
    ring_axis = config.context_sharding
    ulysses_axis = config.ulysses_context_sharding
    axis_names_kv = usp_attention.with_usp_sequence_axes(
        axis_names_kv,
        ring_axis,
        ulysses_axis,
        sequence_dim=2,
    )

  devices_in_data_fsdp = mesh.shape.get("data", 1) * mesh.shape.get("fsdp", 1)
  assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
      "Batch dimension should be shardable among the devices in data and fsdp"
      " axis"
      f" got {query.shape[0]=}/{devices_in_data_fsdp=}"
  )

  sa_config, splash_kernel, segment_axis_names_splash_kernel, mask = tpu_splash_kernels.build_splash_kernel(
      config,
      mesh,
      query,
      key,
      attn_logits_soft_cap,
      attention_type=attention_type,
      num_kv_heads=num_kv_heads,
      tuning=splash_tuning,
      blocks=blocks,
      cp_size=cp_size,
      load_balanced_context_parallel=load_balanced_context_parallel,
      use_tokamax_ring=use_tokamax_ring,
      use_ulysses=use_ulysses,
      use_usp=use_usp,
      axis_names_splash_kernel=axis_names_splash_kernel,
      logical_to_mesh_axes=logical_to_mesh_axes,
      maybe_shard_with_pspec=maybe_shard_with_pspec,
      sliding_window_size=sliding_window_size,
      chunk_attn_window_size=chunk_attn_window_size,
      causal_block_size=causal_block_size,
      compress_ratio=compress_ratio,
      indexer_mask=indexer_mask,
      orig_q_len=orig_q_len,
      orig_kv_len=orig_kv_len,
      pad_kv_total=pad_kv_total,
  )
  max_logit_value = None

  # wrap_flash_attention receives Q/K/V and mask metadata with the shardings
  # specified in the shard_map in_specs below. For the all-gather path Q is
  # sequence-sharded and K/V are replicated. For the Tokamax ring path Q, K,
  # V, and segment IDs are all sequence-sharded over the context axis.
  # For Ulysses Q/K/V are sequence-sharded at the boundary and head-sharded
  # inside the local Splash call. USP performs the same Ulysses exchange over
  # its ulysses axis and runs the ring kernel over the ring axis within each
  # head subset.

  if record_max_logits:
    # max_logits will share similar sharding as query but last dim is unrelated to model
    # Using None for the last dimension of max_logits sharding
    if isinstance(axis_names_q, jax.sharding.PartitionSpec):
      # max_logits is Rank 3 (batch, heads, seq), so drop the last dimension (d_kv)
      max_logits_spec = jax.sharding.PartitionSpec(*axis_names_q[:-1])
    else:
      # Fallback if axis_names_q is not a PartitionSpec (unlikely in this context)
      max_logits_spec = axis_names_q[:-1]
    out_specs = (axis_names_q, max_logits_spec)
  else:
    # Safely pad the specs with None to match the (attention_output, None) return type
    out_specs = (axis_names_q, None)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          axis_names_q,
          axis_names_kv,
          axis_names_kv,
          segment_axis_names_q,
          segment_axis_names_kv,
          None,  # no sharding for config
          segment_axis_names_splash_kernel,
          None,  # no sharding for cp_size
          None,  # no sharding for load_balanced_context_parallel
          sink_axis_names,  # sharding align with query heads
          indexer_mask_axis_names,
      ),
      out_specs=out_specs,
      check_vma=False,
  )
  def wrap_flash_attention(
      query,
      key,
      value,
      decoder_segment_ids_q,
      decoder_segment_ids_kv,
      sa_config,
      splash_kernel,
      cp_size,
      load_balanced_context_parallel,
      sinks,
      indexer_mask,
  ):
    return tpu_flash_shard.local_flash_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        decoder_segment_ids_kv,
        sa_config,
        splash_kernel,
        cp_size,
        load_balanced_context_parallel,
        sinks,
        indexer_mask,
        config=config,
        use_tokamax_ring=use_tokamax_ring,
        use_ulysses=use_ulysses,
        use_usp=use_usp,
        context_axis=context_axis,
        ulysses_axis=ulysses_axis,
        max_logit_value=max_logit_value,
        record_max_logits=record_max_logits,
        attn_logits_soft_cap=attn_logits_soft_cap,
        attention_type=attention_type,
        num_kv_heads=num_kv_heads,
        mask=mask,
        splash_tuning=splash_tuning,
    )

  query = maybe_shard_with_pspec(query, axis_names_q)
  key = maybe_shard_with_pspec(key, axis_names_kv)
  value = maybe_shard_with_pspec(value, axis_names_kv)
  decoder_segment_ids_q = maybe_shard_with_pspec(decoder_segment_ids, segment_axis_names_q)
  decoder_segment_ids_kv = maybe_shard_with_pspec(decoder_segment_ids_kv_in, segment_axis_names_kv)
  sinks = maybe_shard_with_pspec(sinks, sink_axis_names)
  indexer_mask = maybe_shard_with_pspec(indexer_mask, indexer_mask_axis_names)

  ret = wrap_flash_attention(
      query,
      key,
      value,
      decoder_segment_ids_q,
      decoder_segment_ids_kv,
      sa_config,
      None if config.use_jax_splash else splash_kernel,
      cp_size,
      load_balanced_context_parallel,
      sinks,
      indexer_mask,
  )

  x, max_logits = ret
  x = jnp.transpose(x, axes=(0, 2, 1, 3))
  # Slice outputs back to unpadded sequence length for compressed attention
  if attention_type == AttentionType.COMPRESSED and pad_q > 0:
    x = x[:, :orig_q_len, :, :]
    if record_max_logits:
      max_logits = max_logits[:, :, :orig_q_len]

  if record_max_logits:
    # Max over sequence length (dim 2 of max_logits)
    # max_logits from kernel is (batch, heads, q_len)
    # output needs to be (batch, heads)
    # Note: q_len is sharded. We first reduce locally.
    max_logits_local = jnp.max(max_logits, axis=2)

    return x, max_logits_local

  return x, None
