"""Runs TPU flash attention on the shard-local Q/K/V inside the flash attention `shard_map`."""

from functools import partial
import math

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
import jax.numpy as jnp

from lilax._backend.common.common_types import AttentionType, Config, DEFAULT_MASK_VALUE
from lilax._backend.kernels.attention import jax_flash_attention
from lilax._backend.kernels.attention import tokamax_ring_attention
from lilax._backend.kernels.attention import ulysses_attention
from lilax._backend.kernels.attention import usp_attention
from lilax._backend.layers.attention_execution.tokamax_splash import tokamax_splash_kernel
from lilax._backend.layers.attention_execution.tpu_splash_tuning import SplashTuning
from lilax._backend.utils.context_parallel import sequence_reorder


def local_flash_attention(
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
    *,
    config: Config,
    use_tokamax_ring: bool,
    use_ulysses: bool,
    use_usp: bool,
    context_axis,
    ulysses_axis,
    max_logit_value,
    record_max_logits: bool,
    attn_logits_soft_cap: float | None,
    attention_type: AttentionType,
    num_kv_heads: int,
    mask,
    splash_tuning: SplashTuning,
):
  """Returns `(attention_output, max_logits)`, where max_logits is None unless the kernel recorded them."""
  if use_tokamax_ring:
    attention_output = tokamax_ring_attention.call_ring_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        decoder_segment_ids_kv,
        splash_kernel,
        indexer_mask=indexer_mask,
    )
    return attention_output, None

  if use_ulysses:
    return _ulysses_attention(
        query, key, value, decoder_segment_ids_q, splash_kernel, sinks, context_axis, max_logit_value
    ), None

  if use_usp:
    attention_output = usp_attention.call_usp_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        splash_kernel,
        ulysses_axis,
    )
    return attention_output, None

  # The load-balanced all-gather path restores K/V to contiguous order
  # before calling Splash attention.
  if cp_size > 1 and load_balanced_context_parallel:
    key = sequence_reorder.reorder_sequence(tensor=key, cp_size=cp_size, seq_dim=2, to_contiguous=True)
    value = sequence_reorder.reorder_sequence(tensor=value, cp_size=cp_size, seq_dim=2, to_contiguous=True)

    decoder_segment_ids_unpermuted = sequence_reorder.reorder_sequence(
        tensor=decoder_segment_ids_kv,
        cp_size=cp_size,
        seq_dim=1,
        to_contiguous=True,
    )

    if indexer_mask is not None:
      indexer_mask = sequence_reorder.reorder_sequence(
          tensor=indexer_mask,
          cp_size=cp_size,
          seq_dim=2,
          to_contiguous=True,
      )

  if decoder_segment_ids_q is not None:
    if cp_size > 1 and load_balanced_context_parallel:
      decoder_segment_ids_tuple = splash_attention_kernel.SegmentIds(
          decoder_segment_ids_q, decoder_segment_ids_unpermuted
      )
    else:
      # if cp=1, decoder_segment_ids_q is the same as decoder_segment_ids_kv
      decoder_segment_ids_tuple = splash_attention_kernel.SegmentIds(decoder_segment_ids_q, decoder_segment_ids_kv)
  else:
    decoder_segment_ids_tuple = None

  if config.use_tokamax_splash:
    if indexer_mask is not None and cp_size > 1:
      return _block_masked_indexer_attention(
          query, key, value, decoder_segment_ids_tuple, sa_config, sinks, indexer_mask, record_max_logits,
          attn_logits_soft_cap,
      )
    if indexer_mask is not None:
      return _dynamic_mask_splash_attention(
          query, key, value, decoder_segment_ids_tuple, sa_config, sinks, indexer_mask, record_max_logits,
          max_logit_value, attention_type, num_kv_heads,
      )
    return _static_splash_attention(
        query, key, value, decoder_segment_ids_tuple, splash_kernel, sinks, record_max_logits, max_logit_value,
        attention_type, num_kv_heads,
    )

  elif config.use_jax_splash:
    materialized_mask = jnp.asarray(mask[:, :])
    attention_output = jax_flash_attention.flash_attention_block_masked(
        query,
        key,
        value,
        decoder_segment_ids_tuple,
        block_kv=splash_tuning.block_kv,
        block_q=splash_tuning.block_q,
        mask=materialized_mask,
        mask_value=DEFAULT_MASK_VALUE,
        cap=attn_logits_soft_cap,
    )
    if record_max_logits:
      # The native JAX splash attention implementation does not currently expose the softmax statistics
      # (e.g., max_logits) required for QK-Clip. Use tokamax splash attention if max logit recording is needed.
      raise NotImplementedError("record_max_logits not supported for jax_splash")
  else:
    attention_output = jax.vmap(splash_kernel, in_axes=(0, 0, 0, 0, None))(
        query, key, value, decoder_segment_ids_tuple, sinks
    )
    if record_max_logits:
      raise NotImplementedError("record_max_logits not supported for legacy splash")

  return attention_output, None


def _ulysses_attention(query, key, value, decoder_segment_ids_q, splash_kernel, sinks, context_axis, max_logit_value):
  """Exchanges sequence shards for head shards, runs the full-sequence kernel, then exchanges back."""
  query = ulysses_attention.ulysses_all_to_all(query, context_axis)
  key = ulysses_attention.ulysses_all_to_all(key, context_axis)
  value = ulysses_attention.ulysses_all_to_all(value, context_axis)
  if decoder_segment_ids_q is not None:
    # Q and KV segment IDs are the same tensor in this train-only
    # self-attention path, so one gather serves both kernel operands.
    full_segment_ids = jax.lax.all_gather(
        decoder_segment_ids_q,
        context_axis,
        axis=1,
        tiled=True,
    )
    decoder_segment_ids_tuple = tokamax_splash_kernel.SegmentIds(
        full_segment_ids,
        full_segment_ids,
    )
  else:
    decoder_segment_ids_tuple = None
  kernel = partial(splash_kernel, max_logit_value=max_logit_value)
  attention_output = jax.vmap(lambda q, k, v, d, s: kernel(q, k, v, d, sinks=s), in_axes=(0, 0, 0, 0, None))(
      query, key, value, decoder_segment_ids_tuple, sinks
  )
  return ulysses_attention.inverse_ulysses_all_to_all(attention_output, context_axis)


def _block_masked_indexer_attention(
    query, key, value, decoder_segment_ids_tuple, sa_config, sinks, indexer_mask, record_max_logits,
    attn_logits_soft_cap,
):
  """Applies a sparse-indexer mask under all-gather context parallelism with the block-sparse JAX kernel."""
  # Dynamic Tokamax Splash derives its Pallas grid from each shard-local
  # mask, which is unsupported under sharding; the native block-sparse
  # implementation consumes the runtime mask without a data-dependent
  # device-local grid.
  if sinks is not None:
    # cp_size comes from the mesh axis named by context_sharding, which
    # custom mesh rules can point at an axis the config validation does
    # not cover.
    raise ValueError(
        "Sparse indexer with all-gather context parallelism for flash attention does not support"
        " attention sinks."
    )
  indexer_mask = indexer_mask == 0.0
  # sa_config blocks are clamped to the global sequence lengths; inside
  # the shard map the query is a sequence shard, and the blocks must
  # divide the sequence lengths the kernel sees exactly.
  block_q = math.gcd(sa_config.block_q, query.shape[2])
  block_kv = math.gcd(sa_config.block_kv, key.shape[2])
  if record_max_logits:
    attention_output, stats = jax_flash_attention.flash_attention_block_masked(
        query,
        key,
        value,
        decoder_segment_ids_tuple,
        block_kv=block_kv,
        block_q=block_q,
        mask=indexer_mask,
        mask_value=DEFAULT_MASK_VALUE,
        cap=attn_logits_soft_cap,
        save_residuals=True,
        logits_dtype=query.dtype,
        loop_unroll=False,
        fuse_logits=False,
    )
    return attention_output, stats["max_logits"]

  attention_output = jax_flash_attention.flash_attention_block_masked(
      query,
      key,
      value,
      decoder_segment_ids_tuple,
      block_kv=block_kv,
      block_q=block_q,
      mask=indexer_mask,
      mask_value=DEFAULT_MASK_VALUE,
      cap=attn_logits_soft_cap,
      logits_dtype=query.dtype,
      loop_unroll=False,
      fuse_logits=False,
  )
  return attention_output, None


def _dynamic_mask_splash_attention(
    query, key, value, decoder_segment_ids_tuple, sa_config, sinks, indexer_mask, record_max_logits,
    max_logit_value, attention_type, num_kv_heads,
):
  """Runs Tokamax Splash with a per-example runtime indexer mask padded to whole kernel blocks."""
  # Convert additive float mask (0.0=attend, negative=masked) to boolean mask for Tokamax splash kernel
  indexer_mask = indexer_mask == 0.0
  padded_q_len = ((indexer_mask.shape[-2] + sa_config.block_q - 1) // sa_config.block_q) * sa_config.block_q
  padded_kv_len = ((indexer_mask.shape[-1] + sa_config.block_kv - 1) // sa_config.block_kv) * sa_config.block_kv
  pad_q = padded_q_len - indexer_mask.shape[-2]
  pad_kv = padded_kv_len - indexer_mask.shape[-1]
  if pad_q > 0 or pad_kv > 0:
    pad_width = [(0, 0)] * (indexer_mask.ndim - 2) + [(0, pad_q), (0, pad_kv)]
    indexer_mask = jnp.pad(
        indexer_mask,
        pad_width,
        constant_values=False,
    )

  # Construct the splash kernel call with dynamic mask
  def dynamic_mask_splash_kernel(q, k, v, segment, sinks, indexer_mask):
    if attention_type == AttentionType.COMPRESSED and num_kv_heads == 1:
      splash_kernel = tokamax_splash_kernel.make_dynamic_splash_mqa(
          mask=indexer_mask,
          config=sa_config,
      )
      k_in = k[0]
      v_in = v[0]
    else:
      splash_kernel = tokamax_splash_kernel.make_dynamic_splash_mha(
          mask=indexer_mask,
          config=sa_config,
      )
      k_in = k
      v_in = v
    kernel = partial(splash_kernel, max_logit_value=max_logit_value)

    if record_max_logits:
      out, stats = kernel(q, k_in, v_in, segment, sinks=sinks, save_residuals=True)
      return out, stats["max_logits"]
    else:
      return kernel(q, k_in, v_in, segment, sinks=sinks), None

  # Iterate over batch dimension for (query, key, value, segment, sinks, mask)
  attn_fn = jax.vmap(dynamic_mask_splash_kernel, (0, 0, 0, 0, None, 0))

  if record_max_logits:
    attention_output, max_logits = attn_fn(query, key, value, decoder_segment_ids_tuple, sinks, indexer_mask)
    return attention_output, max_logits
  else:
    attention_output, _ = attn_fn(query, key, value, decoder_segment_ids_tuple, sinks, indexer_mask)
    return attention_output, None


def _static_splash_attention(
    query, key, value, decoder_segment_ids_tuple, splash_kernel, sinks, record_max_logits, max_logit_value,
    attention_type, num_kv_heads,
):
  """Runs the prebuilt static-mask Tokamax Splash kernel over the batch."""
  kernel = partial(splash_kernel, max_logit_value=max_logit_value)
  is_mqa_compressed = attention_type == AttentionType.COMPRESSED and num_kv_heads == 1

  if record_max_logits:

    def kernel_fn(q, k, v, d, s):
      # Pass save_residuals=True to force stats generation
      k_in = k[0] if is_mqa_compressed else k
      v_in = v[0] if is_mqa_compressed else v
      out, stats = kernel(q, k_in, v_in, d, sinks=s, save_residuals=True)
      return out, stats["max_logits"]

    attention_output, max_logits = jax.vmap(kernel_fn, in_axes=(0, 0, 0, 0, None))(
        query, key, value, decoder_segment_ids_tuple, sinks
    )
    return attention_output, max_logits
  else:

    def call_kernel_fn(q, k, v, d, s):
      k_in = k[0] if is_mqa_compressed else k
      v_in = v[0] if is_mqa_compressed else v
      return kernel(q, k_in, v_in, d, sinks=s)

    attention_output = jax.vmap(call_kernel_fn, in_axes=(0, 0, 0, 0, None))(
        query, key, value, decoder_segment_ids_tuple, sinks
    )
    return attention_output, None
