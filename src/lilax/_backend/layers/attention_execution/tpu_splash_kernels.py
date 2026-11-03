"""Builds the Splash Attention kernel configuration, mask and kernel object for each TPU flash strategy."""

import dataclasses
from functools import partial
import math
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.sharding import Mesh

from lilax._backend.common.common_types import Array, AttentionType, AxisNames, Config, HEAD, Q_LENGTH
from lilax._backend.kernels.attention import tokamax_ring_attention
from lilax._backend.layers.attention_execution.tokamax_splash import tokamax_splash_kernel, tokamax_splash_mask
from lilax._backend.layers.attention_execution.tpu_splash_tuning import SplashBlockSizes, SplashTuning
from lilax._backend.layers.attention_masks import splash_mask_policy
from lilax._backend.utils.sharding import logical_to_spec

# Number of in-kernel dQ reduction steps used for AttentionType.COMPRESSED.
#
# The Tokamax Splash fused backward kernel only accepts `dq_reduction_steps` of 3 or None
# (see SplashConfig.__post_init__); any other value raises at kernel config construction.
# Selecting 3 enables a 3-slot circular ring buffer for dQ accumulation in SRAM instead of
# writing one unreduced dQ tile per KV step to HBM, reducing unreduced dQ write traffic by a
# factor of `kv_steps / 3`. For the DeepSeek-V4 HCA shapes benchmarked here (kv_steps = 128)
# that is ~42x; the ratio is shape dependent, not a fixed kernel property.
_COMPRESSED_DQ_REDUCTION_STEPS = 3


def create_splash_config(
    config: Config,
    query: Array,
    key: Array,
    attn_logits_soft_cap: float | None,
    *,
    attention_type: AttentionType,
    tuning: SplashTuning,
    blocks: SplashBlockSizes,
    cp_size: int,
    load_balanced_context_parallel: bool,
):
  """Creates the Tokamax `SplashConfig` or the JAX Splash `BlockSizes`, with blocks clamped to the sequences."""
  if config.use_tokamax_splash:
    # Tokamax only allows diagonal sub-tile skipping with a pure `CausalMask`, which
    # GLOBAL/MLA get unless load-balanced context parallelism permutes the sequence.
    use_diag_skip = (
        attention_type in (AttentionType.GLOBAL, AttentionType.MLA)
        and query.shape[2] == key.shape[2]
        and not (cp_size > 1 and load_balanced_context_parallel)
    )
    diag_skip_kwargs = {}
    if use_diag_skip and (tuning.qk_diag_skip or tuning.sv_diag_skip):
      # Only passed when enabled so older Tokamax releases (< 0.0.15) keep working by default.
      diag_skip_kwargs = {
          "qk_diag_skip": tuning.qk_diag_skip,
          "sv_diag_skip": tuning.sv_diag_skip,
          "qk_diag_grid": config.sa_qk_diag_grid,
      }
    sa_config = tokamax_splash_kernel.SplashConfig(
        block_q=min(blocks.block_q, query.shape[2]),
        block_kv=min(blocks.block_kv, key.shape[2]),
        block_kv_compute=min(blocks.block_kv_compute, key.shape[2]),
        block_q_dkv=min(blocks.block_q_dkv, query.shape[2]),
        block_kv_dkv=min(blocks.block_kv_dkv, key.shape[2]),
        block_kv_dkv_compute=min(blocks.block_kv_dkv_compute, key.shape[2]),
        use_fused_bwd_kernel=True,  # tokamax only supports fused bwd kernel
        q_layout=tokamax_splash_kernel.QKVLayout[blocks.q_layout],
        k_layout=tokamax_splash_kernel.QKVLayout[blocks.k_layout],
        v_layout=tokamax_splash_kernel.QKVLayout[blocks.v_layout],
        attn_logits_soft_cap=attn_logits_soft_cap,
        fuse_reciprocal=tuning.fuse_reciprocal,
        use_base2_exp=tuning.use_base2_exp,
        dkv_reuse_scaled_q=getattr(config, "sa_dkv_reuse_scaled_q", False),
        q_prescaled_log2e=getattr(config, "sa_q_prescaled_log2e", False),
        residual_checkpoint_name="context",
        fwd_cost_estimate=pl.CostEstimate(
            flops=config.cost_estimate_flops_fwd,
            transcendentals=0,
            bytes_accessed=0,
        )
        if config.cost_estimate_flops_fwd >= 0
        else None,
        bwd_cost_estimate=pl.CostEstimate(
            flops=config.cost_estimate_flops_bwd,
            transcendentals=0,
            bytes_accessed=0,
        )
        if config.cost_estimate_flops_bwd >= 0
        else None,
        # `dq_reduction_steps` of 0 means "unset" (see configs/base.yml); the Tokamax
        # backward kernel accepts only 3 or None, so COMPRESSED opts into the ring buffer
        # while every other attention type leaves the reduction unconstrained.
        dq_reduction_steps=(
            config.dq_reduction_steps
            if config.dq_reduction_steps > 0
            else (_COMPRESSED_DQ_REDUCTION_STEPS if attention_type == AttentionType.COMPRESSED else None)
        ),
        use_experimental_scheduler=tuning.use_splash_scheduler,
        **diag_skip_kwargs,
    )
  else:
    sa_config = splash_attention_kernel.BlockSizes(
        block_q=min(blocks.block_q, query.shape[2]),
        block_kv=min(blocks.block_kv, key.shape[2]),
        block_kv_compute=min(blocks.block_kv_compute, key.shape[2]),
        block_q_dkv=min(blocks.block_q_dkv, query.shape[2]),
        block_kv_dkv=min(blocks.block_kv_dkv, key.shape[2]),
        block_kv_dkv_compute=min(blocks.block_kv_dkv_compute, key.shape[2]),
        block_q_dq=None if tuning.use_fused_bwd_kernel else min(blocks.block_q_dq, query.shape[2]),
        block_kv_dq=None if tuning.use_fused_bwd_kernel else min(blocks.block_kv_dq, key.shape[2]),
        use_fused_bwd_kernel=tuning.use_fused_bwd_kernel,
        q_layout=splash_attention_kernel.QKVLayout[blocks.q_layout],
        k_layout=splash_attention_kernel.QKVLayout[blocks.k_layout],
        v_layout=splash_attention_kernel.QKVLayout[blocks.v_layout],
    )
  return sa_config


def make_ulysses_splash_kernel(sa_config, mask):
  """Builds the replicated full-sequence Tokamax MHA kernel run after the Ulysses all-to-all."""

  @partial(
      jax.jit,
      static_argnames=[
          "single_head_mask",
      ],
  )
  def wrap_ulysses_splash_kernel(single_head_mask):
    splash_kernel = tokamax_splash_kernel.make_splash_mha(
        mask=single_head_mask,
        config=sa_config,
        q_seq_shards=1,
    )
    return splash_kernel

  return wrap_ulysses_splash_kernel(mask)


def make_tokamax_splash_kernel(
    single_head_mask, sa_config, attention_type: AttentionType, num_kv_heads: int, splash_q_seq_shards: int
):
  """Builds the static-mask Tokamax kernel; single-KV-head compressed attention uses the MQA variant."""

  # Create the splash attention kernel object separately, jit it for performance
  @partial(
      jax.jit,
      static_argnames=[
          "single_head_mask",
      ],
  )
  def wrap_tokamax_splash_kernel(single_head_mask):
    if attention_type == AttentionType.COMPRESSED and num_kv_heads == 1:
      splash_kernel = tokamax_splash_kernel.make_splash_mqa(
          mask=single_head_mask,
          config=sa_config,
          q_seq_shards=splash_q_seq_shards,
      )
    else:
      splash_kernel = tokamax_splash_kernel.make_splash_mha(
          mask=single_head_mask,
          config=sa_config,
          q_seq_shards=splash_q_seq_shards,
      )
    return splash_kernel

  return wrap_tokamax_splash_kernel(single_head_mask)


def make_jax_splash_kernel(
    multi_head_mask, shard_head_size: int, sa_config, cp_size: int, attn_logits_soft_cap: float | None
):
  """Builds the legacy JAX Splash multi-head kernel."""

  # Create the splash attention kernel object separately, jit it for performance
  @partial(
      jax.jit,
      static_argnames=[
          "multi_head_mask",
          "shard_head_size",
      ],
  )
  def wrap_jax_splash_kernel(multi_head_mask, shard_head_size=1):
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=shard_head_size,  # the size of the axis if sharding over heads
        q_seq_shards=cp_size,  # axis for sequence sharding
        block_sizes=sa_config,
        attn_logits_soft_cap=attn_logits_soft_cap,
        residual_checkpoint_name="context",
    )
    return splash_kernel

  return wrap_jax_splash_kernel(multi_head_mask, shard_head_size)


def build_splash_kernel(
    config: Config,
    mesh: Mesh,
    query: Array,
    key: Array,
    attn_logits_soft_cap: float | None,
    *,
    attention_type: AttentionType,
    num_kv_heads: int,
    tuning: SplashTuning,
    blocks: SplashBlockSizes,
    cp_size: int,
    load_balanced_context_parallel: bool,
    use_tokamax_ring: bool,
    use_ulysses: bool,
    use_usp: bool,
    axis_names_splash_kernel: AxisNames,
    logical_to_mesh_axes: Callable,
    maybe_shard_with_pspec: Callable,
    sliding_window_size: int | None,
    chunk_attn_window_size: int | None,
    causal_block_size: int | None,
    compress_ratio: int,
    indexer_mask: Array | None,
    orig_q_len: int,
    orig_kv_len: int,
    pad_kv_total: int,
):
  """Returns `(sa_config, splash_kernel, segment_axis_names_splash_kernel, mask)` for the selected strategy.

  `splash_kernel` is None for JAX-native Splash, and for Tokamax when a runtime `indexer_mask`
  replaces the static mask. `mask` is None for the ring and USP strategies, which build no static mask.
  """
  splash_kernel = None
  mask = None
  if use_tokamax_ring or use_usp:
    sa_config, splash_kernel, segment_axis_names_splash_kernel = (
        tokamax_ring_attention.make_sharded_ring_attention_kernel(
            config,
            query=query,
            key=key,
            context_parallel_size=cp_size,
            ring_axis=config.context_sharding,
            attn_logits_soft_cap=attn_logits_soft_cap,
            maybe_shard_with_pspec=maybe_shard_with_pspec,
            mask=None,
        )
    )
  elif use_ulysses:
    sa_config = create_splash_config(
        config, query, key, attn_logits_soft_cap, attention_type=attention_type, tuning=tuning, blocks=blocks,
        cp_size=cp_size, load_balanced_context_parallel=load_balanced_context_parallel,
    )
    if config.use_max_logit_estimate > 0:
      sa_config = dataclasses.replace(sa_config, max_logit_const=config.use_max_logit_estimate)
    mask_shape = (query.shape[2], key.shape[2])  # (q_seq_len, kv_seq_len)
    mask = tokamax_splash_mask.CausalMask(shape=mask_shape)
    splash_kernel = make_ulysses_splash_kernel(sa_config, mask)
    # After the all-to-all every device runs the kernel over the full
    # sequence, so its mask metadata is replicated (q_seq_shards=1) instead
    # of sequence-sharded as in the all-gather path.
    segment_axis_names_splash_kernel = jax.sharding.PartitionSpec(None)
    splash_kernel = maybe_shard_with_pspec(splash_kernel, segment_axis_names_splash_kernel)
  else:
    sa_config = create_splash_config(
        config, query, key, attn_logits_soft_cap, attention_type=attention_type, tuning=tuning, blocks=blocks,
        cp_size=cp_size, load_balanced_context_parallel=load_balanced_context_parallel,
    )
    mask_shape = (query.shape[2], key.shape[2])  # (q_seq_len, kv_seq_len)

    splash_q_seq_shards = cp_size

    mask = splash_mask_policy.make_static_splash_mask(
        attention_type=attention_type,
        mask_shape=mask_shape,
        mask_module=tokamax_splash_mask if config.use_tokamax_splash else splash_attention_mask,
        use_tokamax_splash=config.use_tokamax_splash,
        cp_size=cp_size,
        splash_q_seq_shards=splash_q_seq_shards,
        use_load_balanced_cp=cp_size > 1 and load_balanced_context_parallel,
        sliding_window_size=sliding_window_size,
        chunk_attn_window_size=chunk_attn_window_size,
        causal_block_size=causal_block_size,
        compress_ratio=compress_ratio,
        indexer_mask=indexer_mask,
        orig_q_len=orig_q_len,
        orig_kv_len=orig_kv_len,
        pad_kv_total=pad_kv_total,
    )
  if not use_tokamax_ring and not use_ulysses and not use_usp and config.use_tokamax_splash:
    # Create mask
    single_head_mask = mask  # tokamax now just uses a single mask and assumes broadcast to all heads
    if config.use_max_logit_estimate > 0:
      sa_config = dataclasses.replace(sa_config, max_logit_const=config.use_max_logit_estimate)

    segment_axis_names_splash_kernel = logical_to_mesh_axes((Q_LENGTH,))
    if indexer_mask is None and mask is not None:
      splash_kernel = make_tokamax_splash_kernel(
          single_head_mask, sa_config, attention_type, num_kv_heads, splash_q_seq_shards
      )
      splash_kernel = maybe_shard_with_pspec(splash_kernel, segment_axis_names_splash_kernel)
    else:
      splash_kernel = None
  elif not use_tokamax_ring and not use_ulysses and not use_usp and config.use_jax_splash:
    if config.use_max_logit_estimate > 0:
      sa_config = dataclasses.replace(sa_config, max_logit_const=config.use_max_logit_estimate)
    segment_axis_names_splash_kernel = logical_to_spec((Q_LENGTH,))
  elif not use_tokamax_ring and not use_ulysses and not use_usp:
    # Create multi-head mask
    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

    head_physical_axes = logical_to_mesh_axes((HEAD,))[0]
    head_physical_axes = (head_physical_axes,) if isinstance(head_physical_axes, str) else (head_physical_axes or ())
    shard_head_size = math.prod(mesh.shape.get(ax, 1) for ax in head_physical_axes)
    splash_kernel = make_jax_splash_kernel(multi_head_mask, shard_head_size, sa_config, cp_size, attn_logits_soft_cap)
    named_sharding = jax.sharding.NamedSharding(mesh, axis_names_splash_kernel)
    segment_axis_names_splash_kernel = splash_kernel.manual_sharding_spec(named_sharding)
    splash_kernel = jax.tree.map(
        lambda arr, spec: None if arr is None else maybe_shard_with_pspec(arr, spec),
        splash_kernel,
        segment_axis_names_splash_kernel,
        is_leaf=lambda x: x is None,
    )
  return sa_config, splash_kernel, segment_axis_names_splash_kernel, mask
