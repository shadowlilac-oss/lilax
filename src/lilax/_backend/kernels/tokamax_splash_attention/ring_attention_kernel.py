"""Implementation of Ring Attention."""

import dataclasses
import functools
from typing import Any

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
from lilax._backend.kernels.tokamax_splash_attention import base
from lilax._backend.kernels.tokamax_splash_attention import ring_attention_utils
from lilax._backend.kernels.tokamax_splash_attention import splash_attention_kernel as splash_kernel
from lilax._backend.kernels.tokamax_splash_attention import splash_attention_mask as mask_lib
from lilax._backend.kernels.tokamax_splash_attention import splash_attention_mask_info as mask_info_lib

P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial

SegmentIds = base.SegmentIds
SplashConfig = splash_kernel.SplashConfig
RingSplashResidualsType = tuple[
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    SegmentIds | None,  # segment_ids
    jax.Array | None,  # sinks
    jax.Array,  # out
    jax.Array,  # logsumexp
    MaskInfo | None,  # dkv_mask_info
    jax.Array | None,  # indexer_mask
]
SplashCustomReturnType = base.SplashCustomReturnType
MaskFunctionType = splash_kernel.MaskFunctionType
_splash_attention_forward = splash_kernel._splash_attention_forward  # pylint: disable=protected-access
_splash_attention_bwd = splash_kernel._splash_attention_bwd  # pylint: disable=protected-access

_dynamic_slice_mask_info = ring_attention_utils.dynamic_slice_mask_info
_offset_q_sequence_for_kv_shard = ring_attention_utils.offset_q_sequence_for_kv_shard
_has_no_active_blocks = ring_attention_utils.has_no_active_blocks
_has_empty_attention_rows = ring_attention_utils.has_empty_attention_rows
_mask_sparsity = ring_attention_utils.mask_sparsity
_has_axis = ring_attention_utils.has_axis


def _validate_ring_axis_size(ring_axis: str, ring_axis_size: int, expected_ring_size: int) -> None:
  if ring_axis_size != expected_ring_size:
    raise ValueError(
        f"Ring axis {ring_axis} has size {ring_axis_size}, but ring attention "
        f"was built for {expected_ring_size} sequence shards."
    )


def _inject_local_indexer_mask(
    local_mask_info: MaskInfo,
    local_idx_mask: jax.Array | None,
    block_shape: tuple[int | None, int | None] = (128, 128),
    is_dkv: bool = False,
) -> MaskInfo:
  """Injects a pre-sliced dynamic Indexer mask shard into local MaskInfo for the current ring step."""
  if local_idx_mask is None:
    return local_mask_info

  bq, bkv = block_shape
  assert bq is not None and bkv is not None, "block_shape must not be None"
  if local_idx_mask.ndim == 3:
    local_idx_mask = local_idx_mask[0]

  q_len, kv_len = local_idx_mask.shape
  combined_mask = local_idx_mask
  q_blocks = q_len // bq
  kv_blocks = kv_len // bkv
  num_blocks = q_blocks * kv_blocks

  blocks = combined_mask.reshape(q_blocks, bq, kv_blocks, bkv)
  blocks = blocks.swapaxes(1, 2)
  if is_dkv:
    blocks = blocks.swapaxes(0, 1)
    blocks = blocks.swapaxes(-1, -2)
  blocks = blocks.reshape(num_blocks, blocks.shape[-2], blocks.shape[-1])
  blocks = blocks.astype(jnp.int8)

  mask_next = jnp.arange(num_blocks, dtype=jnp.int32)
  return local_mask_info._replace(
      mask_next=mask_next,
      active_rows=None,
      active_cols=None,
      block_mask=None,
      num_active_blocks=None,
      partial_mask_blocks=blocks,
  )


def _ring_attention_forward(
    fwd_mask_info: MaskInfo,
    indexer_mask: jax.Array | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    *,
    sinks: jax.Array | None = None,
    ring_axis: str,
    expected_ring_size: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  if q.shape[-1] != k.shape[-1]:
    raise NotImplementedError("Queries and keys must have the same head dimension.")
  if sinks is not None:
    raise NotImplementedError("Sinks aren't supported yet.")

  ring_axis_size = lax.axis_size(ring_axis)
  _validate_ring_axis_size(ring_axis, ring_axis_size, expected_ring_size)
  ring_axis_idx = lax.axis_index(ring_axis)
  shift = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
  )
  splash_fwd_partial = partial(
      _splash_attention_forward,
      save_residuals=True,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=None,
  )
  o_shape = q.shape[:-1] + (v.shape[-1],)
  o_init = jnp.zeros(o_shape, dtype=jnp.float32)
  l_init = jnp.zeros((o_shape[0], o_shape[1]), jnp.float32)
  m_init = jnp.full_like(l_init, mask_value, dtype=jnp.float32)

  if indexer_mask is not None:
    kv_shard_len = k.shape[-2]
    mask_4d = indexer_mask.reshape(*indexer_mask.shape[:-1], ring_axis_size, kv_shard_len)
  else:
    mask_4d = None

  if mask_4d is not None:
    i_vals = jnp.arange(0, ring_axis_size)
    target_indices = (ring_axis_idx - i_vals) % ring_axis_size
    mask_ring_steps = mask_4d[..., target_indices, :]
    mask_ring_steps = jnp.moveaxis(mask_ring_steps, -2, 0)
    xs = (i_vals, mask_ring_steps)
  else:
    xs = jnp.arange(0, ring_axis_size)

  def body(carry, xs_arg):
    if mask_4d is not None:
      i, local_idx_mask = xs_arg
    else:
      i = xs_arg
      local_idx_mask = None

    m_prev, l_prev, o_prev, k_current, v_current, segment_ids_current = carry
    current_kv_shard_idx = (ring_axis_idx - i) % ring_axis_size
    local_fwd_mask_info = _dynamic_slice_mask_info(fwd_mask_info, current_kv_shard_idx, ring_axis_size)
    local_fwd_mask_info = _offset_q_sequence_for_kv_shard(local_fwd_mask_info, current_kv_shard_idx, k_current.shape[-2])
    local_fwd_mask_info = _inject_local_indexer_mask(
        local_fwd_mask_info,
        local_idx_mask,
        block_shape=(config.block_q, config.block_kv),
        is_dkv=False,
    )
    k_next = shift(k_current)
    v_next = shift(v_current)
    if segment_ids is not None:
      kv_segment_ids_next = shift(segment_ids_current.kv)
      segment_ids_next = SegmentIds(segment_ids.q, kv_segment_ids_next)
    else:
      segment_ids_next = None

    out_curr, stats = splash_fwd_partial(
        local_fwd_mask_info, q, k_current, v_current, segment_ids=segment_ids_current, sinks=sinks,
    )
    no_active_blocks = _has_no_active_blocks(local_fwd_mask_info)
    out_curr = jnp.where(no_active_blocks, jnp.zeros_like(out_curr), out_curr)
    lse_curr = jnp.where(no_active_blocks, jnp.full_like(stats["logsumexp"], mask_value), stats["logsumexp"])
    m_curr = jnp.where(no_active_blocks, jnp.full_like(stats["max_logits"], mask_value), stats["max_logits"])
    empty_rows = _has_empty_attention_rows(lse_curr, m_curr, mask_value)
    l_curr = jnp.exp(lse_curr - m_curr)
    l_curr = jnp.where(empty_rows, jnp.zeros_like(l_curr), l_curr)
    o_curr = out_curr.astype(jnp.float32) * l_curr[..., None]
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    o_next = alpha[..., None] * o_prev + beta[..., None] * o_curr
    return (m_next, l_next, o_next, k_next, v_next, segment_ids_next), None

  initial_carry = (m_init, l_init, o_init, k, v, segment_ids)
  (m_final, l_final, o_final, _, _, _), _ = lax.scan(
      body, initial_carry, xs=xs, length=ring_axis_size, unroll=config.ring_scan_unroll,
  )  # type: ignore[arg-type]
  assert l_final.dtype == jnp.float32
  l_inv = jnp.where(l_final == 0.0, 0.0, 1.0 / l_final)
  out = (o_final * l_inv[..., None]).astype(q.dtype)
  lse = jnp.log(l_final) + m_final
  lse = jnp.where(l_final == 0.0, mask_value, lse)
  return out, (lse, m_final)


def _ring_attention_bwd(
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool,
    ring_axis: str,
    expected_ring_size: int,
    res: Any,
    do: jax.Array,
):
  del save_residuals
  (q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info, indexer_mask) = res
  do = do.astype(jnp.float32)
  if dkv_mask_info is None:
    raise ValueError("Need to specify backward blocks.")

  ring_axis_size = lax.axis_size(ring_axis)
  _validate_ring_axis_size(ring_axis, ring_axis_size, expected_ring_size)
  ring_axis_idx = lax.axis_index(ring_axis)
  shift = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
  )
  use_dq_carry = config.dq_reduction_steps == 3
  dq_accum = jnp.zeros((3, *q.shape), dtype=jnp.float32) if use_dq_carry else jnp.zeros(q.shape, dtype=jnp.float32)
  dk_accum = jnp.zeros(k.shape, dtype=jnp.float32)
  dv_accum = jnp.zeros(v.shape, dtype=jnp.float32)
  dsinks = sinks

  def rotate_kv(k_current, v_current, segment_ids_current):
    k_next = shift(k_current)
    v_next = shift(v_current)
    if segment_ids is not None:
      kv_segment_ids_next = shift(segment_ids_current.kv)
      segment_ids_next = SegmentIds(segment_ids.q, kv_segment_ids_next)
    else:
      segment_ids_next = None
    return k_next, v_next, segment_ids_next

  if indexer_mask is not None:
    kv_shard_len = k.shape[-2]
    mask_4d = indexer_mask.reshape(*indexer_mask.shape[:-1], ring_axis_size, kv_shard_len)
    step0_mask_idx = (ring_axis_idx - 0) % ring_axis_size
    step0_mask = mask_4d[..., step0_mask_idx, :]
    i_vals = jnp.arange(1, ring_axis_size)
    target_indices = (ring_axis_idx - i_vals) % ring_axis_size
    mask_ring_steps = mask_4d[..., target_indices, :]
    mask_ring_steps = jnp.moveaxis(mask_ring_steps, -2, 0)
    xs = (i_vals, mask_ring_steps)
  else:
    mask_4d = None
    step0_mask = None
    xs = jnp.arange(1, ring_axis_size)

  def compute_step(i, local_idx_mask, k_current, v_current, segment_ids_current, dq_accum):
    current_kv_shard_idx = (ring_axis_idx - i) % ring_axis_size
    local_dkv_mask_info = _dynamic_slice_mask_info(dkv_mask_info, current_kv_shard_idx, ring_axis_size)
    local_dkv_mask_info = _offset_q_sequence_for_kv_shard(local_dkv_mask_info, current_kv_shard_idx, k_current.shape[-2])
    local_dkv_mask_info = _inject_local_indexer_mask(
        local_dkv_mask_info,
        local_idx_mask,
        block_shape=(config.block_q_dkv, config.block_kv_dkv),
        is_dkv=True,
    )
    residuals_for_chunk = (
        q, k_current, v_current, segment_ids_current, sinks, out, logsumexp, local_dkv_mask_info,
    )
    attn_bwd = functools.partial(
        _splash_attention_bwd,
        save_residuals=False,
        mask_value=mask_value,
        is_mqa=is_mqa,
        config=config,
        mask_function=mask_function,
        fwd_mask_sparsity=fwd_mask_sparsity,
        dkv_mask_sparsity=dkv_mask_sparsity,
        return_fp32_grads=True,
        return_unreduced_dq=use_dq_carry,
    )
    _, _, dq_i, dk_i, dv_i, _, dsinks, _ = attn_bwd(
        res=residuals_for_chunk,
        grads=do,
        dq_carry_in=dq_accum if use_dq_carry else None,
    )
    if not use_dq_carry:
      dq_i = dq_accum + dq_i.astype(jnp.float32)
    return dq_i, dk_i, dv_i, dsinks

  dq_i, dk_pending, dv_pending, dsinks = compute_step(0, step0_mask, k, v, segment_ids, dq_accum)
  dq_accum = dq_i
  k_current, v_current, segment_ids_current = rotate_kv(k, v, segment_ids)

  def body(carry, xs_arg):
    if mask_4d is not None:
      i, local_idx_mask = xs_arg
    else:
      i = xs_arg
      local_idx_mask = None
    (
        dq_accum, dk_accum, dv_accum, dk_pending, dv_pending, k_current, v_current, segment_ids_current, _,
    ) = carry
    dk_next = shift(dk_accum + dk_pending.astype(jnp.float32))
    dv_next = shift(dv_accum + dv_pending.astype(jnp.float32))
    k_next, v_next, segment_ids_next = rotate_kv(k_current, v_current, segment_ids_current)
    dq_i, dk_i, dv_i, dsinks = compute_step(i, local_idx_mask, k_current, v_current, segment_ids_current, dq_accum)
    dq_accum = dq_i
    return (
        dq_accum, dk_next, dv_next, dk_i, dv_i, k_next, v_next, segment_ids_next, dsinks,
    ), None

  initial_carry = (
      dq_accum, dk_accum, dv_accum, dk_pending, dv_pending, k_current, v_current, segment_ids_current, dsinks,
  )
  (dq, dk, dv, dk_pending, dv_pending, _, _, _, dsinks), _ = lax.scan(
      body,
      initial_carry,
      xs=xs,
      length=ring_axis_size - 1,
      unroll=config.ring_scan_unroll,
  )
  dk = shift(dk + dk_pending.astype(jnp.float32))
  dv = shift(dv + dv_pending.astype(jnp.float32))
  if use_dq_carry:
    dq = dq.sum(axis=0)
  if sinks is not None:
    dsinks = jax.lax.psum(dsinks, axis_name=ring_axis)
  return (
      None,  # fwd_mask_info
      None,  # dkv_mask_info
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      dv.astype(v.dtype),
      None,
      dsinks,
      None,  # indexer_mask
  )


def _ring_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    indexer_mask: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool,
    ring_axis: str,
    expected_ring_size: int,
) -> tuple[jax.Array, RingSplashResidualsType]:
  """Forward pass for the custom VJP of ring attention."""
  del dkv_mask_sparsity
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported.")
  out, (logsumexp, max_logits) = _ring_attention_forward(
      fwd_mask_info,
      indexer_mask,
      q,
      k,
      v,
      segment_ids,
      sinks=sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      ring_axis=ring_axis,
      expected_ring_size=expected_ring_size,
  )
  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(out, name=config.residual_checkpoint_name)
    logsumexp = ad_checkpoint.checkpoint_name(logsumexp, name=config.residual_checkpoint_name)
  residuals = (q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info, indexer_mask)
  return out, residuals


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "mask_value",
        "is_mqa",
        "config",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "save_residuals",
        "ring_axis",
        "expected_ring_size",
    ),
)
def _ring_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    indexer_mask: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool,
    ring_axis: str,
    expected_ring_size: int,
) -> SplashCustomReturnType:
  """Performs ring attention with a custom VJP."""
  del dkv_mask_info, dkv_mask_sparsity
  out, _ = _ring_attention_forward(
      fwd_mask_info,
      indexer_mask,
      q,
      k,
      v,
      segment_ids,
      sinks=sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      ring_axis=ring_axis,
      expected_ring_size=expected_ring_size,
  )
  return out


_ring_attention_custom.defvjp(_ring_attention_fwd, _ring_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "save_residuals",
        "ring_axis",
        "expected_ring_size",
    ],
)
def _ring_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    indexer_mask: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool = False,
    ring_axis: str,
    expected_ring_size: int,
) -> SplashCustomReturnType:
  """Performs ring attention using SplashAttention kernels."""
  if indexer_mask is not None and indexer_mask.dtype != jnp.bool_:
    indexer_mask = jnp.isclose(indexer_mask, 0.0)
  if not _has_axis(ring_axis):
    raise ValueError(f"Ring axis {ring_axis} does not exist")
  return _ring_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      indexer_mask,
      is_mqa=is_mqa,
      config=config,
      mask_value=mask_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
      save_residuals=save_residuals,
      ring_axis=ring_axis,
      expected_ring_size=expected_ring_size,
  )


@jax.tree_util.register_pytree_node_class
class RingSplashAttentionKernel:
  """Implements Ring Attention using SplashAttention for sequence parallelism."""

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      ring_axis: str,
      expected_ring_size: int,
      **kwargs,
  ):
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info
    self.ring_axis = ring_axis
    self.expected_ring_size = expected_ring_size
    self.kwargs = kwargs

  def __call__(self, *args, **kwargs):
    reserved_kwargs = set(self.kwargs) | {"ring_axis", "expected_ring_size"}
    overrides = reserved_kwargs.intersection(kwargs)
    if overrides:
      raise ValueError(
          "Ring attention call-time kwargs cannot override validated kernel " f"configuration: {sorted(overrides)}."
      )
    return _ring_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(
            self.kwargs,
            **kwargs,
            ring_axis=self.ring_axis,
            expected_ring_size=self.expected_ring_size,
        ),
    )

  def manual_sharding_spec(self):
    """Ring attention expects MaskInfo metadata to be sharded by Q shard.

    Each local Q shard slices the current KV shard metadata at each ring step.
    Partial mask blocks stay replicated because they are shared by all shards.
    """
    spec = jax.sharding.PartitionSpec(self.ring_axis)
    _resolve_spec = lambda x: spec if x is not None else None

    def mask_info_spec(mask_info):
      if mask_info is None:
        return None
      return MaskInfo(
          mask_next=_resolve_spec(mask_info.mask_next),  # pyrefly: ignore[bad-argument-type]
          active_rows=_resolve_spec(mask_info.active_rows),  # pyrefly: ignore[bad-argument-type]
          active_cols=_resolve_spec(mask_info.active_cols),  # pyrefly: ignore[bad-argument-type]
          num_active_blocks=_resolve_spec(mask_info.num_active_blocks),  # pyrefly: ignore[bad-argument-type]
          block_mask=_resolve_spec(mask_info.block_mask),  # pyrefly: ignore[bad-argument-type]
          partial_mask_blocks=jax.sharding.PartitionSpec()  # pyrefly: ignore[bad-argument-type]
          if mask_info.partial_mask_blocks is not None
          else None,
          q_sequence=_resolve_spec(mask_info.q_sequence),  # pyrefly: ignore[bad-argument-type]
          kv_sequence=jax.sharding.PartitionSpec()  # pyrefly: ignore[bad-argument-type]
          if mask_info.kv_sequence is not None
          else None,
      )

    return RingSplashAttentionKernel(
        mask_info_spec(self.fwd_mask_info),
        mask_info_spec(self.dkv_mask_info),
        ring_axis=self.ring_axis,
        expected_ring_size=self.expected_ring_size,
        **self.kwargs,
    )

  def tree_flatten(self):
    children = (self.fwd_mask_info, self.dkv_mask_info)
    aux_data = self.kwargs.copy()
    aux_data["ring_axis"] = self.ring_axis
    aux_data["expected_ring_size"] = self.expected_ring_size
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    fwd_mask_info, dkv_mask_info = children
    dkv_mask_info = mask_info_lib.MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    return cls(
        mask_info_lib.MaskInfo(*fwd_mask_info),
        dkv_mask_info,
        **aux_data,
    )


def make_ring_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    ring_axis: str,
    q_seq_shards: int = 1,
    kv_seq_shards: int = 1,
):
  """Creates a RingSplashAttentionKernel."""
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")
  if mask.shape[0] != mask.shape[1]:
    raise NotImplementedError(
        "Ring attention currently supports only square self-attention masks; " f"got shape {mask.shape}."
    )
  if save_residuals:
    raise NotImplementedError("Ring attention does not support save_residuals=True.")
  if q_seq_shards <= 0 or kv_seq_shards <= 0:
    raise ValueError("q_seq_shards and kv_seq_shards must be positive.")
  if q_seq_shards != kv_seq_shards:
    raise NotImplementedError("Ring attention currently requires q_seq_shards == kv_seq_shards.")
  if isinstance(mask, (np.ndarray, mask_lib.NumpyMask)):
    raise NotImplementedError(
        "Ring attention does not support dense NumpyMask yet; use FullMask or " "CausalMask with offset=0."
    )
  if not isinstance(mask, (mask_lib.FullMask, mask_lib.CausalMask)):
    raise NotImplementedError(
        "Only FullMask and CausalMask are supported for ring attention; "
        "other lazy masks are intentionally unsupported until rotated-shard "
        f"semantics are covered. Got {type(mask)}."
    )
  if isinstance(mask, mask_lib.CausalMask) and mask.offset != 0:
    raise NotImplementedError("Ring attention supports CausalMask only with offset=0; " f"got offset={mask.offset}.")

  default_config = SplashConfig.get_default()
  if config is None:
    config = dataclasses.replace(default_config, use_base2_exp=False)
  elif config.use_base2_exp:
    raise NotImplementedError(
        "Ring attention does not support use_base2_exp=True because ring "
        "normalization currently combines natural-log softmax statistics."
    )
  if not config.fuse_reciprocal:
    raise NotImplementedError(
        "Ring attention requires fuse_reciprocal=True because ring "
        "normalization combines already-normalized per-shard outputs."
    )

  process_fn = partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
      kv_seq_shards=kv_seq_shards,
  )
  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
      return_dynamic_grid=True,
  )
  fwd_mask_sparsity = _mask_sparsity(fwd_mask_info)
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  dkv_mask_sparsity = 0.0
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3 and not config.bwd_dkv_megacore,
    )
    assert (mask_function_fwd is None) == (mask_function_dkv is None)
    dkv_mask_sparsity = _mask_sparsity(dkv_mask_info)
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

  return RingSplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      ring_axis=ring_axis,
      expected_ring_size=q_seq_shards,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )
