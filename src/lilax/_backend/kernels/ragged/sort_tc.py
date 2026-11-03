"""Ring-of-experts ragged sort / unsort on TensorCore."""

import dataclasses
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lilax._backend.kernels.ragged import gather_tc as tc


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TcRouting:
  token_ids_sorted: jax.Array
  slot_order: jax.Array
  start: jax.Array
  count: jax.Array
  reduce_meta: tc.RaggedGatherReduceMetadata
  num_out_tokens: int = jax.tree.static()
  windowed: bool = jax.tree.static()


def _to3d(x):
  n, d = x.shape
  if d % 128:
    raise ValueError(f"TC ragged sort needs hidden % 128 == 0, got {d}.")
  return x.reshape(n, d // 128, 128)


def _to2d(x):
  return x.reshape(x.shape[0], x.shape[1] * x.shape[2])


def _manual_axis_type(x):
  return getattr(jax.typeof(x), "manual_axis_type", None)


def _flatten_kernel(n_ref, x_ref, o_ref, *, block, zero_fill):
  if not zero_fill:
    o_ref[...] = x_ref[...].reshape(o_ref.shape)
    return
  start = pl.program_id(0) * block
  n = n_ref[0]

  @pl.when(start + block <= n)
  def _():
    o_ref[...] = x_ref[...].reshape(o_ref.shape)

  @pl.when(start >= n)
  def _():
    o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

  @pl.when(jnp.logical_and(start < n, start + block > n))
  def _():
    y = x_ref[...].reshape(o_ref.shape)
    rows = start + jax.lax.broadcasted_iota(jnp.int32, o_ref.shape, 0)
    o_ref[...] = jnp.where(rows < n, y, jnp.zeros((), y.dtype))


def _tc_flatten(x, num_rows, block, zero_fill=False):
  rows, d0, d1 = x.shape
  d = d0 * d1
  block = min(block, rows)
  n = jnp.reshape(jnp.asarray(num_rows, jnp.int32), (1,))
  if zero_fill:
    grid = (pl.cdiv(rows, block),)

    def in_map(i, n_ref):
      return (jnp.minimum(i, jnp.maximum(pl.cdiv(n_ref[0], block) - 1, 0)), 0, 0)

  else:
    grid = (pl.cdiv(n[0], block),)

    def in_map(i, n_ref):
      del n_ref
      return (i, 0, 0)

  return pl.pallas_call(
      functools.partial(_flatten_kernel, block=block, zero_fill=zero_fill),
      out_shape=jax.ShapeDtypeStruct((rows, d), x.dtype, manual_axis_type=_manual_axis_type(x)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=grid,
          in_specs=[pl.BlockSpec((block, d0, d1), in_map)],
          out_specs=pl.BlockSpec((block, d), lambda i, n_ref: (i, 0)),
      ),
      name="ragged_flatten_tc",
  )(n, x)


def _unflatten_kernel(n_ref, x_ref, o_ref):
  del n_ref
  o_ref[...] = x_ref[...].reshape(o_ref.shape)


def _tc_unflatten(x, num_rows, block):
  rows, d = x.shape
  if d % 128:
    raise ValueError(f"TC ragged sort needs hidden % 128 == 0, got {d}.")
  d0, d1 = d // 128, 128
  block = min(block, rows)
  n = jnp.reshape(jnp.asarray(num_rows, jnp.int32), (1,))
  return pl.pallas_call(
      _unflatten_kernel,
      out_shape=jax.ShapeDtypeStruct((rows, d0, d1), x.dtype, manual_axis_type=_manual_axis_type(x)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(pl.cdiv(n[0], block),),
          in_specs=[pl.BlockSpec((block, d), lambda i, n_ref: (i, 0))],
          out_specs=pl.BlockSpec((block, d0, d1), lambda i, n_ref: (i, 0, 0)),
      ),
      name="ragged_unflatten_tc",
  )(n, x)


def _keep_3d(blocks):
  return len(blocks) > 2 and bool(blocks[2])


def _tokens_3d(blocks):
  return len(blocks) > 3 and bool(blocks[3])


def _gather(x2d, routing, cap, blocks, mask_padding):
  block_size, flat_block = blocks[:2]
  if x2d.ndim == 3:
    x3d = x2d
  elif flat_block:
    x3d = _tc_unflatten(x2d, x2d.shape[0], flat_block)
  else:
    x3d = _to3d(x2d)
  out = tc.ragged_gather_tc(
      x3d,
      routing.token_ids_sorted,
      start=routing.start,
      num_tokens=routing.count,
      max_out_tokens=cap,
      block_size=block_size,
  )
  if _keep_3d(blocks):
    if mask_padding:
      valid = jnp.arange(cap, dtype=jnp.int32) < routing.count
      out = jnp.where(valid[:, None, None], out, jnp.zeros((), out.dtype))
    return out
  if flat_block:
    return _tc_flatten(out, routing.count, flat_block, zero_fill=mask_padding)
  out = _to2d(out)
  if mask_padding:
    valid = jnp.arange(cap, dtype=jnp.int32) < routing.count
    out = jnp.where(valid[:, None], out, jnp.zeros((), out.dtype))
  return out


def _gather_reduce(buf2d, routing, num_out_tokens, topk, blocks):
  flat_block = blocks[1]
  if buf2d.ndim == 3:
    buf3d = buf2d
  elif flat_block:
    buf3d = _tc_unflatten(buf2d, routing.count, flat_block)
  else:
    buf3d = _to3d(buf2d)
  out = tc.ragged_gather_reduce_tc(buf3d, routing.reduce_meta, num_out_tokens=num_out_tokens, top_k=topk)
  if _tokens_3d(blocks):
    return out
  if flat_block:
    return _tc_flatten(out, num_out_tokens, flat_block)
  return _to2d(out)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def tc_sort_gather(x2d, routing, cap, topk, blocks, mask_padding):
  return _gather(x2d, routing, cap, blocks, mask_padding)


def _tc_sort_gather_fwd(x2d, routing, cap, topk, blocks, mask_padding):
  return _gather(x2d, routing, cap, blocks, mask_padding), (routing, x2d.shape[0])


def _tc_sort_gather_bwd(cap, topk, blocks, mask_padding, res, g):
  del cap, mask_padding
  routing, n_tokens = res
  return _gather_reduce(g, routing, n_tokens, topk, blocks), None


tc_sort_gather.defvjp(_tc_sort_gather_fwd, _tc_sort_gather_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def tc_unsort_reduce(buf2d, routing, num_out_tokens, topk, blocks, mask_padding):
  return _gather_reduce(buf2d, routing, num_out_tokens, topk, blocks)


def _tc_unsort_reduce_fwd(buf2d, routing, num_out_tokens, topk, blocks, mask_padding):
  return _gather_reduce(buf2d, routing, num_out_tokens, topk, blocks), (routing, buf2d.shape[0])


def _tc_unsort_reduce_bwd(num_out_tokens, topk, blocks, mask_padding, res, g):
  del num_out_tokens, topk
  routing, cap = res
  return _gather(g, routing, cap, blocks, mask_padding), None


tc_unsort_reduce.defvjp(_tc_unsort_reduce_fwd, _tc_unsort_reduce_bwd)


def ring_ragged_sort_tc(
    hidden_states_local,
    topk_indices_local,
    num_experts,
    topk,
    ep_name,
    ep_size,
    buffer_size,
    gather_block_size=1024,
    reduce_block_size=896,
    mask_padding=True,
    flatten_block_size=0,
    tag_routing_fn=None,
    keep_3d=False,
    window_routing=False,
):
  num_tokens_local = hidden_states_local.shape[0]
  topk_indices_flat = topk_indices_local.flatten().astype(jnp.int32)
  n = topk_indices_flat.shape[0]
  if window_routing:
    _, topk_argsort_indices = tc.packed_stable_sort(topk_indices_flat, num_experts - 1)
  else:
    topk_argsort_indices = jnp.argsort(topk_indices_flat, stable=True).astype(jnp.int32)
  group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)

  shard_idx = jax.lax.axis_index(ep_name) if ep_size > 1 else 0
  local_num_experts = num_experts // ep_size
  group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
  start = group_offsets[shard_idx * local_num_experts].astype(jnp.int32)
  end = group_offsets[(shard_idx + 1) * local_num_experts].astype(jnp.int32)
  cap = n if buffer_size is None else min(buffer_size, n)
  count = jnp.minimum(end - start, cap).astype(jnp.int32)

  if window_routing:
    topk_argsort_revert_indices = topk_argsort_indices
    slot_order = jax.lax.dynamic_slice_in_dim(jnp.pad(topk_argsort_indices, (0, cap)), start, cap, axis=0)
    token_ids_sorted = jax.lax.stop_gradient(slot_order // topk)
    reduce_meta = tc.ragged_gather_reduce_tc_metadata(
        token_ids_sorted,
        0,
        count,
        top_k=topk,
        block_size=reduce_block_size,
        num_out_tokens=num_tokens_local,
        packed_sort=True,
    )
    start = jnp.int32(0)
  else:
    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices).astype(jnp.int32)
    slot_order = topk_argsort_indices
    token_ids_sorted = jax.lax.stop_gradient(topk_argsort_indices // topk)
    reduce_meta = tc.ragged_gather_reduce_tc_metadata(
        token_ids_sorted, start, count, top_k=topk, block_size=reduce_block_size
    )
  routing = TcRouting(
      token_ids_sorted=token_ids_sorted,
      slot_order=slot_order,
      start=start,
      count=count,
      reduce_meta=reduce_meta,
      num_out_tokens=num_tokens_local,
      windowed=window_routing,
  )
  if tag_routing_fn is not None:
    routing = jax.tree.map(tag_routing_fn, routing)
    group_sizes_local = tag_routing_fn(group_sizes_local)
    topk_argsort_revert_indices = tag_routing_fn(topk_argsort_revert_indices)
  tokens_3d = hidden_states_local.ndim == 3
  x = tc_sort_gather(
      hidden_states_local,
      routing,
      cap,
      topk,
      (gather_block_size, flatten_block_size, keep_3d, tokens_3d),
      mask_padding,
  )
  return x, group_sizes_local, topk_argsort_revert_indices, routing


def tc_buffer_row_weights(routing, topk_weights, cap):
  w_flat = topk_weights.astype(jnp.float32)
  iota = jnp.arange(cap, dtype=jnp.int32)
  valid = iota < routing.count
  if routing.windowed:
    rows = jnp.where(valid, routing.slot_order, w_flat.shape[0] + iota)
    return w_flat.at[rows].get(mode="fill", fill_value=0.0, unique_indices=True)
  padded_order = jnp.pad(routing.slot_order, (0, cap))
  rows = jax.lax.dynamic_slice_in_dim(padded_order, routing.start, cap, axis=0)
  return jnp.where(valid, jnp.take(w_flat, jnp.where(valid, rows, 0), axis=0), 0.0)


def ring_ragged_unsort_tc(
    sorted_tokens_local,
    routing,
    topk,
    topk_weights,
    gather_block_size=1024,
    mask_padding=True,
    flatten_block_size=0,
    prescaled=False,
    out_3d=False,
):
  cap = sorted_tokens_local.shape[0]
  num_out_tokens = routing.num_out_tokens
  if prescaled:
    scaled = sorted_tokens_local
  else:
    w_rows = tc_buffer_row_weights(routing, topk_weights, cap)
    w_rows = w_rows.reshape(-1, *([1] * (sorted_tokens_local.ndim - 1)))
    scaled = (sorted_tokens_local.astype(jnp.float32) * w_rows).astype(sorted_tokens_local.dtype)
  keep_3d = sorted_tokens_local.ndim == 3
  return tc_unsort_reduce(
      scaled,
      routing,
      num_out_tokens,
      topk,
      (gather_block_size, flatten_block_size, keep_3d, out_3d),
      mask_padding,
  )
