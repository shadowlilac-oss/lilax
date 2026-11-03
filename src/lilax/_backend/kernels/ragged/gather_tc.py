"""TensorCore ragged gather and gather-reduce kernels."""

import dataclasses
import functools
import logging
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


_RAGGED_GATHER_TC_NUM_BUFFERS = 2


def ragged_gather_tc(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    num_tokens: jax.Array | int,
    max_out_tokens: int,
    block_size: int = 2048,
) -> jax.Array:
  if max_out_tokens <= 0:
    raise ValueError(f"max_out_tokens must be positive, got {max_out_tokens}.")
  if isinstance(num_tokens, int) and not 0 <= num_tokens <= max_out_tokens:
    raise ValueError(f"num_tokens must be in [0, {max_out_tokens}], got {num_tokens}.")
  if block_size <= 0:
    raise ValueError(f"block_size must be positive, got {block_size}.")
  num_indices = indices.shape[0]
  _, dim0, dim1 = x.shape
  if num_indices == 0 or (isinstance(num_tokens, int) and num_tokens == 0):
    return jnp.empty((max_out_tokens, dim0, dim1), dtype=x.dtype)

  block_size = min(block_size, max_out_tokens)
  num_buffers = _RAGGED_GATHER_TC_NUM_BUFFERS
  num_full_blocks = max_out_tokens // block_size
  rem = max_out_tokens % block_size
  num_blocks = num_full_blocks + (1 if rem > 0 else 0)
  idx_align = 128
  idx_window = pl.cdiv(block_size, idx_align) * idx_align + idx_align
  idx_pad = max((-num_indices) % idx_align, idx_window - num_indices)
  if idx_pad > 0:
    indices = jnp.pad(indices, (0, idx_pad))
  max_win_start = indices.shape[0] - idx_window

  def _ragged_gather_kernel(
      scalars_smem_ref,
      idx_hbm_ref,
      x_hbm_ref,
      o_hbm_ref,
      vmem_ref,
      idx_smem_ref,
      idx_sem,
      data_recv_sem,
      data_send_sem,
  ):
    block_idx = pl.program_id(0)
    buf_idx = block_idx % num_buffers
    start_idx = scalars_smem_ref[0]
    num_valid = scalars_smem_ref[1]

    def _run_step(step_idx, b_idx, action: str):
      base = step_idx * block_size
      row0 = start_idx + base
      win_start = jnp.minimum((row0 // idx_align) * idx_align, max_win_start)
      idx_off = row0 - win_start

      def _fetch_indices():
        copy = pltpu.make_async_copy(
            idx_hbm_ref.at[pl.ds(pl.multiple_of(win_start, idx_align), idx_window)],
            idx_smem_ref,
            idx_sem,
        )
        copy.start()
        copy.wait()

      def _chunk(off, size: int, chunk_action: str):
        if chunk_action == "data_start":
          step_unroll = math.gcd(size, 128)

          @pl.loop(0, size // step_unroll)
          def _(chunk_i):
            base_i = off + chunk_i * step_unroll
            src_indices = [idx_smem_ref[idx_off + base_i + u] for u in range(step_unroll)]
            for u in range(step_unroll):
              pltpu.make_async_copy(
                  x_hbm_ref.at[pl.ds(src_indices[u], 1), :, :],
                  vmem_ref.at[b_idx, pl.ds(base_i + u, 1), :, :],
                  data_recv_sem.at[b_idx],
              ).start()

        elif chunk_action == "recv_wait":
          pltpu.make_async_copy(
              o_hbm_ref.at[pl.ds(0, size), :, :],
              vmem_ref.at[b_idx, pl.ds(0, size), :, :],
              data_recv_sem.at[b_idx],
          ).wait()
        else:
          copy = pltpu.make_async_copy(
              vmem_ref.at[b_idx, pl.ds(off, size), :, :],
              o_hbm_ref.at[pl.ds(base + off, size), :, :],
              data_send_sem.at[b_idx],
          )
          if chunk_action == "send_start":
            copy.start()
          else:
            copy.wait()

      def _do_len(length: int):
        step_valid = jnp.minimum(jnp.maximum(num_valid - base, 0), length)

        def _over_valid_rows(chunk_action: str):
          @pl.when(step_valid == length)
          def _():
            _chunk(0, length, chunk_action)

          @pl.when(step_valid < length)
          def _():
            off = 0
            for k in reversed(range(length.bit_length())):
              size = 1 << k
              bit = step_valid & size

              def _chunk_if_bit_set(off=off, size=size):
                _chunk(off, size, chunk_action)

              pl.when(bit != 0)(_chunk_if_bit_set)
              off = off + bit

        if action == "data_start":
          @pl.when(step_valid > 0)
          def _():
            _fetch_indices()

          _over_valid_rows("data_start")
        elif action == "copy_out_start":
          _over_valid_rows("recv_wait")
          _over_valid_rows("send_start")
        elif action == "copy_out_wait":
          _over_valid_rows("send_wait")

      if rem == 0:
        _do_len(block_size)
      elif isinstance(step_idx, int):
        _do_len(rem if step_idx == num_blocks - 1 else block_size)
      else:
        jax.lax.cond(step_idx == num_blocks - 1, lambda: _do_len(rem), lambda: _do_len(block_size))

    @pl.when(block_idx == 0)
    def _prologue():
      _run_step(0, 0, "data_start")

    _run_step(block_idx, buf_idx, "copy_out_start")
    num_steps = pl.num_programs(0)
    next_data_block = block_idx + 1

    @pl.when(next_data_block < num_steps)
    def _queue_next():
      next_data_buf = next_data_block % num_buffers

      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        _run_step(block_idx - 1, next_data_buf, "copy_out_wait")

      _run_step(next_data_block, next_data_buf, "data_start")

    @pl.when(block_idx == num_steps - 1)
    def _epilogue():
      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        prev_block = block_idx - 1
        _run_step(prev_block, prev_block % num_buffers, "copy_out_wait")

      _run_step(block_idx, buf_idx, "copy_out_wait")

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct((max_out_tokens, dim0, dim1), x.dtype, manual_axis_type=manual_axis_type)
  scalars = jnp.stack([jnp.asarray(start, dtype=jnp.int32), jnp.asarray(num_tokens, dtype=jnp.int32)])
  num_steps = pl.cdiv(num_tokens if isinstance(num_tokens, int) else scalars[1], block_size)
  return pl.pallas_call(
      _ragged_gather_kernel,
      out_shape=out_shape,
      grid=(num_steps,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
      scratch_shapes=[
          pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
          pltpu.SMEM((idx_window,), indices.dtype),
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA((num_buffers,)),
          pltpu.SemaphoreType.DMA((num_buffers,)),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)),
      name="ragged_gather_tc",
  )(scalars, indices, x)


_RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS = 4
_RAGGED_GATHER_REDUCE_TC_LOOKAHEAD = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS - 2
_RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL = 64
_RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL = 8
_RAGGED_GATHER_REDUCE_TC_ALIGN = 128
_RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK = 64
_RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS = 4
_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT = 64 * 1024 * 1024
_RAGGED_GATHER_REDUCE_TC_VMEM_BUDGET = 56 * 1024 * 1024


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RaggedGatherReduceMetadata:
  src_rows: jax.Array
  tokens: jax.Array
  blocks: jax.Array
  block_size: int = jax.tree.static()
  num_indices: int = jax.tree.static()


def _ragged_gather_reduce_tc_layout(
    num_indices: int,
    top_k: int,
    block_size: int,
    num_out_tokens: int | None = None,
) -> tuple[int, int, int]:
  if num_out_tokens is None:
    num_out_tokens = num_indices // top_k
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  max_blocks = pl.cdiv(num_indices + num_out_tokens, block_size - top_k)
  window = pl.cdiv(block_size, align) * align + align
  slots_len = pl.cdiv(max(num_indices, window), align) * align
  return max_blocks, window, slots_len


def _check_ragged_gather_reduce_tc_args(top_k: int, block_size: int):
  if top_k <= 0:
    raise ValueError(f"top_k must be positive, got {top_k}.")
  if block_size <= top_k:
    raise ValueError(
        f"block_size must exceed top_k ({top_k}) so that a block fits a token" f" and its rows, got {block_size}."
    )


@functools.cache
def _warn_packed_sort_fallback(n: int, max_key: int, pos_bits: int, key_bits: int) -> None:
  logging.warning(
      "packed_stable_sort: %d position bits (n=%d) + %d key bits (max_key=%d) > 32; "
      "falling back to the (key, position) tuple sort.",
      pos_bits,
      n,
      key_bits,
      max_key,
  )


def packed_stable_sort(key: jax.Array, max_key: int) -> tuple[jax.Array, jax.Array]:
  n = key.shape[0]
  positions = jnp.arange(n, dtype=jnp.int32)
  pos_bits = max(1, (n - 1).bit_length())
  key_bits = max(1, int(max_key).bit_length())
  if pos_bits + key_bits > 32:
    _warn_packed_sort_fallback(n, int(max_key), pos_bits, key_bits)
    sorted_key, sorted_positions = jax.lax.sort((key.astype(jnp.int32), positions), num_keys=1, is_stable=True)
    return sorted_key, sorted_positions
  shift = jnp.uint32(pos_bits)
  packed_key = (key.astype(jnp.uint32) << shift) | positions.astype(jnp.uint32)
  sorted_packed = jax.lax.sort(packed_key, is_stable=False)
  sorted_positions = (sorted_packed & jnp.uint32((1 << pos_bits) - 1)).astype(jnp.int32)
  return (sorted_packed >> shift).astype(jnp.int32), sorted_positions


def ragged_gather_reduce_tc_metadata(
    indices: jax.Array,
    start: jax.Array | int,
    num_tokens: jax.Array | int,
    *,
    top_k: int,
    block_size: int = 896,
    num_out_tokens: int | None = None,
    packed_sort: bool = False,
) -> RaggedGatherReduceMetadata:
  _check_ragged_gather_reduce_tc_args(top_k, block_size)
  num_indices = indices.shape[0]
  if num_out_tokens is None:
    if num_indices % top_k:
      raise ValueError(f"indices length ({num_indices}) must be divisible by top_k ({top_k}).")
    num_out_tokens = num_indices // top_k
  if isinstance(num_tokens, int) and num_tokens < 0:
    raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
  if isinstance(start, int) and start < 0:
    raise ValueError(f"start must be non-negative, got {start}.")
  if isinstance(start, int) and isinstance(num_tokens, int) and start + num_tokens > num_indices:
    raise ValueError(f"start + num_tokens ({start + num_tokens}) must not exceed indices" f" length ({num_indices}).")
  max_blocks, _, slots_len = _ragged_gather_reduce_tc_layout(
      num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  if num_out_tokens == 0:
    empty_slots = jnp.zeros((slots_len,), jnp.int32)
    return RaggedGatherReduceMetadata(
        src_rows=empty_slots,
        tokens=empty_slots,
        blocks=jnp.zeros((1,), jnp.int32),
        block_size=block_size,
        num_indices=num_indices,
    )

  start = jnp.asarray(start, dtype=jnp.int32)
  num_tokens = jnp.asarray(num_tokens, dtype=jnp.int32)
  positions = jnp.arange(num_indices, dtype=jnp.int32)
  rel = positions - start
  valid = (rel >= 0) & (rel < num_tokens)
  key = jnp.where(valid, indices, num_out_tokens).astype(jnp.int32)
  if packed_sort:
    sorted_key, sorted_positions = packed_stable_sort(key, num_out_tokens)
  else:
    sorted_key, sorted_positions = jax.lax.sort((key, positions), num_keys=1, is_stable=True)
  pad = slots_len - num_indices
  src_rows = jnp.pad(sorted_positions - start, (0, pad))
  tokens = jnp.pad(sorted_key, (0, pad))

  is_run_start = jnp.concatenate([jnp.ones((1,), bool), sorted_key[1:] != sorted_key[:-1]])
  run_start = jax.lax.cummax(jnp.where(is_run_start, positions, 0))
  int_max = jnp.iinfo(jnp.int32).max
  slot_cost = jnp.where(sorted_key < num_out_tokens, run_start + sorted_key, int_max)
  chunk = _RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK
  num_chunks = pl.cdiv(num_indices, chunk)
  cost_chunks = jnp.pad(slot_cost, (0, num_chunks * chunk - num_indices), constant_values=int_max).reshape(
      num_chunks, chunk
  )
  threshold = jnp.arange(max_blocks + 1, dtype=jnp.int32) * (block_size - top_k)
  full_chunks = jnp.sum(cost_chunks[:, -1] < threshold[:, None], axis=1, dtype=jnp.int32)
  row = jnp.minimum(full_chunks, num_chunks - 1)
  in_chunk = jnp.sum(cost_chunks[row] < threshold[:, None], axis=1, dtype=jnp.int32)
  slot_start = row * chunk + in_chunk
  prev_key = jnp.where(slot_start > 0, sorted_key[jnp.maximum(slot_start - 1, 0)], -1)
  next_key = jnp.where(
      slot_start < num_indices,
      sorted_key[jnp.minimum(slot_start, num_indices - 1)],
      num_out_tokens,
  )
  token_start = jnp.clip(threshold - slot_start, prev_key + 1, next_key)
  table = jnp.stack(
      [slot_start[:-1], jnp.diff(slot_start), token_start[:-1], jnp.diff(token_start)], axis=1
  )
  num_blocks = jnp.sum(token_start[:-1] < num_out_tokens, dtype=jnp.int32)
  blocks = jnp.concatenate([num_blocks[None], table.reshape(-1)])
  return RaggedGatherReduceMetadata(
      src_rows=src_rows,
      tokens=tokens,
      blocks=blocks.astype(jnp.int32),
      block_size=block_size,
      num_indices=num_indices,
  )


def ragged_gather_reduce_tc(
    x: jax.Array,
    metadata: RaggedGatherReduceMetadata,
    *,
    num_out_tokens: int,
    top_k: int,
) -> jax.Array:
  max_input_tokens, dim0, dim1 = x.shape
  block_size = metadata.block_size
  _check_ragged_gather_reduce_tc_args(top_k, block_size)
  if num_out_tokens < 0:
    raise ValueError(f"num_out_tokens must be non-negative, got {num_out_tokens}.")
  num_buffers = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  fields = _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS
  max_blocks, window, slots_len = _ragged_gather_reduce_tc_layout(
      metadata.num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  expected_shapes = ((slots_len,), (slots_len,), (1 + fields * max_blocks,))
  shapes = (
      metadata.src_rows.shape,
      metadata.tokens.shape,
      metadata.blocks.shape,
  )
  if shapes != expected_shapes:
    raise ValueError(
        "metadata (src_rows, tokens, blocks) must have shapes"
        f" {expected_shapes}, got {shapes}; was it derived for"
        f" num_out_tokens={num_out_tokens} with top_k={top_k}?"
    )
  itemsize = jnp.dtype(x.dtype).itemsize
  sublanes = 8 * max(1, 4 // itemsize)
  token_vmem_bytes = pl.cdiv(dim0, sublanes) * sublanes * pl.cdiv(dim1, 128) * 128 * itemsize
  vmem_budget = _RAGGED_GATHER_REDUCE_TC_VMEM_BUDGET
  if num_buffers * block_size * token_vmem_bytes > vmem_budget:
    raise ValueError(
        f"block_size {block_size} does not fit {num_buffers} VMEM buffers of"
        f" ({dim0}, {dim1}) {x.dtype} tokens in {vmem_budget} bytes; use"
        f" block_size <= {vmem_budget // (num_buffers * token_vmem_bytes)}."
    )
  if num_out_tokens == 0 or max_input_tokens == 0:
    return jnp.zeros((num_out_tokens, dim0, dim1), x.dtype)

  lookahead = _RAGGED_GATHER_REDUCE_TC_LOOKAHEAD
  ring = lookahead + 2

  def _ragged_gather_reduce_kernel(
      blocks_smem_ref,
      src_hbm_ref,
      tok_hbm_ref,
      x_hbm_ref,
      o_hbm_ref,
      vmem_ref,
      src_smem_ref,
      tok_smem_ref,
      recv_sem,
      send_sem,
      slots_sem,
  ):
    num_blocks = blocks_smem_ref[0]

    def _block(blk):
      entry = 1 + fields * blk
      return tuple(blocks_smem_ref[entry + f] for f in range(fields))

    def _window_start(slot_start):
      return jnp.minimum((slot_start // align) * align, slots_len - window)

    def _slots_loads(blk):
      r = blk % ring
      window_start = pl.multiple_of(_window_start(_block(blk)[0]), align)
      return [
          pltpu.make_async_copy(
              hbm_ref.at[pl.ds(window_start, window)],
              smem_ref.at[pl.ds(r * window, window)],
              slots_sem.at[r],
          )
          for hbm_ref, smem_ref in (
              (src_hbm_ref, src_smem_ref),
              (tok_hbm_ref, tok_smem_ref),
          )
      ]

    def _slot_base(blk):
      slot_start = _block(blk)[0]
      return (blk % ring) * window + slot_start - _window_start(slot_start)

    def _pow2_chunks(n, fn):
      off = 0
      for k in reversed(range(block_size.bit_length())):
        size = 1 << k
        bit = n & size

        def _fn_if_bit_set(off=off, size=size):
          fn(off, size)

        pl.when(bit != 0)(_fn_if_bit_set)
        off = off + bit

    def _wait_rows(b, n, sem):
      def _wait(off, size):
        del off
        rows = vmem_ref.at[b, pl.ds(0, size), :, :]
        pltpu.make_async_copy(rows, rows, sem.at[b]).wait()

      _pow2_chunks(n, _wait)

    def _dynamic_fori(lo, hi, unroll, body, carry):
      def _chunk(c, carry):
        for u in range(unroll):
          carry = body(lo + c * unroll + u, carry)
        return carry

      num_full = (hi - lo) // unroll
      carry = jax.lax.fori_loop(0, num_full, _chunk, carry)
      return jax.lax.fori_loop(lo + num_full * unroll, hi, body, carry)

    def _issuer(blk):
      b = blk % num_buffers
      num_tokens = _block(blk)[3]
      base = _slot_base(blk)

      def _issue(i, carry):
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(src_smem_ref[base + i], 1), :, :],
            vmem_ref.at[b, pl.ds(num_tokens + i, 1), :, :],
            recv_sem.at[b],
        ).start()
        return carry

      return _issue

    def _in_start(blk):
      _dynamic_fori(0, _block(blk)[1], _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL, _issuer(blk), ())

    def _in_wait(blk):
      _wait_rows(blk % num_buffers, _block(blk)[1], recv_sem)

    def _reducer(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)

      @pl.loop(0, num_tokens)
      def _zero_token(t):
        vmem_ref[b, t, :, :] = jnp.zeros((dim0, dim1), x.dtype)

      base = _slot_base(blk)

      def _reduce(i, carry):
        prev_t, total = carry
        t = tok_smem_ref[base + i] - token_start
        row = vmem_ref[b, num_tokens + i, :, :].astype(jnp.float32)
        total = jnp.where(t == prev_t, total, 0.0) + row
        vmem_ref[b, t, :, :] = total.astype(x.dtype)
        return t, total

      return _reduce

    def _out_start(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)

      def _store(off, size):
        pltpu.make_async_copy(
            vmem_ref.at[b, pl.ds(off, size), :, :],
            o_hbm_ref.at[pl.ds(token_start + off, size), :, :],
            send_sem.at[b],
        ).start()

      _pow2_chunks(num_tokens, _store)

    def _out_wait(blk):
      _wait_rows(blk % num_buffers, _block(blk)[3], send_sem)

    def _slots_wait(blk):
      for load in _slots_loads(blk):
        load.wait()

    for blk in range(ring):
      @pl.when(blk < num_blocks)
      def _load_first_slots(blk=blk):
        for load in _slots_loads(blk):
          load.start()

    for blk in range(lookahead):
      @pl.when(blk < num_blocks)
      def _start_first_rows(blk=blk):
        _slots_wait(blk)
        _in_start(blk)

    @pl.loop(0, num_blocks)
    def _step(i):
      _in_wait(i)
      nxt = i + lookahead
      has_nxt = nxt < num_blocks

      @pl.when(nxt >= num_buffers)
      def _free_buffer():
        _out_wait(nxt - num_buffers)

      @pl.when(has_nxt)
      def _wait_next_slots():
        _slots_wait(nxt)

      nxt = jnp.minimum(nxt, num_blocks - 1)
      num_reduce = _block(i)[1]
      num_issue = jnp.where(has_nxt, _block(nxt)[1], 0)
      num_both = jnp.minimum(num_reduce, num_issue)
      issue = _issuer(nxt)
      reduce = _reducer(i)

      def _issue_and_reduce(j, carry):
        issue(j, ())
        return reduce(j, carry)

      unroll = _RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL
      carry = (jnp.int32(-1), jnp.zeros((dim0, dim1), jnp.float32))
      carry = _dynamic_fori(0, num_both, unroll, _issue_and_reduce, carry)
      _dynamic_fori(num_both, num_reduce, unroll, reduce, carry)
      _dynamic_fori(num_both, num_issue, _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL, issue, ())
      _out_start(i)

      @pl.when(i + ring < num_blocks)
      def _load_later_slots():
        for load in _slots_loads(i + ring):
          load.start()

    for k in range(num_buffers - lookahead):
      @pl.when(num_blocks - 1 - k >= 0)
      def _wait_last_stores(k=k):
        _out_wait(num_blocks - 1 - k)

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct((num_out_tokens, dim0, dim1), x.dtype, manual_axis_type=manual_axis_type)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  dma_sems = pltpu.SemaphoreType.DMA((num_buffers,))
  return pl.pallas_call(
      _ragged_gather_reduce_kernel,
      out_shape=out_shape,
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          hbm_spec,
          hbm_spec,
          hbm_spec,
      ],
      out_specs=hbm_spec,
      scratch_shapes=[
          pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
          pltpu.SMEM((ring * window,), jnp.int32),
          pltpu.SMEM((ring * window,), jnp.int32),
          dma_sems,
          dma_sems,
          pltpu.SemaphoreType.DMA((ring,)),
      ],
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT),
      name="ragged_gather_reduce_tc",
  )(metadata.blocks, metadata.src_rows, metadata.tokens, x)
