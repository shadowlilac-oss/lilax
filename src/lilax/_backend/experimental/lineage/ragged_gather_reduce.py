"""TensorCore Pallas scatter-reduce of ragged gathered rows into output tokens."""

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import ragged_gather_reduce_metadata

_RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS = 4
_RAGGED_GATHER_REDUCE_TC_LOOKAHEAD = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS - 2
_RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL = 64
_RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL = 16
_RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS = 4
_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT = 64 * 1024 * 1024
_RAGGED_GATHER_REDUCE_TC_VMEM_BUDGET = 56 * 1024 * 1024


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_gather_reduce_tc(
    x: jt.Num[jax.Array, "max_input_tokens d0 d1"],
    metadata: ragged_gather_reduce_metadata.RaggedGatherReduceMetadata,
    *,
    num_out_tokens: int,
    top_k: int,
    out: jt.Num[jax.Array, "num_out_tokens d0 d1"] | None = None,
    zero_initialized: bool | None = None,
) -> jt.Num[jax.Array, "num_out_tokens d0 d1"]:
  """Scatter-reduce the valid rows of a ragged gather into output tokens.

  Metadata supplies the routing and output blocks. Duplicate destinations are
  summed in float32 in buffer order. The caller must ensure that the number of
  valid rows does not exceed the input buffer size.
  """
  max_input_tokens, dim0, dim1 = x.shape
  block_size = metadata.block_size
  ragged_gather_reduce_metadata.check_ragged_gather_reduce_tc_args(top_k, block_size)
  if num_out_tokens < 0:
    raise ValueError(f"num_out_tokens must be non-negative, got {num_out_tokens}.")
  if out is not None:
    if out.shape != (num_out_tokens, dim0, dim1):
      raise ValueError(f"out must have shape {(num_out_tokens, dim0, dim1)}, got" f" {out.shape}.")
    if out.dtype != x.dtype:
      raise ValueError(f"out must have dtype {x.dtype}, got {out.dtype}.")
  if zero_initialized is None:
    zero_initialized = out is None

  num_buffers = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS
  align = ragged_gather_reduce_metadata.RAGGED_GATHER_REDUCE_TC_ALIGN
  fields = _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS
  max_blocks, window, slots_len = ragged_gather_reduce_metadata.ragged_gather_reduce_tc_layout(
      num_out_tokens * top_k, top_k, block_size
  )
  expected_shapes = ((slots_len,), (slots_len,), (1 + fields * max_blocks,))
  shapes = (metadata.src_rows.shape, metadata.tokens.shape, metadata.blocks.shape)
  if shapes != expected_shapes:
    raise ValueError(
        "metadata (src_rows, tokens, blocks) must have shapes"
        f" {expected_shapes}, got {shapes}; was it derived from"
        f" {num_out_tokens * top_k} indices with top_k={top_k}?"
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
    if out is not None and not zero_initialized:
      return out
    return jnp.zeros((num_out_tokens, dim0, dim1), x.dtype)
  if not zero_initialized and out is None:
    out = jnp.zeros((num_out_tokens, dim0, dim1), x.dtype)

  lookahead = _RAGGED_GATHER_REDUCE_TC_LOOKAHEAD
  ring = lookahead + 2
  has_out_in = out is not None

  def _ragged_gather_reduce_kernel(*args):
    if has_out_in:
      (blocks_smem_ref, src_hbm_ref, tok_hbm_ref, x_hbm_ref, o_in_hbm_ref, o_hbm_ref,
       vmem_ref, src_smem_ref, tok_smem_ref, recv_sem, send_sem, slots_sem) = args
    else:
      (blocks_smem_ref, src_hbm_ref, tok_hbm_ref, x_hbm_ref, o_hbm_ref, vmem_ref,
       src_smem_ref, tok_smem_ref, recv_sem, send_sem, slots_sem) = args
      o_in_hbm_ref = None

    num_blocks = blocks_smem_ref[0]

    def _block(blk):
      return (
          blocks_smem_ref[1 + blk], blocks_smem_ref[1 + max_blocks + blk],
          blocks_smem_ref[1 + 2 * max_blocks + blk], blocks_smem_ref[1 + 3 * max_blocks + blk],
      )

    def _window_start(slot_start):
      return jnp.minimum((slot_start // align) * align, slots_len - window)

    def _slots_loads(blk):
      r = blk % ring
      window_start = pl.multiple_of(_window_start(_block(blk)[0]), align)
      return [
          pltpu.make_async_copy(
              hbm_ref.at[pl.ds(window_start, window)], smem_ref.at[pl.ds(r * window, window)], slots_sem.at[r],
          )
          for hbm_ref, smem_ref in ((src_hbm_ref, src_smem_ref), (tok_hbm_ref, tok_smem_ref))
      ]

    def _slot_base(blk):
      slot_start = _block(blk)[0]
      return (blk % ring) * window + slot_start - _window_start(slot_start)

    def _pow2_chunks(n, fn):
      off = 0
      for k in reversed(range(block_size.bit_length())):
        size = 1 << k
        bit = n & size

        @pl.when(bit != 0)
        def _(off=off, size=size):
          fn(off, size)

        off = off + bit

    def _unrolled_chunks(n, max_unroll: int, fn):
      def _do_pow2(off, size):
        unroll = min(size, max_unroll)
        if size == unroll:
          fn(off, unroll)
        else:
          @pl.loop(0, size // unroll)
          def _(c):
            fn(off + c * unroll, unroll)

      _pow2_chunks(n, _do_pow2)

    def _wait_rows(b, n, sem):
      def _wait(off, size):
        del off
        rows = vmem_ref.at[b, pl.ds(0, size), :, :]
        pltpu.make_async_copy(rows, rows, sem.at[b]).wait()

      _pow2_chunks(n, _wait)

    def _in_out_start(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)
      out_base = block_size - num_tokens
      if zero_initialized:
        def _zero_out(off, size):
          vmem_ref[b, pl.ds(out_base + off, size), :, :] = jnp.zeros((size, dim0, dim1), x.dtype)

        _unrolled_chunks(num_tokens, 256, _zero_out)
      else:
        assert o_in_hbm_ref is not None

        def _load_out(off, size):
          pltpu.make_async_copy(
              o_in_hbm_ref.at[pl.ds(token_start + off, size), :, :],
              vmem_ref.at[b, pl.ds(out_base + off, size), :, :], recv_sem.at[b],
          ).start()

        _pow2_chunks(num_tokens, _load_out)

    def _issue_in_chunk(b, slot_base, base_i, unroll):
      idx_base = slot_base + base_i
      srcs = [src_smem_ref[idx_base + u] for u in range(unroll)]
      for u in range(unroll):
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(srcs[u], 1), :, :], vmem_ref.at[b, pl.ds(base_i + u, 1), :, :], recv_sem.at[b],
        ).start()

    def _in_start(blk):
      b = blk % num_buffers
      slot_base = _slot_base(blk)
      _in_out_start(blk)
      _unrolled_chunks(
          _block(blk)[1], _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          lambda off, u: _issue_in_chunk(b, slot_base, off, u),
      )

    def _in_wait(blk):
      _, num_slots, _, num_tokens = _block(blk)
      wait_count = num_slots if zero_initialized else num_slots + num_tokens
      _wait_rows(blk % num_buffers, wait_count, recv_sem)

    def _out_start(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)
      out_base = block_size - num_tokens

      def _store(off, size):
        pltpu.make_async_copy(
            vmem_ref.at[b, pl.ds(out_base + off, size), :, :],
            o_hbm_ref.at[pl.ds(token_start + off, size), :, :], send_sem.at[b],
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

      nxt_clamped = jnp.minimum(nxt, num_blocks - 1)

      @pl.when(has_nxt)
      def _start_next_out():
        _in_out_start(nxt_clamped)

      b_i = i % num_buffers
      _, num_reduce, token_start_i, num_tokens_i = _block(i)
      reduce_base = _slot_base(i)
      token_bias_i = token_start_i - (block_size - num_tokens_i)
      b_nxt = nxt_clamped % num_buffers
      num_issue = jnp.where(has_nxt, _block(nxt_clamped)[1], 0)
      issue_base = _slot_base(nxt_clamped)
      unroll = _RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL
      num_both_chunks = jnp.minimum(num_reduce, num_issue) // unroll
      num_reduce_chunks = num_reduce // unroll
      base_issue_rem = num_both_chunks * unroll
      base_reduce_rem = num_reduce_chunks * unroll

      def _reduce_step(t, row, prev_t, total):
        if zero_initialized:
          total = jnp.where(t == prev_t, total, 0.0) + row
        else:
          init = vmem_ref[b_i, t, :, :].astype(jnp.float32)
          total = jnp.where(t == prev_t, total, init) + row
        vmem_ref[b_i, t, :, :] = total.astype(x.dtype)
        return t, total

      def _issue_and_reduce_chunk(c, carry):
        prev_t, total = carry
        base_j = c * unroll
        issue_idx = issue_base + base_j
        reduce_idx = reduce_base + base_j
        srcs = [src_smem_ref[issue_idx + u] for u in range(unroll)]
        toks = [tok_smem_ref[reduce_idx + u] - token_bias_i for u in range(unroll)]
        rows = vmem_ref[b_i, pl.ds(base_j, unroll), :, :].astype(jnp.float32)
        for u in range(unroll):
          pltpu.make_async_copy(
              x_hbm_ref.at[pl.ds(srcs[u], 1), :, :], vmem_ref.at[b_nxt, pl.ds(base_j + u, 1), :, :],
              recv_sem.at[b_nxt],
          ).start()
          prev_t, total = _reduce_step(toks[u], rows[u], prev_t, total)
        return prev_t, total

      def _reduce_chunk(c, carry):
        prev_t, total = carry
        base_j = c * unroll
        reduce_idx = reduce_base + base_j
        toks = [tok_smem_ref[reduce_idx + u] - token_bias_i for u in range(unroll)]
        rows = vmem_ref[b_i, pl.ds(base_j, unroll), :, :].astype(jnp.float32)
        for u in range(unroll):
          prev_t, total = _reduce_step(toks[u], rows[u], prev_t, total)
        return prev_t, total

      def _reduce_one(j, carry):
        prev_t, total = carry
        t = tok_smem_ref[reduce_base + j] - token_bias_i
        row = vmem_ref[b_i, j, :, :].astype(jnp.float32)
        return _reduce_step(t, row, prev_t, total)

      carry = (jnp.int32(-1), jnp.zeros((dim0, dim1), jnp.float32))
      carry = jax.lax.fori_loop(0, num_both_chunks, _issue_and_reduce_chunk, carry)
      _unrolled_chunks(
          num_issue - base_issue_rem, _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          lambda off, u: _issue_in_chunk(b_nxt, issue_base, base_issue_rem + off, u),
      )
      carry = jax.lax.fori_loop(num_both_chunks, num_reduce_chunks, _reduce_chunk, carry)
      jax.lax.fori_loop(base_reduce_rem, num_reduce, _reduce_one, carry)
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
  in_specs = [pl.BlockSpec(memory_space=pltpu.SMEM), hbm_spec, hbm_spec, hbm_spec]
  inputs = [metadata.blocks, metadata.src_rows, metadata.tokens, x]
  input_output_aliases = {}
  if has_out_in:
    in_specs.append(hbm_spec)
    inputs.append(out)
    input_output_aliases[4] = 0
  scratch_shapes = [
      pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
      pltpu.SMEM((ring * window,), jnp.int32), pltpu.SMEM((ring * window,), jnp.int32),
      dma_sems, dma_sems, pltpu.SemaphoreType.DMA((ring,)),
  ]
  return pl.pallas_call(
      _ragged_gather_reduce_kernel, out_shape=out_shape, in_specs=in_specs, out_specs=hbm_spec,
      scratch_shapes=scratch_shapes, input_output_aliases=input_output_aliases,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT, disable_bounds_checks=True,
      ),
      name="ragged_gather_reduce_tc",
  )(*inputs)
