"""TensorCore Pallas ragged gather of token rows, optionally with their absmax."""

import math
from typing import Any

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

_RAGGED_GATHER_TC_NUM_BUFFERS = 2


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_gather_tc(
    x: jt.Num[jax.Array, "num_input_tokens d0 d1"], indices: jt.Int[jax.Array, "num_indices"],
    start: jt.Int[jax.Array, ""] | int, num_tokens: jt.Int[jax.Array, ""] | int,
    max_out_tokens: int, block_size: int = 2048, *, with_absmax: bool = False,
) -> Any:
  """Executes a ragged gather on TensorCore using a Pallas kernel."""
  if max_out_tokens <= 0:
    raise ValueError(f"max_out_tokens must be positive, got {max_out_tokens}.")
  if isinstance(num_tokens, int) and not 0 <= num_tokens <= max_out_tokens:
    raise ValueError(f"num_tokens must be in [0, {max_out_tokens}], got {num_tokens}.")
  if block_size <= 0:
    raise ValueError(f"block_size must be positive, got {block_size}.")
  num_indices = indices.shape[0]
  _, dim0, dim1 = x.shape
  if num_indices == 0 or (isinstance(num_tokens, int) and num_tokens == 0):
    empty = jnp.empty((max_out_tokens, dim0, dim1), dtype=x.dtype)
    if with_absmax:
      return empty, jnp.zeros((1, 1), dtype=jnp.float32)
    return empty
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

  def _ragged_gather_kernel(*args):
    if with_absmax:
      (scalars_smem_ref, idx_hbm_ref, x_hbm_ref, o_hbm_ref, amax_smem_ref, vmem_ref, idx_smem_ref,
       idx_sem, data_recv_sem, data_send_sem, amax_vmem_ref) = args
    else:
      (scalars_smem_ref, idx_hbm_ref, x_hbm_ref, o_hbm_ref, vmem_ref, idx_smem_ref,
       idx_sem, data_recv_sem, data_send_sem) = args
      amax_smem_ref = amax_vmem_ref = None
    block_idx = pl.program_id(0)
    buf_idx = block_idx % num_buffers
    start_idx = scalars_smem_ref[0]
    num_valid = scalars_smem_ref[1]
    if with_absmax:
      assert amax_vmem_ref is not None

      @pl.when(block_idx == 0)
      def _init_amax():
        amax_vmem_ref[...] = jnp.zeros_like(amax_vmem_ref)

    def _run_step(step_idx, b_idx, action: str):
      base = step_idx * block_size
      row0 = start_idx + base
      win_start = jnp.minimum((row0 // idx_align) * idx_align, max_win_start)
      idx_off = row0 - win_start

      def _fetch_indices():
        copy = pltpu.make_async_copy(
            idx_hbm_ref.at[pl.ds(pl.multiple_of(win_start, idx_align), idx_window)], idx_smem_ref, idx_sem,
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
                  vmem_ref.at[b_idx, pl.ds(base_i + u, 1), :, :], data_recv_sem.at[b_idx],
              ).start()
        elif chunk_action == "recv_wait":
          pltpu.make_async_copy(
              o_hbm_ref.at[pl.ds(0, size), :, :], vmem_ref.at[b_idx, pl.ds(0, size), :, :], data_recv_sem.at[b_idx],
          ).wait()
        elif chunk_action == "absmax":
          assert amax_vmem_ref is not None
          step_unroll = math.gcd(size, 16)

          @pl.loop(0, size // step_unroll)
          def _(chunk_i):
            base_i = off + chunk_i * step_unroll
            tile = vmem_ref[b_idx, pl.ds(base_i, step_unroll), :, :]
            row_max = jnp.max(jnp.abs(tile).astype(jnp.float32), axis=(0, 1))
            amax_vmem_ref[0] = jnp.maximum(amax_vmem_ref[0], row_max)
        else:
          copy = pltpu.make_async_copy(
              vmem_ref.at[b_idx, pl.ds(off, size), :, :], o_hbm_ref.at[pl.ds(base + off, size), :, :],
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

              @pl.when(bit != 0)
              def _(off=off, size=size):
                _chunk(off, size, chunk_action)

              off = off + bit

        if action == "data_start":
          @pl.when(step_valid > 0)
          def _():
            _fetch_indices()

          _over_valid_rows("data_start")
        elif action == "copy_out_start":
          _over_valid_rows("recv_wait")
          if with_absmax:
            _over_valid_rows("absmax")
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
      if with_absmax:
        assert amax_smem_ref is not None and amax_vmem_ref is not None
        amax_smem_ref[0, 0] = jnp.max(amax_vmem_ref[0])

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_struct = jax.ShapeDtypeStruct((max_out_tokens, dim0, dim1), x.dtype, manual_axis_type=manual_axis_type)
  scalars = jnp.stack([jnp.asarray(start, dtype=jnp.int32), jnp.asarray(num_tokens, dtype=jnp.int32)])
  num_steps = pl.cdiv(num_tokens if isinstance(num_tokens, int) else scalars[1], block_size)
  scratch_shapes = [
      pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype), pltpu.SMEM((idx_window,), indices.dtype),
      pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.DMA((num_buffers,)), pltpu.SemaphoreType.DMA((num_buffers,)),
  ]
  if with_absmax:
    num_steps = jnp.maximum(1, num_steps)
    amax_struct = jax.ShapeDtypeStruct((1, 1), jnp.float32, manual_axis_type=manual_axis_type)
    out_shape = (out_struct, amax_struct)
    out_specs = (pl.BlockSpec(memory_space=pltpu.HBM), pl.BlockSpec(memory_space=pltpu.SMEM))
    scratch_shapes.append(pltpu.VMEM((1, dim1), jnp.float32))
    call_name = "ragged_gather_tc-absmax"
  else:
    out_shape = out_struct
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
    call_name = "ragged_gather_tc"
  return pl.pallas_call(
      _ragged_gather_kernel, out_shape=out_shape, grid=(num_steps,),
      in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM)],
      out_specs=out_specs, scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)), name=call_name,
  )(scalars, indices, x)
