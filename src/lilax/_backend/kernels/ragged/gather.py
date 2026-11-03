"""Ragged gather kernel implementation from tpu-inference."""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def main_kernel(
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    weights_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    indices_vmem_ref: jax.Ref,
    weights_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    has_weights: bool,
):
  """Core ragged gather operation with per-row weighting."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_lanes = tpu_info.num_lanes
  hidden_size = in_hbm_ref.shape[-1]
  col_size = out_vmem_ref.shape[-1]
  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
  block_size = num_simd_lanes * num_cores

  recv_sem = sem_ref.at[0]
  send_sem = sem_ref.at[1]
  dma_list = []
  dma = pltpu.make_async_copy(start_ref.at[:1], start_vmem_ref.at[:1], recv_sem)
  dma_list.append(dma)
  dma = pltpu.make_async_copy(end_ref.at[:1], end_vmem_ref.at[:1], recv_sem)
  dma_list.append(dma)
  jax.tree.map(lambda x: x.start(), dma_list)
  jax.tree.map(lambda x: x.wait(), dma_list)

  start = start_vmem_ref[...][0]
  end = end_vmem_ref[...][0]
  block_start = start // block_size
  block_end = pl.cdiv(end, block_size)
  num_blocks = block_end - block_start
  num_blocks = jnp.where(end == start, 0, num_blocks)
  aligned_start = block_start * block_size
  num_cols = pl.cdiv(hidden_size, col_size)

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_blocks, num_cores, num_cols),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL, pltpu.ARBITRARY),
  )
  def inner_kernel():
    block_id = pl.program_id(0)
    core_id = pl.program_id(1)
    col_id = pl.program_id(2)
    row_tile_start = aligned_start + block_id * block_size + core_id * num_simd_lanes
    col_tile_start = col_id * col_size

    @pl.when(col_id == 0)
    def _():
      pltpu.sync_copy(indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)], indices_vmem_ref)
      if has_weights:
        pltpu.sync_copy(weights_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)], weights_vmem_ref)

    indices = indices_vmem_ref[...]
    if has_weights:
      weights = weights_vmem_ref[...]
    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)  # pyrefly: ignore[missing-attribute]
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)  # pyrefly: ignore[missing-attribute]

    for col_vmem_start in range(0, col_size, num_lanes):
      col_hbm_start = col_tile_start + col_vmem_start
      for row_vmem in range(num_simd_lanes):
        row_hbm = indices[row_vmem] // packing
        pltpu.make_async_copy(
            in_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
            out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
            recv_sem,
        ).start()

    @pl.loop(0, col_size, step=num_lanes)
    @jax.named_scope("dma_write_loop")
    def dma_write_loop(col_vmem_start):
      col_hbm_start = col_tile_start + col_vmem_start
      for _ in range(num_simd_lanes):
        pltpu.make_async_copy(in_32b_hbm_ref.at[0, :num_lanes], out_vmem_ref.at[0, :num_lanes], recv_sem).wait()

      if packing > 1:
        for col_compute_offset in range(0, num_lanes, num_simd_lanes):
          col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)
          out = None
          for row_src in range(num_simd_lanes):
            row_src_pack = indices[row_src] % packing
            row_dst_pack = row_src % packing
            rightshift_bits = row_src_pack * dtype_bits
            leftshift_bits = row_dst_pack * dtype_bits
            data = out_vmem_ref[row_src, col_slice]
            data = jnp.bitwise_right_shift(data, rightshift_bits)
            data = jnp.bitwise_and(data, 2**dtype_bits - 1)
            data = jnp.bitwise_left_shift(data, leftshift_bits)
            if row_dst_pack == 0:
              out = data
            else:
              assert out is not None
              out = jnp.bitwise_or(out, data)
            if row_dst_pack == packing - 1:
              row_dst = row_src // packing
              out_vmem_ref[row_dst, col_slice] = out

      if has_weights:
        if packing == 1:
          for col_compute_offset in range(0, num_lanes, num_simd_lanes):
            col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)
            for row_vmem in range(num_simd_lanes):
              data = out_vmem_ref[row_vmem, col_slice]
              data_f32 = jax.lax.bitcast_convert_type(data, jnp.float32)
              data_f32 = data_f32 * weights[row_vmem]
              out_vmem_ref[row_vmem, col_slice] = jax.lax.bitcast_convert_type(data_f32, jnp.uint32)
        else:
          for col_compute_offset in range(0, num_lanes, num_simd_lanes):
            col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)
            for row_dst in range(num_simd_lanes // packing):
              packed_data = out_vmem_ref[row_dst, col_slice]
              result = jnp.zeros_like(packed_data)
              for sub in range(packing):
                row_src = row_dst * packing + sub
                shift_right = sub * dtype_bits
                shift_left = sub * dtype_bits
                elem = jnp.bitwise_right_shift(packed_data, shift_right)
                elem = jnp.bitwise_and(elem, 2**dtype_bits - 1)
                elem_f32 = jnp.bitwise_left_shift(elem, 16)
                elem_f32 = jax.lax.bitcast_convert_type(elem_f32, jnp.float32)
                elem_f32 = elem_f32 * weights[row_src]
                elem_u32 = jax.lax.bitcast_convert_type(elem_f32, jnp.uint32)
                elem_bf16 = jnp.bitwise_right_shift(elem_u32, 16)
                elem_bf16 = jnp.bitwise_and(elem_bf16, 2**dtype_bits - 1)
                elem_bf16 = jnp.bitwise_left_shift(elem_bf16, shift_left)
                result = jnp.bitwise_or(result, elem_bf16)
              out_vmem_ref[row_dst, col_slice] = result

      for row_vmem in range(num_simd_lanes // packing):
        row_hbm = row_tile_start // packing + row_vmem
        pltpu.make_async_copy(
            out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
            out_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
            send_sem,
        ).start()

    for _ in range(0, col_size, num_lanes):
      for _ in range(num_simd_lanes // packing):
        pltpu.make_async_copy(out_vmem_ref.at[0, :num_lanes], out_32b_hbm_ref.at[0, :num_lanes], send_sem).wait()

  inner_kernel()


def get_cost_estimate(
    out_size: int,
    hidden_size: int,
    dtype_bytes: int,
    has_weights: bool,
    flops_override: int = -1,
    bytes_accessed_override: int = -1,
) -> pl.CostEstimate:
  """Returns a cost estimate for the ragged gather kernel."""
  if flops_override > 0:
    flops = flops_override
  else:
    flops = out_size * hidden_size if has_weights else 0
  if bytes_accessed_override > 0:
    bytes_accessed = bytes_accessed_override
  else:
    bytes_in = out_size * hidden_size * dtype_bytes
    bytes_in += out_size * 4
    if has_weights:
      bytes_in += out_size * 4
    bytes_out = out_size * hidden_size * dtype_bytes
    bytes_accessed = bytes_in + bytes_out
  return pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)


def _fallback_implementation(
    x: jax.Array, indices: jax.Array, weights: jax.Array | None = None, has_weights: bool = False,
) -> jax.Array:
  """Fallback to (non-ragged) JAX implementation for ragged gather."""
  out = x[indices]
  if has_weights:
    out = out * weights[:, None]  # pyrefly: ignore[unsupported-operation]
  return out


def calculate_col_size(hidden_size: int) -> int:
  """Calculate col size for ragged gather kernel."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_lanes = tpu_info.num_lanes
  num_simd_lanes = sc_info.num_lanes
  match tpu_info.chip_version:
    case 6:
      target_bytes = (256 * 1024) * 0.9
    case 7:
      target_bytes = (512 * 1024) * 0.9
    case _:
      target_bytes = (128 * 1024) * 0.9
  base_bytes = num_simd_lanes * hidden_size * (32 // 8)
  num_cols = 1
  while pl.cdiv(base_bytes, num_cols * num_lanes) * num_lanes > target_bytes:
    num_cols += 1
  return pl.cdiv(hidden_size, (num_cols * num_lanes)) * num_lanes


@functools.partial(
    jax.jit,
    static_argnames=("has_weights", "enforce_fallback", "flops_override", "bytes_accessed_override", "use_single_sparsecore"),
)
def ragged_gather(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    weights: jax.Array | None = None,
    has_weights: bool = False,
    enforce_fallback: bool = False,
    flops_override: int = -1,
    bytes_accessed_override: int = -1,
    use_single_sparsecore: bool = False,
) -> jax.Array:
  """Perform gather on indices within dynamic array start and end."""
  assert x.ndim == 2, "Ragged gather only supports 2d inputs."
  assert indices.ndim == 1, "Ragged gather only supports 1d indices."
  if jnp.ndim(start) == 0:
    start = jnp.repeat(start, 2)
  elif start.shape == (1,):
    start = jnp.pad(start, (0, 1))
  if jnp.ndim(end) == 0:
    end = jnp.repeat(end, 2)
  elif end.shape == (1,):
    end = jnp.pad(end, (0, 1))
  dtype = x.dtype
  if enforce_fallback or not pltpu.is_tpu_device():
    return _fallback_implementation(x, indices, weights, has_weights)
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return _fallback_implementation(x, indices, weights, has_weights)
  hidden_size = x.shape[-1]
  out_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_sc_cores = 1 if use_single_sparsecore else sc_info.num_cores
  num_cores = num_sc_cores * sc_info.num_subcores
  block_size = num_simd_lanes * num_cores
  col_size = calculate_col_size(hidden_size)
  out_pad_size = pl.cdiv(out_size, block_size) * block_size - out_size
  indices = jnp.pad(indices, ((0, out_pad_size)))
  if has_weights:
    weights = jnp.pad(weights, ((0, out_pad_size)), constant_values=1.0)  # pyrefly: ignore[bad-argument-type]
  else:
    weights = jnp.ones((out_size + out_pad_size,), dtype=jnp.float32)
  aligned_hidden_size = pl.cdiv(hidden_size, col_size) * col_size
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=num_sc_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  return pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          has_weights=has_weights,
      ),
      out_type=jax.ShapeDtypeStruct((out_size + out_pad_size, aligned_hidden_size), dtype),
      compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True, disable_bounds_checks=True),
      cost_estimate=get_cost_estimate(
          out_size=out_size + out_pad_size,
          hidden_size=aligned_hidden_size,
          dtype_bytes=jax.dtypes.itemsize_bits(dtype) // 8,
          has_weights=has_weights,
          flops_override=flops_override,
          bytes_accessed_override=bytes_accessed_override,
      ),
      scratch_types=[
          pltpu.VMEM((num_simd_lanes,), jnp.int32),
          pltpu.VMEM((num_simd_lanes,), jnp.int32),
          pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
          pltpu.VMEM((num_simd_lanes,), jnp.int32),
          pltpu.VMEM((num_simd_lanes,), jnp.float32),
          pltpu.SemaphoreType.DMA((2,)),
      ],
      mesh=vector_mesh,
      name="sc_ragged_gather",
  )(start, end, x, indices, weights)[:out_size, :hidden_size]
