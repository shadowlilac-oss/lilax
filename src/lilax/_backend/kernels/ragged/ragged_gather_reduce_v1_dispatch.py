"""JIT dispatch for the v1 ragged gather reduce kernel."""

import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


def ragged_gather_reduce(
    x,
    indices,
    topk_weights,
    valid_rows_mask,
    reduce_group_size,
    enforce_fallback,
    flops_override,
    bytes_accessed_override,
    use_single_sparsecore,
    align_to,
    get_cost_estimate,
    fallback_implementation,
    preprocess,
    main_kernel,
):
  assert x.ndim == 2, "ragged_gather_reduce only supports 2d inputs."
  assert indices.ndim == 1, "ragged_gather_reduce only supports 1d indices."
  assert topk_weights.ndim == 1, "ragged_gather_reduce only supports 1d topk_weights."
  assert valid_rows_mask.ndim == 1, "ragged_gather_reduce only supports 1d valid_rows_mask."

  if enforce_fallback or not pltpu.is_tpu_device():
    return fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  dtype = x.dtype
  dtype_bytes = jax.dtypes.itemsize_bits(dtype) // 8

  hidden_size = x.shape[-1]
  input_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_sc_cores = 1 if use_single_sparsecore else sc_info.num_cores
  num_cores = num_sc_cores * sc_info.num_subcores

  num_column_partitions = 8
  assert num_cores % num_column_partitions == 0
  num_rows_partitions = num_cores // num_column_partitions

  aligned_hidden_size = align_to(hidden_size, 128 * num_column_partitions)
  col_size = aligned_hidden_size // num_column_partitions
  row_tile_size = num_simd_lanes
  padded_input_size = align_to(
      input_size,
      math.lcm(num_rows_partitions * row_tile_size, reduce_group_size),
  )
  pad_input_size = padded_input_size - input_size

  x = jnp.pad(
      x,
      ((0, pad_input_size), (0, aligned_hidden_size - hidden_size)),
      constant_values=0,
  )
  indices = jnp.pad(indices, (0, pad_input_size), constant_values=0)
  topk_weights = jnp.pad(topk_weights, (0, pad_input_size), constant_values=0)
  valid_rows_mask = jnp.pad(valid_rows_mask, (0, pad_input_size), constant_values=False)

  (
      src_indices,
      dst_indices,
      topk_weights,
      num_src_rows_per_row_partition,
      mask,
  ) = preprocess(
      indices,
      topk_weights,
      valid_rows_mask,
      reduce_group_size,
      num_rows_partitions,
      num_simd_lanes,
  )

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=num_sc_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  out = pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          num_row_partitions=num_rows_partitions,
          num_column_partitions=num_column_partitions,
      ),
      out_type=jax.ShapeDtypeStruct(
          (padded_input_size // reduce_group_size, aligned_hidden_size),
          jnp.float32,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      cost_estimate=get_cost_estimate(
          padded_input_size=padded_input_size,
          aligned_hidden_size=aligned_hidden_size,
          reduce_group_size=reduce_group_size,
          input_dtype_bytes=dtype_bytes,
          flops_override=flops_override,
          bytes_accessed_override=bytes_accessed_override,
      ),
      scratch_types=dict(
          num_rows_per_row_partition_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
          out_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
          prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size), jnp.uint32),
          src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
          dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
          topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.float32),
          sem_ref=pltpu.SemaphoreType.DMA((2,)),
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather_reduce",
  )(num_src_rows_per_row_partition, x, src_indices, dst_indices, topk_weights)

  return jnp.where(
      mask[:, None],
      out.astype(x.dtype),
      jnp.zeros_like(out, dtype=x.dtype),
  )[: (input_size // reduce_group_size), :hidden_size]
