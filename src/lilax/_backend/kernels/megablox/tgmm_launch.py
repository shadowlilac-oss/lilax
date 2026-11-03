"""Pallas launch construction for transposed grouped matrix multiplication."""

import json

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import qwix.pallas as qpl


def make_launch(
    kernel, lhs, rhs, num_active_tiles, lhs_block_spec, rhs_block_spec, out_block_spec, in_out_block_spec,
    input_output_aliases, m, k, n, tm, tk, tn, tiles_k, tiles_n, num_actual_groups,
    preferred_element_type, interpret, varying_axes, calculate_bytes,
):
  lhs_bytes = calculate_bytes(lhs)
  rhs_bytes = calculate_bytes(rhs)
  out_bytewidth = jnp.dtype(preferred_element_type).itemsize
  out_bytes = (num_actual_groups * k * n) * out_bytewidth
  bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * tiles_k) + out_bytes
  flops = 2 * m * k * n
  cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)
  lhs = lhs.swapaxes(0, 1)
  metadata = {
      "tiling": {"tile_m": tm, "tile_k": tk, "tile_n": tn},
      "prefer_element_type": jnp.dtype(preferred_element_type).name,
      "num_actual_groups": num_actual_groups,
  }
  call_gmm = qpl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(
          (num_actual_groups, k, n),
          preferred_element_type,
          manual_axis_type=jax.sharding.ManualAxisType(varying=frozenset(varying_axes)),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[lhs_block_spec, rhs_block_spec, in_out_block_spec],
          out_specs=out_block_spec,
          grid=(tiles_n, tiles_k, num_active_tiles),
          scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
      ),
      input_output_aliases=input_output_aliases,
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
      interpret=interpret,
      cost_estimate=cost_estimate,
      metadata={"xprof_metadata": json.dumps(metadata)},
  )
  return lhs, call_gmm
