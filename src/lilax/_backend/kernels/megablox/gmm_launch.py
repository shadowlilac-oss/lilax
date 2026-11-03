"""Pallas launch construction for grouped matrix multiplication."""

import json

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import qwix.pallas as qpl


def make_launch(
    kernel, lhs, rhs, group_metadata, num_active_tiles, lhs_block_spec, rhs_block_spec,
    out_block_spec, in_out_block_spec, input_output_aliases, m, k, n, tm, tk, tn, tiles_k, tiles_n,
    preferred_element_type, transpose_rhs, interpret, varying_axes, calculate_bytes,
):
  lhs_bytes = calculate_bytes(lhs)
  if isinstance(rhs, qpl.QArray):
    rhs_bytes = (k * n) * rhs.qvalue.itemsize
  else:
    rhs_bytes = (k * n) * rhs.itemsize
  out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
  max_active_tiles = group_metadata[1].size
  bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * max_active_tiles) + out_bytes
  flops = 2 * m * k * n
  cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)
  metadata = {
      "preferred_element_type": jnp.dtype(preferred_element_type).name,
      "tiling": {"tile_m": tm, "tile_k": tk, "tile_n": tn},
      "transpose_rhs": transpose_rhs,
  }
  return qpl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(
          (m, n), preferred_element_type, manual_axis_type=jax.sharding.ManualAxisType(varying=frozenset(varying_axes))
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[lhs_block_spec, rhs_block_spec, in_out_block_spec],
          out_specs=out_block_spec,
          grid=(tiles_n, num_active_tiles, tiles_k),
          scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
      ),
      input_output_aliases=input_output_aliases,
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
      interpret=interpret,
      cost_estimate=cost_estimate,
      metadata={"xprof_metadata": json.dumps(metadata)},
  )
