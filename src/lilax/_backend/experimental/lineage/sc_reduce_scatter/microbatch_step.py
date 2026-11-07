"""A vector subcore's ring step and microbatch, and the rows its tiles address."""

import dataclasses
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

from lilax._backend.experimental.lineage.sc_reduce_scatter import plan as plan_lib


@dataclasses.dataclass(frozen=True)
class MicrobatchStep:
  """Ring step `s` of microbatch `m` on vector subcore `subcore` of die `c` at ring position `z`."""

  plan: plan_lib.ReduceScatterPlan
  z: Any
  c: Any
  subcore: Any
  s: Any
  m: Any
  three_way: bool
  to_final_out: bool

  def tile_offsets(self, tile_idx):
    """Returns the x, receive buffer, previous hop and output rows of a tile."""
    plan = self.plan
    ring_size, num_microbatches, mb_rows = plan.ring_size, plan.num_microbatches, plan.mb_rows
    z, c, s, m = self.z, self.c, self.s, self.m
    d = jax.lax.div(tile_idx, plan.tiles_per_dir)
    t = jax.lax.rem(tile_idx, plan.tiles_per_dir)
    cw_z_dst = jax.lax.rem(z + ring_size - 1 - s, ring_size)
    ccw_z_dst = jax.lax.rem(z + 1 + s, ring_size)
    z_dst = jnp.where(d == 0, cw_z_dst, ccw_z_dst)
    sc_off = self.subcore * plan.subcore_rows + t * plan.vmem_tile_rows
    x_row = pl.multiple_of((((z_dst * 2 + c) * 2 + d) * num_microbatches + m) * mb_rows + sc_off, 8)
    buf_row = pl.multiple_of(((s * num_microbatches + m) * 2 + d) * mb_rows + sc_off, 8)
    prev_c2c_row = pl.multiple_of((((s - 1) * num_microbatches + m) * 2 + d) * mb_rows + sc_off, 8)
    out_row = pl.multiple_of((d * num_microbatches + m) * mb_rows + sc_off, 8)
    return x_row, buf_row, prev_c2c_row, out_row
