"""Vector subcore sum of the local, die partner and previous hop rows of one VMEM tile."""

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

from lilax._backend.experimental.lineage.sc_reduce_scatter import fp8_scaling
from lilax._backend.experimental.lineage.sc_reduce_scatter import microbatch_step

_FP8_MAX = fp8_scaling.FP8_MAX


def _reduce_rows(mb_step: microbatch_step.MicrobatchStep, a_vmem, b_vmem, c_vmem, slot, r_slice):
  three_way, to_final_out = mb_step.three_way, mb_step.to_final_out
  if mb_step.plan.quantized_input:
    acc = fp8_scaling.to_compressed_bf16(a_vmem[slot, r_slice, :]) + fp8_scaling.to_compressed_bf16(
        b_vmem[slot, r_slice, :]
    )
    if three_way:
      acc = acc + fp8_scaling.to_compressed_bf16(c_vmem[slot, r_slice, :])
    a_vmem[slot, r_slice, :] = fp8_scaling.from_compressed_bf16(acc)
  else:
    acc = a_vmem[slot, r_slice, :] + b_vmem[slot, r_slice, :]
    if three_way:
      acc = acc + c_vmem[slot, r_slice, :].astype(jnp.bfloat16)
    if to_final_out:
      a_vmem[slot, r_slice, :] = acc
    else:
      c_vmem[slot, r_slice, :] = jnp.clip(acc, -_FP8_MAX, _FP8_MAX).astype(jnp.float8_e4m3fn)


def _reduce_scaled_rows(mb_step: microbatch_step.MicrobatchStep, a_vmem, b_vmem, c_vmem, slot, r_slice, inv_scale,
                        prev_scale):
  three_way, to_final_out = mb_step.three_way, mb_step.to_final_out
  acc = a_vmem[slot, r_slice, :].astype(jnp.float32) + b_vmem[slot, r_slice, :].astype(jnp.float32)
  if not to_final_out:
    acc = acc * inv_scale
  if three_way:
    prev = c_vmem[slot, r_slice, :].astype(jnp.bfloat16).astype(jnp.float32)
    acc = acc + prev if prev_scale is None else acc + prev * prev_scale
  if to_final_out:
    a_vmem[slot, r_slice, :] = acc.astype(jnp.bfloat16)
  else:
    c_vmem[slot, r_slice, :] = jnp.clip(acc, -_FP8_MAX, _FP8_MAX).astype(jnp.bfloat16).astype(jnp.float8_e4m3fn)


def _reduce_tile_global_scaled(mb_step: microbatch_step.MicrobatchStep, a_vmem, b_vmem, c_vmem, slot, amax_scratch):
  """Reduces a tile with the global per-token scales loaded alongside it."""
  plan = mb_step.plan
  inv_vmem = amax_scratch["inv_vmem"]

  @plsc.parallel_loop(0, plan.tokens_per_tile * plan.dim0, step=4)
  def _token_row_loop_global(r):
    j = jax.lax.div(r, plan.dim0)
    inv_scale = jnp.broadcast_to(inv_vmem[slot, pl.ds(j, 1), :], (4, 128))
    prev_scale = None
    if mb_step.to_final_out:
      prev_scale = jax.lax.bitcast_convert_type(
          (254 << 23) - jax.lax.bitcast_convert_type(inv_scale, jnp.int32), jnp.float32,
      )
    _reduce_scaled_rows(mb_step, a_vmem, b_vmem, c_vmem, slot, pl.ds(r, 4), inv_scale, prev_scale)


def _reduce_tile_fused_hop(mb_step: microbatch_step.MicrobatchStep, a_vmem, b_vmem, c_vmem, slot, tok_base,
                           amax_scratch):
  """Reduces a tile in two passes per token, deriving its hop scale from the token's partial sum."""
  plan = mb_step.plan
  dim0, dim1, three_way = plan.dim0, plan.dim1, mb_step.three_way
  sum_vmem = amax_scratch["sum_vmem"]
  hop_recv_vmem = amax_scratch["hop_recv_vmem"]
  hop_bound_vmem = amax_scratch["hop_bound_vmem"]

  @pl.loop(0, plan.tokens_per_tile)
  def _tok_loop(j):
    r_tok = pl.multiple_of(j * dim0, 8)
    idx = tok_base + j
    row_max_init = jnp.zeros((4, dim1), dtype=jnp.float32)

    @plsc.parallel_loop(0, dim0, step=4, carry=row_max_init)
    def _pass1(dr, cur_max):
      r = pl.multiple_of(r_tok + dr, 4)
      acc = a_vmem[slot, pl.ds(r, 4), :].astype(jnp.float32) + b_vmem[slot, pl.ds(r, 4), :].astype(jnp.float32)
      sum_vmem[pl.ds(dr, 4), :] = acc
      return jnp.maximum(cur_max, jnp.abs(acc))

    m4 = _pass1
    m2 = jnp.maximum(m4[:2, :], m4[2:, :])
    m1 = jnp.maximum(m2[:1, :], m2[1:, :])
    v64 = jnp.maximum(m1[:, :64], m1[:, 64:])
    v32 = jnp.maximum(v64[:, :32], v64[:, 32:])
    v16 = jnp.maximum(v32[:, :16], v32[:, 16:])
    vals = [v16[0, k] for k in range(16)]
    while len(vals) > 1:
      vals = [jnp.maximum(vals[2 * k], vals[2 * k + 1]) for k in range(len(vals) // 2)]
    pair_amax_1 = jnp.broadcast_to(vals[0], (1,))
    if three_way:
      recv_bound_1 = hop_recv_vmem[pl.ds(idx, 1)]
      bound_1 = pair_amax_1 + recv_bound_1
      e_new_1 = fp8_scaling.pow2_exponent(bound_1)
      prev_scale = jnp.broadcast_to(
          fp8_scaling.pow2_from_exponent(jnp.clip(127 + fp8_scaling.pow2_exponent(recv_bound_1) - e_new_1, 0, 254)),
          (4, dim1),
      )
    else:
      bound_1 = pair_amax_1
      e_new_1 = fp8_scaling.pow2_exponent(bound_1)
      prev_scale = None
    hop_bound_vmem[pl.ds(idx, 1)] = bound_1
    inv_scale = jnp.broadcast_to(fp8_scaling.pow2_from_exponent(254 - e_new_1), (4, dim1))

    @plsc.parallel_loop(0, dim0, step=4)
    def _pass2(dr):
      r = pl.multiple_of(r_tok + dr, 4)
      acc = sum_vmem[pl.ds(dr, 4), :] * inv_scale
      if three_way:
        prev = c_vmem[slot, pl.ds(r, 4), :].astype(jnp.bfloat16).astype(jnp.float32)
        acc = acc + prev * prev_scale
      c_vmem[slot, pl.ds(r, 4), :] = (
          jnp.clip(acc, -_FP8_MAX, _FP8_MAX).astype(jnp.bfloat16).astype(jnp.float8_e4m3fn)
      )


def _reduce_tile_hop_scaled(mb_step: microbatch_step.MicrobatchStep, a_vmem, b_vmem, c_vmem, slot, tok_base,
                            amax_scratch):
  """Reduces a tile with the hop scales derived before the tile loop."""
  plan = mb_step.plan
  hop_prev_scale_vmem = amax_scratch["hop_prev_scale_vmem"]

  @plsc.parallel_loop(0, plan.tokens_per_tile * plan.dim0, step=4)
  def _token_row_loop_hop(r):
    j = jax.lax.div(r, plan.dim0)
    idx = tok_base + j
    if mb_step.to_final_out:
      inv_scale = None
    else:
      inv_scale = jnp.broadcast_to(amax_scratch["hop_inv_vmem"][pl.ds(idx, 1)], (4, 128))
    if not mb_step.three_way:
      prev_scale = None
    else:
      prev_scale = jnp.broadcast_to(hop_prev_scale_vmem[pl.ds(idx, 1)], (4, 128))
    _reduce_scaled_rows(mb_step, a_vmem, b_vmem, c_vmem, slot, pl.ds(r, 4), inv_scale, prev_scale)


def reduce_tile(mb_step: microbatch_step.MicrobatchStep, slot, tile_idx, a_vmem, b_vmem, c_vmem, amax_scratch):
  """Sums the tile in `slot`, writing the running sum or the final output rows."""
  plan = mb_step.plan
  if not plan.scaled:
    @plsc.parallel_loop(0, plan.vmem_tile_rows, step=4)
    def _row_loop(r):
      _reduce_rows(mb_step, a_vmem, b_vmem, c_vmem, slot, pl.ds(r, 4))

    return

  if plan.global_scaling:
    _reduce_tile_global_scaled(mb_step, a_vmem, b_vmem, c_vmem, slot, amax_scratch)
    return

  d = jax.lax.div(tile_idx, plan.tiles_per_dir)
  t = jax.lax.rem(tile_idx, plan.tiles_per_dir)
  tok_base = d * plan.block_tokens + t * plan.tokens_per_tile
  if plan.fused_per_hop and not mb_step.to_final_out:
    _reduce_tile_fused_hop(mb_step, a_vmem, b_vmem, c_vmem, slot, tok_base, amax_scratch)
    return
  _reduce_tile_hop_scaled(mb_step, a_vmem, b_vmem, c_vmem, slot, tok_base, amax_scratch)
