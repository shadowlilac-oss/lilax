"""Per-hop token scale bounds exchanged with the ring neighbours of each microbatch block."""

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lilax._backend.experimental.lineage.sc_reduce_scatter import fp8_scaling
from lilax._backend.experimental.lineage.sc_reduce_scatter import sparsecore_dma
from lilax._backend.experimental.lineage.sc_reduce_scatter import microbatch_step


def _load_prev_hop_bounds(mb_step: microbatch_step.MicrobatchStep, block_offsets, extra, amax_scratch):
  """Copies the bounds received from the previous hop into VMEM."""
  dim0, block_tokens = mb_step.plan.dim0, mb_step.plan.block_tokens
  in_copies = []
  for d, (_, _, prev_c2c_row, _) in enumerate(block_offsets):
    dst = pl.ds(d * block_tokens, block_tokens)
    prev_tok0 = pl.multiple_of(jax.lax.div(prev_c2c_row, dim0), 8)
    in_copies.append(pltpu.make_async_copy(
        extra["amax_recv"].at[pl.ds(prev_tok0, block_tokens)], amax_scratch["hop_recv_vmem"].at[dst],
        amax_scratch["hop_in_sem"],
    ))
  for cp in in_copies:
    cp.start()
  for cp in in_copies:
    cp.wait()


def unfused_hop_token_scales(mb_step: microbatch_step.MicrobatchStep, amax_ref, extra, amax_scratch):
  """Derives hop scales from the local, partner and received token bounds and starts sending the new bounds."""
  plan = mb_step.plan
  dim0, tiles_per_dir, block_tokens = plan.dim0, plan.tiles_per_dir, plan.block_tokens
  three_way, to_final_out = mb_step.three_way, mb_step.to_final_out
  local_vmem = amax_scratch["hop_local_vmem"]
  partner_vmem = amax_scratch["hop_partner_vmem"]
  recv_vmem = amax_scratch["hop_recv_vmem"]
  bound_vmem = amax_scratch["hop_bound_vmem"]
  inv_vmem = amax_scratch["hop_inv_vmem"]
  prev_scale_vmem = amax_scratch["hop_prev_scale_vmem"]
  in_sem = amax_scratch["hop_in_sem"]
  block_offsets = [mb_step.tile_offsets(d * tiles_per_dir) for d in range(2)]
  in_copies = []
  for d, (x_row, _, prev_c2c_row, _) in enumerate(block_offsets):
    dst = pl.ds(d * block_tokens, block_tokens)
    if not to_final_out:
      tok0 = pl.multiple_of(jax.lax.div(x_row, dim0), 8)
      in_copies.append(pltpu.make_async_copy(amax_ref.at[pl.ds(tok0, block_tokens)], local_vmem.at[dst], in_sem))
      in_copies.append(pltpu.make_async_copy(
          extra["amax_partner"].at[pl.ds(tok0, block_tokens)], partner_vmem.at[dst], in_sem,
      ))
    if three_way:
      prev_tok0 = pl.multiple_of(jax.lax.div(prev_c2c_row, dim0), 8)
      in_copies.append(pltpu.make_async_copy(
          extra["amax_recv"].at[pl.ds(prev_tok0, block_tokens)], recv_vmem.at[dst], in_sem,
      ))
  for cp in in_copies:
    cp.start()
  for cp in in_copies:
    cp.wait()

  out_copies = []
  for d, (_, buf_row, _, _) in enumerate(block_offsets):
    sl = pl.ds(d * block_tokens, block_tokens)
    if to_final_out:
      prev_scale_vmem[sl] = fp8_scaling.pow2_from_exponent(fp8_scaling.pow2_exponent(recv_vmem[sl]))
      continue
    bound = local_vmem[sl] + partner_vmem[sl]
    if three_way:
      recv_bound = recv_vmem[sl]
      bound = bound + recv_bound
      e_new = fp8_scaling.pow2_exponent(bound)
      prev_scale_vmem[sl] = fp8_scaling.pow2_from_exponent(
          jnp.clip(127 + fp8_scaling.pow2_exponent(recv_bound) - e_new, 0, 254)
      )
    else:
      e_new = fp8_scaling.pow2_exponent(bound)
    inv_vmem[sl] = fp8_scaling.pow2_from_exponent(254 - e_new)
    bound_vmem[sl] = bound
    send_tok0 = pl.multiple_of(jax.lax.div(buf_row, dim0), 8)
    out_copies.append(sparsecore_dma.push_copy(
        bound_vmem.at[sl], extra["amax_send"].at[pl.ds(send_tok0, block_tokens)], amax_scratch["hop_out_sem"],
    ))
  for cp in out_copies:
    cp.start()
  return out_copies


def fused_hop_prologue(mb_step: microbatch_step.MicrobatchStep, extra, amax_scratch):
  """Loads the received bounds the fused tile reduction rescales the previous hop with."""
  plan = mb_step.plan
  tiles_per_dir, block_tokens = plan.tiles_per_dir, plan.block_tokens
  prev_scale_vmem = amax_scratch["hop_prev_scale_vmem"]
  block_offsets = [mb_step.tile_offsets(d * tiles_per_dir) for d in range(2)]
  if mb_step.three_way:
    _load_prev_hop_bounds(mb_step, block_offsets, extra, amax_scratch)
  if mb_step.to_final_out:
    recv_vmem = amax_scratch["hop_recv_vmem"]
    for d in range(2):
      sl = pl.ds(d * block_tokens, block_tokens)
      prev_scale_vmem[sl] = fp8_scaling.pow2_from_exponent(fp8_scaling.pow2_exponent(recv_vmem[sl]))


def fused_hop_epilogue(mb_step: microbatch_step.MicrobatchStep, extra, amax_scratch):
  """Sends the bounds the fused tile reduction derived to the next hop."""
  plan = mb_step.plan
  dim0, tiles_per_dir, block_tokens = plan.dim0, plan.tiles_per_dir, plan.block_tokens
  bound_vmem = amax_scratch["hop_bound_vmem"]
  out_copies = []
  for d in range(2):
    _, buf_row, _, _ = mb_step.tile_offsets(d * tiles_per_dir)
    sl = pl.ds(d * block_tokens, block_tokens)
    send_tok0 = pl.multiple_of(jax.lax.div(buf_row, dim0), 8)
    out_copies.append(sparsecore_dma.push_copy(
        bound_vmem.at[sl], extra["amax_send"].at[pl.ds(send_tok0, block_tokens)], amax_scratch["hop_out_sem"],
    ))
  for cp in out_copies:
    cp.start()
  for cp in out_copies:
    cp.wait()
