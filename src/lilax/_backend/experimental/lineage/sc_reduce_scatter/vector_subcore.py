"""Vector subcore pipeline that loads, reduces and stores the tiles of every ring step and microbatch."""

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu

from lilax._backend.experimental.lineage.sc_reduce_scatter import global_scales
from lilax._backend.experimental.lineage.sc_reduce_scatter import hop_scales
from lilax._backend.experimental.lineage.sc_reduce_scatter import microbatch_step
from lilax._backend.experimental.lineage.sc_reduce_scatter import plan as plan_lib
from lilax._backend.experimental.lineage.sc_reduce_scatter import sparsecore_dma
from lilax._backend.experimental.lineage.sc_reduce_scatter import tile_reduce


def vector_subcore_kernel(
    plan: plan_lib.ReduceScatterPlan, *refs, scs_to_tec, tec_to_scs, d2d_send_sem, d2d_recv_sem, c2c_send_sem,
    c2c_recv_sem, a_vmem, b_vmem, c_vmem, tec_in_sem, tec_out_sem, **amax_scratch,
):
  """Reduces each ring step's microbatches once the scalar subcore signals that their rows arrived."""
  del d2d_send_sem, d2d_recv_sem, c2c_send_sem, c2c_recv_sem
  ring_size, num_microbatches, vmem_tile_rows = plan.ring_size, plan.num_microbatches, plan.vmem_tile_rows
  dim0, tiles_per_dir, num_tiles, block_tokens = plan.dim0, plan.tiles_per_dir, plan.num_tiles, plan.block_tokens
  x_ref, amax_ref, out_ref, running_sum_ref, d2d_recv_ref, c2c_recv_ref, extra = plan.unpack_refs(refs)
  z = jax.lax.axis_index(plan.ring_axis)
  c = jax.lax.axis_index(plan.core_axis)
  subcore = jax.lax.axis_index("subcore")
  if plan.global_scaling:
    global_scales.global_token_scales(
        plan, subcore, extra["amax_gather"], extra["inv_rows"], tec_in_sem, tec_out_sem, **amax_scratch
    )

  def _run_microbatch(s, m, *, three_way: bool, to_final_out: bool):
    mb_step = microbatch_step.MicrobatchStep(
        plan=plan, z=z, c=c, subcore=subcore, s=s, m=m, three_way=three_way, to_final_out=to_final_out,
    )
    sync_slot = jax.lax.rem(s * num_microbatches + m, 2)
    pl.semaphore_wait(scs_to_tec.at[sync_slot], 1)

    if plan.unfused_per_hop:
      bound_copies = hop_scales.unfused_hop_token_scales(mb_step, amax_ref, extra, amax_scratch)
    else:
      bound_copies = []
      if plan.fused_per_hop:
        hop_scales.fused_hop_prologue(mb_step, extra, amax_scratch)

    def _load_copies(tile_idx, slot):
      x_row, buf_row, prev_c2c_row, _ = mb_step.tile_offsets(tile_idx)
      in_sem = tec_in_sem.at[slot]
      copies = [
          pltpu.make_async_copy(x_ref.at[pl.ds(x_row, vmem_tile_rows), :], a_vmem.at[slot], sem=in_sem),
          pltpu.make_async_copy(d2d_recv_ref.at[pl.ds(buf_row, vmem_tile_rows), :], b_vmem.at[slot], sem=in_sem),
      ]
      if three_way:
        copies.append(pltpu.make_async_copy(
            c2c_recv_ref.at[pl.ds(prev_c2c_row, vmem_tile_rows), :], c_vmem.at[slot], sem=in_sem,
        ))
      if plan.global_scaling:
        tile_tok0 = pl.multiple_of(jax.lax.div(x_row, dim0), plan.tokens_per_tile)
        copies.append(pltpu.make_async_copy(
            extra["inv_rows"].at[pl.ds(tile_tok0, plan.tokens_per_tile), :], amax_scratch["inv_vmem"].at[slot],
            sem=in_sem,
        ))
      return copies

    def _store_copy(tile_idx, slot):
      _, buf_row, _, out_row = mb_step.tile_offsets(tile_idx)
      dst_ref = out_ref if to_final_out else running_sum_ref
      dst_row = out_row if to_final_out else buf_row
      src_vmem = a_vmem if plan.quantized_input or to_final_out else c_vmem
      return sparsecore_dma.push_copy(
          src_vmem.at[slot], dst_ref.at[pl.ds(dst_row, vmem_tile_rows), :], tec_out_sem.at[slot]
      )

    for cp in _load_copies(0, 0):
      cp.start()

    @pl.loop(0, num_tiles)
    def _tile_loop(tile_idx):
      slot = jax.lax.rem(tile_idx, 2)
      nxt_slot = 1 - slot
      for cp in _load_copies(tile_idx, slot):
        cp.wait()

      @pl.when(tile_idx >= 1)
      def _wait_prev_store():
        _store_copy(tile_idx - 1, nxt_slot).wait()

      @pl.when(tile_idx + 1 < num_tiles)
      def _start_next_loads():
        for cp in _load_copies(tile_idx + 1, nxt_slot):
          cp.start()

      tile_reduce.reduce_tile(mb_step, slot, tile_idx, a_vmem, b_vmem, c_vmem, amax_scratch)
      _store_copy(tile_idx, slot).start()

    _store_copy(num_tiles - 1, (num_tiles - 1) % 2).wait()
    if plan.fused_per_hop and not to_final_out:
      hop_scales.fused_hop_epilogue(mb_step, extra, amax_scratch)
    for cp in bound_copies:
      cp.wait()
    if not to_final_out:
      src_v = a_vmem if plan.quantized_input else c_vmem
      for k in range(min(2, num_tiles)):
        t_idx = num_tiles - 1 - k
        t_slot = t_idx % 2
        _, t_buf_row, _, _ = mb_step.tile_offsets(t_idx)
        tail_row = pl.multiple_of(t_buf_row + vmem_tile_rows - 8, 8)
        sparsecore_dma.fence_readback(
            running_sum_ref.at[pl.ds(tail_row, 8), :], src_v.at[t_slot, pl.ds(vmem_tile_rows - 8, 8), :],
            tec_out_sem.at[t_slot],
        )
      if plan.per_hop_scaling:
        _, d1_buf_row, _, _ = mb_step.tile_offsets(tiles_per_dir)
        d1_send_tok = pl.multiple_of(jax.lax.div(d1_buf_row, dim0) + block_tokens - 8, 8)
        sparsecore_dma.fence_readback(
            extra["amax_send"].at[pl.ds(d1_send_tok, 8)],
            amax_scratch["hop_bound_vmem"].at[pl.ds(2 * block_tokens - 8, 8)], amax_scratch["hop_out_sem"],
        )
    pl.semaphore_signal(tec_to_scs.at[sync_slot], 1)

  @pl.loop(0, num_microbatches)
  def _tec_step0(m):
    _run_microbatch(0, m, three_way=False, to_final_out=False)

  if ring_size > 2:
    @pl.loop(0, ring_size - 2)
    def _tec_mid_steps(h):
      @pl.loop(0, num_microbatches)
      def _tec_mid_mb(m):
        _run_microbatch(h + 1, m, three_way=True, to_final_out=False)

  @pl.loop(0, num_microbatches)
  def _tec_final_step(m):
    _run_microbatch(ring_size - 1, m, three_way=True, to_final_out=True)
