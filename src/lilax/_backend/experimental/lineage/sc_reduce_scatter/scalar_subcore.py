"""Scalar subcore schedule of the die-to-die and chip-to-chip transfers of the reduce-scatter."""

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu

from lilax._backend.experimental.lineage.sc_reduce_scatter import plan as plan_lib


def _wait_for_peers(plan: plan_lib.ReduceScatterPlan, my_slot, partner_c, right_z, left_z):
  """Signals and waits on the barrier of every device the kernel exchanges data with."""
  ring_axis, core_axis, ring_size, num_devices = plan.ring_axis, plan.core_axis, plan.ring_size, plan.num_devices
  barrier_sem = pltpu.get_barrier_semaphore()
  if plan.global_scaling:
    for k in range(1, num_devices):
      slot = jax.lax.rem(my_slot + k, num_devices)
      pl.semaphore_signal(barrier_sem, 1, device_id={ring_axis: jax.lax.div(slot, 2), core_axis: jax.lax.rem(slot, 2)})
    pl.semaphore_wait(barrier_sem, num_devices - 1)
  else:
    pl.semaphore_signal(barrier_sem, 1, device_id={core_axis: partner_c})
    pl.semaphore_signal(barrier_sem, 1, device_id={ring_axis: right_z})
    if ring_size > 2:
      pl.semaphore_signal(barrier_sem, 1, device_id={ring_axis: left_z})
      pl.semaphore_wait(barrier_sem, 3)
    else:
      pl.semaphore_wait(barrier_sem, 2)


def scalar_subcore_kernel(
    plan: plan_lib.ReduceScatterPlan, *refs, scs_to_tec, tec_to_scs, d2d_send_sem, d2d_recv_sem, c2c_send_sem,
    c2c_recv_sem, a_vmem, b_vmem, c_vmem, tec_in_sem, tec_out_sem, **amax_scratch,
):
  """Issues the remote copies of every ring step and hands each microbatch to the vector subcores."""
  del a_vmem, b_vmem, c_vmem, tec_in_sem, tec_out_sem
  ring_axis, core_axis, ring_size, num_devices = plan.ring_axis, plan.core_axis, plan.ring_size, plan.num_devices
  num_tokens, num_microbatches = plan.num_tokens, plan.num_microbatches
  mb_rows, mb_tokens = plan.mb_rows, plan.mb_tokens
  global_scaling, unfused_per_hop, per_hop_scaling = plan.global_scaling, plan.unfused_per_hop, plan.per_hop_scaling
  x_ref, amax_ref, _, running_sum_ref, d2d_recv_ref, c2c_recv_ref, extra = plan.unpack_refs(refs)
  z = jax.lax.axis_index(ring_axis)
  c = jax.lax.axis_index(core_axis)
  right_z = jax.lax.rem(z + 1, ring_size)
  left_z = jax.lax.rem(z + ring_size - 1, ring_size)
  partner_c = 1 - c
  my_slot = z * 2 + c
  _wait_for_peers(plan, my_slot, partner_c, right_z, left_z)

  def _amax_copy(k):
    slot = jax.lax.rem(my_slot + k, num_devices)
    return pltpu.make_async_remote_copy(
        amax_ref, extra["amax_gather"].at[pl.ds(my_slot * num_tokens, num_tokens)],
        amax_scratch["amax_send_sem"], amax_scratch["amax_recv_sem"],
        device_id={ring_axis: jax.lax.div(slot, 2), core_axis: jax.lax.rem(slot, 2)},
    )

  def _d2d_copies(s, m):
    cw_z_dst = jax.lax.rem(z + ring_size - 1 - s, ring_size)
    ccw_z_dst = jax.lax.rem(z + 1 + s, ring_size)
    src_cw = pl.multiple_of((((cw_z_dst * 2 + partner_c) * 2 + 0) * num_microbatches + m) * mb_rows, 8)
    src_ccw = pl.multiple_of((((ccw_z_dst * 2 + partner_c) * 2 + 1) * num_microbatches + m) * mb_rows, 8)
    dst_cw = pl.multiple_of(((s * num_microbatches + m) * 2 + 0) * mb_rows, 8)
    dst_ccw = pl.multiple_of(((s * num_microbatches + m) * 2 + 1) * mb_rows, 8)
    d2d_cw = pltpu.make_async_remote_copy(
        x_ref.at[pl.ds(src_cw, mb_rows), :], d2d_recv_ref.at[pl.ds(dst_cw, mb_rows), :],
        d2d_send_sem.at[s, m, 0], d2d_recv_sem.at[s, m, 0], device_id={core_axis: partner_c},
    )
    d2d_ccw = pltpu.make_async_remote_copy(
        x_ref.at[pl.ds(src_ccw, mb_rows), :], d2d_recv_ref.at[pl.ds(dst_ccw, mb_rows), :],
        d2d_send_sem.at[s, m, 1], d2d_recv_sem.at[s, m, 1], device_id={core_axis: partner_c},
    )
    return d2d_cw, d2d_ccw

  def _c2c_copies(h, m):
    copies = []
    for d, dst_z in ((0, right_z), (1, left_z)):
      row = pl.multiple_of(((h * num_microbatches + m) * 2 + d) * mb_rows, 8)
      copies.append(pltpu.make_async_remote_copy(
          running_sum_ref.at[pl.ds(row, mb_rows), :], c2c_recv_ref.at[pl.ds(row, mb_rows), :],
          c2c_send_sem.at[h, m, d], c2c_recv_sem.at[h, m, d], device_id={ring_axis: dst_z},
      ))
      if per_hop_scaling:
        tok = pl.multiple_of(((h * num_microbatches + m) * 2 + d) * mb_tokens, 8)
        copies.append(pltpu.make_async_remote_copy(
            extra["amax_send"].at[pl.ds(tok, mb_tokens)], extra["amax_recv"].at[pl.ds(tok, mb_tokens)],
            amax_scratch["c2c_amax_send_sem"].at[h, m, d], amax_scratch["c2c_amax_recv_sem"].at[h, m, d],
            device_id={ring_axis: dst_z},
        ))
    return copies

  def _amax_partner_copy():
    return pltpu.make_async_remote_copy(
        amax_ref, extra["amax_partner"], amax_scratch["amax_d2d_send_sem"],
        amax_scratch["amax_d2d_recv_sem"], device_id={core_axis: partner_c},
    )

  def _signal_and_wait_tec(slot):
    for subcore in range(plan_lib.NUM_SUBCORES):
      pl.semaphore_signal(scs_to_tec.at[slot], 1, device_id={"subcore": subcore})
    pl.semaphore_wait(tec_to_scs.at[slot], plan_lib.NUM_SUBCORES)

  d2d0_cw, d2d0_ccw = _d2d_copies(0, 0)
  d2d0_cw.start()
  d2d0_ccw.start()
  if global_scaling:
    for k in range(num_devices):
      _amax_copy(k).start()
    for k in range(num_devices):
      _amax_copy(k).wait_recv()
    for subcore in range(plan_lib.NUM_SUBCORES):
      pl.semaphore_signal(amax_scratch["amax_ready"], 1, device_id={"subcore": subcore})
  elif unfused_per_hop:
    _amax_partner_copy().start()
    _amax_partner_copy().wait_recv()

  @pl.loop(0, num_microbatches)
  def _step0_mb(m):
    slot = jax.lax.rem(m, 2)

    @pl.when(m + 1 < num_microbatches)
    def _prefetch_step0_d2d():
      nxt_cw, nxt_ccw = _d2d_copies(0, m + 1)
      nxt_cw.start()
      nxt_ccw.start()

    cur_cw, cur_ccw = _d2d_copies(0, m)
    cur_cw.wait_recv()
    cur_ccw.wait_recv()
    nxt_hop_cw, nxt_hop_ccw = _d2d_copies(1, m)
    nxt_hop_cw.start()
    nxt_hop_ccw.start()
    _signal_and_wait_tec(slot)
    for cp in _c2c_copies(0, m):
      cp.start()

  if ring_size > 2:
    @pl.loop(0, ring_size - 2)
    def _mid_steps(h):
      s = h + 1

      @pl.loop(0, num_microbatches)
      def _mid_mb(m):
        slot = jax.lax.rem(s * num_microbatches + m, 2)
        for cp in _c2c_copies(h, m):
          cp.wait_recv()
        cur_d2d_cw, cur_d2d_ccw = _d2d_copies(s, m)
        cur_d2d_cw.wait_recv()
        cur_d2d_ccw.wait_recv()
        nxt_d2d_cw, nxt_d2d_ccw = _d2d_copies(s + 1, m)
        nxt_d2d_cw.start()
        nxt_d2d_ccw.start()
        _signal_and_wait_tec(slot)
        for cp in _c2c_copies(s, m):
          cp.start()

  last_h = ring_size - 2
  last_s = ring_size - 1

  @pl.loop(0, num_microbatches)
  def _final_mb(m):
    slot = jax.lax.rem(last_s * num_microbatches + m, 2)
    for cp in _c2c_copies(last_h, m):
      cp.wait_recv()
    cur_d2d_cw, cur_d2d_ccw = _d2d_copies(last_s, m)
    cur_d2d_cw.wait_recv()
    cur_d2d_ccw.wait_recv()
    _signal_and_wait_tec(slot)

  if global_scaling:
    for k in range(num_devices):
      _amax_copy(k).wait_send()
  elif unfused_per_hop:
    _amax_partner_copy().wait_send()

  @pl.loop(0, ring_size)
  def _wait_d2d_sends(s):
    @pl.loop(0, num_microbatches)
    def _wait_d2d_mb(m):
      d2d_cw, d2d_ccw = _d2d_copies(s, m)
      d2d_cw.wait_send()
      d2d_ccw.wait_send()

  @pl.loop(0, ring_size - 1)
  def _wait_c2c_sends(h):
    @pl.loop(0, num_microbatches)
    def _wait_c2c_mb(m):
      for cp in _c2c_copies(h, m):
        cp.wait_send()
