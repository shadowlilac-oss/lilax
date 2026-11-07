"""Global per-token inverse scales from the token absmax gathered across all devices."""

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

from lilax._backend.experimental.lineage.sc_reduce_scatter import fp8_scaling
from lilax._backend.experimental.lineage.sc_reduce_scatter import plan as plan_lib
from lilax._backend.experimental.lineage.sc_reduce_scatter import sparsecore_dma


def global_token_scales(
    plan: plan_lib.ReduceScatterPlan, subcore, amax_gather_ref, inv_rows_ref, tec_in_sem, tec_out_sem, *, amax_ready,
    amax_vmem, acc_vmem, rows_vmem, inv_vmem, **unused_scs_scratch,
):
  """Sums the gathered bounds of this subcore's tokens and writes their lane-broadcast inverse scales."""
  del inv_vmem, unused_scs_scratch
  num_tokens, num_devices, dim1 = plan.num_tokens, plan.num_devices, plan.dim1
  tokens_per_subcore = plan.tokens_per_subcore
  amax_chunk = plan_lib.AMAX_CHUNK
  pl.semaphore_wait(amax_ready, 1)
  tok_base = pl.multiple_of(subcore * tokens_per_subcore, tokens_per_subcore)

  def _slot_copy(k, buf):
    return pltpu.make_async_copy(
        amax_gather_ref.at[pl.ds(k * num_tokens + tok_base, tokens_per_subcore)], amax_vmem.at[buf], tec_in_sem.at[buf],
    )

  _slot_copy(0, 0).start()
  for k in range(num_devices):
    _slot_copy(k, k % 2).wait()
    if k + 1 < num_devices:
      _slot_copy(k + 1, (k + 1) % 2).start()
    if k == 0:
      acc_vmem[...] = amax_vmem[0]
    else:
      acc_vmem[...] += amax_vmem[k % 2]
  acc_vmem[...] = fp8_scaling.pow2_from_exponent(254 - fp8_scaling.pow2_exponent(acc_vmem[...]))

  def _row_copy(ci, buf):
    return sparsecore_dma.push_copy(
        rows_vmem.at[buf], inv_rows_ref.at[pl.ds(tok_base + ci * amax_chunk, amax_chunk), :], tec_out_sem.at[buf],
    )

  num_chunks = tokens_per_subcore // amax_chunk

  @pl.loop(0, num_chunks)
  def _chunk(ci):
    buf = jax.lax.rem(ci, 2)

    @pl.when(ci >= 2)
    def _wait_prev():
      _row_copy(ci - 2, buf).wait()

    v = acc_vmem[pl.ds(pl.multiple_of(ci * amax_chunk, amax_chunk), amax_chunk)]
    for j in range(amax_chunk):
      rows_vmem[buf, pl.ds(j, 1), :] = jnp.broadcast_to(v[j], (1, dim1))
    _row_copy(ci, buf).start()

  for ci in range(max(num_chunks - 2, 0), num_chunks):
    _row_copy(ci, ci % 2).wait()
  for ci in range(max(num_chunks - 2, 0), num_chunks):
    sparsecore_dma.fence_readback(
        inv_rows_ref.at[pl.ds(tok_base + ci * amax_chunk, amax_chunk), :], rows_vmem.at[ci % 2], tec_out_sem.at[ci % 2],
    )
  plsc.subcore_barrier()
