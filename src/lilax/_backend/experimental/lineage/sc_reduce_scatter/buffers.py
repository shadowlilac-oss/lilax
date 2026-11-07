"""HBM outputs and per-subcore scratch of the reduce-scatter kernel."""

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lilax._backend.experimental.lineage.sc_reduce_scatter import plan as plan_lib


def kernel_out_types(plan: plan_lib.ReduceScatterPlan, x_dtype, manual_axis_type) -> tuple[jax.ShapeDtypeStruct, ...]:
  """Returns the output, running sum and receive buffers, then the scaling buffers named by `plan.extra_names`."""
  dim1, c2c_buf_rows = plan.dim1, plan.c2c_buf_rows
  out_struct = jax.ShapeDtypeStruct((plan.chunk_rows, dim1), x_dtype, manual_axis_type=manual_axis_type)
  running_sum_struct = jax.ShapeDtypeStruct((c2c_buf_rows, dim1), jnp.float8_e4m3fn, manual_axis_type=manual_axis_type)
  d2d_recv_struct = jax.ShapeDtypeStruct(
      (plan.ring_size * plan.num_microbatches * 2 * plan.mb_rows, dim1), x_dtype, manual_axis_type=manual_axis_type,
  )
  c2c_recv_struct = jax.ShapeDtypeStruct((c2c_buf_rows, dim1), jnp.float8_e4m3fn, manual_axis_type=manual_axis_type)
  out_type = (out_struct, running_sum_struct, d2d_recv_struct, c2c_recv_struct)

  def _f32_struct(shape):
    return jax.ShapeDtypeStruct(shape, jnp.float32, manual_axis_type=manual_axis_type)

  if plan.global_scaling:
    out_type += (_f32_struct((plan.num_devices * plan.num_tokens,)), _f32_struct((plan.num_tokens, dim1)))
  elif plan.per_hop_scaling:
    c2c_buf_tokens = c2c_buf_rows // plan.dim0
    if plan.unfused_per_hop:
      out_type += (_f32_struct((plan.num_tokens,)),)
    out_type += (_f32_struct((c2c_buf_tokens,)), _f32_struct((c2c_buf_tokens,)))
  return out_type


def kernel_scratch_types(plan: plan_lib.ReduceScatterPlan, x_dtype, scs_mesh, tec_mesh) -> dict:
  """Returns the semaphores and VMEM buffers of both subcore meshes, keyed by kernel argument name."""
  ring_size, num_microbatches, dim0, dim1 = plan.ring_size, plan.num_microbatches, plan.dim0, plan.dim1
  vmem_shape = (2, plan.vmem_tile_rows, dim1)
  scratch_types = dict(
      scs_to_tec=pltpu.SemaphoreType.REGULAR((2,)) @ tec_mesh,
      tec_to_scs=pltpu.SemaphoreType.REGULAR((2,)) @ scs_mesh,
      d2d_send_sem=pltpu.SemaphoreType.DMA((ring_size, num_microbatches, 2)) @ scs_mesh,
      d2d_recv_sem=pltpu.SemaphoreType.DMA((ring_size, num_microbatches, 2)) @ scs_mesh,
      c2c_send_sem=pltpu.SemaphoreType.DMA((ring_size - 1, num_microbatches, 2)) @ scs_mesh,
      c2c_recv_sem=pltpu.SemaphoreType.DMA((ring_size - 1, num_microbatches, 2)) @ scs_mesh,
      a_vmem=pltpu.VMEM(vmem_shape, x_dtype) @ tec_mesh,
      b_vmem=pltpu.VMEM(vmem_shape, x_dtype) @ tec_mesh,
      c_vmem=pltpu.VMEM(vmem_shape, jnp.float8_e4m3fn) @ tec_mesh,
      tec_in_sem=pltpu.SemaphoreType.DMA((2,)) @ tec_mesh,
      tec_out_sem=pltpu.SemaphoreType.DMA((2,)) @ tec_mesh,
  )
  if plan.global_scaling:
    scratch_types.update(
        amax_send_sem=pltpu.SemaphoreType.DMA(()) @ scs_mesh,
        amax_recv_sem=pltpu.SemaphoreType.DMA(()) @ scs_mesh,
        amax_ready=pltpu.SemaphoreType.REGULAR(()) @ tec_mesh,
        amax_vmem=pltpu.VMEM((2, plan.tokens_per_subcore), jnp.float32) @ tec_mesh,
        acc_vmem=pltpu.VMEM((plan.tokens_per_subcore,), jnp.float32) @ tec_mesh,
        rows_vmem=pltpu.VMEM((2, plan_lib.AMAX_CHUNK, dim1), jnp.float32) @ tec_mesh,
        inv_vmem=pltpu.VMEM((2, plan.tokens_per_tile, dim1), jnp.float32) @ tec_mesh,
    )
  elif plan.per_hop_scaling:
    c2c_sem_shape = (ring_size - 1, num_microbatches, 2)

    def _hop_vmem():
      return pltpu.VMEM((2 * plan.block_tokens,), jnp.float32) @ tec_mesh

    scratch_types.update(
        c2c_amax_send_sem=pltpu.SemaphoreType.DMA(c2c_sem_shape) @ scs_mesh,
        c2c_amax_recv_sem=pltpu.SemaphoreType.DMA(c2c_sem_shape) @ scs_mesh,
        hop_in_sem=pltpu.SemaphoreType.DMA(()) @ tec_mesh,
        hop_out_sem=pltpu.SemaphoreType.DMA(()) @ tec_mesh,
        hop_recv_vmem=_hop_vmem(), hop_bound_vmem=_hop_vmem(), hop_prev_scale_vmem=_hop_vmem(),
    )
    if plan.unfused_per_hop:
      scratch_types.update(
          amax_d2d_send_sem=pltpu.SemaphoreType.DMA(()) @ scs_mesh,
          amax_d2d_recv_sem=pltpu.SemaphoreType.DMA(()) @ scs_mesh,
          hop_local_vmem=_hop_vmem(), hop_partner_vmem=_hop_vmem(), hop_inv_vmem=_hop_vmem(),
      )
    else:
      scratch_types.update(sum_vmem=pltpu.VMEM((dim0, dim1), jnp.float32) @ tec_mesh)
  return scratch_types
