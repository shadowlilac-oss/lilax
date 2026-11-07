"""SparseCore local VMEM to HBM pushes across Pallas async copy API versions."""

import inspect

from jax.experimental.pallas import tpu as pltpu

_ASYNC_COPY_HAS_SRC_SEM = "src_sem" in inspect.signature(pltpu.make_async_copy).parameters


def push_copy(src_ref, dst_ref, sem):
  """Local VMEM to HBM copy on SparseCore, tracked by sem."""
  if _ASYNC_COPY_HAS_SRC_SEM:
    return pltpu.make_async_copy(src_ref, dst_ref, sem=None, src_sem=sem)
  return pltpu.make_async_copy(src_ref, dst_ref, sem)


def fence_readback(hbm_slice, vmem_slice, sem):
  """Minimal read-back on sem to drain prior VMEM to HBM pushes."""
  if not _ASYNC_COPY_HAS_SRC_SEM:
    rb = pltpu.make_async_copy(hbm_slice, vmem_slice, sem)
    rb.start()
    rb.wait()
