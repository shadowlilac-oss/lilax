"""FSDP all-gather of the expert weights and biases ahead of the sparse MoE shard_map."""

import jax
from lilax._backend.layers.moe_ops.sparse.partition_specs import reduced_axes


def gather_expert_weights(
    w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias, shardings, pin_kernel_gather, fsdp_all_gather
):
  """All-gathers the expert kernels and biases to their shard_map pspecs; `reduced` kernels are already gathered.

  With `pin_kernel_gather`, an optimization barrier keeps the kernel gather inside the layer scan.
  """
  if pin_kernel_gather:
    w0_kernel, w1_kernel, wo_kernel = jax.lax.optimization_barrier((w0_kernel, w1_kernel, wo_kernel))

  w0_kernel = w0_kernel if reduced_axes(w0_kernel) else fsdp_all_gather(w0_kernel, shardings.w0)
  w1_kernel = w1_kernel if reduced_axes(w1_kernel) else fsdp_all_gather(w1_kernel, shardings.w1)
  wo_kernel = wo_kernel if reduced_axes(wo_kernel) else fsdp_all_gather(wo_kernel, shardings.wo)
  if w0_bias is not None:
    w0_bias = fsdp_all_gather(w0_bias, shardings.w0_bias)
  if w1_bias is not None:
    w1_bias = fsdp_all_gather(w1_bias, shardings.w1_bias)
  if wo_bias is not None:
    wo_bias = fsdp_all_gather(wo_bias, shardings.wo_bias)
  return w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
