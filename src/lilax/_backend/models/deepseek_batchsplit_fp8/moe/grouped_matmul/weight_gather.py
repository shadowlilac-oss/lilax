"""Grouped matmul weight-gather axes for the DeepSeek FP8 batch-split routed experts."""

from lilax._backend.layers import moe as moe_lib
from lilax._backend.utils.sharding import logical_to_spec


def gmm_weight_gather_axes(config, mesh):
  """Returns the (wi, wo) megablox weight-gather axes, which are only populated for qwix quantization."""
  wi_gather_axes = []
  wo_gather_axes = []
  if config.quantization and config.use_qwix_quantization:
    gating_pspec, linear_pspec = moe_lib.get_batchsplit_init_kernel_axes()
    w0_pspec = logical_to_spec(gating_pspec)
    wo_pspec = logical_to_spec(linear_pspec)
    ignored_axes = ("expert", "tensor")

    def get_active_sharding_axes(pspec_dim_axes, tensor_dim_index):
      if pspec_dim_axes is None:
        return []
      axes = (pspec_dim_axes,) if isinstance(pspec_dim_axes, str) else pspec_dim_axes
      active = []
      for ax in axes:
        if ax and ax not in ignored_axes and mesh.shape.get(ax, 1) > 1:
          active.append((ax, tensor_dim_index))
      return active

    wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[0], 0))
    wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[2], 2))

    wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[0], 0))
    wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[1], 1))
  return wi_gather_axes, wo_gather_axes
