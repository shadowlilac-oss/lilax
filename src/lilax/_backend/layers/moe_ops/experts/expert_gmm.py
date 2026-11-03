"""Grouped matmuls of this shard's local experts: routing parameters, weight-gather axes and tiling."""

import functools

import jax
import jax.numpy as jnp
from lilax._backend.layers.moe_ops.experts import up_projection


class LocalExpertGmm:
  """Configures the grouped matmul for the experts held by this expert shard.

  `config`, `mesh`, `expert_parallelism_name` and `expert_parallelism_size` are zero-arg callables resolved at the
  point of use, so each layer read happens where, and as often as, the layer performs it. `fwd_tile` is the layer's
  forward-tile reader, bound to the config the layer passed by value.
  """

  def __init__(
      self, config, fwd_tile, mesh, expert_parallelism_name, expert_parallelism_size, weight_gather, w0_pspec, wo_pspec,
      gmm,
  ):
    self.config = config
    self.fwd_tile = fwd_tile
    self.mesh = mesh
    self.expert_parallelism_name = expert_parallelism_name
    self.expert_parallelism_size = expert_parallelism_size
    self.weight_gather = weight_gather
    self.w0_pspec = w0_pspec
    self.wo_pspec = wo_pspec
    self.gmm = gmm

  def _non_expert_axes(self, pspec_dim_axes):
    """Drops the expert-parallel mesh axis name(s) from one PartitionSpec entry (flat dim 0 = (expert, fsdp))."""
    if pspec_dim_axes is None:
      return None
    ep = self.expert_parallelism_name()
    ep_names = set(ep) if isinstance(ep, tuple) else {ep}
    axes = (pspec_dim_axes,) if isinstance(pspec_dim_axes, str) else tuple(pspec_dim_axes)
    return tuple(ax for ax in axes if ax not in ep_names)

  def _active_sharding_axes(self, pspec_dim_axes, tensor_dim_index):
    if pspec_dim_axes is None:
      return []
    axes = (pspec_dim_axes,) if isinstance(pspec_dim_axes, str) else pspec_dim_axes
    active = []
    for ax in axes:
      if ax and self.mesh().shape.get(ax, 1) > 1:
        active.append((ax, tensor_dim_index))
    return active

  def wi_params(self):
    wi_gather_axes = []
    if self.weight_gather:
      # weight_gather implies either exp or embed_moe is sharded.
      if self.config().moe_flat_fsdp_weights:
        # Flat wi [E * In, Hidden]: gather the non-expert (FSDP) axes of dim 0, then Hidden(1).
        wi_gather_axes.extend(self._active_sharding_axes(self._non_expert_axes(self.w0_pspec[0]), 0))
        wi_gather_axes.extend(self._active_sharding_axes(self.w0_pspec[1], 1))
      else:
        if self.config().shard_exp_on_fsdp:
          # wi [Experts, In, Hidden] -> Gather Exp(0)
          wi_gather_axes.extend(self._active_sharding_axes(self.w0_pspec[0], 0))
        else:
          # Gather In(1) where embed_moe is sharded.
          wi_gather_axes.extend(self._active_sharding_axes(self.w0_pspec[1], 1))
        # Gather Hidden(2)
        wi_gather_axes.extend(self._active_sharding_axes(self.w0_pspec[2], 2))
    wi_tile_size = (
        self.fwd_tile("wi_tile_fwd_batch_seq"),  # m (LHS batch)
        self.fwd_tile("wi_tile_fwd_embed_dim"),  # k  (contracting)
        self.fwd_tile("wi_tile_fwd_mlp_dim"),  # n (RHS batch)
        self.config().wi_tile_dlhs_batch_seq,  # m (LHS batch)
        self.config().wi_tile_dlhs_mlp_dim,  # k (contracting)
        self.config().wi_tile_dlhs_embed_dim,  # n (RHS batch)
        self.config().wi_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
        self.config().wi_tile_drhs_embed_dim,  # Called k in megablox, but this is LHS batch dim
        self.config().wi_tile_drhs_mlp_dim,  # Called n in megablox, and indeed is RHS batch dim
    )
    return wi_gather_axes, wi_tile_size

  def wo_params(self):
    wo_gather_axes = []
    if self.weight_gather:
      # weight_gather implies either exp or embed_moe is sharded.
      if self.config().moe_flat_fsdp_weights:
        # Flat wo [E * Hidden, Out]: gather the non-expert (FSDP) axes of dim 0, then Out(1).
        wo_gather_axes.extend(self._active_sharding_axes(self._non_expert_axes(self.wo_pspec[0]), 0))
        wo_gather_axes.extend(self._active_sharding_axes(self.wo_pspec[1], 1))
      else:
        if self.config().shard_exp_on_fsdp:
          # wo [Experts, Hidden, Out] -> Gather Exp(0)
          wo_gather_axes.extend(self._active_sharding_axes(self.wo_pspec[0], 0))
        else:
          # Gather Out(2) where embed_moe is sharded.
          wo_gather_axes.extend(self._active_sharding_axes(self.wo_pspec[2], 2))
        # Gather Hidden(1)
        wo_gather_axes.extend(self._active_sharding_axes(self.wo_pspec[1], 1))
    wo_tile_size = (
        self.fwd_tile("wo_tile_fwd_batch_seq"),  # m (LHS batch)
        self.fwd_tile("wo_tile_fwd_mlp_dim"),  # k (contracting)
        self.fwd_tile("wo_tile_fwd_embed_dim"),  # n (RHS batch)
        self.config().wo_tile_dlhs_batch_seq,  # m (LHS batch)
        self.config().wo_tile_dlhs_embed_dim,  # k (contracting)
        self.config().wo_tile_dlhs_mlp_dim,  # n (RHS)
        self.config().wo_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
        self.config().wo_tile_drhs_mlp_dim,  # Called k in megablox, but this is LHS batch dim
        self.config().wo_tile_drhs_embed_dim,  # Called n in megablox, and indeed is the RHS batch dim
    )
    return wo_gather_axes, wo_tile_size

  def up(
      self,
      x,
      w0,
      w1,
      w0_bias,
      w1_bias,
      gmm_fn,
      partial_accum0=None,
      partial_accum1=None,
      mask=None,
      gather_weights=True,
      return_weights=False,
  ):
    """Run the two up-projections (gate + up) and apply the FFN activation."""
    wi_gather_axes, wi_tile_size = self.wi_params()
    if not gather_weights:
      wi_gather_axes = []
    return up_projection.up_projection(
        x, w0, w1, w0_bias, w1_bias, gmm_fn, wi_gather_axes, wi_tile_size, self.config(),
        partial_accum0=partial_accum0, partial_accum1=partial_accum1, mask=mask, return_weights=return_weights,
    )

  def valid_token_count(self, x, routing, route_metadata):
    """Rows of `x` the gmm actually computes; the rest is ragged-buffer padding."""
    num_ep = self.expert_parallelism_size()
    num_experts_per_shard = self.config().num_experts // num_ep
    if self.config().use_ring_of_experts and x.shape[0] < routing.sorted_selected_experts.shape[0]:
      return jnp.sum(routing.local_group_sizes)
    if self.config().use_ragged_sort and self.config().use_ring_of_experts:
      experts_start = route_metadata.expert_shard_id * num_experts_per_shard
      return jnp.sum(jax.lax.dynamic_slice_in_dim(routing.group_sizes, experts_start, num_experts_per_shard))
    return jnp.sum(routing.group_sizes)

  def for_local_experts(self, x, routing, route_metadata):
    """Return a partial GMM function with preconfigured routing params."""
    num_ep = self.expert_parallelism_size()
    num_experts_per_shard = self.config().num_experts // num_ep
    if self.config().use_ring_of_experts and x.shape[0] < routing.sorted_selected_experts.shape[0]:
      local_group_sizes = routing.local_group_sizes
      return functools.partial(
          self.gmm,
          group_sizes=local_group_sizes,
          expert_assignments=routing.selected_experts,
          group_offset=0,
      )
    if self.config().use_ragged_sort and self.config().use_ring_of_experts:
      experts_start = route_metadata.expert_shard_id * num_experts_per_shard
    else:
      experts_start = 0
    return functools.partial(
        self.gmm,
        group_sizes=routing.group_sizes,
        expert_assignments=routing.selected_experts,
        group_offset=experts_start,
    )
