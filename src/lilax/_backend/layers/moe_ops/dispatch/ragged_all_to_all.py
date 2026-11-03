"""Ragged all-to-all dispatch: send each routed token to its expert shard and return the expert outputs."""

import jax
import jax.numpy as jnp
from lilax._backend.kernels.ragged.sort import a2a_ragged_unsort
from lilax._backend.layers.moe_ops.dispatch import sorting
from lilax._backend.layers.moe_ops.dispatch.route_state import RouteMetadata, RouteOutput


class RaggedAllToAllDispatch:
  """Exchanges routed tokens between expert shards with ragged all-to-all, and undoes the exchange after the experts.

  Layer-owned values are zero-arg callables and `lookup_*` return the layer's (or class's) method; both are resolved
  at the point of use, so each layer read happens where, and as often as, the layer performs it.
  """

  def __init__(
      self,
      config,
      expert_parallelism_name,
      expert_parallelism_size,
      moe_expert_input_dim,
      ragged_buffer_factor,
      is_batch_sharded_by_expert,
      bias_axis_names,
      use_megatron_seq_aux_loss,
      lookup_permute,
      lookup_get_ragged_buffer_size,
      lookup_get_all_to_all_params,
      lookup_local_permute,
  ):
    self.config = config
    self.expert_parallelism_name = expert_parallelism_name
    self.expert_parallelism_size = expert_parallelism_size
    self.moe_expert_input_dim = moe_expert_input_dim
    self.ragged_buffer_factor = ragged_buffer_factor
    self.is_batch_sharded_by_expert = is_batch_sharded_by_expert
    self.bias_axis_names = bias_axis_names
    self.use_megatron_seq_aux_loss = use_megatron_seq_aux_loss
    self.lookup_permute = lookup_permute
    self.lookup_get_ragged_buffer_size = lookup_get_ragged_buffer_size
    self.lookup_get_all_to_all_params = lookup_get_all_to_all_params
    self.lookup_local_permute = lookup_local_permute

  def route(
      self,
      x,
      logits,
      pre_bias_logits,
      num_ep,
      expert_shard_id,
      rngs,
      input_ids=None,
      forced_routed_experts=None,
      force_dropless=False,
      per_expert_scale=None,
  ):
    local_sorted_indices = None
    all_shards_group_sizes = None
    reshaped_group_sizes = None
    # force_dropless (the in-layer dropless fallback) sizes the receive buffer for the worst case
    # instead of ragged_buffer_factor * balanced, so no tokens can be dropped.
    ragged_buffer_factor = -1.0 if force_dropless else self.ragged_buffer_factor()
    (
        x,
        sorted_selected_experts,
        weights,
        group_sizes,
        selected_experts,
        lb_loss,
        bias_updates,
        local_group_sizes,
        _,
        _,
        _,
        _,
    ) = self.lookup_permute()(
        x,
        logits,
        pre_bias_logits,
        self.config().use_custom_sort_vjp,
        rngs,
        input_ids=input_ids,
        forced_routed_experts=forced_routed_experts,
        mesh_axis_names=self.bias_axis_names,
        compute_lb_loss=not self.use_megatron_seq_aux_loss,
        return_expert_counts=True,
        per_expert_scale=per_expert_scale,
    )

    if num_ep > 1:
      batch_axis = self.expert_parallelism_name() if self.is_batch_sharded_by_expert else "data"
      # get group sizes for all shards
      local_expert_size = self.config().num_experts // num_ep
      reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
      global_group_sizes = group_sizes

      if self.is_batch_sharded_by_expert:
        all_shards_group_sizes = jax.lax.all_gather(reshaped_group_sizes, axis_name=batch_axis)
        buffer_size = self.lookup_get_ragged_buffer_size()(
            jnp.shape(x)[0],
            num_ep,
            self.config().num_experts,
            self.config().num_experts_per_tok,
            ragged_buffer_factor,
        )
        input_offsets, send_sizes, output_offsets, recv_sizes = self.lookup_get_all_to_all_params()(
            all_shards_group_sizes,
            expert_shard_id,
            num_ep,
            ragged_buffer_factor=ragged_buffer_factor,
            buffer_size=buffer_size,
        )

        output_shape = jax.lax.empty((buffer_size, self.moe_expert_input_dim()), dtype=x.dtype)

        x = jax.lax.ragged_all_to_all(
            x,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=self.expert_parallelism_name(),
        )
        global_group_sizes = jax.lax.all_gather(group_sizes, axis_name=self.expert_parallelism_name())
        x, local_sorted_indices, group_sizes, selected_experts = self.lookup_local_permute()(
            x,
            global_group_sizes,
            local_expert_size,
            shard_index=expert_shard_id,
            use_custom_sort_vjp=self.config().use_custom_sort_vjp,
            use_ragged_sort=self.config().use_ragged_sort,
            ragged_buffer_factor=ragged_buffer_factor,
            use_single_sparsecore=self.config().ragged_sort_use_single_sparsecore,
        )
      else:
        x, local_sorted_indices, group_sizes, selected_experts = self.lookup_local_permute()(
            x,
            global_group_sizes[None, :],
            local_expert_size,
            shard_index=expert_shard_id,
            is_offset=True,
            global_sorted_experts=selected_experts,
            use_custom_sort_vjp=self.config().use_custom_sort_vjp,
            use_ragged_sort=self.config().use_ragged_sort,
            ragged_buffer_factor=ragged_buffer_factor,
            use_single_sparsecore=self.config().ragged_sort_use_single_sparsecore,
        )

    return (
        x,
        RouteOutput(
            group_sizes=group_sizes,
            selected_experts=selected_experts,
            sorted_selected_experts=sorted_selected_experts,
            weights=weights,
            lb_loss=lb_loss,
            bias_updates=bias_updates,
            local_group_sizes=local_group_sizes,
        ),
        RouteMetadata(
            expert_shard_id=expert_shard_id,
            local_sorted_indices=local_sorted_indices,
            all_shards_group_sizes=all_shards_group_sizes,
            reshaped_group_sizes=reshaped_group_sizes,
        ),
    )

  def unsort_output_and_ra2a(self, intermediate_output, routing, route_metadata, output_shape):
    """Unsort tokens and return them to original shards using ragged all-to-all."""
    if self.is_batch_sharded_by_expert:
      # locally unpermute back to the original order
      if self.config().use_ragged_sort:
        # Mirror the ragged-prefix gather used in `local_permute`. The
        # un-permute can use the same valid-prefix length because the
        # routed token count is identical for forward and backward.
        valid_end = jnp.sum(routing.group_sizes).astype(jnp.int32)
        local_output = a2a_ragged_unsort(
            intermediate_output,
            jnp.argsort(route_metadata.local_sorted_indices),  # pylint: disable=undefined-variable
            valid_end,
            use_single_sparsecore=self.config().ragged_sort_use_single_sparsecore,
        )
      else:
        local_output = sorting._sort_activations(
            intermediate_output,
            jnp.argsort(route_metadata.local_sorted_indices),
            self.config().use_custom_sort_vjp,
        )

      buffer_size = intermediate_output.shape[0]
      input_offsets, send_sizes, output_offsets, recv_sizes = self.lookup_get_all_to_all_params()(
          route_metadata.all_shards_group_sizes,
          route_metadata.expert_shard_id,
          self.expert_parallelism_size(),
          ragged_buffer_factor=self.ragged_buffer_factor(),
          buffer_size=buffer_size,
          is_dispatch=False,
      )
      return jax.lax.ragged_all_to_all(
          local_output,
          output_shape,
          input_offsets,
          send_sizes,
          output_offsets,
          recv_sizes,
          axis_name=self.expert_parallelism_name(),
      )

    # If batch is replicated across EP shards then each shard should send
    # 0..local_shard_size data to the other shards and receive the
    # local_shard data from all of the other shards using ragged_all_to_all.
    input_offsets, send_sizes, output_offsets, recv_sizes = self.lookup_get_all_to_all_params()(
        route_metadata.reshaped_group_sizes,
        route_metadata.expert_shard_id,
        self.expert_parallelism_size(),
        is_batch_sharded=False,
        is_dispatch=False,
    )
    return jax.lax.ragged_all_to_all(
        intermediate_output,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=self.expert_parallelism_name(),
    )
