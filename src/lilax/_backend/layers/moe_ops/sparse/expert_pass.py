"""One sparse MoE pass over a token chunk: route the tokens, run the expert FFN and combine the expert outputs."""

from jax import ad_checkpoint as adc
import jax
import jax.numpy as jnp
from lilax._backend.kernels import megablox as mblx
from lilax._backend.kernels.ragged.sort_tc import tc_buffer_row_weights
from lilax._backend.layers.moe_ops.dispatch import combine, ring_of_experts
from lilax._backend.layers.moe_ops.sparse import emb_chunking
import qwix.pallas as qpl


class SparseExpertPass:
  """Routes a token chunk to its experts, runs the grouped-matmul expert FFN and returns the outputs in token order.

  Layer-owned values are zero-arg callables and `lookup_*` return the layer's bound method; both are resolved at the
  point of use, so each layer read happens where, and as often as, the layer performs it.
  """

  def __init__(
      self,
      config,
      num_experts_per_tok,
      dtype,
      moe_expert_input_dim,
      expert_parallelism_name,
      tensor_parallelism_name,
      expert_parallelism_size,
      tensor_parallelism_size,
      ragged_buffer_factor,
      expert_gmm,
      ring_dispatch,
      all_to_all_dispatch,
      lookup_apply_ffn_activation,
      lookup_transform_bias,
      lookup_unpermute,
  ):
    self.config = config
    self.num_experts_per_tok = num_experts_per_tok
    self.dtype = dtype
    self.moe_expert_input_dim = moe_expert_input_dim
    self.expert_parallelism_name = expert_parallelism_name
    self.tensor_parallelism_name = tensor_parallelism_name
    self.expert_parallelism_size = expert_parallelism_size
    self.tensor_parallelism_size = tensor_parallelism_size
    self.ragged_buffer_factor = ragged_buffer_factor
    self.expert_gmm = expert_gmm
    self.ring_dispatch = ring_dispatch
    self.all_to_all_dispatch = all_to_all_dispatch
    self.lookup_apply_ffn_activation = lookup_apply_ffn_activation
    self.lookup_transform_bias = lookup_transform_bias
    self.lookup_unpermute = lookup_unpermute

  def _route(
      self,
      x,
      logits,
      pre_bias_logits,
      rngs,
      input_ids=None,
      forced_routed_experts=None,
      force_dropless=False,
      per_expert_scale=None,
  ):
    """Performs both across device and within device token routing/sorting"""
    num_ep = self.expert_parallelism_size()
    expert_shard_id = jax.lax.axis_index(self.expert_parallelism_name()) if num_ep > 1 else 0

    if self.config().use_ring_of_experts:
      return self.ring_dispatch(
          x,
          logits,
          pre_bias_logits,
          num_ep,
          expert_shard_id,
          rngs,
          input_ids=input_ids,
          forced_routed_experts=forced_routed_experts,
          force_dropless=force_dropless,
          per_expert_scale=per_expert_scale,
      )
    else:
      return self.all_to_all_dispatch.route(
          x,
          logits,
          pre_bias_logits,
          num_ep,
          expert_shard_id,
          rngs,
          input_ids=input_ids,
          forced_routed_experts=forced_routed_experts,
          force_dropless=force_dropless,
          per_expert_scale=per_expert_scale,
      )

  def _ring_of_experts_combine(
      self,
      intermediate_output,
      routing,
      batch_size,
      sequence_length,
      tc_prescaled,
      x_is_3d,
      use_fused_tc_unsort,
      use_fused_combine_bwd_qarray,
  ):
    """Restores token order and reduce-scatters the expert outputs over the expert-parallel shards."""
    if use_fused_combine_bwd_qarray:
      return intermediate_output
    if use_fused_tc_unsort:
      output = intermediate_output.reshape(-1, sequence_length, *intermediate_output.shape[1:]).astype(self.dtype())
    else:
      output = self.lookup_unpermute()(
          intermediate_output,
          routing.sorted_selected_experts,
          routing.weights,
          batch_size=batch_size,
          sequence_length=sequence_length,
          use_custom_sort_vjp=self.config().use_custom_sort_vjp,
          group_sizes=routing.group_sizes,
          topk_argsort_indices=routing.topk_argsort_indices,
          tc_routing=routing.tc_routing,
          tc_prescaled=tc_prescaled,
          tc_out_3d=x_is_3d
          and ring_of_experts.uses_tc_3d_dispatch(self.config, self.tensor_parallelism_size, self.moe_expert_input_dim),
      )

    if output.ndim == 4:
      output = jnp.reshape(output, (-1, sequence_length, *output.shape[2:]))
    else:
      output = jnp.reshape(
          output,
          (
              -1,
              sequence_length,
              self.moe_expert_input_dim() // self.tensor_parallelism_size(),
          ),
      )
    combine_bwd_method = self.config().moe_quantize_combine_bwd_method
    if combine_bwd_method:
      output = combine._moe_combine_psum_scatter(
          output,
          self.expert_parallelism_name(),
          scatter_dimension=0,
          tiled=True,
          bwd_method=combine_bwd_method,
      )
    else:
      output = jax.lax.psum_scatter(
          output,
          self.expert_parallelism_name(),
          scatter_dimension=0,
          tiled=True,
      )
    if output.ndim == 4:
      output = jnp.reshape(output, (*output.shape[:2], -1))
    return output

  def _all_to_all_combine(self, intermediate_output, routing, route_metadata, batch_size, sequence_length):
    """Returns the expert outputs to their source shards and restores token order."""
    if self.expert_parallelism_size() > 1:
      original_inputs_first_dim = batch_size * sequence_length * self.config().num_experts_per_tok
      if routing.sorted_selected_experts.shape[0] != original_inputs_first_dim:
        raise ValueError("original_inputs_first_dim does not match the original tensor" " shape!")
      buffer_init = jnp.zeros if self.ragged_buffer_factor() > 0.0 else jax.lax.empty
      output_shape = buffer_init(
          (
              original_inputs_first_dim,
              self.moe_expert_input_dim() // self.tensor_parallelism_size(),
          ),
          dtype=intermediate_output.dtype,
      )

      intermediate_output = self.all_to_all_dispatch.unsort_output_and_ra2a(
          intermediate_output,
          routing,
          route_metadata,
          output_shape,
      )

    return self.lookup_unpermute()(
        intermediate_output,
        routing.sorted_selected_experts,
        routing.weights,
        batch_size=batch_size,
        sequence_length=sequence_length,
        use_custom_sort_vjp=self.config().use_custom_sort_vjp,
        group_sizes=routing.group_sizes,
    )

  def __call__(
      self,
      x,
      logits,
      pre_bias_logits,
      w0,
      w1,
      wo,
      w0_bias,
      w1_bias,
      wo_bias,
      sharded_input_ids,
      rngs,
      forced_routed_experts=None,
      force_dropless=False,
      gather_weights=True,
      return_weights=False,
      per_expert_scale=None,
  ):
    batch_size, sequence_length, embed_dim = x.shape
    if self.config().num_moe_emb_chunks > 0:
      output0, output1, gmm_fn, routing, route_metadata, wo_bias = emb_chunking.emb_chunked_up_projection(
          x,
          logits,
          pre_bias_logits,
          w0,
          w1,
          w0_bias,
          w1_bias,
          wo_bias,
          sharded_input_ids,
          rngs,
          embed_dim,
          forced_routed_experts=forced_routed_experts,
          per_expert_scale=per_expert_scale,
          config=self.config,
          expert_parallelism_size=self.expert_parallelism_size,
          expert_parallelism_name=self.expert_parallelism_name,
          tensor_parallelism_size=self.tensor_parallelism_size,
          ring_dispatch=self.ring_dispatch,
          expert_gmm=self.expert_gmm,
          lookup_transform_bias=self.lookup_transform_bias,
      )
    else:
      x, routing, route_metadata = self._route(
          x,
          logits,
          pre_bias_logits,
          rngs,
          input_ids=sharded_input_ids,
          forced_routed_experts=forced_routed_experts,
          force_dropless=force_dropless,
          per_expert_scale=per_expert_scale,
      )

      def _cn(t):
        return adc.checkpoint_name(t, "moe_x_sorted") if isinstance(t, jax.Array) else t

      x = jax.tree.map(_cn, x)
      routing = jax.tree.map(_cn, routing)
      route_metadata = jax.tree.map(_cn, route_metadata)
      mask = jnp.arange(x.shape[0]) < self.expert_gmm.valid_token_count(x, routing, route_metadata)

      if self.config().mlp_bias:
        if not self.config().use_te_grouped_gemm:
          w0_bias, w1_bias, wo_bias = self.lookup_transform_bias()(routing.selected_experts, w0_bias, w1_bias, wo_bias)
        elif self.tensor_parallelism_size() > 1:
          (wo_bias,) = self.lookup_transform_bias()(routing.selected_experts, wo_bias)

      gmm_fn = self.expert_gmm.for_local_experts(x, routing, route_metadata)
      up_res = self.expert_gmm.up(
          x,
          w0,
          w1,
          w0_bias,
          w1_bias,
          gmm_fn,
          mask=mask,
          gather_weights=gather_weights,
          return_weights=return_weights,
      )
      if return_weights:
        output0, output1, w0, w1 = up_res
      else:
        output0, output1 = up_res

    tc_prescaled = (
        self.config().use_ring_of_experts
        and getattr(routing, "tc_routing", None) is not None
        and self.config().moe_tc_ragged_weights_on_activation
    )
    if tc_prescaled:
      w_rows = tc_buffer_row_weights(routing.tc_routing, jnp.ravel(routing.weights), output0.shape[0])
      if self.config().moe_tc_routing_checkpoint != "remat":
        w_rows = adc.checkpoint_name(w_rows, "moe_tc_routing_checkpoint")
    intermediate_layer = self.lookup_apply_ffn_activation()(output0, output1)
    if tc_prescaled:
      intermediate_layer = (intermediate_layer * w_rows[:, None]).astype(intermediate_layer.dtype)
    wo_gather_axes, wo_tile_size = self.expert_gmm.wo_params()
    use_te_wo_bias = (
        self.config().use_te_grouped_gemm
        and self.config().mlp_bias
        and wo_bias is not None
        and self.tensor_parallelism_size() == 1
    )
    if not gather_weights:
      wo_gather_axes = []
    x_is_3d = self.config().num_moe_emb_chunks <= 0 and (x.qvalue if isinstance(x, qpl.QArray) else x).ndim == 3
    if x_is_3d and (self.tensor_parallelism_size() > 1 or self.config().mlp_bias):
      raise NotImplementedError("moe_tc_ragged_3d_gmm does not support tensor parallelism or mlp_bias.")
    use_fused_tc_unsort = self.config().moe_bwd_prequant_before_unsort and tc_prescaled and x_is_3d
    use_fused_combine_bwd_qarray = use_fused_tc_unsort and self.config().moe_combine_bwd_direct_qarray
    tc_unsort_kwargs = {}
    if use_fused_tc_unsort:
      tc_unsort_kwargs = {
          "tc_routing": routing.tc_routing,
          "tc_unsort_cfg": mblx.TcUnsortCfg(
              unpadded_cap=intermediate_layer.shape[0],
              num_out_tokens=routing.tc_routing.num_out_tokens,
              topk=self.num_experts_per_tok(),
              blocks=(
                  self.config().moe_tc_ragged_gather_block_size,
                  self.config().moe_tc_ragged_flatten_block_size,
                  True,
                  True,
              ),
              mask_padding=self.config().moe_tc_ragged_mask_padding,
              combine_rs_cfg=(
                  (self.expert_parallelism_name(), sequence_length) if use_fused_combine_bwd_qarray else None
              ),
          ),
      }
    intermediate_output = gmm_fn(
        intermediate_layer,
        wo,
        tiling=wo_tile_size,
        weight_gather_axes=wo_gather_axes,
        bias=wo_bias if use_te_wo_bias else None,
        return_rhs=return_weights,
        **({"out_is_3d": True} if x_is_3d else {}),
        **tc_unsort_kwargs,
    )
    if return_weights:
      intermediate_output, wo = intermediate_output
    if self.tensor_parallelism_size() > 1:
      intermediate_output = jax.lax.psum_scatter(
          intermediate_output,
          self.tensor_parallelism_name(),
          scatter_dimension=1,
          tiled=True,
      )
    if self.config().mlp_bias:
      mask = jnp.arange(intermediate_output.shape[0]) < self.expert_gmm.valid_token_count(
          intermediate_output, routing, route_metadata
      )
      if not use_te_wo_bias:
        if tc_prescaled:
          wo_bias = (wo_bias * w_rows[:, None]).astype(wo_bias.dtype)
        intermediate_output = intermediate_output + wo_bias
      intermediate_output = jnp.where(mask[:, None], intermediate_output, 0)
    intermediate_output = adc.checkpoint_name(adc.checkpoint_name(intermediate_output, "mlpwo"), "moe_mlpwo")

    if self.config().use_ring_of_experts:
      output = self._ring_of_experts_combine(
          intermediate_output, routing, batch_size, sequence_length, tc_prescaled, x_is_3d,
          use_fused_tc_unsort, use_fused_combine_bwd_qarray,
      )
    else:
      output = self._all_to_all_combine(intermediate_output, routing, route_metadata, batch_size, sequence_length)
    return (
        output,
        routing.lb_loss,
        routing.bias_updates,
        routing.has_overflow,
        routing.required_rbf,
        routing.max_load_ratio,
        (w0, w1, wo),
    )
