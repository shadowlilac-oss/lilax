"""Ring-of-experts dispatch: all-gather the tokens over the expert-parallel axis, then route them within each shard."""

import dataclasses
import functools

import jax
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
from lilax._backend.layers.moe_ops.dispatch.route_state import RouteMetadata, RouteOutput
import qwix.pallas as qpl


def uses_tc_3d_dispatch(config, tensor_parallelism_size, moe_expert_input_dim):
  """moe_tc_ragged_3d_dispatch: tokens are all-gathered, TC-sorted and combined in (emb // 128, 128) layout.

  Raises instead of silently falling back to the 2D layout when the flag is set but cannot apply.
  `config`, `tensor_parallelism_size` and `moe_expert_input_dim` are zero-arg callables, resolved at each use.
  """
  if not config().moe_tc_ragged_3d_dispatch:
    return False
  requirements = {
      "moe_tc_ragged_sort=True": config().moe_tc_ragged_sort,
      "moe_tc_ragged_3d_gmm=True": config().moe_tc_ragged_3d_gmm,
      "use_ring_of_experts=True": config().use_ring_of_experts,
      "use_ragged_sort=True": config().use_ragged_sort,
      "num_moe_emb_chunks <= 0": config().num_moe_emb_chunks <= 0,
      "no tensor parallelism": tensor_parallelism_size() == 1,
      "mlp_bias=False": not config().mlp_bias,
      f"MoE expert input dim ({moe_expert_input_dim()}) % 128 == 0": moe_expert_input_dim() % 128 == 0,
  }
  unmet = [name for name, ok in requirements.items() if not ok]
  if unmet:
    raise ValueError(f"moe_tc_ragged_3d_dispatch=True requires: {', '.join(unmet)}.")
  return True


def quantize_and_all_gather_tokens(
    x: jax.Array,
    axis_name: str,
    calibration_method: str,
    all_gather_fn=jax.lax.all_gather,
) -> qpl.QArray:
  """Quantizes tokens to act_qtype (e.g., FP8) before All-Gather across EP shards in Ring of Experts."""
  # Static calibration guarantees uniform dequantization bounds across all EP shards
  # without cross-shard dynamic scale synchronization collectives.
  if not calibration_method.lower().startswith("fixed"):
    raise ValueError(
        "moe_quantize_token_all_gather currently only supports static"
        f" (fixed) calibration, got {calibration_method}."
    )
  quantization_rule = qpl.get_current_rule("gmm")
  if quantization_rule is None:
    raise ValueError("Quantization rule is None, cannot quantize activation.")
  # The gmm backward (ops._lhs_rhs_share_fixed_scale) assumes this QArray's fixed scale is the rule's act
  # scale, so it must be quantized with the rule's act calibration.
  if calibration_method != quantization_rule.act_calibration_method:
    raise ValueError(
        f"moe_quantize_token_all_gather calibration {calibration_method!r} must match the gmm rule's"
        f" act_calibration_method {quantization_rule.act_calibration_method!r}."
    )
  act_qtype = quantization_rule.act_qtype

  # Quantize local token activations using the GMM activation quantization rule.
  x_input = qpl.quantize(
      x,
      qtype=act_qtype,
      channelwise_axes=(),
      calibration_method=calibration_method,
  )
  # All-gather quantized byte payloads to reduce inter-chip communication volume.
  x_gathered_qval = all_gather_fn(x_input.qvalue)
  return dataclasses.replace(x_input, qvalue=x_gathered_qval)


def _ep_all_gather_fn(config, expert_parallelism_name):
  """Tiled all-gather over the expert-parallel axis, pinned to a SparseCore when configured."""
  if config().moe_pin_sparse_core_all_gathers or getattr(config(), "moe_pin_sparse_core_ep_all_gathers", False):

    @functools.partial(
        compute_on,
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [config().moe_ep_all_gather_sparse_core_id]}},
    )
    def _ep_all_gather(z):
      return jax.lax.all_gather(z, axis_name=expert_parallelism_name(), tiled=True)

  else:

    def _ep_all_gather(z):
      return jax.lax.all_gather(z, axis_name=expert_parallelism_name(), tiled=True)

  return _ep_all_gather


class RingOfExpertsDispatch:
  """Duplicates the tokens (and their routing inputs) on every expert shard, then routes them to the local experts.

  Layer-owned values (including `config`) are zero-arg callables and `lookup_*` return the layer's bound method; both
  are resolved at the point of use, so each layer read happens where, and as often as, the layer performs it.
  """

  def __init__(
      self,
      config,
      expert_parallelism_name,
      is_hash_routing,
      dtype,
      tensor_parallelism_size,
      moe_expert_input_dim,
      bias_axis_names,
      use_megatron_seq_aux_loss,
      defer_reductions,
      lookup_get_topk,
      lookup_permute,
      lookup_required_ragged_buffer_factor,
  ):
    self.config = config
    self.expert_parallelism_name = expert_parallelism_name
    self.is_hash_routing = is_hash_routing
    self.dtype = dtype
    self.tensor_parallelism_size = tensor_parallelism_size
    self.moe_expert_input_dim = moe_expert_input_dim
    self.bias_axis_names = bias_axis_names
    self.use_megatron_seq_aux_loss = use_megatron_seq_aux_loss
    self.defer_reductions = defer_reductions
    self.lookup_get_topk = lookup_get_topk
    self.lookup_permute = lookup_permute
    self.lookup_required_ragged_buffer_factor = lookup_required_ragged_buffer_factor

  def __call__(
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
    _ep_all_gather = _ep_all_gather_fn(self.config, self.expert_parallelism_name)

    if uses_tc_3d_dispatch(self.config, self.tensor_parallelism_size, self.moe_expert_input_dim):
      # moe_tc_ragged_3d_dispatch: all-gather the tokens in the TC kernels' (emb // 128, 128) layout; the
      # reshape is on the local tokens, so the gathered tensor needs no relayout before the TC sort.
      def token_all_gather(z):
        return _ep_all_gather(z.reshape(*z.shape[:-1], z.shape[-1] // 128, 128))

    else:
      token_all_gather = _ep_all_gather

    # Duplicate token inputs across all expert shards.
    if self.config().moe_quantize_token_all_gather:
      # If enabled, tokens are quantized to FP8 before all-gather to minimize bandwidth.
      # Quantization runs on TensorCore/Device memory; the all-gather collective itself is pinned.
      x = quantize_and_all_gather_tokens(
          x,
          axis_name=self.expert_parallelism_name(),
          calibration_method=self.config().act_quantization_calibration_method,
          all_gather_fn=token_all_gather,
      )
    else:
      x = token_all_gather(x)

    local_tokens = logits.shape[0] * logits.shape[1]
    precomputed_topk = None
    precomputed_density_prob = None
    topk_before_all_gather = (
        self.config().moe_topk_before_ep_all_gather
        and forced_routed_experts is None
        and not self.is_hash_routing()
        and not self.config().use_random_routing
    )
    if topk_before_all_gather:
      # Top-k is per token, so run it on the local tokens and all-gather only the (batch, seq, k)
      # weights and expert ids instead of the (batch, seq, num_experts) logits: every shard
      # otherwise redoes top-k on the full gathered batch, and the logit gather (and its
      # reduce-scatter in the backward) moves num_experts / k times more bytes. Only the index map
      # is computed here; permute() below still sorts the gathered tokens.
      weights, selected_experts = self.lookup_get_topk()(
          logits, pre_bias_logits, rngs, input_ids, per_expert_scale=per_expert_scale
      )
      precomputed_topk = (_ep_all_gather(weights), _ep_all_gather(selected_experts))
      if self.config().load_balance_loss_weight > 0.0 and not self.use_megatron_seq_aux_loss:
        # Per-sequence statistic, so gathering it over EP gives the same loss as the full batch.
        probs_logits = pre_bias_logits if pre_bias_logits is not None else logits
        softmax_probs = jax.nn.softmax(probs_logits.astype(jnp.float32), axis=-1).astype(self.dtype())
        precomputed_density_prob = _ep_all_gather(jnp.mean(softmax_probs, axis=1))
      logits, pre_bias_logits = None, None
    else:
      logits, pre_bias_logits, forced_routed_experts = tuple(
          _ep_all_gather(z) if z is not None else None for z in (logits, pre_bias_logits, forced_routed_experts)
      )

    # "Route" tokens within each shard.
    num_experts_per_shard = self.config().num_experts // num_ep
    (
        x,
        sorted_selected_experts,
        weights,
        group_sizes,
        selected_experts,
        lb_loss,
        bias_updates,
        local_group_sizes,
        has_overflow,
        max_load_ratio,
        topk_argsort_indices,
        tc_routing,
    ) = self.lookup_permute()(
        x,
        logits,
        pre_bias_logits,
        self.config().use_custom_sort_vjp,
        roll_to_expert_id=num_experts_per_shard * expert_shard_id,
        rngs=rngs,
        input_ids=input_ids,
        forced_routed_experts=forced_routed_experts,
        force_dropless=force_dropless,
        mesh_axis_names=self.bias_axis_names,
        compute_lb_loss=not self.use_megatron_seq_aux_loss,
        return_expert_counts=True,
        defer_reductions=self.defer_reductions,
        precomputed_topk=precomputed_topk,
        precomputed_density_prob=precomputed_density_prob,
        per_expert_scale=per_expert_scale,
    )
    required_rbf = None
    if getattr(self.config(), "log_required_ragged_buffer_factor", False):
      required_rbf = self.lookup_required_ragged_buffer_factor()(
          group_sizes, local_tokens * num_ep, num_ep, expert_shard_id
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
            has_overflow=has_overflow,
            required_rbf=required_rbf,
            max_load_ratio=max_load_ratio,
            topk_argsort_indices=topk_argsort_indices,
            tc_routing=tc_routing,
        ),
        RouteMetadata(
            expert_shard_id=expert_shard_id,
            local_sorted_indices=None,
            all_shards_group_sizes=None,
            reshaped_group_sizes=None,
        ),
    )
