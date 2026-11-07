"""MoE related Layers."""

import dataclasses
import functools
import inspect
import math
from typing import Iterable, Optional, Tuple, Union

from aqt.jax.v2 import aqt_tensor as aqt
from flax import nnx
import jax
from jax import ad_checkpoint as adc
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from lilax._backend.common import common_types as ctypes
from lilax._backend.common.common_types import ShardMode
from lilax._backend.common import ffn_flops
from lilax._backend.kernels.ragged.sort import ring_ragged_sort
from lilax._backend.kernels.ragged.sort import ring_ragged_unsort
from lilax._backend.kernels.ragged.sort_tc import ring_ragged_sort_tc
from lilax._backend.kernels.ragged.sort_tc import ring_ragged_unsort_tc
from lilax._backend.layers import attentions, linears, nnx_wrappers
from lilax._backend.layers.moe_ops import routing as moe_routing
from lilax._backend.layers.moe_ops.dispatch import capacity, expert_parallel, ragged_all_to_all, ring_of_experts, sorting
from lilax._backend.layers.moe_ops.experts import activation as expert_activation
from lilax._backend.layers.moe_ops.experts import expert_gmm, grouped_matmul
from lilax._backend.layers.moe_ops.sparse import expert_pass, expert_weights, partition_specs, token_chunks
from lilax._backend.layers.initializers import NdInitializer, default_bias_init, nd_dense_init
from lilax._backend.layers.quantization.aqt import AqtQuantization
from lilax._backend.layers.quantization.fused_moe import (
    get_fused_moe_rule,
    prepare_fused_gmm_scale,
    quantize_weight_for_fused_moe,
    without_qwix_interception,
)
from lilax._backend.layers.quantization.gpu_fp8 import Fp8Quantization, NANOOFp8Quantization
from lilax._backend.layers.quantization.modes import in_serve_mode
from lilax._backend.layers.quantization.nnx_ops import apply_einsum_in_nnx, create_fp8_einsum, make_dot_general
from lilax._backend.layers.quantization.serve_fp8 import ServeFp8WeightQuantization, infer_scale_granularity
from lilax._backend.layers.quantization.weights import WeightQuantConfig, dequantize_weight, get_weight_quant_config
from lilax._backend.observability import max_logging
from lilax._backend.kernels.tiling import eval_tiles
from lilax._backend.integration.transformer_engine import moe_bootstrap
from lilax._backend.pytree import partitioning
from lilax._backend.distributed.sharding import (
    CompoundLogicalAxis,
    create_sharding,
    get_logical_axis_rules,
    logical_to_mesh_axes,
    maybe_shard_with_logical,
    maybe_shard_with_pspec,
    remove_expert_from_partition_spec,
)
import numpy as np
import qwix
from qwix.contrib.sparsity import sparsity_module
import qwix.pallas as qpl


DISPATCH = "dispatch"
COMBINE = "combine"
WI_0 = "wi_0"
WI_1 = "wi_1"
WO = "wo"

_TE_MOE_CHECKPOINT_NAMES = {
    "wi_0_checkpoint_name": "moe_mlpwi_0",
    "wi_1_checkpoint_name": "moe_mlpwi_1",
    "wo_checkpoint_name": "moe_mlpwo",
    "dispatch_checkpoint_name": "te_moe_dispatch_remat_location",
}


def _get_te_moe_checkpoint_kwargs(moe_fn) -> dict[str, str]:
  """Return supported TE MoE output remat names."""
  parameters = inspect.signature(moe_fn).parameters
  return {name: label for name, label in _TE_MOE_CHECKPOINT_NAMES.items() if name in parameters}


def _get_te_moe_cudnn_gmm_fusion_kwargs(disable_cudnn_gmm_fusion: bool) -> dict[str, bool]:
  """Disable cuDNN GMM fusion when explicitly requested, otherwise preserve TE's default."""
  return {"use_cudnn_fusion": False} if disable_cudnn_gmm_fusion else {}


def get_batchsplit_init_kernel_axes():
  return (
      ("expert_only", "embed_moe", None),
      ("expert_only", None, "embed_moe"),
  )


def calculate_load_balance_updates(top_k_indices, num_experts, rate, axis_names=None, return_counts=False):
  """Computes a bias adjustment update based on expert load.

  Used in DeepSeek V3: https://arxiv.org/html/2412.19437v1.
  Implementation reference: https://arxiv.org/pdf/2408.15664.

  Args:
      top_k_indices: Shape (batch, sequence, top_k).
      num_experts: Total number of experts.
      rate: The update rate.
      axis_names: Optional mesh axis names to reduce expert counts across.
        Defaults to None for backward compatibility. When provided, expert token
        counts are collectively reduced (psum) across these axes before
        computing the load balance updates.
      return_counts: If True, return the (reduced) per-expert token counts instead of the update, so a caller that
        splits the batch into microbatches (pipeline parallelism) can sum them and take the sign of the whole-batch
        load, as the reference does. Convert with `rate * sign(mean(counts) - counts)`.

  Returns:
      update: The value to add to the expert bias (or the expert counts). Shape (num_experts,).
  """
  expert_counts = calculate_expert_counts(top_k_indices, num_experts, axis_names=axis_names)
  if return_counts:
    return expert_counts
  return load_balance_updates_from_counts(expert_counts, num_experts, rate)


def calculate_expert_counts(top_k_indices, num_experts, axis_names=None):
  """Counts the token assignments routed to each expert.

  Args:
      top_k_indices: Integer expert ids of any shape, e.g. (batch, sequence, top_k).
      num_experts: Total number of experts.
      axis_names: Optional mesh axis names to psum the counts across.

  Returns:
      int32 array of shape (num_experts,).
  """
  flat_indices = top_k_indices.ravel()
  # one_hot rather than bincount: bincount clips out-of-range values, so the -1
  # padding that forced routing uses would all be counted as expert 0.
  expert_counts = jnp.sum(jax.nn.one_hot(flat_indices, num_experts, dtype=jnp.int32), axis=0)
  if axis_names:
    expert_counts = jax.lax.psum(expert_counts, axis_names)
  return expert_counts


def load_balance_updates_from_counts(expert_counts, num_experts, rate):
  """Loss-free-balancing bias update, rate * sign(mean_load - load), from expert counts.

  Matches Megatron-LM `get_updated_expert_bias` when `expert_counts` covers the full global batch.
  Supports shape (..., num_experts), e.g. (num_experts,) or scanned (num_layers, num_experts).
  """
  total_tokens = jnp.sum(expert_counts, axis=-1, keepdims=True)
  average_load = total_tokens / num_experts
  direction = jnp.sign(average_load - expert_counts)
  output = direction * rate
  return output


def finalize_deferred_bias_signal(partial_counts, config):
  """Reduces the stacked partial expert counts of defer_small_all_reduces into the routed-bias signal.

  Under defer_small_all_reduces each ring-of-experts MoE layer emits its local (per-shard) expert counts, per token
  chunk, instead of all-reducing them inside the layer. The layer scan stacks them, so the value arrives here with
  shape (..., num_parts, num_chunks, num_experts), where num_parts is the product of the mesh axes the counts were
  summed over (the batch axes except EP) and the leading axes are the stacked layers, if any. The sum over num_parts
  is the one all-reduce that replaces the per-layer ones. The rest repeats what the layer does without deferral: the
  counts are summed over chunks, then either one rate * sign(mean - counts) per layer is taken
  (load_balance_updates_from_counts), or, under gradient accumulation, the summed integer counts are returned so
  gradient_accumulation_loss_and_grad sums them over microbatches before its single sign. Integer counts make the
  result identical to the non-deferred path.

  Returns:
      (..., num_experts): the float bias update, or the int32 counts when gradient_accumulation_steps > 1, as the
      layer emits it without deferral.
  """
  counts = jnp.sum(partial_counts, axis=(-3, -2))  # (..., num_experts)
  if getattr(config, "gradient_accumulation_steps", 1) > 1:
    return counts
  return load_balance_updates_from_counts(counts, counts.shape[-1], config.routed_bias_update_rate)


def finalize_deferred_intermediates(intermediate_outputs, config):
  """Applies finalize_deferred_bias_signal to every "moe_bias_updates" leaf when defer_small_all_reduces is on.

  The "moe_has_overflow" leaves need no rewrite: the layers emit them unreduced (per-device local flags) with or
  without the flag, and loss_fn reduces them once with jnp.any.
  """
  if not getattr(config, "defer_small_all_reduces", False):
    return intermediate_outputs

  def _fix(path, leaf):
    keys = [getattr(k, "key", getattr(k, "name", None)) for k in path]
    if "moe_bias_updates" in keys and leaf is not None and getattr(leaf, "ndim", 0) >= 3:
      return finalize_deferred_bias_signal(leaf, config)
    return leaf

  return jax.tree_util.tree_map_with_path(_fix, intermediate_outputs)


def routing_tag_fn(config):
  """Returns the checkpoint_name tagger for the routing arrays, or None under moe_routing_maps=remat.

  With moe_routing_maps=device (or offload) and remat_policy=custom, the tagged top-k expert ids and the ragged-sort
  permutations (and, on the TC path with moe_tc_window_routing, the group sizes) are saved across the remat boundary,
  so the backward re-runs neither top-k nor the sorts. Saving them instead of recomputing them does not change any value.
  """
  if getattr(config, "moe_routing_maps", "remat") == "remat":
    return None
  return lambda t: adc.checkpoint_name(t, "moe_routing_maps")


def uses_megatron_seq_aux_loss(config) -> bool:
  """Whether the MoE aux loss is Megatron-LM's sequence-wise seq_aux_loss (DeepSeek-V3 Eqs 17-20).

  When moe_use_megatron_seq_aux_loss=True, sigmoid routers (DeepSeek-V3 family) and Sqrt(Softplus)
  routers (DeepSeek-V4) use it: fp32 per-token normalized affinity scores, token fractions from an
  unbiased, ungrouped top-k, one loss per full sequence, summed over MoE layers. By default
  (moe_use_megatron_seq_aux_loss=False), and for other routers or te_moe_block (whose aux loss
  TransformerEngine computes), MaxText uses the per-layer Switch-style loss averaged over MoE layers.
  """
  return (
      getattr(config, "moe_use_megatron_seq_aux_loss", False)
      and getattr(config, "routed_score_func", "") in ("sigmoid", "sqrtsoftplus")
      and not getattr(config, "te_moe_block", False)
  )


def megatron_aux_loss_inputs(scores, num_experts_per_tok):
  """Aux-loss probabilities and token assignments of the Megatron-LM sigmoid router.

  Mirrors Megatron-LM `compute_routing_scores_for_aux_loss(score_function="sigmoid")`: the
  probabilities are the fp32 affinity scores (sigmoid, or Sqrt(Softplus) for DeepSeek-V4)
  normalized per token, and the token assignments
  come from a plain top-k over the scores, i.e. without the expert bias and without the
  group-limited mask used for dispatch.

  Args:
      scores: Router affinity scores before the expert bias, shape (batch, sequence, num_experts).
      num_experts_per_tok: Router top-k.

  Returns:
      (aux_probs, aux_top_k_indices) with shapes (batch, sequence, num_experts) and
      (batch, sequence, num_experts_per_tok).
  """
  scores_f32 = scores.astype(jnp.float32)
  aux_probs = scores_f32 / (jnp.sum(scores_f32, axis=-1, keepdims=True) + 1e-20)
  _, aux_top_k_indices = jax.lax.top_k(scores_f32, k=num_experts_per_tok)
  return aux_probs, aux_top_k_indices


def max_expert_shard_load_ratio(group_sizes, num_expert_shards, balanced_load):
  """Largest per-expert-shard load divided by the perfectly balanced load.

  Args:
      group_sizes: Token assignments per expert, shape (num_experts,), experts ordered by shard.
      num_expert_shards: Number of expert-parallel shards.
      balanced_load: Token assignments per shard under a perfectly balanced routing.

  Returns:
      float32 scalar. The ring-of-experts ragged buffer overflows iff this exceeds
      ragged_buffer_factor (up to the int() rounding of the buffer size).
  """
  shard_loads = jnp.sum(group_sizes.reshape(num_expert_shards, -1), axis=-1)
  return jnp.max(shard_loads).astype(jnp.float32) / balanced_load


def _batch_axis_names(pspec) -> tuple[str, ...] | None:
  """Returns the mesh axes the (batch, sequence) dims are partitioned over."""
  if not pspec:
    return None
  # 'stage' shards the batch only where it acts as data parallelism (the layers outside the pipeline); inside
  # the pipeline it is the vmapped stage axis and never appears in these per-stage partition specs.
  allowed_batch_axes = frozenset(
      {
          "data",
          "stage",
          "fsdp",
          "fsdp_transpose",
          "expert",
          "context",
          "context_usp_ulysses",
          "context_autoregressive",
          "tensor_sequence",
      }
  )
  axes = []
  for dim in pspec[:2]:
    items = (dim,) if isinstance(dim, str) else (dim or ())
    for ax in items:
      if ax in allowed_batch_axes and ax not in axes:
        axes.append(ax)
  return tuple(axes) or None


def _filter_axis_names(
    axis_names: tuple[str, ...] | None,
    exclude_axes: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | None:
  """Filters out specified axes from a tuple of axis names.

  Args:
    axis_names: Mesh axis names to filter, or None.
    exclude_axes: An axis name or collection of axis names to exclude.

  Returns:
    Filtered axis names tuple, or None if no axes remain.
  """
  if not axis_names or not exclude_axes:
    return axis_names
  exclude = (exclude_axes,) if isinstance(exclude_axes, str) else tuple(exclude_axes)
  filtered = tuple(ax for ax in axis_names if ax not in exclude)
  return filtered or None


def fused_moe_lhs_scale_kwargs(quant, fused_moe_fn) -> dict[str, jax.Array]:
  """Static activation-scale kwargs for tpu-inference's `fused_moe_func`.

  With `serve_fp8_weight` and a fixed `act_quantization_calibration_method`, both expert
  grouped matmuls quantize their input with the same static per-tensor scale as the dense
  layers, instead of the kernel's dynamic per-row, per-K-block scale. Returns {} (dynamic
  scaling, the kernel default) otherwise.
  """
  if not isinstance(quant, ServeFp8WeightQuantization):
    return {}
  scale = quant.static_act_scale()
  if scale is None:
    return {}
  if "w1_lhs_scale" not in inspect.signature(fused_moe_fn).parameters:
    raise ValueError(
        f"act_quantization_calibration_method={quant.act_calibration_method!r} requires a tpu-inference whose"
        " fused_moe_func accepts w1_lhs_scale / w2_lhs_scale; upgrade tpu-inference or use 'absmax'."
    )
  return {"w1_lhs_scale": scale, "w2_lhs_scale": scale}


class Tid2EidVar(nnx.Variable):
  """Custom variable to hold tid2eid without trainable param overhead."""


class MoEBiasVar(nnx.Variable):
  """Custom NNX Variable for Auxiliary-Loss-Free MoE Routing Bias (DSV4)."""


class GateLogit(nnx.Module):
  """A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing."""

  def __init__(
      self,
      in_features_shape: Union[Iterable[int], int],
      out_features_shape: Union[Iterable[int], int],
      model_name: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      axis: Union[Iterable[int], int] = -1,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: Tuple[Optional[str], ...] = (),
      use_bias: bool = False,
      bias_dtype: Optional[ctypes.DType] = None,
      score_func: str = "",
      quant: Optional[AqtQuantization] = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
  ):
    """Initializes the GateLogit module.

    Attributes:
      in_features_shape: The shape of the input features.
      out_features_shape: The shape of the output features, typically the number of experts.
      model_name: The name of the model.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      axis: The axis or axes over transformation is applied.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      use_bias: Whether to add learnable bias in gate logit scores. When enabled,
        this bias aids expert load balancing (like in DeepSeek V3), and is not
        part of the loss calculation.
      bias_dtype: Data type of the learnable routing bias; defaults to weight_dtype.
      score_func: Scoring function for output normalization before applying bias.
      quant: The quantization configuration. If None, no quantization is applied.
      matmul_precision: The precision level for the matrix multiplication.
    """
    if isinstance(quant, ServeFp8WeightQuantization) and not ctypes.is_fp8_dtype(weight_dtype):
      quant = None
    self.in_features_shape = linears.canonicalize_tuple(in_features_shape)
    self.out_features_shape = linears.canonicalize_tuple(out_features_shape)
    self.model_name = model_name
    self.mesh = mesh
    self.axis = linears.canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.use_bias = use_bias
    self.bias_dtype = bias_dtype or self.weight_dtype
    self.score_func = score_func
    self.quant = quant
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not in_serve_mode(self.quant):
      self.kernel = nnx.Param(
          self.kernel_init(
              rngs.params(),
              kernel_shape,
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      if self.model_name.startswith(("deepseek3", "deepseek4", "kimi-k2", "ming", "hy4")):
        # DeepSeek uses MoEBiasVar to naturally isolate from sequence-wise updates.
        self.bias = MoEBiasVar(
            default_bias_init(rngs.params(), bias_shape, self.bias_dtype),
            out_sharding=bias_axes,
        )
      else:
        self.bias = nnx.Param(
            default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
            out_sharding=bias_axes,
        )
    else:
      self.bias = None

    if quant and not isinstance(quant, ServeFp8WeightQuantization):
      self._quant_dot_general_name, quant_dot_general = make_dot_general(quant, kernel_axes, rngs)
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      # See the matching comments in linears.py.
      if isinstance(quant_dot_general, nnx_wrappers.ToNNX):
        block_size = getattr(quant, "get_block_size", lambda: 1)()  # needed for TE block scaling
        dummy_inputs = jnp.zeros((block_size, *self.in_features_shape), dtype=self.dtype)
        self(dummy_inputs, _initializing=True)
        if not quant.needs_apply_rngs:
          quant_dot_general.release_rngs()
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx.Module | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def __call__(self, inputs: jax.Array, _initializing: bool = False) -> Tuple[jax.Array, Optional[jax.Array]]:
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = linears.normalize_axes(self.axis, inputs.ndim)

    if in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = self.kernel[...]
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(norm_axis)))
    output_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None))
        if self.shard_mode == ShardMode.EXPLICIT
        else None
    )
    output = linears._compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general,
        _initializing,
        out_sharding=output_sharding,
    )
    pre_bias_logits = None

    if self.score_func:
      output = linears._convert_to_activation_function(self.score_func)(output)

    # NOTE: deepseek2 has a different pattern
    if self.model_name.startswith(("deepseek3", "deepseek4", "kimi-k2", "ming", "hy4")):
      pre_bias_logits = output

    if self.use_bias:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output, pre_bias_logits


class RoutedMoE(nnx.Module):
  """Implements a routed MoE block."""

  def deepseek_scale_weights(self, weights):
    return moe_routing.ExpertGroupRouting.deepseek_scale_weights(
        weights, self.config.routed_score_func, self.config.routed_scaling_factor
    )

  def expert_group_mask(self, gate_logits: jax.Array) -> jax.Array:
    return moe_routing.ExpertGroupRouting.expert_group_mask(
        gate_logits, self.config.n_routing_groups, self.config.topk_routing_group, self.num_experts
    )

  def apply_ffn_activation(self, layer_w0, layer_w1):
    return expert_activation.ExpertActivation.apply_ffn_activation(
        layer_w0, layer_w1, self.config.decoder_block, self.config.mlp_activations_limit, self.activation_fn, self.dtype
    )

  @staticmethod
  def _maybe_truncate_local_group_size(
      all_shard_local_sizes: jax.Array, buffer_size: int, ragged_buffer_factor: float
  ) -> jax.Array:
    return expert_parallel.ExpertDispatch._maybe_truncate_local_group_size(
        all_shard_local_sizes, buffer_size, ragged_buffer_factor
    )

  @staticmethod
  def local_permute(
      inputs,
      global_group_sizes,
      local_expert_size,
      shard_index,
      is_offset=False,
      global_sorted_experts=None,
      use_custom_sort_vjp=True,
      use_ragged_sort=False,
      ragged_buffer_factor=-1.0,
      use_single_sparsecore=False,
  ):
    return expert_parallel.ExpertDispatch.local_permute(
        inputs, global_group_sizes, local_expert_size, shard_index, is_offset, global_sorted_experts,
        use_custom_sort_vjp, use_ragged_sort, ragged_buffer_factor, use_single_sparsecore,
        truncate_local_group_size=lambda sizes, size, factor: RoutedMoE._maybe_truncate_local_group_size(
            sizes, size, factor
        ),
    )

  @staticmethod
  def get_all_to_all_params(
      all_shards_group_sizes,
      shard_id,
      num_expert_parallelism,
      is_batch_sharded=True,
      ragged_buffer_factor=-1.0,
      buffer_size=None,
      is_dispatch=True,
  ):
    return expert_parallel.ExpertDispatch.get_all_to_all_params(
        all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded,
        ragged_buffer_factor, buffer_size, is_dispatch,
    )

  def required_ragged_buffer_factor(self, group_sizes, bsz_times_seq_len, num_ep, expert_shard_id):
    """log_required_ragged_buffer_factor probe: the smallest ragged_buffer_factor that would not have dropped tokens.

    group_sizes are the global per-expert token counts of the (EP-gathered) batch, as ring_ragged_sort computes them
    for the overflow check. The per-shard buffer is int(balanced_size * factor) rows, so a shard drops tokens iff
    sum(local group sizes) > balanced_size * factor. The result is max over all devices of
    sum(local group sizes) / balanced_size: one float32 scalar max all-reduce over the mesh axes (no other change).
    """
    local_num_experts = self.config.num_experts // num_ep
    local_group_sizes = jax.lax.dynamic_slice_in_dim(
        group_sizes, expert_shard_id * local_num_experts, local_num_experts, axis=0
    )
    balanced_size = (bsz_times_seq_len // num_ep) * self.num_experts_per_tok
    required = jnp.sum(local_group_sizes).astype(jnp.float32) / jnp.float32(balanced_size)
    if self.mesh is not None and self.mesh.axis_names:
      required = jax.lax.pmax(required, tuple(self.mesh.axis_names))
    return required

  def __init__(
      self,
      config: ctypes.Config,
      num_experts: int,
      num_experts_per_tok: int,
      mesh: jax.sharding.Mesh,
      kernel_init: attentions.NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      intermediate_dim: int = 2048,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[AqtQuantization] = None,
      is_hash_routing: bool = False,
      force_dropless: bool = False,
      weight_quant: Optional[WeightQuantConfig] = None,
  ):
    """Initializes the RoutedMoE module.

    Attributes:
      config: The main config setting.
      num_experts: Number of experts.
      num_experts_per_tok: Number of experts for each token.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      intermediate_dim: Intermediate dimension of MoE.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
      is_hash_routing: Whether this layer uses deterministic hash routing instead of top-K routing.
      weight_quant: Optional WeightQuantConfig decoupling weight quantization settings.
    """
    if weight_quant is not None:
      weight_dtype = weight_quant.weight_dtype
    elif ctypes.is_module_unquantized(config, "routed_experts"):
      weight_dtype = ctypes.get_weight_dtype(config, "routed_experts")
      quant = ctypes.get_quant(config, "routed_experts", quant)

    self.config = config
    self.force_dropless = force_dropless
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.intermediate_dim = intermediate_dim
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs
    self.is_hash_routing = is_hash_routing
    self.weight_quant = weight_quant

    # DeepSeek V4 Hash Routing
    if self.is_hash_routing:
      # Token-ID to Expert-ID lookup table for static routing
      # Must be stored as float32 because MaxText passes the entire variable tree
      # through jax.value_and_grad, which strictly requires all leaves to be inexact types
      # (even if they receive no gradients). We cast to int32 dynamically during routing.
      self.tid2eid = Tid2EidVar(
          jnp.zeros(
              (self.config.vocab_size, self.num_experts_per_tok),
              dtype=jnp.float32,
          ),
          out_sharding=None,  # Replicated across shards for local lookup
      )
    else:
      self.tid2eid = None

    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    if self.config.custom_mesh_and_rule == ctypes.CustomRule.EXPERT_ON_FSDP_EP:
      if any(
          (
              self.config.shard_exp_on_fsdp,
              self.config.use_2d_fsdp_sharding,
              self.config.use_batch_split_schedule,
              self.config.moe_flat_fsdp_weights,
          )
      ):
        raise ValueError(
            "expert-on-fsdp-ep requires the standard unflattened expert weight axes; "
            "disable shard_exp_on_fsdp, use_2d_fsdp_sharding, use_batch_split_schedule, and moe_flat_fsdp_weights."
        )

    if self.config.shard_exp_on_fsdp:
      # special sharding for dsv3
      self.wi_kernel_axes = ("embed_moe", None, "mlp_moe")
      self.wo_kernel_axes = ("embed_moe", "mlp_moe", None)
    elif self.config.use_2d_fsdp_sharding:
      self.wi_kernel_axes = ("embed_moe", "mlp_moe", None)
      self.wo_kernel_axes = ("embed_moe", "mlp_moe", None)
    elif self.config.use_batch_split_schedule:
      self.wi_kernel_axes, self.wo_kernel_axes = get_batchsplit_init_kernel_axes()
    else:
      self.wi_kernel_axes = ("exp", "embed_moe", "mlp_moe")
      self.wo_kernel_axes = ("exp", "mlp_moe", "embed_moe")
      if self.config.moe_flat_fsdp_weights:
        # Flat layout: wi [E * embed, mlp], wo [E * mlp, embed]; dim 0 is sharded over (expert, fsdp) via the
        # compound logical axis ("exp", "embed_moe") so every FSDP shard is a contiguous tile-aligned row block
        # and the FSDP-gathered weight is a free bitcast of the [E_local, rows, cols] operand the gmm kernels pin.
        # wo's rows are mlp but reuse the "embed_moe" logical name to get the same (expert, fsdp) dim-0 sharding.
        self.wi_kernel_axes = (CompoundLogicalAxis(("exp", "embed_moe")), "mlp_moe")
        self.wo_kernel_axes = (CompoundLogicalAxis(("exp", "embed_moe")), None)

    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa"):
      # vLLM uses 'model' as the tensor parallelism axis name
      self._tensor_parallelism_name = ("model", "attn_dp")
    else:
      self._tensor_parallelism_name = "tensor"

    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa"):
      expert_axes = ["attn_dp_expert", "expert"]
      if self.mesh is not None and "dcp" in self.mesh.shape and self.mesh.shape["dcp"] > 1:
        expert_axes.append("dcp")
      self._expert_parallelism_name = tuple(expert_axes)
    elif self.config.custom_mesh_and_rule == ctypes.CustomRule.CP_AS_EP:
      # when custom mesh and rule is cp-as-ep, context axis is same with expert in MoE component
      self._expert_parallelism_name = ("context", "expert")
    else:
      self._expert_parallelism_name = "expert"

    # TE v1 grouped GEMM stream-syncs a D2H copy inside shard_map, which deadlocks with several local devices.
    if self.config.use_te_grouped_gemm and self.mesh is not None and len(self.mesh.local_devices) > 1:
      raise ValueError(
          "use_te_grouped_gemm=True requires one local device per process "
          f"(got {len(self.mesh.local_devices)} local devices in the mesh). "
          "Launch with one GPU per rank (e.g. tools/orchestration/gpu_per_device_run.sh)."
      )

    if isinstance(self.quant, (Fp8Quantization, NANOOFp8Quantization)):
      einsum_names = [WI_0, WI_1, WO]
      if self.config.capacity_factor > 0:
        einsum_names += [DISPATCH, COMBINE]
      self.quant_einsums = nnx.Dict(
          {name: create_fp8_einsum(self.quant, self.dtype, self.rngs) for name in einsum_names}
      )
    else:
      self.quant_einsums = None

    gate_weight_dtype = ctypes.get_weight_dtype(self.config, "gate")
    self.gate = GateLogit(
        in_features_shape=self.moe_expert_input_dim,
        out_features_shape=self.num_experts,
        mesh=self.mesh,
        model_name=self.config.model_name,
        dtype=jnp.float32 if self.config.float32_gate_logits else self.dtype,
        weight_dtype=gate_weight_dtype,
        quant=ctypes.get_quant(self.config, "gate", self.quant),
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        use_bias=self.config.routed_bias and not self.is_hash_routing,
        bias_dtype=jnp.float32 if self.config.float32_gate_logits else gate_weight_dtype,
        # tpu-inference applies the score function in the fused_moe_gmm kernel,
        # so we don't apply it here to avoid redundant computation.
        # See https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/common/fused_moe_gmm.py#L58.
        score_func="" if self.config.attention in ("vllm_rpa", "vllm_batched_rpa") else self.config.routed_score_func,
        matmul_precision=self.config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )
    rule = qpl.get_current_rule("gmm")
    sparsity_rule = None
    if rule is not None:
      if not isinstance(rule, qwix.QtRule):
        raise ValueError("Expect a QtRule for quantized training.")
      if rule.additional_qt_config and "sparsity_rule" in rule.additional_qt_config:
        q_s_rule = rule.additional_qt_config["sparsity_rule"]
        if q_s_rule and q_s_rule.weight_sparsity_n and q_s_rule.weight_sparsity_m:
          sparsity_rule = q_s_rule

    if sparsity_rule is not None:
      self.wi_0_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wi_1_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wo_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.intermediate_dim, self.config.emb_dim),
          sharding_axes=self.wo_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
    else:
      self.wi_0_sparsity_module = None
      self.wi_1_sparsity_module = None
      self.wo_sparsity_module = None

    # pylint: disable=protected-access
    self.activation_fn = linears._convert_to_activation_function(self.config.mlp_activations[0])

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    # Pad the MoE input weight kernels for GMM_v2 execution when configured.
    moe_intermediate_dim = (
        self.config.padded_base_moe_mlp_dim if self.config.padded_base_moe_mlp_dim is not None else self.intermediate_dim
    )

    if in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save
      # memory. Instead they are retrieved from the tensors stored in the 'aqt'
      # collection.
      self.wi_0 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wi_1 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wo = jnp.zeros((num_experts, intermediate_dim, self.moe_expert_input_dim))
    elif self.config.prefuse_moe_weights:
      self.wi = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim * 2),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (
                  self.num_experts,
                  moe_intermediate_dim,
                  self.moe_expert_input_dim,
              ),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wo_kernel_axes,
      )
    else:

      def _init_expert_kernel(shape, dtype):
        # Always init in 3D [E, rows, cols] so the fan-in/out (and RNG draw) match the 3D layout, then view it
        # as the flat [E * rows, cols] matrix under moe_flat_fsdp_weights (see wi_kernel_axes above).
        kernel = self.kernel_init(self.rngs.params(), shape, dtype, kernel_in_axis, kernel_out_axis)
        if self.config.moe_flat_fsdp_weights:
          kernel = kernel.reshape(shape[0] * shape[1], shape[2])
        return kernel

      self.wi_0 = nnx.Param(
          _init_expert_kernel((num_experts, self.moe_expert_input_dim, moe_intermediate_dim), weight_dtype),
          out_sharding=self.wi_kernel_axes,
      )
      self.wi_1 = nnx.Param(
          _init_expert_kernel((num_experts, self.moe_expert_input_dim, moe_intermediate_dim), weight_dtype),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          _init_expert_kernel((self.num_experts, moe_intermediate_dim, self.moe_expert_input_dim), self.weight_dtype),
          out_sharding=self.wo_kernel_axes,
      )

    if self.config.mlp_bias:
      wi_bias_axes = ("exp", "activation_mlp")
      wo_bias_axes = ("exp", "activation_embed")
      wi_bias_shape = (self.num_experts, self.intermediate_dim)
      wo_bias_shape = (self.num_experts, self.moe_expert_input_dim)
      self.wi_0_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wi_1_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wo_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wo_bias_shape, self.weight_dtype),
          out_sharding=wo_bias_axes,
      )
    else:
      self.wi_0_bias = None
      self.wi_1_bias = None
      self.wo_bias = None

    if self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
      self.per_expert_scale = nnx.Param(
          jnp.ones((self.num_experts,), dtype=self.weight_dtype),
          out_sharding=("exp",),
      )
    else:
      self.per_expert_scale = None

    if not in_serve_mode(self.quant) and ctypes.is_fp8_dtype(self.weight_dtype):
      resolved_weight_quant = (
          self.weight_quant
          if self.weight_quant is not None
          else get_weight_quant_config(self.config, "routed_experts")
      )
      scale_dtype = resolved_weight_quant.scale_dtype if resolved_weight_quant is not None else jnp.float32
      block_size = (
          resolved_weight_quant.block_size
          if resolved_weight_quant is not None
          else getattr(self.config, "weight_block_size", None)
      )

      # Phase 1: Resolve scale shape based on quantization granularity
      # - Block scaling (e.g. weight_block_size=[128, 128] or 128): expert weights with shape
      #   (num_experts, in_dim, out_dim) are partitioned into 2D block grids per expert,
      #   yielding a 3D scale tensor (num_experts, in_blocks, out_blocks).
      #   For prefused weights (wi), the out_dim is doubled (moe_intermediate_dim * 2).
      # - Per-expert / per-tensor scaling (weight_block_size is None): a 1D scale tensor
      #   with shape (num_experts,) containing one scalar scale per expert.
      if block_size is not None:
        if isinstance(block_size, (list, tuple)):
          b_in = block_size[0]
          b_out = block_size[1] if len(block_size) > 1 else block_size[0]
        else:
          b_in = block_size
          b_out = block_size
        in_blocks = self.moe_expert_input_dim // b_in if self.moe_expert_input_dim >= b_in else self.moe_expert_input_dim
        out_blocks = moe_intermediate_dim // b_out if moe_intermediate_dim >= b_out else moe_intermediate_dim
        if self.config.prefuse_moe_weights:
          fused_out_dim = moe_intermediate_dim * 2
          fused_out_blocks = fused_out_dim // b_out if fused_out_dim >= b_out else fused_out_dim
          wi_scale_shape = (num_experts, in_blocks, fused_out_blocks)
          wo_scale_shape = (self.num_experts, out_blocks, in_blocks)
        else:
          wi_scale_shape = (num_experts, in_blocks, out_blocks)
          wo_scale_shape = (self.num_experts, out_blocks, in_blocks)
      else:
        wi_scale_shape = (num_experts,)
        wo_scale_shape = (self.num_experts,)

      # Phase 2: Resolve scale mesh sharding
      # - 3D block scale grids inherit full 3D kernel axes ("exp", "embed_moe", "mlp_moe")
      #   to partition synchronously with the weight tensors across expert and tensor meshes.
      # - 1D per-expert scales (num_experts,) only span the expert axis, so sharding must
      #   use (self.wi_kernel_axes[0],) to prevent a rank mismatch with the 3D kernel axes.
      wi_scale_sharding = self.wi_kernel_axes if block_size is not None else (self.wi_kernel_axes[0],)
      wo_scale_sharding = self.wo_kernel_axes if block_size is not None else (self.wo_kernel_axes[0],)
      self.wi_scale_axes = wi_scale_sharding
      self.wo_scale_axes = wo_scale_sharding

      # Phase 3: Instantiate scale parameters matching prefused or unfused weight layout
      if self.config.prefuse_moe_weights:
        self.wi_scale = nnx.Param(
            jnp.ones(wi_scale_shape, dtype=scale_dtype),
            sharding=wi_scale_sharding,
        )
        self.wo_scale = nnx.Param(
            jnp.ones(wo_scale_shape, dtype=scale_dtype),
            sharding=wo_scale_sharding,
        )
        self.wi_0_scale = None
        self.wi_1_scale = None
      else:
        self.wi_0_scale = nnx.Param(
            jnp.ones(wi_scale_shape, dtype=scale_dtype),
            sharding=wi_scale_sharding,
        )
        self.wi_1_scale = nnx.Param(
            jnp.ones(wi_scale_shape, dtype=scale_dtype),
            sharding=wi_scale_sharding,
        )
        self.wo_scale = nnx.Param(
            jnp.ones(wo_scale_shape, dtype=scale_dtype),
            sharding=wo_scale_sharding,
        )
        self.wi_scale = None
    else:
      self.wi_scale = None
      self.wi_0_scale = None
      self.wi_1_scale = None
      self.wo_scale = None
      self.wi_scale_axes = None
      self.wo_scale_axes = None

    # Scale the output projection ahead of time during inference for higher generation throughput.
    if (
        self.per_expert_scale is not None
        and self.config.model_call_mode == "inference"
        and self.config.fuse_expert_scales
    ):
      self.wo.value = self.wo.value * self.per_expert_scale.value[:, None, None]

  def _maybe_shard_with_logical(self, inputs, logical_name):
    return maybe_shard_with_logical(
        inputs,
        logical_name,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def _logical_to_mesh_axes(self, logical_name):
    logical_rules = get_logical_axis_rules()
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=logical_rules)

  def _maybe_shard_with_pspec(self, inputs, pspec: jax.sharding.PartitionSpec | None, logical_axes=None):
    return maybe_shard_with_pspec(
        inputs,
        pspec,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
        logical_axes=logical_axes,
    )

  def _maybe_shard_moe_dispatch(self, inputs, logical_axis, peel_expert):
    """Shard a MoE dispatch/MLP activation. When `peel_expert` is set, drop the 'expert'
    mesh axis from the batch dim (index 1) so the GEMM stays expert-parallel (AllToAll)
    instead of double-mapping E and B onto 'expert'. Each logical dim is resolved
    independently so the shared 'expert' axis is not deduped off the expert dim before
    the peel."""
    if not peel_expert:
      return self._maybe_shard_with_logical(inputs, logical_axis)
    spec = [None if name is None else self._logical_to_mesh_axes((name,))[0] for name in logical_axis]
    pspec = remove_expert_from_partition_spec(jax.sharding.PartitionSpec(*spec), dims_to_peel=(1,))
    return self._maybe_shard_with_pspec(inputs, pspec)

  def get_expert_parallelism_size(self):
    # When expert parallelism has more than one physical axes, take product of their shapes
    if isinstance(self._expert_parallelism_name, tuple):
      return math.prod(self.mesh.shape.get(name, 1) for name in self._expert_parallelism_name)
    return self.mesh.shape.get(self._expert_parallelism_name, 1)

  def get_tensor_parallelism_size(self):
    if isinstance(self._tensor_parallelism_name, tuple):
      size = 1
      for axis in self._tensor_parallelism_name:
        size *= self.mesh.shape.get(axis, 1)
      return size
    return self.mesh.shape.get(self._tensor_parallelism_name, 1)

  def get_context_autoregressive_parallelism_size(self):
    return self.mesh.shape.get("context_autoregressive", 1)

  def should_update_load_balance(self):
    """Determines if loss-free load balancing updates should be applied.

    The bias update logic is only applicable to Top-K router.
    Hash router does not use routed bias.
    """
    return self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and not self.is_hash_routing

  def get_topk(
      self,
      gate_logits,
      pre_bias_logits,
      rngs=None,
      input_ids=None,
      forced_routed_experts=None,
      per_expert_scale=None,
  ):
    """get topk.

    `per_expert_scale` (Gemma 4 only) is normally read off the module, but the
    sparse path runs inside a `jax.shard_map` that cannot close over an array
    living on explicit mesh axes, so it hands the value in instead.
    """
    if per_expert_scale is None and self.per_expert_scale is not None:
      per_expert_scale = self.per_expert_scale[...]
    # shape of top_k_weights & top_k_indices:
    # (batch, sequence, num_experts_per_tok).
    valid_token_mask = None
    if forced_routed_experts is not None:
      top_k_indices = forced_routed_experts
      # Any id outside [0, num_experts) is an unused slot, same as the -1 sentinel.
      valid_token_mask = moe_routing.valid_expert_mask(top_k_indices, self.num_experts)
      # Gather at 0, not -1: take_along_axis would wrap -1 onto the last expert and
      # that value reaches the softmax denominator before the mask zeroes it.
      gather_indices = jnp.where(valid_token_mask, top_k_indices, 0)
      if self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
        router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
        top_k_weights = jnp.take_along_axis(router_probs, gather_indices, axis=-1).astype(self.dtype)
      else:
        top_k_weights = jnp.take_along_axis(gate_logits, gather_indices, axis=-1)
    else:
      if self.config.use_random_routing:
        if rngs is None:
          raise ValueError("The random key cannot be None for random routing.")
        # Reuse the 'params' RNG stream to ensure random routing
        rng = rngs.params() if hasattr(rngs, "params") and callable(getattr(rngs, "params")) else rngs
        top_k_weights, top_k_indices = moe_routing.random_routing(rng, gate_logits, self.num_experts_per_tok)
        return top_k_weights, top_k_indices

      if self.is_hash_routing:
        if input_ids is None:
          raise ValueError("input_ids cannot be None when is_hash_routing is True")
        # Access the static routing table
        tid2eid_int = self.tid2eid.value
        # Cast the float32 array to int32 (JAX automatically assigns 0.0 gradients to integer casts)
        tid2eid_int = tid2eid_int.astype(jnp.int32)
        # Cast input_ids to int32 to safely index the hash routing table
        top_k_indices = tid2eid_int[input_ids.astype(jnp.int32)]
        top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
      # NOTE: deepseek2 has a different pattern
      elif self.config.model_name.startswith(("deepseek3", "deepseek4", "kimi-k2", "ming", "hy4")):
        top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
      elif self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
        router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
        _, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
        top_k_weights = jnp.take_along_axis(router_probs, top_k_indices, axis=-1).astype(self.dtype)
      else:
        top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

    if self.config.decoder_block in (
        ctypes.DecoderBlockType.DEEPSEEK,
        ctypes.DecoderBlockType.DEEPSEEK4,
        ctypes.DecoderBlockType.MING,
        ctypes.DecoderBlockType.HY4,
    ):
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
      if valid_token_mask is not None:
        top_k_weights = top_k_weights * valid_token_mask
    else:
      if self.config.decoder_block not in (ctypes.DecoderBlockType.LLAMA4, ctypes.DecoderBlockType.GEMMA4):
        if valid_token_mask is not None:
          # Padding must stay out of the softmax denominator or it rescales the
          # real slots. Large-negative, not -inf: a fully-padded token would be NaN.
          top_k_weights = jnp.where(valid_token_mask, top_k_weights, jnp.finfo(jnp.float32).min / 2)
        top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)

      if valid_token_mask is not None:
        top_k_weights = top_k_weights * valid_token_mask

      # Normalization of router weights (e.g. used by Qwen3, Gemma4).
      if self.config.norm_topk_prob:
        weight_sum = top_k_weights.sum(axis=-1, keepdims=True)
        if valid_token_mask is not None:
          # A fully-padded token sums to zero; 0/0 would NaN the whole batch loss.
          weight_sum = jnp.where(weight_sum == 0, 1.0, weight_sum)
        top_k_weights /= weight_sum

      if per_expert_scale is not None and not (
          self.config.model_call_mode == "inference" and self.config.fuse_expert_scales
      ):
        per_expert_scale_topk = jnp.take_along_axis(per_expert_scale[None, None, :], top_k_indices, axis=-1)
        top_k_weights = top_k_weights * per_expert_scale_topk.astype(top_k_weights.dtype)

    return top_k_weights, top_k_indices

  def deepseek_routing(self, gate_logits: jax.Array, pre_bias_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """DeepSeek routing logit.

    If the configuration does not specify routing groups (`n_routing_groups` is
    -1), we use a standard top-k routing mechanism. Otherwise, we force all
    selected experts to be from the a subset of the highest rated expert groups.

    The selection process uses post_bias logits, while the return weights use
    pre_bias logits.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.
      pre_bias_logits: Array of shape `(batch, seq,num_experts)`.

    Returns:
      - top_k_weights: `(batch, seq, num_experts_per_tok)` array of weight values for
        each selected expert.
      - top_k_indices: `(batch, seq, num_experts_per_tok)` array of indices
        identifying the selected experts for each token.
    """
    expert_mask = 1 if self.config.n_routing_groups == -1 else self.expert_group_mask(gate_logits)
    _, top_k_indices = jax.lax.top_k(
        jnp.where(expert_mask > 0, gate_logits, -jnp.inf),
        k=self.num_experts_per_tok,
    )
    # Tag before the weight gather, so the gather's backward reads the saved ids instead of re-running top-k.
    tag_routing_fn = routing_tag_fn(self.config)
    if tag_routing_fn is not None:
      top_k_indices = tag_routing_fn(top_k_indices)
    if getattr(self.config, "router_topk_matmul_vjp", False):
      top_k_weights = moe_routing.take_along_last_axis_dense_vjp(pre_bias_logits, top_k_indices, pre_bias_logits.shape[-1])
    else:
      top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    return top_k_weights, top_k_indices

  def permute(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      use_custom_sort_vjp=True,
      rngs=None,
      roll_to_expert_id=None,
      input_ids=None,
      forced_routed_experts=None,
      force_dropless=False,
      mesh_axis_names=None,
      compute_lb_loss=True,
      return_expert_counts=False,
      defer_reductions=False,
      precomputed_topk=None,
      precomputed_density_prob=None,
      per_expert_scale=None,
  ):
    is_qarray = isinstance(inputs, qpl.QArray)
    raw_inputs = inputs.qvalue if is_qarray else inputs
    inputs_shape = raw_inputs.shape
    bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    inputs_2d = jnp.reshape(raw_inputs, (bsz_times_seq_len, *inputs_shape[2:]))
    if precomputed_topk is not None:
      weights, selected_experts = precomputed_topk
    else:
      weights, selected_experts = self.get_topk(
          gate_logits, pre_bias_logits, rngs, input_ids, forced_routed_experts, per_expert_scale=per_expert_scale
      )
    lb_loss = None
    if compute_lb_loss and self.config.load_balance_loss_weight > 0.0 and not self.is_hash_routing:
      if precomputed_density_prob is not None:
        lb_loss = self.load_balance_loss(selected_experts, None, density_prob=precomputed_density_prob)
      else:
        probs_logits = pre_bias_logits if pre_bias_logits is not None else gate_logits
        softmax_probs = jax.nn.softmax(probs_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        lb_loss = self.load_balance_loss(selected_experts, softmax_probs)

    if self.should_update_load_balance():
      if defer_reductions:
        bias_updates = calculate_expert_counts(selected_experts, self.config.num_experts)
      elif return_expert_counts or getattr(self.config, "using_pipeline_parallelism", False):
        bias_updates = calculate_expert_counts(selected_experts, self.config.num_experts, axis_names=mesh_axis_names)
      else:
        bias_updates = calculate_load_balance_updates(
            selected_experts, self.config.num_experts, self.config.routed_bias_update_rate, axis_names=mesh_axis_names
        )
    else:
      bias_updates = None

    if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
      router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))
      inputs_2d = inputs_2d * router_scores.reshape(bsz_times_seq_len, -1)

    num_expert_parallelism = self.get_expert_parallelism_size()
    use_ragged_in_permute = self.config.use_ragged_sort and self.config.use_ring_of_experts
    buffer_size = None
    topk_argsort_indices = None
    use_tc_sort = False
    tc_routing = None
    if use_ragged_in_permute:
      topk_indices_2d = jnp.reshape(selected_experts, (bsz_times_seq_len, selected_experts.shape[2]))
      if forced_routed_experts is not None:
        topk_indices_2d = jnp.where(
            moe_routing.valid_expert_mask(topk_indices_2d, self.config.num_experts), topk_indices_2d, self.config.num_experts
        )
      ragged_buffer_factor = self.get_ragged_buffer_factor()
      if not force_dropless and ragged_buffer_factor > 0.0:
        balanced_size = (bsz_times_seq_len // num_expert_parallelism) * self.num_experts_per_tok
        buffer_size = self.get_ragged_buffer_size(
            balanced_size, num_expert_parallelism, self.config.num_experts,
            self.num_experts_per_tok, ragged_buffer_factor,
        )
      else:
        buffer_size = None

      use_tc_sort = (
          getattr(self.config, "moe_tc_ragged_sort", False)
          and buffer_size is not None
          and buffer_size < bsz_times_seq_len * self.num_experts_per_tok
      )
      if getattr(self.config, "moe_tc_ragged_sort", False) and not use_tc_sort:
        max_logging.warning(
            "moe_tc_ragged_sort=True but the TensorCore ragged sort needs a truncated buffer "
            f"(buffer_size={buffer_size}, total slots={bsz_times_seq_len * self.num_experts_per_tok}); "
            "using the SparseCore ragged sort for this call."
        )
    if inputs_2d.ndim == 3 and not use_tc_sort:
      inputs_2d = jnp.reshape(inputs_2d, (bsz_times_seq_len, -1))
    if use_ragged_in_permute and use_tc_sort:
      sorted_inputs, group_size, sorted_selected_experts, tc_routing = ring_ragged_sort_tc(
          inputs_2d, topk_indices_2d, self.config.num_experts, self.num_experts_per_tok,
          self._expert_parallelism_name, num_expert_parallelism, buffer_size,
          gather_block_size=self.config.moe_tc_ragged_gather_block_size,
          reduce_block_size=self.config.moe_tc_ragged_reduce_block_size,
          mask_padding=self.config.moe_tc_ragged_mask_padding,
          flatten_block_size=self.config.moe_tc_ragged_flatten_block_size,
          tag_routing_fn=routing_tag_fn(self.config),
          keep_3d=self.config.moe_tc_ragged_3d_gmm,
          window_routing=self.config.moe_tc_window_routing,
      )
    elif use_ragged_in_permute:
      tag_routing_fn = routing_tag_fn(self.config)
      sorted_inputs, group_size, sorted_selected_experts, topk_argsort_indices = ring_ragged_sort(
          inputs_2d, topk_indices_2d, self.config.num_experts, self.num_experts_per_tok,
          self._expert_parallelism_name, num_expert_parallelism, buffer_size=buffer_size,
          enforce_gather_fallback=self.config.ragged_gather_fallback,
          enforce_gather_reduce_fallback=self.config.ragged_gather_reduce_fallback,
          gather_flops_override=self.config.ragged_gather_cost_estimate_flops,
          gather_reduce_flops_override=self.config.ragged_gather_reduce_cost_estimate_flops,
          gather_bytes_accessed_override=self.config.ragged_gather_cost_estimate_bytes_accessed,
          gather_reduce_bytes_accessed_override=self.config.ragged_gather_reduce_cost_estimate_bytes_accessed,
          use_single_sparsecore=self.config.ragged_sort_use_single_sparsecore,
          tag_routing_fn=tag_routing_fn,
      )
      if tag_routing_fn is None:
        topk_argsort_indices = None
    else:
      flatten_selected_experts = jnp.ravel(selected_experts)
      if forced_routed_experts is not None:
        valid_mask = moe_routing.valid_expert_mask(flatten_selected_experts, self.num_experts)
      if roll_to_expert_id is not None:
        flatten_selected_experts = (flatten_selected_experts - roll_to_expert_id) % self.num_experts
      if forced_routed_experts is not None:
        dummy_indices = jnp.arange(flatten_selected_experts.shape[0]) % self.num_experts
        flatten_selected_experts_safe = jnp.where(valid_mask, flatten_selected_experts, dummy_indices)
      else:
        flatten_selected_experts_safe = flatten_selected_experts

      sorted_selected_experts = jnp.argsort(flatten_selected_experts_safe)
      if self.config.moe_use_direct_token_gather:
        sorted_inputs = sorting._route_activations(inputs_2d, flatten_selected_experts_safe)
      else:
        replicated_inputs_2d = jnp.repeat(inputs_2d, self.num_experts_per_tok, axis=0)
        sorted_inputs = sorting._sort_activations(replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp)
      if not is_qarray:
        sorted_inputs = sorted_inputs.astype(self.dtype)
      group_size = jnp.bincount(flatten_selected_experts_safe, length=self.num_experts)

    num_tokens = bsz_times_seq_len * self.num_experts_per_tok
    use_truncated_buffer = use_ragged_in_permute and buffer_size is not None and buffer_size < num_tokens
    has_overflow = False
    max_load_ratio = None
    if use_ragged_in_permute and getattr(self.config, "moe_log_max_load_ratio", False):
      balanced_load = (bsz_times_seq_len / num_expert_parallelism) * self.num_experts_per_tok
      max_load_ratio = max_expert_shard_load_ratio(group_size, num_expert_parallelism, balanced_load)
      if self.mesh is not None and self.mesh.axis_names:
        max_load_ratio = jax.lax.pmax(max_load_ratio, tuple(self.mesh.axis_names))
    if use_truncated_buffer:
      local_num_experts = self.config.num_experts // num_expert_parallelism
      shard_idx = jax.lax.axis_index(self._expert_parallelism_name) if num_expert_parallelism > 1 else 0
      experts_start = shard_idx * local_num_experts
      local_group_size = jax.lax.dynamic_slice_in_dim(group_size, experts_start, local_num_experts, axis=0)
      has_overflow = jnp.sum(local_group_size) > buffer_size
      local_group_size = expert_parallel._truncate_matrix(local_group_size[:, None], buffer_size)[:, 0]
      if use_tc_sort and self.config.moe_tc_window_routing and (tag_fn := routing_tag_fn(self.config)):
        local_group_size = tag_fn(local_group_size)
      expert_indices = jnp.arange(local_num_experts)
      sorted_experts = jnp.repeat(expert_indices, repeats=local_group_size, total_repeat_length=buffer_size)
    else:
      local_group_size = None
      expert_indices = jnp.arange(self.num_experts)
      sorted_experts = jnp.repeat(expert_indices, repeats=group_size, total_repeat_length=math.prod(selected_experts.shape))

    if is_qarray:
      sorted_inputs = dataclasses.replace(inputs, qvalue=sorted_inputs)

    return (
        sorted_inputs, sorted_selected_experts, weights, group_size, sorted_experts, lb_loss,
        bias_updates, local_group_size, has_overflow, max_load_ratio, topk_argsort_indices, tc_routing,
    )

  def unpermute(
      self,
      intermediate,
      sorted_selected_experts,
      weights,
      batch_size,
      sequence_length,
      use_custom_sort_vjp=True,
      group_sizes=None,
      topk_argsort_indices=None,
      tc_routing=None,
      tc_prescaled=False,
      tc_out_3d=False,
  ):
    if tc_routing is not None:
      output = ring_ragged_unsort_tc(
          intermediate,
          tc_routing,
          self.num_experts_per_tok,
          jnp.ravel(weights).astype(jnp.float32),
          gather_block_size=self.config.moe_tc_ragged_gather_block_size,
          mask_padding=self.config.moe_tc_ragged_mask_padding,
          flatten_block_size=self.config.moe_tc_ragged_flatten_block_size,
          prescaled=tc_prescaled,
          out_3d=tc_out_3d,
      )
      if tc_out_3d:
        return output.reshape(-1, sequence_length, *output.shape[1:]).astype(self.dtype)
    elif self.config.use_ragged_sort and self.config.use_ring_of_experts:
      local_num_experts = self.config.num_experts // self.get_expert_parallelism_size()
      flat_weights = jnp.ravel(weights).astype(jnp.float32)
      output = ring_ragged_unsort(
          intermediate,
          group_sizes,
          sorted_selected_experts,
          self.num_experts_per_tok,
          local_num_experts,
          self._expert_parallelism_name,
          topk_weights=flat_weights,
          enforce_gather_fallback=self.config.ragged_gather_fallback,
          enforce_gather_reduce_fallback=self.config.ragged_gather_reduce_fallback,
          gather_flops_override=self.config.ragged_gather_cost_estimate_flops,
          gather_reduce_flops_override=self.config.ragged_gather_reduce_cost_estimate_flops,
          gather_bytes_accessed_override=self.config.ragged_gather_cost_estimate_bytes_accessed,
          gather_reduce_bytes_accessed_override=self.config.ragged_gather_reduce_cost_estimate_bytes_accessed,
          use_single_sparsecore=self.config.ragged_sort_use_single_sparsecore,
          topk_argsort_indices=topk_argsort_indices,
      )
    else:
      output = sorting._unsort_and_weight_sum(
          intermediate, sorted_selected_experts, weights, self.num_experts_per_tok, use_custom_sort_vjp,
          self.config.matmul_precision, self.config.decoder_block, self.config.float32_weight_sum,
      )
    return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

  def transform_bias(self, experts_index, *biases):
    return tuple(bias[experts_index] for bias in biases)

  def get_ragged_buffer_factor(self):
    override = getattr(self, "ragged_buffer_factor_override", None)
    if override is not None and override > 0:
      return override
    if eval_tiles.is_eval(self.config):
      return self.config.eval_ragged_buffer_factor
    return self.config.ragged_buffer_factor

  @staticmethod
  def get_ragged_buffer_size(local_batch, ep_degree, global_experts, top_k, ragged_buffer_factor):
    balanced_size = local_batch
    if ragged_buffer_factor > 0.0:
      return int(balanced_size * ragged_buffer_factor)
    else:
      worst_case_factor = min(ep_degree, global_experts / top_k)
      return int(balanced_size * worst_case_factor)

  def get_dropless_fallback_num_chunks(self, seq_len, n_chunks):
    ep = self.get_expert_parallelism_size()
    factor = self.get_ragged_buffer_factor()
    if factor <= 0.0 or factor >= min(ep, self.config.num_experts / self.num_experts_per_tok):
      return None
    target = min(math.ceil(n_chunks * ep / factor), seq_len)
    return next(d for d in range(target, seq_len + 1) if seq_len % d == 0)

  def _check_ragged_overflow(
      self, logits, pre_bias_logits, rngs, input_ids, forced_routed_experts, n_chunks, per_expert_scale=None
  ):
    num_ep = self.get_expert_parallelism_size()
    _, selected_experts = self.get_topk(
        logits, pre_bias_logits, rngs, input_ids, forced_routed_experts, per_expert_scale=per_expert_scale
    )
    batch, seq_len, top_k = selected_experts.shape
    target_shard = selected_experts // (self.config.num_experts // num_ep)
    counts = jax.nn.one_hot(target_shard, num_ep, dtype=jnp.int32)
    counts = counts.reshape(batch, n_chunks, -1, num_ep).sum(axis=(0, 2))
    if num_ep > 1:
      counts = jax.lax.psum(counts, self._expert_parallelism_name)
    buffer_size = self.get_ragged_buffer_size(
        batch * (seq_len // n_chunks) * top_k, num_ep, self.config.num_experts, top_k, self.get_ragged_buffer_factor()
    )
    overflow = jnp.any(counts > buffer_size).astype(jnp.int32)
    return jax.lax.psum(overflow, tuple(self.mesh.axis_names)) > 0

  def sparse_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
      input_ids=None,
      forced_routed_experts=None,
  ):
    """Perform sparse matrix multiplication of inputs and Experts."""

    fwd_tile = functools.partial(grouped_matmul.forward_tile, self.config)
    gmm = grouped_matmul.GroupedMatmul(
        self.config, self.mesh, self.quant, self.dtype, self.num_experts // self.get_expert_parallelism_size()
    )
    allow_batch_replication = self.get_expert_parallelism_size() == 1
    # Layer reads in the moe_ops units are deferred to their point of use: values through zero-arg callables, methods
    # through `lookup_*` callables that return the bound method, so the read order matches the layer's own code.
    config = lambda: self.config
    is_batch_sharded_by_expert = partition_specs.is_batch_sharded_by_ep(inputs)
    weight_gather = partition_specs.explicitly_weight_ag(config)
    shardings = partition_specs.get_routed_moe_shardings(
        config, lambda: self._logical_to_mesh_axes, is_batch_sharded_by_expert, input_ids is not None,
        wi_kernel_axes=lambda: self.wi_kernel_axes, wo_kernel_axes=lambda: self.wo_kernel_axes,
    )
    shardings = partition_specs.with_aqt_partition(shardings, w0_kernel, w1_kernel, wo_kernel)

    output_pspec = self._logical_to_mesh_axes(
        (
            shardings.batch_logical_axis,
            "activation_norm_length",
            "activation_embed",
        )
    )
    shardings, output_pspec = partition_specs.with_replicated_incompatible_batches(
        shardings, output_pspec, inputs, gate_logits, pre_bias_logits, input_ids, allow_batch_replication,
        lambda: self.mesh,
    )
    bias_update_axis_names = _batch_axis_names(shardings.gate_logits)
    # In RoE, logits are already all-gathered across self._expert_parallelism_name before
    # routing, so each shard evaluates the full batch and computes identical token counts.
    # Exclude the EP axis so psum doesn't redundantly reduce and scale counts by num_ep
    # for calculate_load_balance_updates().
    roe_bias_axis_names = _filter_axis_names(bias_update_axis_names, self._expert_parallelism_name)
    # Sigmoid and Sqrt(Softplus) routers use Megatron-LM's seq_aux_loss, computed once on the full sequences after the
    # shard_map (below), so the per-chunk loss inside the ring-of-experts loop is skipped.
    use_megatron_seq_aux_loss = (
        self.config.load_balance_loss_weight > 0.0
        and not self.is_hash_routing
        and uses_megatron_seq_aux_loss(self.config)
    )
    # Debug only: the shard_map gets a 5th (max_load_ratio) output only when requested, so the
    # default shard_map signature is unchanged. Stub configs in tests may lack the field.
    log_max_load_ratio = getattr(self.config, "moe_log_max_load_ratio", False)

    def finalize_bias_updates(summed_expert_counts):
      """Loss-free-balancing update from expert counts summed over all token chunks (the full batch)."""
      if summed_expert_counts is None:
        return None
      if getattr(self.config, "gradient_accumulation_steps", 1) > 1:
        # Under gradient accumulation, emit raw per-microbatch expert counts so
        # gradient_accumulation_loss_and_grad sums counts across microbatches
        # before taking a single sign() over the global batch.
        return summed_expert_counts
      return load_balance_updates_from_counts(
          summed_expert_counts, self.config.num_experts, self.config.routed_bias_update_rate
      )

    # defer_small_all_reduces: the ring-of-experts layer emits local partial counts (leading axis sharded over the
    # axes the per-layer psum would have used) and loss_fn reduces the stacked arrays once after the layer loop.
    defer_small_ars = bool(getattr(self.config, "defer_small_all_reduces", False)) and self.config.use_ring_of_experts
    per_expert_scale = None if self.per_expert_scale is None else self.per_expert_scale[...]
    in_specs = partition_specs.shard_map_in_specs(
        shardings, w0_kernel, w1_kernel, wo_kernel, forced_routed_experts is not None, per_expert_scale is not None
    )
    overflow_mesh_axes = partition_specs.overflow_mesh_axes(in_specs, lambda: self.mesh.axis_names)

    def sparse_matmul_route_and_compute(
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
        per_expert_scale=None,
    ):
      override_chunks = getattr(self, "num_moe_token_chunks", None)
      n_chunks = override_chunks if override_chunks is not None else self.config.num_moe_token_chunks
      barrier_enabled = getattr(self, "moe_chunk_barrier", None)
      if barrier_enabled is None:
        barrier_enabled = self.config.moe_chunk_barrier

      expert_parallelism_name = lambda: self._expert_parallelism_name
      expert_parallelism_size = lambda: self.get_expert_parallelism_size()
      tensor_parallelism_size = lambda: self.get_tensor_parallelism_size()
      ragged_buffer_factor = lambda: self.get_ragged_buffer_factor()
      moe_expert_input_dim = lambda: self.moe_expert_input_dim
      dtype = lambda: self.dtype
      local_expert_gmm = expert_gmm.LocalExpertGmm(
          config, fwd_tile, lambda: self.mesh, expert_parallelism_name, expert_parallelism_size, weight_gather,
          shardings.w0, shardings.wo, gmm,
      )
      ring_dispatch = ring_of_experts.RingOfExpertsDispatch(
          config, expert_parallelism_name, lambda: self.is_hash_routing, dtype, tensor_parallelism_size,
          moe_expert_input_dim, roe_bias_axis_names, use_megatron_seq_aux_loss, defer_small_ars,
          lambda: self.get_topk, lambda: self.permute, lambda: self.required_ragged_buffer_factor,
      )
      all_to_all_dispatch = ragged_all_to_all.RaggedAllToAllDispatch(
          config, expert_parallelism_name, expert_parallelism_size, moe_expert_input_dim, ragged_buffer_factor,
          is_batch_sharded_by_expert, bias_update_axis_names, use_megatron_seq_aux_loss,
          lambda: self.permute, lambda: self.get_ragged_buffer_size,
          lambda: RoutedMoE.get_all_to_all_params, lambda: RoutedMoE.local_permute,
      )
      sparse_expert_pass = expert_pass.SparseExpertPass(
          config, lambda: self.num_experts_per_tok, dtype, moe_expert_input_dim, expert_parallelism_name,
          lambda: self._tensor_parallelism_name, expert_parallelism_size, tensor_parallelism_size,
          ragged_buffer_factor, local_expert_gmm, ring_dispatch, all_to_all_dispatch,
          lambda: self.apply_ffn_activation, lambda: self.transform_bias, lambda: self.unpermute,
      )
      chunks = token_chunks.TokenChunkExecution(
          sparse_expert_pass, x, logits, pre_bias_logits, w0, w1, wo, w0_bias, w1_bias, wo_bias, sharded_input_ids,
          rngs, forced_routed_experts, per_expert_scale,
          config=config, n_chunks=n_chunks, barrier_enabled=barrier_enabled, defer_reductions=defer_small_ars,
          log_max_load_ratio=log_max_load_ratio, finalize_bias_updates=finalize_bias_updates,
      )

      force_dropless = getattr(self, "force_dropless", False)
      fallback_chunks = None
      if self.config.moe_dropless_fallback == "layer":
        fallback_chunks = self.get_dropless_fallback_num_chunks(x.shape[1], n_chunks)
      if fallback_chunks is None:
        return chunks.without_fallback(force_dropless)

      took_fallback = self._check_ragged_overflow(
          logits, pre_bias_logits, rngs, sharded_input_ids, forced_routed_experts, n_chunks, per_expert_scale
      )
      return chunks.with_dropless_fallback(took_fallback, fallback_chunks)

    route_and_compute = jax.shard_map(
        sparse_matmul_route_and_compute,
        mesh=self.mesh,
        in_specs=in_specs,
        out_specs=partition_specs.shard_map_out_specs(
            output_pspec, roe_bias_axis_names, defer_small_ars, overflow_mesh_axes, log_max_load_ratio
        ),
        check_vma=self.config.check_vma,
    )

    if self.config.moe_fsdp_use_two_stage_all_gather:
      w0_kernel = self._maybe_shard_with_logical(w0_kernel, ("exp_with_fsdp", None, "mlp"))
      w1_kernel = self._maybe_shard_with_logical(w1_kernel, ("exp_with_fsdp", None, "mlp"))
      wo_kernel = self._maybe_shard_with_logical(wo_kernel, ("exp_with_fsdp", "mlp", None))
      w0_kernel = jax.lax.optimization_barrier(w0_kernel)
      w1_kernel = jax.lax.optimization_barrier(w1_kernel)
      wo_kernel = jax.lax.optimization_barrier(wo_kernel)
      w0_kernel = self._maybe_shard_with_logical(w0_kernel, ("exp_with_fsdp", None, "mlp_no_fsdp"))
      w1_kernel = self._maybe_shard_with_logical(w1_kernel, ("exp_with_fsdp", None, "mlp_no_fsdp"))
      wo_kernel = self._maybe_shard_with_logical(wo_kernel, ("exp_with_fsdp", "mlp_no_fsdp", None))

    batch_logical_axis = shardings.batch_logical_axis
    input_logical_axes = (batch_logical_axis, "activation_norm_length", None)
    gate_logits_logical_axes = (batch_logical_axis, "activation_norm_length", None)
    pre_bias_logits_logical_axes = (
        (batch_logical_axis, "activation_norm_length", None)
        if self.config.model_name.startswith(("deepseek3", "deepseek4", "kimi-k2", "ming", "hy4"))
        else None
    )
    inputs = self._maybe_shard_with_pspec(inputs, shardings.input, logical_axes=input_logical_axes)
    gate_logits = self._maybe_shard_with_pspec(
        gate_logits,
        shardings.gate_logits,
        logical_axes=gate_logits_logical_axes,
    )
    pre_bias_logits = self._maybe_shard_with_pspec(
        pre_bias_logits,
        shardings.pre_bias_logits,
        logical_axes=pre_bias_logits_logical_axes,
    )
    if forced_routed_experts is not None:
      forced_routed_experts = self._maybe_shard_with_pspec(
          forced_routed_experts,
          shardings.gate_logits,
          logical_axes=gate_logits_logical_axes,
      )

    if self.config.moe_pin_sparse_core_all_gathers or getattr(self.config, "moe_pin_sparse_core_fsdp_all_gathers", False):
      fwd_only = getattr(self.config, "moe_pin_sparse_core_fsdp_all_gathers_fwd_only", False)

      def _fsdp_all_gather(w, pspec):
        if w is None or pspec is None:
          return w

        @functools.partial(
            compute_on,
            compute_type="tpu_sparsecore",
            out_memory_spaces=jax.memory.Space.Device,
            compiler_options={"sparse_core_config": {"core_ids": [self.config.moe_fsdp_all_gather_sparse_core_id]}},
        )
        def _reshard_fn(x):
          return self._maybe_shard_with_pspec(x, pspec)

        if not fwd_only:
          return _reshard_fn(w)

        @jax.custom_vjp
        def _pinned_fwd_reshard(x):
          return _reshard_fn(x)

        def _pinned_fwd_reshard_fwd(x):
          return _reshard_fn(x), None

        def _pinned_fwd_reshard_bwd(_, g):
          return (self._maybe_shard_with_pspec(g, pspec),)

        _pinned_fwd_reshard.defvjp(_pinned_fwd_reshard_fwd, _pinned_fwd_reshard_bwd)
        return _pinned_fwd_reshard(w)

    else:

      def _fsdp_all_gather(w, pspec):
        if w is None or pspec is None:
          return w
        return self._maybe_shard_with_pspec(w, pspec)

    pin_kernel_gather = (
        jnp.dtype(self.weight_dtype) == jnp.dtype(self.dtype)
        and self.config.scan_layers
        and not getattr(self.config, "prefetch_fsdp_weights", False)
    )
    w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias = expert_weights.gather_expert_weights(
        w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias, shardings, pin_kernel_gather, _fsdp_all_gather
    )
    if per_expert_scale is not None and self.config.shard_mode == ShardMode.EXPLICIT:
      per_expert_scale = self._maybe_shard_with_pspec(per_expert_scale, P())

    route_and_compute_outputs = route_and_compute(
        inputs,
        gate_logits,
        pre_bias_logits,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        w0_bias,
        w1_bias,
        wo_bias,
        input_ids,
        self.rngs,
        forced_routed_experts,
        per_expert_scale,
    )
    output, lb_loss, bias_updates, has_overflow, took_fallback, required_rbf = route_and_compute_outputs[:6]
    max_load_ratio = route_and_compute_outputs[6] if log_max_load_ratio else None
    if use_megatron_seq_aux_loss:
      lb_loss = self.megatron_seq_aux_loss(pre_bias_logits if pre_bias_logits is not None else gate_logits)
    if getattr(self, "force_dropless", False):
      self.sow(nnx.Intermediate, "moe_has_overflow", jnp.zeros((), jnp.bool_))
    else:
      self.sow(nnx.Intermediate, "moe_has_overflow", has_overflow)
    if self.config.moe_dropless_fallback == "layer":
      self.sow(nnx.Intermediate, "moe_dropless_fallback", took_fallback)
    if required_rbf is not None:
      self.sow(nnx.Intermediate, "moe_required_rbf", required_rbf)
    if max_load_ratio is not None:
      self.sow(nnx.Intermediate, "moe_max_load_ratio", max_load_ratio)
    return output, lb_loss, bias_updates

  def reshape_and_update_weights(self, weights, indices, safe_updates=False):
    """Reshape and update weights.

    safe_updates (forced routing): padding is masked to (index 0, weight 0) and
    the scatter accumulates instead of setting.
    """
    update_weights = jnp.zeros(
        (weights.shape[0], weights.shape[1], self.num_experts),
        dtype=weights.dtype if self.config.float32_weight_sum else self.dtype,
    )
    if safe_updates:
      valid_mask = indices >= 0
      safe_indices = jnp.where(valid_mask, indices, 0)
      safe_weights = jnp.where(valid_mask, weights, 0.0)
    else:
      safe_indices = indices
      safe_weights = weights.astype(update_weights.dtype)

    index_update = (
        self._maybe_shard_with_logical(
            jnp.arange(weights.shape[0])[:, None, None],
            ("activation_batch", None, None),
        ),
        self._maybe_shard_with_logical(jnp.arange(weights.shape[1])[:, None], ("activation_length", None)),
        safe_indices,
    )
    weight_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None))
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )
    if safe_updates:
      update_weights = update_weights.at[index_update].add(safe_weights, out_sharding=weight_sharding)
    else:
      update_weights = update_weights.at[index_update].set(safe_weights, out_sharding=weight_sharding)
    return update_weights

  def get_context_partition_and_sub_seq(self, seq_len):
    cp = self.get_context_autoregressive_parallelism_size()
    if seq_len % cp != 0:
      cp = 1
    sub_seq = seq_len // cp
    return cp, sub_seq

  def generate_masks_subgroup(self, top_k_indices, softmax_probs):
    """Subgroup mask generation for inference only."""
    cp, sub_seq = self.get_context_partition_and_sub_seq(top_k_indices.shape[1])
    return capacity.subgroup_capacity_masks(
        top_k_indices, softmax_probs, self.num_experts, self.num_experts_per_tok, self.config.capacity_factor,
        cp, sub_seq, self._maybe_shard_with_logical,
    )

  def generate_masks(self, top_k_indices, softmax_probs):
    """Generate masks."""
    return capacity.capacity_masks(
        top_k_indices, softmax_probs, self.num_experts, self.num_experts_per_tok, self.config.capacity_factor,
        self._maybe_shard_with_logical,
    )

  def load_balance_loss(self, top_k_indices, logits, density_prob=None) -> jax.Array:
    """Compute the sequence-wise load balance loss.

    For DeepSeek V4 like models, standard load balancing across an entire batch can
    be inadequate due to heterogeneous prompt lengths and varying sequence
    characteristics. This method implements sequence-wise load balancing by
    computing the token density and routing probabilities on a per-sequence basis.

    The resulting loss is scaled by `self.config.load_balance_loss_weight`.
    When this configuration value is set > 0, the computed loss is aggregated
    into the total training loss. By minimizing this scaled auxiliary loss,
    the optimizer updates the routing parameters to actively enforce an even
    distribution of tokens to experts within each individual sequence.

    `density_prob` optionally passes the precomputed per-sequence mean of `logits` (batch, num_experts).
    """
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    density = jnp.mean(summed_expert_mask, axis=1) / self.num_experts_per_tok
    if density_prob is None:
      density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def megatron_seq_aux_loss(self, scores) -> jax.Array:
    """Sequence-wise aux loss of the Megatron-LM sigmoid router (seq_aux_loss).

    Used for sigmoid and Sqrt(Softplus) routers (see `uses_megatron_seq_aux_loss`). Pass whole sequences: the token
    fractions and probabilities are per-sequence statistics.

    Args:
      scores: Router affinity scores before the expert bias, shape (batch, sequence, num_experts).

    Returns:
      Scalar loss averaged over the batch, already scaled by load_balance_loss_weight.
    """
    aux_probs, aux_top_k_indices = megatron_aux_loss_inputs(scores, self.num_experts_per_tok)
    return self.load_balance_loss(aux_top_k_indices, aux_probs)

  def get_einsum(
      self,
      rhs_mesh_axes: Tuple[Optional[str], ...] = (),
      einsum_name: str | None = None,
  ):
    """Get the Einstein summation."""
    if self.config.model_call_mode == "inference" and einsum_name in (
        DISPATCH,
        COMBINE,
    ):
      return jnp.einsum

    if self.quant and not isinstance(self.quant, ServeFp8WeightQuantization):
      op_id = einsum_name if einsum_name is not None else "einsum"

      def quant_einsum(*args, **kwargs):
        if self.quant_einsums is not None:
          if op_id not in self.quant_einsums:
            raise ValueError(
                f"Einsum name '{op_id}' is not registered in quant_einsums. "
                f"Available names: {list(self.quant_einsums.keys())}"
            )
          return self.quant_einsums[op_id](*args)
        einsum = self.quant.einsum(mesh_axes=rhs_mesh_axes)
        return apply_einsum_in_nnx(self, op_id, einsum, ["aqt"], *args)

      einsum_op = quant_einsum
    else:
      einsum_op = jnp.einsum
    return einsum_op

  def maybe_all_gather_kernel_weight_in_expert_parallelism(
      self, kernel: jax.Array, kernel_axes: Tuple[Optional[str], ...]
  ):
    """All-gather kernel weight in expert parallelism if needed."""
    if self.get_expert_parallelism_size() > 1:
      kernel = self._maybe_shard_with_logical(kernel, kernel_axes)
    return kernel

  def dense_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
      input_ids=None,
      forced_routed_experts=None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Dense matrix multiplication."""
    # gate_logits: batch, length, expert
    gate_logits = self._maybe_shard_with_logical(gate_logits, ("activation_batch_moe", "activation_length_moe", None))
    # NOTE: deepseek2 has a different pattern
    if self.config.model_name.startswith(("deepseek3", "deepseek4", "kimi-k2", "ming", "hy4")):
      # pre_bias_logits is None for non-deepseek3/4 models, including deepseek2
      pre_bias_logits = self._maybe_shard_with_logical(
          pre_bias_logits, ("activation_batch_moe", "activation_length_moe", None)
      )
    if forced_routed_experts is not None:
      forced_routed_experts = self._maybe_shard_with_logical(
          forced_routed_experts,
          ("activation_batch_moe", "activation_length_moe", None),
      )
    top_k_weights, top_k_indices = self.get_topk(
        gate_logits,
        pre_bias_logits,
        self.rngs,
        input_ids=input_ids,
        forced_routed_experts=forced_routed_experts,
    )
    is_llama4_decoder_layer = self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4
    if is_llama4_decoder_layer:
      router_scores = jax.nn.sigmoid(top_k_weights.astype(jnp.float32)).astype(self.dtype)
      inputs = inputs * router_scores
    else:
      weights = self.reshape_and_update_weights(
          top_k_weights,
          top_k_indices,
          safe_updates=(forced_routed_experts is not None),
      )
    matmul_precision = jax.lax.Precision(self.config.matmul_precision)

    # Calculate load balance loss
    # DeepSeek V4 uses Hash Routing for the first `first_num_hash_layers` layers.
    # These layers route deterministically based on token IDs and do not generate auxiliary loss.
    if self.config.model_call_mode != "inference" and not self.is_hash_routing:
      if self.config.load_balance_loss_weight > 0.0 and uses_megatron_seq_aux_loss(self.config):
        # Same loss as sparse_matmul: Megatron-LM's seq_aux_loss on the pre-bias affinity scores.
        lb_loss = self.megatron_seq_aux_loss(pre_bias_logits if pre_bias_logits is not None else gate_logits)
      else:
        softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        lb_loss = (
            self.load_balance_loss(top_k_indices, softmax_probs) if self.config.load_balance_loss_weight > 0.0 else None
        )
    else:
      lb_loss = None

    # Calculate routed bias updates (loss-free)
    # The bias update logic is only applicable to Top-K routed layers.
    if self.should_update_load_balance():
      # Under plain JIT (dense_matmul), GSPMD automatically inserts the all-reduce
      # across partitioned batch/sequence axes on global arrays, so axis_names=None.
      if getattr(self.config, "gradient_accumulation_steps", 1) > 1 or getattr(
          self.config, "using_pipeline_parallelism", False
      ):
        bias_updates = calculate_expert_counts(
            top_k_indices,
            self.config.num_experts,
            axis_names=None,
        )
      else:
        bias_updates = calculate_load_balance_updates(
            top_k_indices,
            self.config.num_experts,
            self.config.routed_bias_update_rate,
            axis_names=None,
        )
    else:
      bias_updates = None

    batch_size = inputs.shape[0]
    seq_len = inputs.shape[1]

    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    if self.config.capacity_factor > 0:
      # token dropping if needed
      moe_peel_expert = False  # only the training dispatch/MLP path peels 'expert' from the batch dim
      if self.config.model_call_mode != "inference":
        # TODO(b/425930949): remove this pylint by refactoring the logic here.
        dispatch_mask, combine_mask = self.generate_masks(
            top_k_indices, weights  # pylint: disable=undefined-variable,possibly-used-before-assignment
        )
        mask_axes = (
            "activation_batch_moe",
            "activation_norm_length_moe",
            None,
            None,
        )
        # Dispatch/MLP are already expert-sharded via "activation_exp". With
        # moe_dispatch_no_expert_sharding we peel 'expert' off the batch dim of these specs
        # (see _maybe_shard_moe_dispatch) so the GEMM stays expert-parallel (AllToAll) instead
        # of double-mapping E and B onto 'expert' (FSDP-style fallback).
        moe_peel_expert = self.config.moe_dispatch_no_expert_sharding
        dispatch_axis = (
            "activation_exp",
            "activation_batch_moe",
            None,
            "activation_embed_moe",
        )
        mlp_axis = (
            "activation_exp",
            "activation_batch_moe",
            None,
            "activation_mlp",
        )
        dispatch_eimsum = "BSM,BSEC -> EBCM"
        mlp_up_einsum = "EBCM,EMH -> EBCH"
        mlp_down_einsum = "EBCH,EHM -> EBCM"
        output_einsum = "EBCM,BSEC -> BSM"
      else:
        # TODO(b/425930507): Try replacing `softmax_probs` with padded weights
        # and verify with decode acc tests.
        softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        dispatch_mask, combine_mask = self.generate_masks_subgroup(top_k_indices, softmax_probs)
        if self.get_context_autoregressive_parallelism_size() > 0 and cp == 1:
          mask_axes = (
              "activation_norm_length_moe",
              "activation_batch_moe",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_norm_length_moe",
              "activation_batch_moe",
              None,
              "activation_embed_moe",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_embed_moe",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_mlp",
          )
        else:
          mask_axes = (
              "activation_batch_moe",
              "activation_norm_length_moe",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_batch_moe",
              "activation_norm_length_moe",
              None,
              "activation_embed_moe",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_embed_moe",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_mlp",
          )
        dispatch_eimsum = "BNSM,BNSEC -> EBNCM"
        mlp_up_einsum = "EBNCM,EMH -> EBNCH"
        mlp_down_einsum = "EBNCH,EHM -> EBNCM"
        output_einsum = "EBNCM,BNSEC -> BNSM"

        inputs = jnp.reshape(inputs, (batch_size, cp, sub_seq, inputs.shape[2]))
        inputs = self._maybe_shard_with_logical(inputs, input_axis)

      dispatch_mask = self._maybe_shard_with_logical(dispatch_mask, mask_axes)
      combine_mask = self._maybe_shard_with_logical(combine_mask, mask_axes)

      with jax.named_scope("dispatch"):
        # only cp during prefill
        dispatch = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=DISPATCH)(
            dispatch_eimsum, inputs, dispatch_mask, precision=matmul_precision
        )
        if cp > 1:
          dispatch = self._maybe_shard_with_logical(
              dispatch,
              (
                  None,
                  "activation_batch_moe",
                  "activation_norm_length_moe",
                  None,
                  "activation_embed_moe",
              ),
          )
        dispatch = self._maybe_shard_moe_dispatch(dispatch, dispatch_axis, moe_peel_expert)
      with jax.named_scope("wi_0"):
        w0_kernel_axes = ("exp", None, "mlp")
        w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w0_kernel, w0_kernel_axes)
        layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes, einsum_name=WI_0)(
            mlp_up_einsum, dispatch, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w0_bias = w0_bias[:, None, None, :]
          layer_w0 = layer_w0 + w0_bias

        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = self._maybe_shard_moe_dispatch(layer_w0, mlp_axis, moe_peel_expert)
        layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes, einsum_name=WI_1)(
            mlp_up_einsum, dispatch, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w1_bias = w1_bias[:, None, None, :]
          layer_w1 = layer_w1 + w1_bias
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = self._maybe_shard_moe_dispatch(layer_w1, mlp_axis, moe_peel_expert)
        layer_w1 = adc.checkpoint_name(adc.checkpoint_name(layer_w1, "mlpwi_1"), "moe_mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes, einsum_name=WO)(
            mlp_down_einsum,
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          wo_bias = wo_bias[:, None, None, :]
          intermediate_layer = intermediate_layer + wo_bias
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        if self.config.model_call_mode != "inference":
          intermediate_layer = self._maybe_shard_with_logical(
              intermediate_layer,
              (
                  "activation_exp",
                  "activation_batch_moe",
                  None,
                  "activation_embed_moe",
              ),
          )
        intermediate_layer = adc.checkpoint_name(adc.checkpoint_name(intermediate_layer, "mlpwo"), "moe_mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
            output_einsum,
            intermediate_layer,
            combine_mask,
            precision=matmul_precision,
        )
        if output.ndim == 4:
          output = jnp.reshape(
              output,
              (
                  output.shape[0],
                  output.shape[1] * output.shape[2],
                  output.shape[3],
              ),
          )
      return output, lb_loss, bias_updates
    else:
      inputs = self._maybe_shard_with_logical(
          inputs,
          (
              "activation_batch_moe",
              "activation_norm_length_moe",
              "activation_embed_moe",
          ),
      )
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes, einsum_name=WI_0)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w0 = layer_w0 + w0_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes, einsum_name=WI_1)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w1 = layer_w1 + w1_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = adc.checkpoint_name(adc.checkpoint_name(layer_w1, "mlpwi_1"), "moe_mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)

      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes, einsum_name=WO)(
            "BSEH,EHM -> BSEM",
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          intermediate_layer = intermediate_layer + wo_bias[None, None, :, :]
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = adc.checkpoint_name(adc.checkpoint_name(intermediate_layer, "mlpwo"), "moe_mlpwo")
      with jax.named_scope("weight_sum"):
        if is_llama4_decoder_layer:
          weights = self.reshape_and_update_weights(
              jnp.ones_like(top_k_weights),
              top_k_indices,
              safe_updates=(forced_routed_experts is not None),
          )
        if self.config.float32_weight_sum:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
          weights = weights.astype(jnp.float32)
        # cast to f32 for sum up in einsum op
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,
            precision=matmul_precision,
        ).astype(self.dtype)
      return output, lb_loss, bias_updates

  def fused_moe_matmul(
      self,
      inputs,
      gate_logits,
      wo_kernel,
      w0_kernel=None,
      w1_kernel=None,
      fused_kernel=None,
      forced_routed_experts=None,
      native_w1_scale=None,
      native_w2_scale=None,
  ) -> tuple[jax.Array, None, None]:
    """Fused MoE via tpu_inference fused_moe_func (vllm_rpa path only).

    fused_moe_func handles routing, GMM, and weighted combination internally.
    It does not compute lb_loss or bias_updates (inference-only).
    """
    if forced_routed_experts is not None:
      raise NotImplementedError("Forced routing via forced_routed_experts is not supported with" " fused_moe_matmul.")
    try:
      # pylint: disable=import-outside-toplevel
      # pytype: disable=import-error
      from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
      from tpu_inference import envs as tpu_inference_envs
    except ImportError as e:
      raise ImportError("fused_moe_matmul requires the tpu-inference package.") from e

    # Reshape 3D [B, S, D] -> 2D [T, D] (fused_moe_func expects 2D input)
    batch_size, seq_len, emb_dim = inputs.shape
    hidden_states = jnp.reshape(inputs, (batch_size * seq_len, emb_dim))
    gating_output = jnp.reshape(gate_logits, (batch_size * seq_len, self.num_experts))

    if self.config.return_routed_experts:
      _, top_k_indices = jax.lax.top_k(gating_output, self.num_experts_per_tok)
      self.selected_experts = nnx.Intermediate(top_k_indices)

    # Concatenate gate and up projections: [E, D, H] + [E, D, H] -> [E, D, 2H]
    # fused_moe_func splits this internally: gate=w1[..., :H], up=w1[..., H:]
    if fused_kernel is None:
      fused_kernel = jnp.concatenate([w0_kernel, w1_kernel], axis=-1)

    # Use expert parallelism if the expert axis has size > 1
    use_ep = self.get_expert_parallelism_size() > 1

    # Map MaxText config fields to fused_moe_func args
    activation = self.config.mlp_activations[0]  # e.g. "silu"
    scoring_fn = self.config.routed_score_func if self.config.routed_score_func else "softmax"

    # Check if the model architecture intrinsically renormalizes weights
    renormalize = self.config.norm_topk_prob or (
        self.config.decoder_block not in (ctypes.DecoderBlockType.LLAMA4, ctypes.DecoderBlockType.GEMMA4)
    )

    if native_w1_scale is not None and native_w2_scale is not None:
      # Checkpoint-native FP8: kernels are already raw FP8, scales already in
      # gmm_v2's rhs_scale layout (see prepare_fused_gmm_scale) -- no requantize.
      quantized_w1, quantized_w2 = fused_kernel, wo_kernel
      w1_scale, w2_scale = native_w1_scale, native_w2_scale
    else:
      # The fused kernel quantizes for itself: expert weights go in pre-quantized with
      # per-block scales (when the qwix rule covers the grouped matmul) and the kernel
      # quantizes the activations in-kernel, accumulating in f32. The call runs outside
      # qwix's interception: qwix only guards pallas_call, while the kernel is launched
      # through pl.kernel, so under a QtProvider its tiled matmuls would otherwise be
      # fake-quantized into fp8 x fp8 with a bf16 accumulator, which Mosaic rejects.
      rule = get_fused_moe_rule()
      quantized_w1, w1_scale = quantize_weight_for_fused_moe(fused_kernel, rule)
      quantized_w2, w2_scale = quantize_weight_for_fused_moe(wo_kernel, rule)
    # serve_fp8_weight with a fixed act_quantization_calibration_method: the experts use the
    # same static activation scale as the dense layers ({} keeps the kernel's dynamic scale).
    lhs_scale_kwargs = fused_moe_lhs_scale_kwargs(getattr(self, "quant", None), fused_moe_func)
    fused_moe = without_qwix_interception(fused_moe_func)

    output_2d = fused_moe(
        hidden_states=hidden_states,
        w1=quantized_w1,
        w2=quantized_w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=None,
        w2_bias=None,
        gating_output=gating_output,
        topk=self.num_experts_per_tok,
        renormalize=renormalize,
        mesh=self.mesh,
        use_ep=use_ep,
        activation=activation,
        scoring_fn=scoring_fn,
        **lhs_scale_kwargs,
        # Forward the same environment-backed kernel knobs that tpu-inference passes on its
        # own serving path (tpu_inference/layers/common/moe.py). Without these, env vars such
        # as ONEHOT_MOE_PERMUTE_THRESHOLD and VLLM_MOE_CHUNK_SIZE are silently ignored when a
        # MaxText model is served through vLLM, so the two paths run the same kernel with
        # different configurations. In particular, ONEHOT_MOE_PERMUTE_THRESHOLD selects the
        # one-hot permute path instead of the SparseCore ragged_gather_reduce kernel, which
        # is what makes expert parallelism usable on TPU generations whose SparseCore has
        # fewer SIMD lanes than the kernel requires (e.g. v5p).
        enable_rs_kernel=tpu_inference_envs.ENABLE_RS_KERNEL,
        use_gmm_fused_rs_kernel=tpu_inference_envs.USE_GMM_FUSED_RS_KERNEL,
        onehot_moe_permute_threshold=tpu_inference_envs.ONEHOT_MOE_PERMUTE_THRESHOLD,
        moe_chunk_size=tpu_inference_envs.VLLM_MOE_CHUNK_SIZE,
        scatter_results=(
            self.mesh is not None
            and (
                self.mesh.shape.get("data", 1)
                * self.mesh.shape.get("attn_dp", 1)
                * self.mesh.shape.get("attn_dp_expert", 1)
            )
            > 1
        ),
    )

    # Reshape output 2D [T, D] -> 3D [B, S, D]
    output = jnp.reshape(output_2d, (batch_size, seq_len, emb_dim))
    return output, None, None

  def retrieve_quantized_weight(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[aqt.QTensor, aqt.QTensor, aqt.QTensor]:
    """Retrieve quantized weights."""
    # This is called only during tracing. This is to invoke creation of
    # quantized tensor inside AqtEinsum.  After jit, this will become no-op and
    # will not affect performance.
    _ = self.dense_matmul(
        inputs,
        gate_logits,
        pre_bias_logits,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        w0_bias,
        w1_bias,
        wo_bias,
    )

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"]["frozen"]

    w0_kernel = partitioning.unbox_logicallypartioned(w0_kernel)
    w1_kernel = partitioning.unbox_logicallypartioned(w1_kernel)
    wo_kernel = partitioning.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  def _te_moe_block(
      self,
      inputs: jax.Array,
      gate_kernel: jax.Array,
      wi_kernel: jax.Array,
      wo_kernel: jax.Array,
      w0_bias: jax.Array | None,
      w1_bias: jax.Array | None,
      wo_bias: jax.Array | None,
      out_sharding: NamedSharding | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Run TransformerEngine's fused EP MoEBlock using MaxText-owned params."""
    try:
      from transformer_engine.jax import moe as te_moe  # pylint: disable=import-outside-toplevel
      from transformer_engine.jax.sharding import MeshResource  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
      raise ImportError(
          "te_moe_block=True requires TransformerEngine JAX MoE support. "
          "Please upgrade to the latest version of TransformerEngine."
      ) from exc

    if self.quant is None or not hasattr(self.quant, "get_moe_block_quantizer_sets"):
      raise ValueError("te_moe_block=True requires TransformerEngine quantization or te_gmm_quantization=te_no_quant.")

    expert_bias = None
    if self.config.routed_bias:
      expert_bias = jnp.asarray(self.gate.bias[...], jnp.float32)

    fsdp_size = self.mesh.shape.get("fsdp", 1)
    ep_size = self.mesh.shape.get(self._expert_parallelism_name, 1)
    if self.config.custom_mesh_and_rule == ctypes.CustomRule.EXPERT_ON_FSDP_EP:
      if self._expert_parallelism_name != "expert":
        raise ValueError("TE expert FSDP sharding currently requires the standard expert mesh axis.")
      if self.num_experts % (ep_size * fsdp_size) != 0:
        raise ValueError(f"num_experts={self.num_experts} must be divisible by EP x FSDP={ep_size * fsdp_size}.")
    if self.num_experts % ep_size != 0:
      raise ValueError(f"num_experts={self.num_experts} must be divisible by EP size={ep_size}.")

    fc1_quantizer_set, fc2_quantizer_set = self.quant.get_moe_block_quantizer_sets(
        self.config.te_gmm_quantization,
        # Dispatch buffers have one group per (fsdp, ep, local_expert),
        # while model weights retain their global expert-group descriptor.
        # TE owns either the BF16 gather or the opt-in quantized gather.
        n_token_groups=self.mesh.shape.get("data", 1) * fsdp_size * self.num_experts,
        n_expert_groups=self.num_experts,
    )

    checkpoint_kwargs = _get_te_moe_checkpoint_kwargs(te_moe.moe)
    cudnn_gmm_fusion_kwargs = _get_te_moe_cudnn_gmm_fusion_kwargs(self.config.te_moe_disable_cudnn_gmm_fusion)

    output, lb_loss, total_recv_tokens = te_moe.moe(
        inputs,
        gate_kernel,
        wi_kernel,
        wo_kernel,
        w0_bias,
        w1_bias,
        wo_bias,
        expert_bias,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        activation_type=self.config.mlp_activations[0],
        score_function=self.config.routed_score_func or "softmax",
        use_pre_softmax=False,
        num_groups=None if self.config.n_routing_groups <= 0 else self.config.n_routing_groups,
        group_topk=None if self.config.topk_routing_group <= 0 else self.config.topk_routing_group,
        scaling_factor=self.config.routed_scaling_factor,
        aux_loss_coeff=self.config.load_balance_loss_weight,
        apply_topk_weights_early=True,
        quantizer_sets=(fc1_quantizer_set, fc2_quantizer_set),
        mesh_resource=MeshResource(dp_resource="data", fsdp_resource="fsdp", ep_resource=self._expert_parallelism_name),
        input_axes=("activation_batch", "activation_norm_length", None),
        gate_kernel_axes=self.kernel_axes,
        wi_kernel_axes=self.wi_kernel_axes,
        wo_kernel_axes=self.wo_kernel_axes,
        dtype=self.dtype,
        recv_capacity_per_rank=moe_bootstrap.get_te_moe_recv_capacity_per_rank(),
        quant_before_fsdp_ag=self.config.te_moe_quantize_before_fsdp_all_gather,
        **cudnn_gmm_fusion_kwargs,
        **checkpoint_kwargs,
    )
    recv_capacity_per_rank = moe_bootstrap.get_te_moe_recv_capacity_per_rank()
    output = output.astype(self.dtype)
    if lb_loss is not None:
      lb_loss = lb_loss.astype(self.dtype)
    if out_sharding is not None:
      output = jax.lax.with_sharding_constraint(output, out_sharding)
    return output, lb_loss, (total_recv_tokens, jnp.asarray(recv_capacity_per_rank, dtype=jnp.int32))

  def __call__(
      self,
      inputs: jax.Array,
      input_ids: jax.Array | None = None,
      gate_inputs: jax.Array | None = None,
      out_sharding: NamedSharding | None = None,
      forced_routed_experts: jax.Array | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Executes the routed MoE block.

    Args:
      inputs: The input activations.
      input_ids: Optional token IDs corresponding to the inputs, used strictly for Hash Routing
        in DeepSeek V4's early MoE layers. If None, routing relies purely on `gate_inputs` or `inputs`.
      gate_inputs: Optional alternate inputs to feed into the routing gate.
      out_sharding: Optional sharding specification for the output.

    Returns:
      A tuple containing the MoE output, the load balance loss (if applicable),
      and any routed bias updates.
    """
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)

    if cfg.te_moe_block and gate_inputs is not None:
      raise ValueError("te_moe_block=True does not support separate gate_inputs.")
    if cfg.te_moe_block and in_serve_mode(self.quant):
      raise ValueError("te_moe_block=True does not support serve-mode quantized weights.")

    gate_dtype = jnp.float32 if cfg.float32_gate_logits else cfg.dtype
    routing_inputs = inputs if gate_inputs is None else gate_inputs.astype(gate_dtype)

    if cfg.te_moe_block:
      gate_kernel = jnp.asarray(self.gate.kernel[...], self.dtype)
      wi_kernel = jnp.asarray(self.wi[...], self.dtype)
      wo_kernel = jnp.asarray(self.wo[...], self.dtype)
      if self.per_expert_scale is not None:
        wo_kernel = wo_kernel * jnp.asarray(self.per_expert_scale[...], self.dtype)[:, None, None]
      if cfg.mlp_bias:
        w0_bias = jnp.asarray(self.wi_0_bias[...], self.dtype)
        w1_bias = jnp.asarray(self.wi_1_bias[...], self.dtype)
        wo_bias = jnp.asarray(self.wo_bias[...], self.dtype)
      else:
        w0_bias, w1_bias, wo_bias = None, None, None
      return self._te_moe_block(
          inputs,
          gate_kernel,
          wi_kernel,
          wo_kernel,
          w0_bias,
          w1_bias,
          wo_bias,
          out_sharding,
      )

    gate_logits, pre_bias_logits = self.gate(routing_inputs)

    is_fused_moe_path = cfg.attention in ("vllm_rpa", "vllm_batched_rpa") and not self.is_hash_routing

    native_gmm = (
        isinstance(self.quant, ServeFp8WeightQuantization)
        and cfg.sparse_matmul
        and cfg.use_gmm_v2
        and not is_fused_moe_path
    )
    is_kernel_quantized = isinstance(self.quant, ServeFp8WeightQuantization) and is_fused_moe_path

    def _maybe_native_gmm_weight(kernel, scale):
      """Non-fused (gmm_v2) path: a qpl.QArray wrapping kernel+scale when
      native_gmm applies, else the plain dequantized kernel."""
      if native_gmm and ctypes.is_fp8_dtype(kernel.dtype) and scale is not None:
        scheme, channel_axis = infer_scale_granularity(scale.shape[1:])
        if scheme == "per_tensor":
          per_expert_scale = jnp.max(scale.reshape(scale.shape[0], -1), axis=1).reshape(-1, 1, 1)
          return qpl.QArray(qvalue=kernel, scale=per_expert_scale)
        elif scheme == "per_channel" and channel_axis == 1:
          # gmm_v2 only natively broadcasts per-output-channel (N-axis) scales.
          per_channel_scale = scale.reshape(scale.shape[0], 1, -1)
          return qpl.QArray(qvalue=kernel, scale=per_channel_scale)
      return dequantize_weight(kernel, scale, self.dtype)

    def _maybe_native_fused_weight(kernel, scale):
      """Fused (tpu_inference) path: returns (kernel, native_scale). Native
      (is_kernel_quantized, FP8 kernel, scale present): kernel stays raw FP8,
      scale is expanded to gmm_v2's rhs_scale layout. Otherwise: kernel is
      dequantized and native_scale is None."""
      if is_kernel_quantized and ctypes.is_fp8_dtype(kernel.dtype) and scale is not None:
        return kernel, prepare_fused_gmm_scale(scale, kernel.shape)
      return dequantize_weight(kernel, scale, self.dtype), None

    wo_scale = self.wo_scale[...] if self.wo_scale is not None else None
    if is_fused_moe_path:
      wo_kernel, wo_native_scale = _maybe_native_fused_weight(self.wo[...], wo_scale)
    else:
      wo_kernel, wo_native_scale = _maybe_native_gmm_weight(self.wo[...], wo_scale), None

    fused_kernel = None
    w0_kernel = None
    w1_kernel = None
    wi_native_scale = None
    if cfg.prefuse_moe_weights and is_fused_moe_path:
      wi_scale = self.wi_scale[...] if self.wi_scale is not None else None
      fused_kernel, wi_native_scale = _maybe_native_fused_weight(self.wi[...], wi_scale)
    elif cfg.prefuse_moe_weights:
      wi_scale = self.wi_scale[...] if self.wi_scale is not None else None
      wi = dequantize_weight(self.wi[...], wi_scale, self.dtype)
      n = wi.shape[-1] // 2
      w0_kernel = wi[..., :n]
      w1_kernel = wi[..., n:]
    else:
      wi_0_scale = self.wi_0_scale[...] if self.wi_0_scale is not None else None
      wi_1_scale = self.wi_1_scale[...] if self.wi_1_scale is not None else None
      if is_fused_moe_path:
        w0_kernel, w0_native_scale = _maybe_native_fused_weight(self.wi_0[...], wi_0_scale)
        w1_kernel, w1_native_scale = _maybe_native_fused_weight(self.wi_1[...], wi_1_scale)
        if w0_native_scale is not None and w1_native_scale is not None:
          wi_native_scale = jnp.concatenate([w0_native_scale, w1_native_scale], axis=-1)
      else:
        w0_kernel = _maybe_native_gmm_weight(self.wi_0[...], wi_0_scale)
        w1_kernel = _maybe_native_gmm_weight(self.wi_1[...], wi_1_scale)

    # For fused MoE path (inference only), if we have not fused expert
    # scales at init, we must apply them to wo_kernel here because
    # fused_moe_func doesn't support them. Other paths (dense/sparse
    # matmul) apply them to top_k_weights in get_topk.
    if is_fused_moe_path:
      if self.per_expert_scale is not None and not (cfg.model_call_mode == "inference" and cfg.fuse_expert_scales):
        if wo_native_scale is not None:
          # Fold into the companion scale -- wo_kernel is still raw FP8 here.
          per_expert = jnp.asarray(self.per_expert_scale[...], jnp.float32)[:, None, None, None]
          wo_native_scale = wo_native_scale * per_expert
        else:
          wo_kernel = wo_kernel * jnp.asarray(self.per_expert_scale[...], self.dtype)[:, None, None]

    if self.wi_0_sparsity_module is not None:
      _, w0_kernel = self.wi_0_sparsity_module(jnp.zeros_like(w0_kernel), w0_kernel)
      _, w1_kernel = self.wi_1_sparsity_module(jnp.zeros_like(w1_kernel), w1_kernel)
      _, wo_kernel = self.wo_sparsity_module(jnp.zeros_like(wo_kernel), wo_kernel)
    if cfg.mlp_bias:
      w0_bias = jnp.asarray(self.wi_0_bias[...], self.dtype)
      w1_bias = jnp.asarray(self.wi_1_bias[...], self.dtype)
      wo_bias = jnp.asarray(self.wo_bias[...], self.dtype)
    else:
      w0_bias, w1_bias, wo_bias = None, None, None

    # vllm_rpa codepath uses fused_moe_func from tpu_inference for optimized inference.
    # The fused MoE kernel currently only supports standard Top-K routing with associated
    # weights. Hash routed layers bypass this kernel and fall back
    # to the sparse matmul implementation.
    if is_fused_moe_path:
      output, lb_loss, bias_updates = self.fused_moe_matmul(
          inputs,
          gate_logits,
          wo_kernel,
          w0_kernel=w0_kernel,
          w1_kernel=w1_kernel,
          fused_kernel=fused_kernel,
          forced_routed_experts=forced_routed_experts,
          native_w1_scale=wi_native_scale,
          native_w2_scale=wo_native_scale,
      )
    elif cfg.sparse_matmul:
      if in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs,
            gate_logits,
            pre_bias_logits,
            w0_kernel,
            w1_kernel,
            wo_kernel,
            w0_bias,
            w1_bias,
            wo_bias,
        )
      output, lb_loss, bias_updates = self.sparse_matmul(
          inputs,
          gate_logits,
          pre_bias_logits,
          w0_kernel,
          w1_kernel,
          wo_kernel,
          w0_bias,
          w1_bias,
          wo_bias,
          input_ids=input_ids,
          forced_routed_experts=forced_routed_experts,
      )
    else:
      output, lb_loss, bias_updates = self.dense_matmul(
          inputs,
          gate_logits,
          pre_bias_logits,
          w0_kernel,
          w1_kernel,
          wo_kernel,
          w0_bias,
          w1_bias,
          wo_bias,
          input_ids=input_ids,
          forced_routed_experts=forced_routed_experts,
      )
    return output, lb_loss, bias_updates


class RoutedAndSharedMoE(nnx.Module):
  """Implements a block which combines shared and routed experts."""

  def __init__(
      self,
      config: ctypes.Config,
      mesh: jax.sharding.Mesh,
      kernel_init: NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[AqtQuantization] = None,
      is_hash_routing: bool = False,
  ):
    """Initializes the RoutedAndSharedMoE module.

    Attributes:
      config: The main config setting.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
      is_hash_routing: Passed down to the internal `RoutedMoE` to determine routing behavior (e.g. hash vs top-K).
    """
    self.config = config
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs
    self.is_hash_routing = is_hash_routing
    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    # The router ('embed_router') has its own logical axis so a rule set can
    # replicate or shard it independently from the routed experts ('embed_moe').
    # NOTE: the name MoeBlock_0 is to ensure reverse compatibility with
    # existing checkpoints for routed experts.
    self.MoeBlock_0 = RoutedMoE(
        config=self.config,
        num_experts=self.config.num_experts,
        num_experts_per_tok=self.config.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_router", None),
        intermediate_dim=self.config.moe_mlp_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        rngs=self.rngs,
        is_hash_routing=self.is_hash_routing,
    )

    shared_expert_mlp_dim = ffn_flops.get_shared_expert_mlp_dim(self.config)
    # DeepSeek-V4 clamps the shared expert's SwiGLU like the routed experts (apply_ffn_activation).
    clamp_shared = (
        self.config.decoder_block == ctypes.DecoderBlockType.DEEPSEEK4 and self.config.mlp_activations_limit > 0.0
    )
    self.shared_experts = linears.MlpBlock(
        mesh=self.mesh,
        in_features=self.moe_expert_input_dim,
        intermediate_dim=self.config.shared_experts * shared_expert_mlp_dim,
        activations=self.config.mlp_activations,
        kernel_init=self.kernel_init,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=self.quant,
        rngs=self.rngs,
        activations_limit=self.config.mlp_activations_limit if clamp_shared else None,
    )

  @property
  def routed_moe(self):
    return self.MoeBlock_0

  def __call__(
      self,
      inputs: jax.Array,
      original_inputs: jax.Array | None = None,
      gate_inputs: jax.Array | None = None,
      intermediate_sharding: NamedSharding | None = None,
      out_sharding: NamedSharding | None = None,
      input_ids: jax.Array | None = None,
      forced_routed_experts: jax.Array | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Executes both the routed experts and the shared expert block.

    Args:
      inputs: The input activations.
      original_inputs: The original pre-normalization inputs (unused by default).
      gate_inputs: Optional alternate inputs to feed into the routing gate.
      intermediate_sharding: Optional sharding spec for the shared experts intermediate output.
      out_sharding: Optional sharding spec for the final combined output.
      input_ids: Optional token IDs corresponding to the inputs, used strictly for Hash Routing
        in DeepSeek V4's early MoE layers.

    Returns:
      A tuple containing the combined MoE output (routed + shared),
      the load balance loss, and any routed bias updates.
    """
    routed_experts, load_balance_loss, moe_bias_updates = self.routed_moe(
        inputs,
        gate_inputs=gate_inputs,
        out_sharding=out_sharding,
        input_ids=input_ids,
        forced_routed_experts=forced_routed_experts,
    )
    shared_experts = self.shared_experts(
        inputs,
        intermediate_sharding=intermediate_sharding,
        out_sharding=out_sharding,
    )
    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates
