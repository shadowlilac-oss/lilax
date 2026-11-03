#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  pytype: disable=module-attr
"""Attentions Ops Layers."""

import math
from typing import Any, Callable, Optional

from flax import linen as nn
from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh
from lilax._backend.common.common_types import (
    Array,
    AttentionType,
    AxisIdxes,
    AxisNames,
    BATCH_ATTN,
    CACHE_BATCH,
    CACHE_BATCH_PREFILL,
    CACHE_HEADS,
    CACHE_KV,
    CACHE_SCALE_BATCH,
    CACHE_SCALE_HEADS,
    CACHE_SCALE_KV,
    CACHE_SCALE_SEQUENCE,
    CACHE_SEQUENCE,
    Config,
    SEGMENT_ID_BATCH,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    DEFAULT_MASK_VALUE,
    DType,
    D_KV,
    HEAD,
    KV_LENGTH,
    LENGTH,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    Q_LENGTH,
)
from lilax._backend.inference.kvcache import KVQuant, KVTensor
from lilax._backend.kernels.attention import tokamax_ring_attention
from lilax._backend.kernels.attention import ulysses_attention
from lilax._backend.kernels.attention import usp_attention
from lilax._backend.layers import nnx_wrappers
from lilax._backend.layers.attention_execution import context_parallel_layout_validation
from lilax._backend.layers.attention_execution import context_parallel_runtime_validation
from lilax._backend.layers.attention_execution import cudnn_jax
from lilax._backend.layers.attention_execution import cudnn_te
from lilax._backend.layers.attention_execution import dot_product
from lilax._backend.layers.attention_execution import gpu_flash
from lilax._backend.layers.attention_execution import gpu_ragged
from lilax._backend.layers.attention_execution import normalization
from lilax._backend.layers.attention_execution import tpu_flash
from lilax._backend.layers.attention_execution import tpu_ragged
from lilax._backend.layers.attention_execution.gpu_flash_validation import validate_gpu_flash_attention
from lilax._backend.layers.attention_execution.tpu_splash_tuning import SplashTuning, select_block_sizes
from lilax._backend.layers.attention_masks import dense_masks
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant
from lilax._backend.layers.context_parallel import sequence_reorder
from lilax._backend.kernels.tiling import eval_tiles
from lilax._backend.distributed.sharding import get_logical_axis_rules, logical_to_mesh_axes, maybe_shard_with_pspec

# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


dynamic_vector_slice_in_dim = jax.vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


def _resolve_attention_type(config: Config, attention_type: AttentionType | str | None) -> AttentionType:
  configured_attention_type = AttentionType(getattr(config, "attention_type", AttentionType.GLOBAL.value))
  if attention_type is None:
    return configured_attention_type
  resolved_attention_type = AttentionType(attention_type)
  if configured_attention_type == AttentionType.BLOCK_DIFFUSION and resolved_attention_type == AttentionType.GLOBAL:
    return configured_attention_type
  return resolved_attention_type


class AttentionOp(nnx.Module):
  """Attention operation"""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      attention_kernel: str,
      max_target_length: int,
      num_query_heads: int,
      num_kv_heads: int,
      float32_qk_product: bool = False,
      max_prefill_predict_length: int = -1,
      float32_logits: bool = False,
      flash_axis_names_q: AxisNames = (BATCH_ATTN, HEAD, LENGTH, D_KV),
      flash_axis_names_kv: AxisNames = (BATCH_ATTN, HEAD, KV_LENGTH, D_KV),
      flash_axis_names_splash_kernel: AxisNames = (HEAD, LENGTH),
      prefill_cache_logical_axis_names: AxisNames = (
          CACHE_BATCH_PREFILL,
          CACHE_SEQUENCE,
          CACHE_HEADS,
          CACHE_KV,
      ),
      cache_logical_axis_names: AxisNames = (
          CACHE_BATCH,
          CACHE_SEQUENCE,
          CACHE_HEADS,
          CACHE_KV,
      ),
      cache_scale_logical_axis_names: AxisNames = (
          CACHE_SCALE_BATCH,
          CACHE_SCALE_SEQUENCE,
          CACHE_SCALE_HEADS,
          CACHE_SCALE_KV,
      ),
      ragged_qkv_axis_names: AxisNames = (
          CACHE_BATCH,
          CACHE_HEADS,
          CACHE_SEQUENCE,
          CACHE_KV,
      ),
      ragged_lengths_names: AxisNames = (CACHE_BATCH,),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      key_axis_order: AxisIdxes = (2, 0, 1, 3),
      reshape_q: bool = False,
      dropout_rate: float = 0.0,
      dtype: DType = jnp.float32,
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      chunk_attn_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      rngs: nnx.Rngs | None = None,
  ):
    """Initializes the AttentionOp module.

    Args:
      config: The configuration for the model.
      mesh: The device mesh.
      attention_kernel: The attention kernel to use.
      max_target_length: The maximum target length.
      num_query_heads: The number of query heads.
      num_kv_heads: The number of key/value heads.
      float32_qk_product: Whether to compute qk_product in float32.
      max_prefill_predict_length: The maximum prefill predict length.
      float32_logits: Whether to compute logits in float32.
      flash_axis_names_kv: The logical axis names for the KV cache in flash
        attention.
      flash_axis_names_q: The logical axis names for the query in flash
        attention.
      flash_axis_names_splash_kernel: The logical axis names for the splash
        attention kernel.
      prefill_cache_logical_axis_names: The logical axis names for the prefill
        cache.
      cache_logical_axis_names: The logical axis names for the cache.
      cache_scale_logical_axis_names: The logical axis names for the cache
        scale.
      ragged_qkv_axis_names: The logical axis names for ragged QKV tensors.
      ragged_lengths_names: The logical axis names for ragged lengths.
      compute_axis_order: The order of axes for computation.
      key_axis_order: The order of axes for the key. ... and other configuration
        parameters.
      rngs: The random number generators for initialization, passed by the
        nnx.to_linen wrapper.
    """
    self.config = config
    self.mesh = mesh
    self.attention_kernel = attention_kernel
    self.max_target_length = max_target_length
    self.num_query_heads = num_query_heads
    self.num_kv_heads = num_kv_heads
    self.attention_type = _resolve_attention_type(self.config, attention_type)
    self.is_absorbed_mqa = getattr(config, "use_mla_absorbed_mqa", False) and self.attention_type == AttentionType.MLA
    self.float32_qk_product = float32_qk_product
    self.max_prefill_predict_length = max_prefill_predict_length
    self.float32_logits = float32_logits
    self.flash_axis_names_q = flash_axis_names_q
    self.flash_axis_names_kv = flash_axis_names_kv
    self.flash_axis_names_splash_kernel = flash_axis_names_splash_kernel
    self.prefill_cache_logical_axis_names = prefill_cache_logical_axis_names
    self.cache_logical_axis_names = cache_logical_axis_names
    self.cache_scale_logical_axis_names = cache_scale_logical_axis_names
    self.ragged_qkv_axis_names = ragged_qkv_axis_names
    self.ragged_lengths_names = ragged_lengths_names
    self.compute_axis_order = compute_axis_order
    self.key_axis_order = key_axis_order
    self.reshape_q = reshape_q
    self.dropout_rate = dropout_rate
    self.dtype = dtype
    self.quant = quant
    self.kv_quant = kv_quant
    self.causal_block_size = getattr(self.config, "causal_block_size", None)
    if self.attention_type == AttentionType.BLOCK_DIFFUSION:
      if self.causal_block_size is None or self.causal_block_size <= 0:
        raise ValueError("causal_block_size must be positive for block-diffusion attention")
      if self.attention_kernel not in ("autoselected", "dot_product", "flash"):
        raise ValueError("Block-diffusion attention is supported only by dot_product attention and TPU Splash attention.")
    # Block sizes are only used by TPU splash attention kernels. Exclude non-splash kernels
    if self.attention_kernel not in (
        "dot_product",
        "paged",
        "vllm_rpa",
        "vllm_batched_rpa",
        "cudnn_flash_te",
        "cudnn_flash_jax",
    ):
      if self.attention_type == AttentionType.LOCAL_SLIDING:
        self.block_q = self.config.local_sa_block_q
        self.block_kv = self.config.local_sa_block_kv
        self.block_kv_compute = self.config.local_sa_block_kv_compute
        self.eval_block_q = self.config.eval_local_sa_block_q
        self.eval_block_kv = self.config.eval_local_sa_block_kv
        self.eval_block_kv_compute = self.config.eval_local_sa_block_kv_compute
        self.block_q_dkv = self.config.local_sa_block_q_dkv
        self.block_kv_dkv = self.config.local_sa_block_kv_dkv
        self.block_kv_dkv_compute = self.config.local_sa_block_kv_dkv_compute
        self.block_q_dq = self.config.local_sa_block_q_dq
        self.block_kv_dq = self.config.local_sa_block_kv_dq
        self.use_fused_bwd_kernel = self.config.local_sa_use_fused_bwd_kernel
        self.q_layout = self.config.local_sa_q_layout
        self.k_layout = self.config.local_sa_k_layout
        self.v_layout = self.config.local_sa_v_layout
        self.eval_q_layout = self.config.eval_local_sa_q_layout
        self.eval_k_layout = self.config.eval_local_sa_k_layout
        self.eval_v_layout = self.config.eval_local_sa_v_layout
        self.use_splash_scheduler = self.config.local_use_splash_scheduler
        self.fuse_reciprocal = self.config.local_sa_fuse_reciprocal
        self.use_base2_exp = self.config.local_sa_use_base2_exp
        # Diagonal sub-tile skipping requires a pure causal mask.
        self.qk_diag_skip = False
        self.sv_diag_skip = False
      else:
        self.block_q = self.config.sa_block_q
        self.block_kv = self.config.sa_block_kv
        self.block_kv_compute = self.config.sa_block_kv_compute
        self.eval_block_q = self.config.eval_sa_block_q
        self.eval_block_kv = self.config.eval_sa_block_kv
        self.eval_block_kv_compute = self.config.eval_sa_block_kv_compute
        self.block_q_dkv = self.config.sa_block_q_dkv
        self.block_kv_dkv = self.config.sa_block_kv_dkv
        self.block_kv_dkv_compute = self.config.sa_block_kv_dkv_compute
        self.block_q_dq = self.config.sa_block_q_dq
        self.block_kv_dq = self.config.sa_block_kv_dq
        self.use_fused_bwd_kernel = self.config.sa_use_fused_bwd_kernel
        self.q_layout = self.config.sa_q_layout
        self.k_layout = self.config.sa_k_layout
        self.v_layout = self.config.sa_v_layout
        self.eval_q_layout = self.config.eval_sa_q_layout
        self.eval_k_layout = self.config.eval_sa_k_layout
        self.eval_v_layout = self.config.eval_sa_v_layout
        self.use_splash_scheduler = self.config.use_splash_scheduler
        self.fuse_reciprocal = self.config.sa_fuse_reciprocal
        self.use_base2_exp = self.config.sa_use_base2_exp
        self.qk_diag_skip = self.config.sa_qk_diag_skip
        self.sv_diag_skip = self.config.sa_sv_diag_skip
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.sliding_window_size = sliding_window_size
    self.chunk_attn_window_size = chunk_attn_window_size
    self.use_ragged_attention = use_ragged_attention
    self.ragged_block_size = ragged_block_size
    # Only the cudnn_flash_te bridge draws from these, and only with attention dropout
    # on; holding them otherwise leaves dead RNG state in the model.
    self.rngs = rngs if dropout_rate > 0.0 else None
    if self.attention_kernel == "flash" and tokamax_ring_attention.is_context_parallel_ring_requested(self.config):
      context_parallel_layout_validation.validate_tokamax_ring_layout(
          self.config, self.mesh, self.attention_type, self.num_query_heads, self.num_kv_heads,
          self.flash_axis_names_q, self.flash_axis_names_kv, self._logical_to_mesh_axes,
      )
    if self.attention_kernel == "flash" and ulysses_attention.is_context_parallel_ulysses_requested(self.config):
      context_parallel_layout_validation.validate_ulysses_layout(
          self.config, self.mesh, self.attention_type, self.num_query_heads, self.num_kv_heads, self.dropout_rate,
          self.use_ragged_attention, self.flash_axis_names_q, self.flash_axis_names_kv, self._logical_to_mesh_axes,
      )
    if self.attention_kernel == "flash" and usp_attention.is_context_parallel_usp_requested(self.config):
      context_parallel_layout_validation.validate_usp_layout(
          self.config, self.mesh, self.attention_type, self.num_query_heads, self.num_kv_heads, self.dropout_rate,
          self.use_ragged_attention, self.flash_axis_names_q, self.flash_axis_names_kv, self._logical_to_mesh_axes,
      )

    def maybe_create_nnx(einsum, *args):
      if isinstance(einsum, nn.Module):
        return nnx_wrappers.ToNNX(einsum, rngs=rngs).lazy_init(*args)
      return einsum

    # qk_product
    if self.kv_quant:
      # Dummy inputs for lazy initialization
      b = 1
      t_prefill = self.max_prefill_predict_length
      t_ar = 1  # Autoregressive mode has a query length of 1
      n = self.num_query_heads
      n_kv = self.num_kv_heads
      d = self.config.head_dim
      g = n // n_kv
      s_prefill = self.max_prefill_predict_length
      s_ar = self.max_target_length

      # Dummy query/key/value shapes as before...
      dummy_query_prefill = jnp.zeros((b, t_prefill, n_kv, g, d), dtype=self.dtype)
      dummy_key_prefill = jnp.zeros((b, s_prefill, n_kv, d), dtype=self.dtype)
      dummy_query_ar = jnp.zeros((b, t_ar, n_kv, g, d), dtype=self.dtype)
      dummy_key_ar = jnp.zeros((b, s_ar, n_kv, d), dtype=self.dtype)

      dummy_attn_weights_prefill = jnp.zeros((b, n_kv, g, t_prefill, s_prefill), dtype=jnp.float32)
      dummy_value_prefill = jnp.zeros((b, s_prefill, n_kv, d), dtype=self.dtype)
      dummy_attn_weights_ar = jnp.zeros((b, n_kv, g, t_ar, s_ar), dtype=jnp.float32)
      dummy_value_ar = jnp.zeros((b, s_ar, n_kv, d), dtype=self.dtype)

      # Prefill AqtEinsum instances
      self.AqtEinsum_0 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor(),
          "btkgd,bskd->bkgts",
          dummy_query_prefill,
          dummy_key_prefill,
      )
      self.AqtEinsum_1 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd",
          dummy_attn_weights_prefill,
          dummy_value_prefill,
      )
      # Autoregressive AqtEinsum instances
      self.AqtEinsum_2 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor(),
          "btkgd,bskd->bkgts",
          dummy_query_ar,
          dummy_key_ar,
      )
      self.AqtEinsum_3 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd",
          dummy_attn_weights_ar,
          dummy_value_ar,
      )
    else:
      self.AqtEinsum_0 = jnp.einsum
      self.AqtEinsum_1 = jnp.einsum
      self.AqtEinsum_2 = jnp.einsum
      self.AqtEinsum_3 = jnp.einsum

  def _logical_to_mesh_axes(self, logical_name):
    logical_rules = get_logical_axis_rules()
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=logical_rules)

  def _context_parallel_size(self) -> int:
    """Number of ways the query sequence is actually sharded, per the active rules.

    `config.context_sharding` is derived once from `logical_axis_rules`, but the
    ambient rules can differ from those - the eval loop runs under
    `logical_axis_rules_for_eval` when `custom_mesh_and_rule_for_eval` is set.
    Resolving `activation_q_length` against the ambient rules keeps this count
    consistent with the PartitionSpec the kernel is sharded with; reading the
    static field instead lets the two disagree (e.g. a Splash kernel built for
    one shard handed to a 4-way `shard_map`).

    The Ulysses axis is excluded: under USP the query sequence is split over both
    the ring and Ulysses axes, but `cp_size` refers to the ring extent alone.
    """
    if self.mesh is None:
      return 1
    q_seq_axes = self._logical_to_mesh_axes((Q_LENGTH,))[0]
    if q_seq_axes is None:
      return 1
    if isinstance(q_seq_axes, str):
      q_seq_axes = (q_seq_axes,)
    return math.prod(
        self.mesh.shape[axis]
        for axis in q_seq_axes
        if axis is not None and axis != self.config.ulysses_context_sharding and axis in self.mesh.shape
    )

  def _load_balanced_context_parallel(self) -> bool:
    """Whether the load-balanced CP path is valid for this call.

    `context_parallel_load_balance` on its own is not enough. The load-balanced
    path assumes the batch arrives in DUAL_CHUNK_SWAP order: the mask bakes that
    permutation into a static `q_sequence`, and `wrap_flash_attention` restores
    K/V to contiguous order. The input pipeline applies that order once, for one
    specific cp_size, so the kernel may only take this path when it shards the
    query exactly that many ways.

    Equality, not `> 1`: a cp=2 train mesh with a cp=4 eval mesh is as wrong as
    1 vs 4. A mismatch does not raise - it reorders K/V that were never permuted
    and hands rows a `q_sequence` that lets them attend to the future - so fall
    back to the plain causal path, which is correct at any shard count.
    """
    cp_size = self._context_parallel_size()
    return cp_size > 1 and sequence_reorder.reordered_cp_size(self.config, self.mesh) == cp_size

  def check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, f"k (dim {key.ndim}), v (dim {value.ndim}) must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        f"{query.shape[:-3]=}, {key.shape[:-3]=}, {value.shape[:-3]=} batch" " dims must match."
    )
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  def _maybe_shard_with_pspec(self, inputs, pspec: jax.sharding.PartitionSpec | None):
    if inputs is None or self.mesh is None or pspec is None:
      return inputs
    return maybe_shard_with_pspec(
        inputs,
        pspec,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def generate_attention_mask(
      self,
      query,
      key,
      decoder_segment_ids: Array | None,
      model_mode: str,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      compressed_mask: Optional[Array] = None,
      segment_positions: Array | None = None,
      pad_kv_total: int = 0,
      decoder_segment_ids_kv: Optional[Array] = None,
      attention_type: AttentionType | None = None,
      kv_cache_part: str | None = None,
  ) -> Array | None:
    """Generates a combined attention mask for Transformer models.

    This function constructs an attention mask by potentially combining
    several types of masks based on the input parameters and model
    configuration. The generated mask dictates which query-key pairs are
    allowed to attend to each other.

    The masking logic can enforce:
    1.  **Sequence Separation:** Using `decoder_segment_ids`, attention is
      confined within distinct sequences in a batch. This is crucial when
      multiple unrelated sequences are packed together.
    2.  **Causality:** Preventing attention to future positions. This is
      standard for autoregressive decoding. For chunked prefill, as
      described in the SARATHI paper [2], causality is adjusted based
      on `previous_chunk` information.
    3.  **Specialized Attention Patterns:** Depending on `attention_type`,
      it can apply:
      * Local Sliding Window Attention: Restricts attention to a
          fixed-size window around each query position.
      * Compressed Attention (DeepSeek-V4): Combines an uncompressed sliding
          window attention mask over the recent prefix of length
          s_len = kv_seq_len - c_len with a pre-computed compressed mask over
          the compressed KV memory blocks of length c_len.
      * Chunk Attention: Divides sequences into chunks and applies
          masking at the chunk level.
    4.  **Bidirectional Attention for Sub-sequences:** If `bidirectional_mask`
      is provided (e.g., for image tokens in a multimodal model),
      those parts of the sequence can attend bidirectionally, and this
      mask is OR-ed with other generated masks.

    The overall approach and specific masking techniques are influenced by
    efficient attention mechanisms like those found in the Pallas MHA
    Flash Attention reference [1].

    Args:
      query: The query tensor, typically of shape `[batch_size,
        q_sequence_length, num_heads, head_dim]`. Used primarily for deriving
        sequence length.
      key: The key tensor, typically of shape `[batch_size, kv_sequence_length,
        num_heads, head_dim]`. Used primarily for deriving sequence length.
      decoder_segment_ids: Optional `Array` of shape `[batch_size,
        q_sequence_length]`. Identifies distinct sequences within the batch.
        Attention is restricted to elements within the same segment ID. In
        autoregressive mode, specific values (e.g.,
        `common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR`) can mark the
        currently active sequence for decoding.
      model_mode: A string (e.g., `common_types.MODEL_MODE_AUTOREGRESSIVE`,
        `MODEL_MODE_PREFILL`) indicating the operational mode. This
        significantly influences mask generation, particularly how causality and
        segment separation are handled.
      previous_chunk: Optional. Information about previously processed key/value
        chunks, often a tensor representing the previous keys/values. Used to
        correctly offset causal masks in chunked attention or streaming
        scenarios. Its shape might be `[batch_size, prev_kv_sequence_length,
        ...]`.
      bidirectional_mask: Optional `Array` of shape `[batch_size,
        kv_sequence_length]`. If provided, this boolean mask indicates tokens
        (e.g., image tokens) that are allowed to attend bidirectionally. The
        resulting block-wise bidirectional mask is combined with other masks
        using a logical OR.
      compressed_mask: Optional `Array`. A pre-computed attention mask for
        compressed kv blocks (e.g., DeepSeek-V4 compressed attention). If provided, it is
        concatenated with the dynamically generated uncompressed mask.
      segment_positions: Optional `Array` of shape `[batch_size,
        q_sequence_length]`. Identifies original positions for load-balanced
        context-parallel inputs.
      pad_kv_total: Number of padding tokens prepended to compressed KV blocks to satisfy exact
        block-size alignment required by Splash kernels.
      decoder_segment_ids_kv: Optional `Array` of shape `[batch_size,
        kv_sequence_length]`. Identifies distinct sequences for keys/values.
      attention_type: Optional `AttentionType` overriding the layer's own for
        this call. Lets a caller ask for a subset of the usual mask, e.g. a
        kernel that applies causality and the sliding window itself and only
        needs sequence separation. Defaults to `self.attention_type`.
      kv_cache_part: In single-token decode, which KV cache `key` comes from:
        "prefill" or "ar". With `segment_positions`, the sliding window is then
        measured between true query and key positions. None keeps the
        slot-index approximation.

    Returns:
      An `Array` representing the attention mask, with shape
       `[batch_size, 1, 1, q_sequence_length, kv_sequence_length]`.
      It is broadcastable to the shape
       `[batch_size, num_kv_heads, group_size=n_q // n_kv, q_sequence_length,
       kv_sequence_length]`.
      Positions with `0.0` allow attention, while positions with
       `DEFAULT_MASK_VALUE` (a large negative number) prevent it.
      Returns `None` if no masking is determined to be necessary based on
       the inputs and configuration.

    References:
      [1] JAX Pallas MHA Flash Attention:
          https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
      [2] SARATHI: Efficient LLM Inference by Piggybacking Decodes with
          Chunked Prefills - ArXiv:2308.16369 (https://arxiv.org/abs/2308.16369)
    """
    attention_type = self.attention_type if attention_type is None else attention_type
    mask = None
    if model_mode == MODEL_MODE_AUTOREGRESSIVE and decoder_segment_ids is not None:
      mask = decoder_segment_ids[:, None, None, None, :] == DECODING_ACTIVE_SEQUENCE_INDICATOR
    elif decoder_segment_ids is not None:
      seg_kv = decoder_segment_ids_kv if decoder_segment_ids_kv is not None else decoder_segment_ids

      # With TSP/CP, all-gather seq dim prior to broadcast to avoid large all-to-all on broadcasted ids.
      key_sharding = self._logical_to_mesh_axes((SEGMENT_ID_BATCH, None))
      decoder_segment_ids = self._maybe_shard_with_pspec(decoder_segment_ids, key_sharding)
      seg_kv = self._maybe_shard_with_pspec(seg_kv, key_sharding)

      mask = (decoder_segment_ids[:, :, None] == seg_kv[:, None, :]) & (seg_kv[:, None, :] >= 0)
      mask = mask[:, None, None, :, :]

    _, q_seq_len, _, _ = query.shape
    _, kv_seq_len, _, _ = key.shape
    next_pos = 0
    if previous_chunk is not None:
      next_pos = previous_chunk.shape[1]
      if mask is not None:
        mask = mask[:, :, :, next_pos : next_pos + q_seq_len, :]
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE and q_seq_len == 1:
      # In autoregression, the query position is the last position in the KV sequence.
      next_pos = kv_seq_len - 1
    use_segment_positions = (
        segment_positions is not None
        and self._load_balanced_context_parallel()
        and previous_chunk is None
        and model_mode != MODEL_MODE_AUTOREGRESSIVE
    )
    position_row_ids = position_col_ids = None
    if use_segment_positions:
      position_row_ids = segment_positions[:, :, None]
      position_col_ids = segment_positions[:, None, :]

    causal_mask = None
    if model_mode != MODEL_MODE_AUTOREGRESSIVE and attention_type not in (
        AttentionType.FULL,
        AttentionType.COMPRESSED,
        AttentionType.BLOCK_DIFFUSION,
    ):
      if use_segment_positions:
        causal_mask = (position_col_ids <= position_row_ids)[:, None, None, :, :]
      else:
        mask_shape = (q_seq_len, kv_seq_len)
        # row_ids indicates the position of query
        # col_ids indicates the position of kv
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        # Attention mask for chunked prefill is generated in the same way
        # as mentioned in SARATHI - https://arxiv.org/abs/2308.16369
        causal_mask = (col_ids <= row_ids + next_pos)[None, None, None, :, :]

    output_mask = None
    if (mask is not None) and (causal_mask is not None):
      output_mask = jnp.logical_and(mask, causal_mask)
    elif mask is not None:
      output_mask = mask
    elif causal_mask is not None:
      output_mask = causal_mask

    if attention_type == AttentionType.LOCAL_SLIDING and output_mask is not None:
      if self.sliding_window_size is None:
        raise ValueError("Sliding_window_size must be set if Local Sliding attention type")

      sliding_mask = dense_masks.local_sliding_attention_mask(
          q_seq_len, kv_seq_len, next_pos, self.sliding_window_size, use_segment_positions, position_row_ids,
          position_col_ids, segment_positions, model_mode, decoder_segment_ids, kv_cache_part,
      )
      output_mask = sliding_mask * output_mask
    elif attention_type == AttentionType.COMPRESSED:
      # DeepSeek-V4 Compressed Attention decomposes the full KV sequence of length
      # `kv_seq_len` into two distinct contiguous partitions:
      #   1. Uncompressed Recent Prefix (length `s_len = kv_seq_len - c_len`):
      #      Standard full-resolution tokens subject to causal and local sliding
      #      window constraints (`self.sliding_window_size`).
      #   2. Compressed Memory Blocks (length `c_len`):
      #      Pre-compressed / indexed KV representations (e.g. CSA / HCA) governed
      #      by `compressed_mask`.
      #
      # The resulting attention mask is the concatenation of the uncompressed
      # sliding mask and the compressed mask along the KV sequence dimension:
      #   mask = [expanded_uncompressed_mask, compressed_mask]  (shape: [B, 1, 1, Q, S + C])
      c_len = compressed_mask.shape[-1] if compressed_mask is not None else 0
      s_len = kv_seq_len - c_len

      uncompressed_mask = dense_masks.compressed_sliding_window_mask(
          q_seq_len, s_len, next_pos, self.sliding_window_size, use_segment_positions, position_row_ids,
          position_col_ids, segment_positions, model_mode, decoder_segment_ids, kv_cache_part,
      )

      return dense_masks.combine_compressed_attention_masks(
          uncompressed_mask, compressed_mask, output_mask, s_len, pad_kv_total
      )

    elif attention_type == AttentionType.CHUNK and output_mask is not None:
      if use_segment_positions:
        same_chunk = (position_row_ids // self.chunk_attn_window_size) == (
            position_col_ids // self.chunk_attn_window_size
        )
        chunk_mask = (same_chunk & (position_row_ids >= position_col_ids))[:, None, None, :, :]
      else:
        chunk_mask = dense_masks._generate_chunk_attention_mask(
            mask_shape=(q_seq_len, kv_seq_len),
            chunk_size=self.chunk_attn_window_size,
            q_offset=next_pos,
        )
      output_mask = chunk_mask * output_mask

    # For standard token-by-token autoregressive decoding, keep the existing
    # causal mask path unchanged. BD3LM generation instead uses block-level
    # parallel sampling.
    elif attention_type == AttentionType.BLOCK_DIFFUSION and model_mode != MODEL_MODE_AUTOREGRESSIVE:
      if use_segment_positions:
        block_mask = ((position_row_ids // self.causal_block_size) >= (position_col_ids // self.causal_block_size))[
            :, None, None, :, :
        ]
      else:
        block_mask = dense_masks._generate_block_causal_attention_mask(
            mask_shape=(q_seq_len, kv_seq_len),
            causal_block_size=self.causal_block_size,
            q_offset=next_pos,
        )[None, None, None, :, :]
      output_mask = block_mask if output_mask is None else jnp.logical_and(output_mask, block_mask)

    if bidirectional_mask is not None:
      image_mask = dense_masks._make_bidirectional_block_mask(bidirectional_mask)
      output_mask = output_mask | image_mask[:, None, None, ...]

    return jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE) if output_mask is not None else None

  def calculate_moba_gate_logic(self, q_item, k_item, q_pos_item):
    """Computes the block-level MoBA gating intermediates for one batch item.

    Args:
      q_item: Query tensor shaped `[q_len, n_q_heads, head_dim]`.
      k_item: Key tensor shaped `[kv_len, n_kv_heads, head_dim]`.
      q_pos_item: Absolute query positions shaped `[q_len]`, used to derive the
        chunk index for each query. For example, during prefill after 128 tokens
        have been processed `q_pos_item` is `jnp.arange(128, 128 + q_len)`,
        while in autoregressive decode with a single query token it is
        `jnp.array([kv_len - 1])`.

    Returns:
      `need_attend`, a boolean mask of shape `[n_kv_heads, g, q_len, num_block]`
      indicating which key blocks each query should attend to. The additional
      values in the returned tuple are debug intermediates used for logging and
      diagnostics when inspecting the gating behaviour.
    """
    q_len, n_q_heads, head_dim = q_item.shape
    kv_len, n_kv_heads, _ = k_item.shape
    g = n_q_heads // n_kv_heads

    q_item_f32 = q_item.astype(jnp.float32).reshape(q_len, n_kv_heads, g, head_dim)  # grouped-query attention (GQA)

    moba_chunk_size = self.config.moba_chunk_size
    moba_topk = self.config.moba_topk

    num_block = math.ceil(kv_len / moba_chunk_size)

    block_ids = jnp.arange(kv_len, dtype=jnp.int32) // moba_chunk_size  # chunk index for each key position
    # Sum key vectors per chunk so we can later average within each block.
    key_gate_weight_sum = jax.ops.segment_sum(
        k_item.astype(jnp.float32), block_ids, num_segments=num_block
    )  # [num_block, n_kv_heads, head_dim]
    # Count how many tokens end up in each chunk so we can take the mean.
    block_counts = jax.ops.segment_sum(
        jnp.ones((kv_len,), dtype=jnp.float32),
        block_ids,
        num_segments=num_block,
    )  # [num_block]
    # Mean Pooling, Avoid division by zero for empty blocks.
    key_gate_weight = key_gate_weight_sum / jnp.maximum(
        block_counts[:, None, None], 1
    )  # [num_block, n_kv_heads, head_dim]

    # Take the dot product between each query and every key chunk to get a score.
    gate = jnp.einsum("skgd,Nkd->kgsN", q_item_f32, key_gate_weight)  # [n_kv_heads, g, q_len, num_block]
    gate_before_masking = gate

    q_block_idx = q_pos_item // moba_chunk_size  # chunk id for each query
    block_indices = jnp.arange(num_block)  # list every key chunk index

    q_block_idx_b = jnp.expand_dims(q_block_idx, axis=-1)  # [q_len, 1]
    block_indices_b = jnp.expand_dims(block_indices, axis=0)  # [1, num_block]

    # Block-causal masking: a query can't attend to future key blocks,
    # and must attend to its own key block.
    mask_future = q_block_idx_b > block_indices_b
    gate = jnp.where(mask_future, gate, -float("inf"))
    mask_diag = q_block_idx_b == block_indices_b
    gate = jnp.where(mask_diag, float("inf"), gate)
    gate_after_masking = gate

    k_for_topk = min(moba_topk, num_block)
    gate_top_k_val, gate_top_k_idx = jax.lax.top_k(gate, k=k_for_topk)  # [n_kv_heads, g, q_len, k_for_topk]
    gate_top_k_val_min = jnp.min(gate_top_k_val, axis=-1, keepdims=True)  # [n_kv_heads, g, q_len, 1]
    need_attend_threshold_mask = gate >= gate_top_k_val_min  # [n_kv_heads, g, q_len, num_block]

    # Tie-breaking: if multiple blocks have the same gate value as the k-th
    # block, we only select the ones that appear in the top-k indices.
    gate_idx_mask = jnp.sum(
        jax.nn.one_hot(gate_top_k_idx, num_block, dtype=jnp.bool_), axis=-2
    )  # [n_kv_heads, g, q_len, num_block]
    need_attend = jnp.logical_and(need_attend_threshold_mask, gate_idx_mask)  # [n_kv_heads, g, q_len, num_block]

    return (
        key_gate_weight,
        gate_before_masking,
        gate_after_masking,
        gate_top_k_val,
        gate_top_k_idx,
        gate_top_k_val_min,
        need_attend_threshold_mask,
        gate_idx_mask,
        need_attend,  # [n_kv_heads, g, q_len, num_block]
    )

  def generate_moba_mask_single_item(self, q_item, k_item, q_positions):
    """Generates the token-level MoBA additive mask for a single batch item."""
    q_len, _, _ = q_item.shape
    kv_len, _, _ = k_item.shape
    moba_chunk_size = self.config.moba_chunk_size

    # Run the gating logic to find which key blocks this query cares about.
    *_, need_attend = self.calculate_moba_gate_logic(q_item, k_item, q_positions)

    # Expand the block-level `need_attend` mask to a token-level mask.
    k_block_indices = jnp.arange(kv_len, dtype=jnp.int32) // moba_chunk_size
    token_level_need_attend = need_attend[..., k_block_indices]

    # Convert the boolean mask to float mask values.
    gate = jnp.where(token_level_need_attend, 0.0, -float("inf"))

    # Apply a final per-token causal mask to ensure causality within chunks.
    k_indices = jax.lax.broadcasted_iota(jnp.int32, (q_len, kv_len), 1)
    q_indices = q_positions[:, None]
    causal_mask = q_indices >= k_indices
    gate = jnp.where(causal_mask, gate, -float("inf"))

    # Return the additive mask for this batch item.
    return gate

  def _generate_moba_mask(self, query: Array, key: Array, q_positions: Array) -> Array:
    """Builds the token-level MoBA additive mask for the whole batch.

    Args:
      query: Query tensor shaped `[batch, q_len, n_q_heads, head_dim]`.
      key: Key tensor shaped `[batch, kv_len, n_kv_heads, head_dim]`.
      q_positions: Absolute query positions shaped `[q_len]`, shared across the
        batch, identifying the starting offset of each query token. For example,
        in prefill after 128 tokens we pass `jnp.arange(128, 128 + q_len)`,
        while in autoregressive decode with a single new token the vector is
        `[kv_len - 1]` for each batch element.

    Returns:
      Additive attention mask with shape
      `[batch, n_kv_heads, n_q_heads // n_kv_heads, q_len, kv_len]` containing
      `0.` for permitted positions and `-inf` for masked ones.
    """
    # vmap over the batch dimension of query and key. q_positions is constant across the batch.
    moba_mask = jax.vmap(self.generate_moba_mask_single_item, in_axes=(0, 0, None))(query, key, q_positions)
    return moba_mask

  def _validate_tpu_tokamax_ring_runtime(
      self,
      *,
      model_mode: str,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      use_ragged_attention: bool = False,
      record_max_logits: bool = False,
  ) -> None:
    """Validates runtime constraints for the TPU Tokamax ring path."""
    context_parallel_runtime_validation.validate_tpu_tokamax_ring_runtime(
        self.config,
        self.attention_type,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        bidirectional_mask=bidirectional_mask,
        sinks=sinks,
        indexer_mask=indexer_mask,
        use_ragged_attention=use_ragged_attention,
        record_max_logits=record_max_logits,
    )

  def _validate_tpu_ulysses_runtime(
      self,
      *,
      model_mode: str,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      use_ragged_attention: bool = False,
      record_max_logits: bool = False,
  ) -> None:
    """Validates runtime constraints for the TPU Ulysses path."""
    context_parallel_runtime_validation.validate_tpu_ulysses_runtime(
        self.attention_type,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        bidirectional_mask=bidirectional_mask,
        sinks=sinks,
        indexer_mask=indexer_mask,
        use_ragged_attention=use_ragged_attention,
        record_max_logits=record_max_logits,
    )

  def _validate_tpu_usp_runtime(
      self,
      *,
      model_mode: str,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      use_ragged_attention: bool = False,
      record_max_logits: bool = False,
  ) -> None:
    """Validates runtime constraints for the TPU USP path."""
    context_parallel_runtime_validation.validate_tpu_usp_runtime(
        self.attention_type,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        bidirectional_mask=bidirectional_mask,
        sinks=sinks,
        indexer_mask=indexer_mask,
        use_ragged_attention=use_ragged_attention,
        record_max_logits=record_max_logits,
    )

  def apply_attention(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      segment_positions: Array | None,
      lengths: Array | None,
      model_mode: str,
      use_ragged_attention: bool = False,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      compressed_mask: Optional[Array] = None,
      record_max_logits: bool = False,
      decoder_segment_ids_kv: Optional[Array] = None,
      pad_kv_total: int = 0,
      compress_ratio: int = 0,
      kv_cache_part: str | None = None,
      *,
      qk_product_einsum: Callable[..., Array],
      wv_product_einsum: Callable[..., Array],
  ):
    """Apply attention"""
    if self.attention_kernel == "cudnn_flash_jax":
      validate_gpu_flash_attention(sinks, record_max_logits)
      if isinstance(key, KVTensor):
        key = key.dequant()
      if isinstance(value, KVTensor):
        value = value.dequant()
      query, key, value = self._align_qkv_for_cudnn_flash(query, key, value)

    self.check_attention_inputs(query, key, value)
    length = query.shape[-3]
    target_hardware = self.mesh.devices[(0,) * self.mesh.devices.ndim].platform
    cp_size = self._context_parallel_size()
    context_parallel_runtime_validation.validate_context_parallel_dispatch(
        self.config, self.attention_type, self.attention_kernel, target_hardware, cp_size
    )

    if use_ragged_attention and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      if lengths is None:
        lengths = jnp.sum(decoder_segment_ids, axis=-1)

      if target_hardware == "tpu":
        impl = self.tpu_ragged_attention
      elif target_hardware == "gpu":
        impl = self.gpu_ragged_attention
      else:
        raise NotImplementedError(target_hardware)

      local_out, local_max, local_sum = impl(query, key, value, lengths, self.ragged_block_size)
      if record_max_logits:
        self.max_logits = nnx.Intermediate(local_max)
      return local_out, local_max, local_sum

    # 'vllm_rpa' uses the same dot-attention wrapper but routes to the vLLM
    # ragged paged attention kernel in `Attention.__call__`.
    elif (
        self.attention_kernel == "dot_product"
        or (self.attention_kernel == "autoselected" and model_mode == MODEL_MODE_AUTOREGRESSIVE)
        or (self.attention_kernel == "autoselected" and length < 128)
        or (self.attention_kernel == "autoselected" and target_hardware == "cpu")
        or (self.attention_kernel == "paged")
        or (self.attention_kernel in ("vllm_rpa", "vllm_batched_rpa"))
    ):
      return self.apply_attention_dot(
          query,
          key,
          value,
          decoder_segment_ids,
          model_mode,
          previous_chunk,
          segment_positions=segment_positions,
          bidirectional_mask=bidirectional_mask,
          sinks=sinks,
          indexer_mask=indexer_mask,
          compressed_mask=compressed_mask,
          record_max_logits=record_max_logits,
          decoder_segment_ids_kv=decoder_segment_ids_kv,
          kv_cache_part=kv_cache_part,
          qk_product_einsum=qk_product_einsum,
          wv_product_einsum=wv_product_einsum,
      )
    elif self.attention_kernel in ("flash", "autoselected"):
      if target_hardware == "tpu":
        if isinstance(key, KVTensor):
          key = key.dequant()
        if isinstance(value, KVTensor):
          value = value.dequant()

        if model_mode == MODEL_MODE_AUTOREGRESSIVE:
          raise ValueError(
              """Decode not supported with flash attention.
                              Use `dot_product` instead."""
          )

        out, max_logits = self.tpu_flash_attention(
            query,
            key,
            value,
            decoder_segment_ids,
            self.attn_logits_soft_cap,
            sinks,
            indexer_mask=indexer_mask,
            model_mode=model_mode,
            previous_chunk=previous_chunk,
            bidirectional_mask=bidirectional_mask,
            use_ragged_attention=use_ragged_attention,
            record_max_logits=record_max_logits,
            decoder_segment_ids_kv=decoder_segment_ids_kv,
            pad_kv_total=pad_kv_total,
            compress_ratio=compress_ratio,
        )
        if max_logits is not None:
          self.max_logits = nnx.Intermediate(max_logits)
        return out, None, None

      else:
        if self.attention_type == AttentionType.BLOCK_DIFFUSION:
          raise ValueError(
              "Block-diffusion flash attention is supported only by TPU Splash; "
              "use attention='dot_product' on other hardware."
          )
        if model_mode == MODEL_MODE_AUTOREGRESSIVE:
          # fallback to dot_product as pallas gpu flash attention doesn't support decode stage
          return self.apply_attention_dot(
              query,
              key,
              value,
              decoder_segment_ids,
              model_mode,
              bidirectional_mask=bidirectional_mask,
              record_max_logits=record_max_logits,
              qk_product_einsum=qk_product_einsum,
              wv_product_einsum=wv_product_einsum,
          )
        else:
          validate_gpu_flash_attention(sinks, record_max_logits, attention_kernel=self.attention_kernel)
          out = gpu_flash.gpu_flash_attention(query, key, value, decoder_segment_ids)
          return out, None, None
    elif self.attention_kernel == "cudnn_flash_te":
      validate_gpu_flash_attention(sinks, record_max_logits, attention_kernel=self.attention_kernel)
      if isinstance(key, KVTensor):
        key = key.dequant()
      if isinstance(value, KVTensor):
        value = value.dequant()
      if model_mode == MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            """Decode not supported with flash attention.
                           Use `dot_product` instead."""
        )
      return (
          self.cudnn_flash_attention(
              query,
              key,
              value,
              decoder_segment_ids,
              segment_positions,
              model_mode,
              sinks=sinks,
          ),
          None,
          None,
      )
    elif self.attention_kernel == "cudnn_flash_jax":
      return (
          *self.cudnn_jax_flash_attention(query, key, value, decoder_segment_ids, model_mode),
          None,
      )
    else:
      raise ValueError(f"Unexpected attention kernel {self.attention_kernel=}.")

  def gpu_ragged_attention(
      self,
      q: Array,
      k: Array | KVTensor,
      v: Array | KVTensor,
      lengths: Array,
      block_size: int,
  ):
    """gpu ragged attention"""
    return gpu_ragged.gpu_ragged_attention(
        q, k, v, lengths, block_size, mesh=self.mesh, ragged_lengths_names=self.ragged_lengths_names,
        cache_logical_axis_names=self.cache_logical_axis_names, logical_to_mesh_axes=self._logical_to_mesh_axes,
    )

  def tpu_ragged_attention(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      lengths: Array,
      block_size: int,
  ) -> tuple[Array, Array, Array]:
    """Ragged Attention."""
    return tpu_ragged.tpu_ragged_attention(
        query, key, value, lengths, block_size, mesh=self.mesh, ragged_lengths_names=self.ragged_lengths_names,
        cache_logical_axis_names=self.cache_logical_axis_names, logical_to_mesh_axes=self._logical_to_mesh_axes,
    )

  def tpu_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      attn_logits_soft_cap: float | None = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      use_ragged_attention: bool = False,
      record_max_logits: bool = False,
      decoder_segment_ids_kv: Array | None = None,
      pad_kv_total: int = 0,
      compress_ratio: int = 0,
  ) -> tuple[Array, Array]:
    """TPU Flash Attention."""

    use_tokamax_ring = tokamax_ring_attention.is_context_parallel_ring_requested(self.config)
    use_ulysses = ulysses_attention.is_context_parallel_ulysses_requested(self.config)
    use_usp = usp_attention.is_context_parallel_usp_requested(self.config)
    cp_size = self._context_parallel_size()
    load_balanced_context_parallel = self._load_balanced_context_parallel()
    if use_tokamax_ring:
      self._validate_tpu_tokamax_ring_runtime(
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          bidirectional_mask=bidirectional_mask,
          sinks=sinks,
          indexer_mask=indexer_mask,
          use_ragged_attention=use_ragged_attention,
          record_max_logits=record_max_logits,
      )
    elif use_ulysses:
      self._validate_tpu_ulysses_runtime(
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          bidirectional_mask=bidirectional_mask,
          sinks=sinks,
          indexer_mask=indexer_mask,
          use_ragged_attention=use_ragged_attention,
          record_max_logits=record_max_logits,
      )
    elif use_usp:
      self._validate_tpu_usp_runtime(
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          bidirectional_mask=bidirectional_mask,
          sinks=sinks,
          indexer_mask=indexer_mask,
          use_ragged_attention=use_ragged_attention,
          record_max_logits=record_max_logits,
      )

    query, key, value, decoder_segment_ids, decoder_segment_ids_kv_in, orig_q_len, orig_kv_len = (
        tpu_flash.prepare_flash_inputs(self.config, query, key, value, decoder_segment_ids, decoder_segment_ids_kv)
    )

    # The splash attributes are set together in __init__, so reading them all where the original read
    # the selected block sizes raises (if they are absent) at the same point.
    is_eval = eval_tiles.is_eval(self.config)
    splash_tuning = SplashTuning(
        block_q=self.block_q, block_kv=self.block_kv, block_kv_compute=self.block_kv_compute,
        eval_block_q=self.eval_block_q, eval_block_kv=self.eval_block_kv,
        eval_block_kv_compute=self.eval_block_kv_compute, block_q_dkv=self.block_q_dkv,
        block_kv_dkv=self.block_kv_dkv, block_kv_dkv_compute=self.block_kv_dkv_compute, block_q_dq=self.block_q_dq,
        block_kv_dq=self.block_kv_dq, use_fused_bwd_kernel=self.use_fused_bwd_kernel, q_layout=self.q_layout,
        k_layout=self.k_layout, v_layout=self.v_layout, eval_q_layout=self.eval_q_layout,
        eval_k_layout=self.eval_k_layout, eval_v_layout=self.eval_v_layout,
        use_splash_scheduler=self.use_splash_scheduler, fuse_reciprocal=self.fuse_reciprocal,
        use_base2_exp=self.use_base2_exp, qk_diag_skip=self.qk_diag_skip, sv_diag_skip=self.sv_diag_skip,
    )

    return tpu_flash.run_flash_attention(
        query, key, value, decoder_segment_ids, decoder_segment_ids_kv_in, orig_q_len, orig_kv_len,
        attn_logits_soft_cap, sinks, indexer_mask, record_max_logits, pad_kv_total, compress_ratio,
        config=self.config, mesh=self.mesh, attention_type=self.attention_type, num_kv_heads=self.num_kv_heads,
        splash_tuning=splash_tuning, blocks=select_block_sizes(splash_tuning, is_eval),
        sliding_window_size=self.sliding_window_size, chunk_attn_window_size=self.chunk_attn_window_size,
        causal_block_size=self.causal_block_size, flash_axis_names_q=self.flash_axis_names_q,
        flash_axis_names_kv=self.flash_axis_names_kv, flash_axis_names_splash_kernel=self.flash_axis_names_splash_kernel,
        use_tokamax_ring=use_tokamax_ring, use_ulysses=use_ulysses, use_usp=use_usp, cp_size=cp_size,
        load_balanced_context_parallel=load_balanced_context_parallel,
        logical_to_mesh_axes=self._logical_to_mesh_axes, maybe_shard_with_pspec=self._maybe_shard_with_pspec,
    )

  def cudnn_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      segment_positions: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
      sinks: Array | None = None,
  ) -> Array:
    """CUDNN Flash Attention with Transformer Engine.
    1. Stable API, supports MHA, GQA, SWA, Packing and Context Parallelism
    2. Context Parallelism currently only supports causal masking
    3. Only Ring attention has packing support with striped load balancing
      (context_parallel_strategy="ring" and context_parallel_load_balance=true)
    4. Breaks with TE 2.12 and 2.13 (known bug); works with TE stable release <=2.11 or >=2.14.
    """
    return cudnn_te.cudnn_flash_attention(
        query, key, value, decoder_segment_ids, segment_positions, model_mode, sinks, config=self.config,
        mesh=self.mesh, attention_type=self.attention_type, sliding_window_size=self.sliding_window_size,
        num_query_heads=self.num_query_heads, num_kv_heads=self.num_kv_heads, dropout_rate=self.dropout_rate,
        dtype=self.dtype, float32_logits=self.float32_logits, max_target_length=self.max_target_length,
        rngs=self.rngs, generate_attention_mask=self.generate_attention_mask,
    )

  def _align_qkv_for_cudnn_flash(
      self,
      query: Array,
      key: Array,
      value: Array,
  ) -> tuple[Array, Array, Array]:
    """Broadcasts key/value batch dim to match query batch dim for cuDNN flash attention."""
    return cudnn_jax.align_qkv_for_cudnn_flash(query, key, value)

  def cudnn_jax_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> tuple[Array, Array]:
    """CUDNN Flash Attention with JAX SDPA API."""
    return cudnn_jax.cudnn_jax_flash_attention(
        query, key, value, decoder_segment_ids, model_mode, dropout_rate=self.dropout_rate
    )

  def compute_local_attention(
      self,
      attn_weights: Array,
      value: Array | KVTensor,
      q_seq_len: int,
      model_mode: str,
      wv_product_einsum: Callable[..., Array],
      sinks: Array | None = None,
  ) -> tuple[Array, Array, Array]:
    """Computes the attention of a local subset of the kv cache."""
    return dot_product.compute_local_attention(
        attn_weights, value, q_seq_len, model_mode, wv_product_einsum, sinks, reshape_q=self.reshape_q,
        wv_product=self.wv_product, is_partition_in_decode=self.is_partition_in_decode,
    )

  def is_partition_in_decode(self, seq_len):
    return self.config.ici_context_autoregressive_parallelism > 0 and seq_len == 1

  def apply_attention_dot(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      segment_positions: Array | None = None,
      bidirectional_mask: Any = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      compressed_mask: Optional[Array] = None,
      record_max_logits: bool = False,
      decoder_segment_ids_kv: Optional[Array] = None,
      kv_cache_part: str | None = None,
      *,
      qk_product_einsum: Callable[..., Array],
      wv_product_einsum: Callable[..., Array],
  ):
    """Apply Attention."""
    query, key, value, attn_weights = dot_product.attention_logits(
        query, key, value, model_mode, qk_product_einsum, compute_axis_order=self.compute_axis_order,
        float32_qk_product=self.float32_qk_product, float32_logits=self.float32_logits,
        attn_logits_soft_cap=self.attn_logits_soft_cap, is_absorbed_mqa=self.is_absorbed_mqa, kv_quant=self.kv_quant,
        qk_product=self.qk_product, is_partition_in_decode=self.is_partition_in_decode,
    )
    q_seq_len = query.shape[1]

    attn_mask = self.generate_attention_mask(
        query, key, decoder_segment_ids, model_mode, previous_chunk, bidirectional_mask,
        compressed_mask=compressed_mask, segment_positions=segment_positions,
        decoder_segment_ids_kv=decoder_segment_ids_kv, kv_cache_part=kv_cache_part,
    )
    if self.config.moba:
      kv_seq_len = key.shape[1]
      next_pos = 0
      if previous_chunk is not None:
        next_pos = previous_chunk.shape[1]
      elif model_mode == MODEL_MODE_AUTOREGRESSIVE and q_seq_len == 1:
        next_pos = kv_seq_len - 1
      q_positions = jnp.arange(next_pos, next_pos + q_seq_len)
      moba_mask = self._generate_moba_mask(query, key, q_positions)
      attn_weights += moba_mask

    attn_weights = dot_product.apply_attention_masks(
        attn_weights, attn_mask, indexer_mask, q_seq_len, model_mode, is_partition_in_decode=self.is_partition_in_decode,
    )

    if record_max_logits:
      self.max_logits = nnx.Intermediate(dot_product.max_logits_per_head(attn_weights))

    return self.compute_local_attention(attn_weights, value, q_seq_len, model_mode, wv_product_einsum, sinks)

  def qk_product(
      self, query: Array, key: Array | KVTensor, q_seq_len: int, model_mode: str, einsum: Callable[..., Array],
  ) -> Array:
    """Query-Key product."""
    return dot_product.qk_product(
        query, key, q_seq_len, model_mode, einsum, config=self.config, num_kv_heads=self.num_kv_heads,
        compute_axis_order=self.compute_axis_order, reshape_q=self.reshape_q,
    )

  def wv_product(
      self, attn_weights: Array, value: Array | KVTensor, model_mode: str, einsum: Callable[..., Array],
  ) -> Array:
    """weighted value product."""
    return dot_product.wv_product(
        attn_weights, value, model_mode, einsum, config=self.config, kv_quant=self.kv_quant,
        compute_axis_order=self.compute_axis_order, reverse_transepose=self.reverse_transepose,
    )

  def reverse_transepose(self, transposed_array, transpose_axis_order):
    return dot_product.reverse_transepose(transposed_array, transpose_axis_order)

  def normalize_cudnn_attention(self, local_outs, local_stats):
    """Normalize across two cuDNN attentions."""
    return normalization.normalize_cudnn_attention(local_outs, local_stats)

  def normalize_attention(self, local_outs, local_maxes, local_sums):
    """Normalize across multiple localized attentions."""
    return normalization.normalize_attention(local_outs, local_maxes, local_sums)

  def __call__(
      self,
      query,
      key,
      value,
      decoder_segment_ids,
      inputs_positions,
      model_mode,
      cached_values=None,
      previous_chunk=None,
      bidirectional_mask=None,
      sinks=None,
      indexer_mask: Optional[Array] = None,
      compressed_mask: Optional[Array] = None,
      compressed_kv: Optional[Array] = None,
      slot: Optional[int] = None,
      record_max_logits: bool = False,
      decoder_segment_ids_kv: Optional[Array] = None,
      pad_kv_total: int = 0,
      compress_ratio: int = 0,
  ):
    if cached_values is None:
      prefill_kv_cache, ar_kv_cache = None, None
    else:
      prefill_kv_cache, ar_kv_cache = cached_values[0], cached_values[1]
    if model_mode != MODEL_MODE_TRAIN:
      assert prefill_kv_cache
      key, value, decoder_segment_ids = prefill_kv_cache
    pass_comp_to_prefill = (compressed_kv is not None) and (ar_kv_cache is None)
    k_prefill = key
    v_prefill = value
    if pass_comp_to_prefill:
      k_prefill = jnp.concatenate([k_prefill, compressed_kv], axis=1)
      v_prefill = jnp.concatenate([v_prefill, compressed_kv], axis=1)

    indexer_mask_prefill = None
    indexer_mask_ar = None
    if indexer_mask is not None:
      if pass_comp_to_prefill:
        indexer_mask_prefill = indexer_mask
      else:
        prefill_len = key.shape[1]
        indexer_mask_prefill = indexer_mask[:, :, :prefill_len]
        if ar_kv_cache is not None:
          indexer_mask_ar = indexer_mask[:, :, prefill_len:]

    prefill_unnormalized_output, prefill_exponentials_max, prefill_exponentials_sum = self.apply_attention(
        query=query,
        key=k_prefill,
        value=v_prefill,
        decoder_segment_ids=decoder_segment_ids,
        segment_positions=inputs_positions,
        lengths=None,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
        bidirectional_mask=bidirectional_mask,
        sinks=sinks,
        indexer_mask=indexer_mask_prefill,
        compressed_mask=compressed_mask if pass_comp_to_prefill else None,
        record_max_logits=record_max_logits,
        qk_product_einsum=self.AqtEinsum_0,
        wv_product_einsum=self.AqtEinsum_1,
        decoder_segment_ids_kv=decoder_segment_ids_kv,
        pad_kv_total=pad_kv_total,
        compress_ratio=compress_ratio,
        kv_cache_part="prefill" if model_mode == MODEL_MODE_AUTOREGRESSIVE else None,
    )

    if ar_kv_cache is None:
      if prefill_exponentials_sum is not None:
        return prefill_unnormalized_output / prefill_exponentials_sum
      return prefill_unnormalized_output

    key_ar, value_ar, decoder_segment_ids_ar, lengths = ar_kv_cache
    if compressed_kv is not None:
      key_ar = jnp.concatenate([key_ar, compressed_kv], axis=1)
      value_ar = jnp.concatenate([value_ar, compressed_kv], axis=1)

    ar_unnormalized_output, ar_exponentials_max, ar_exponentials_sum = self.apply_attention(
        query=query,
        key=key_ar,
        value=value_ar,
        decoder_segment_ids=decoder_segment_ids_ar,
        segment_positions=inputs_positions,
        lengths=lengths,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        bidirectional_mask=bidirectional_mask,
        indexer_mask=indexer_mask_ar,
        compressed_mask=compressed_mask,
        kv_cache_part="ar",
        qk_product_einsum=self.AqtEinsum_2,
        wv_product_einsum=self.AqtEinsum_3,
    )

    if ar_unnormalized_output is not None:
      unnormalized_outputs = [
          prefill_unnormalized_output,
          ar_unnormalized_output,
      ]
      exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
      exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
      if prefill_exponentials_max is not None and prefill_exponentials_sum is None:
        prefill_stat = prefill_exponentials_max
        ar_stat = ar_exponentials_max
        stats = [prefill_stat, ar_stat]
        return self.normalize_cudnn_attention(unnormalized_outputs, stats)
      else:
        return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)
    else:
      return prefill_unnormalized_output / prefill_exponentials_sum
