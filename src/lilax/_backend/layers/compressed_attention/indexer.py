"""Compressed Sparse Attention (CSA) top-k block indexer (DeepSeek-V4)."""


import functools
import math
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import nnx

from lilax._backend.common.common_types import (
    Array,
    MODEL_MODE_TRAIN,
    MODEL_MODE_AUTOREGRESSIVE,
    DEFAULT_MASK_VALUE,
)

from lilax._backend.layers.compressed_attention.compressor_cache import (
    prime_prefill_cache_state,
    update_ar_cache_and_get_validity_mask,
)
from lilax._backend.layers.compressed_attention.csa_pooling import (
    compute_csa_prefill_chunk_pooling,
    csa_overlap_pooling,
)
from lilax._backend.layers.initializers import nd_dense_init, NdInitializer
from lilax._backend.layers.linears import DenseGeneral
from lilax._backend.layers.normalizations import RMSNorm
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant
from lilax._backend.kernels.attention import csa_streamindex


class DeepseekV4Indexer(nnx.Module):
  """Indexer module for Compressed Sparse Attention (DeepSeek-V4 paper §2.3.1).

  Evaluates query representations against compressed KV blocks to identify the top-k
  most relevant blocks to attend to.

  Shape Transformations:
    1. Pool KV: `[batch, seq, emb_dim]` -> `[batch, n_windows, index_head_dim]`
    2. Broadcast KV: -> `[batch, index_n_heads, n_windows, index_head_dim]`
    3. Project Q: `[batch, seq, emb_dim]` -> `[batch, index_n_heads, seq, index_head_dim]`
    4. Einsum Q*KV: -> `[batch, index_n_heads, seq, n_windows]`
    5. Project Weights: `[batch, seq, emb_dim]` -> `[batch, seq, index_n_heads]`
    6. Combine Scores: -> `[batch, seq, n_windows]`
  """

  def __init__(
      self,
      config: Any,
      compress_ratio: int,
      rotary_embedding: Any,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[Mesh] = None,
  ):
    """Initializes the Indexer for CSA.

    Args:
      config: Model configuration containing indexer parameters.
      compress_ratio: The compression ratio (e.g., config.compress_rate_csa).
      rotary_embedding: Rotary embedding instance for injecting position info into index representations.
      kernel_init: Weight initializer for the indexer projections.
      quant: Optional quantization scheme.
      rngs: Optional random state initialization.
      mesh: Device mesh. Required to shard the StreamIndex Pallas kernel; without
        it the kernel runs unsharded and GSPMD all-gathers its operands.
    """
    self.config = config
    self.compress_rate = compress_ratio
    self.mesh = mesh
    self.index_n_heads = config.indexer_n_heads
    self.index_head_dim = config.indexer_head_dim
    self.index_topk = config.indexer_topk
    self.softmax_scale = self.index_head_dim**-0.5
    self.weights_scaling = self.index_n_heads**-0.5
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.rngs = rngs

    self.q_proj = DenseGeneral(
        in_features_shape=config.q_lora_rank,
        out_features_shape=self.index_n_heads * self.index_head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("q_lora", "indexer_q"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.kv_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.index_head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )
    self.gate_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.index_head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.position_bias = nnx.Param(jnp.zeros((self.compress_rate, 2 * self.index_head_dim), dtype=self.weight_dtype))

    self.kv_norm = RMSNorm(
        num_features=self.index_head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.weights_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.index_n_heads,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "indexer_weights"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.rotary_emb = rotary_embedding

  def _streamindex_scores_sharded(self, q: Array, compressed: Array, weights: Array) -> Array:
    """Runs the StreamIndex Pallas kernel under a batch-only `shard_map`.

    A bare `pallas_call` lowers to an opaque `tpu_custom_call` that GSPMD cannot
    partition, so it conservatively all-gathers every operand to full global
    shape on each device — reconstructing the whole `[B, H, S, D]` query and
    defeating the memory saving the kernel exists to provide. `shard_map` makes
    the partitioning explicit and removes those collectives.

    Only the batch dimension is sharded. Context/sequence parallelism is not
    supported: `compressed` would have to be replicated across the context axis
    (causality means late query rows need every earlier compressed block), which
    is a separate change.

    Args:
      q: `[B, H, S, D]`.
      compressed: `[B, W, D]`.
      weights: `[B, S, H]`.

    Returns:
      Index scores, `[B, S, W]`.
    """
    batch_size = q.shape[0]

    batch_axes = tuple(
        axis for axis in ("data", "fsdp", "fsdp_transpose", "expert") if self.mesh is not None and axis in self.mesh.shape
    )
    num_batch_shards = math.prod(self.mesh.shape[a] for a in batch_axes) if batch_axes else 1

    if self.mesh is None or num_batch_shards == 1:
      return csa_streamindex.csa_streamindex_score(q, compressed, weights, self.softmax_scale)

    if batch_size % num_batch_shards != 0:
      raise ValueError(
          f"CSA StreamIndex kernel requires batch_size ({batch_size}) to be divisible by the "
          f"number of batch shards ({num_batch_shards}) across mesh axes {batch_axes}. "
          "Adjust the batch size or disable use_csa_streamindex_kernel."
      )

    q_pspec = jax.sharding.PartitionSpec(batch_axes, None, None, None)
    # `compressed`, `weights` and the output are all 3D and batch-sharded alike.
    pspec_3d = jax.sharding.PartitionSpec(batch_axes, None, None)

    # check_vma=False is required, not defensive: `pallas_call` builds its
    # `out_shape` from a plain `jax.ShapeDtypeStruct`, which carries no
    # `manual_axis_type`, and check_vma=True rejects that outright.
    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(q_pspec, pspec_3d, pspec_3d),
        out_specs=pspec_3d,
        check_vma=False,
    )
    def _sharded(local_q, local_compressed, local_weights):
      return csa_streamindex.csa_streamindex_score(local_q, local_compressed, local_weights, self.softmax_scale)

    return _sharded(q, compressed, weights)

  def __call__(
      self,
      hidden_states: Array,
      q_latent: Array,
      position_ids: Array,
      attention_mask: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      cache: Optional[Any] = None,
      return_scores: bool = False,
      true_length: Optional[Array] = None,
  ) -> Tuple[Array, Optional[Array]]:
    """Forward pass for the DeepSeek-V4 Indexer.

    Args:
      hidden_states: Input token embeddings.
      q_latent: Latent query representation for index score computation.
      position_ids: Absolute token positions.
      attention_mask: Optional attention mask.
      model_mode: Execution mode (train, prefill, or autoregressive).
      cache: Optional Indexer KV cache instance for inference.
      return_scores: Whether to return (final_indices, index_scores).
      true_length: Optional unpadded prompt length per batch row (prefill only), shape `[batch]`.

    Returns:
      A tuple (top_k_indices, indexer_scores) where indexer_scores is None when return_scores=False.
    """
    batch_size, seq_len, _ = hidden_states.shape
    # Stop gradient on indexer inputs so indexer loss does not backprop into main model projections
    hidden_states = jax.lax.stop_gradient(hidden_states)
    q_latent = jax.lax.stop_gradient(q_latent)

    future_mask = None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:

      def indexer_compressor_fn(buf_kv, buf_gate):
        chunk_kv_reshaped = jnp.swapaxes(buf_kv, 1, 2)
        chunk_gate_reshaped = jnp.swapaxes(buf_gate, 1, 2) + self.position_bias.value

        assert cache.overlap_kv is not None, "cache.overlap_kv cannot be None"
        assert cache.overlap_gate is not None, "cache.overlap_gate cannot be None"
        prior_kv = cache.overlap_kv.get_value()
        prior_gate = cache.overlap_gate.get_value()

        compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
            chunk_kv_reshaped, chunk_gate_reshaped, self.kv_norm, self.index_head_dim, prior_kv, prior_gate
        )

        block_pos = position_ids - (self.compress_rate - 1)
        compressed = self.rotary_emb(compressed, block_pos, unsqueeze_dim=None)

        return (
            jnp.expand_dims(compressed, 2),
            next_prior_kv,
            next_prior_gate,
        )

      compressed, is_valid = update_ar_cache_and_get_validity_mask(
          kv=kv,
          gate=gate,
          cache=cache,
          model_mode=model_mode,
          compressor_fn=indexer_compressor_fn,
          comp_dim=self.index_head_dim,
          batch_size=batch_size,
          mask_ndims=3,
      )
      compressed_len = compressed.shape[1]
      future_mask = ~is_valid

    # --- PREFILL CHUNKING & PRIMING ---
    else:
      compressed, compressed_len, next_prior_kv, next_prior_gate, usable = compute_csa_prefill_chunk_pooling(
          kv=kv,
          gate=gate,
          seq_len=seq_len,
          batch_size=batch_size,
          compress_rate=self.compress_rate,
          position_ids=position_ids,
          position_bias=self.position_bias.value,
          kv_norm=self.kv_norm,
          rotary_emb=self.rotary_emb,
          head_dim=self.index_head_dim,
          dtype=self.dtype,
          cache=cache,
          true_length=true_length,
      )

      # Prefill Cache Insertion
      if cache is not None:
        prime_prefill_cache_state(
            kv=kv,
            gate=gate,
            compressed_kv=compressed,
            seq_len=seq_len,
            usable=usable,
            compress_rate=self.compress_rate,
            cache=cache,
            next_prior_kv=next_prior_kv,
            next_prior_gate=next_prior_gate,
            true_length=true_length,
        )

    if compressed_len == 0:
      empty_indices = jnp.zeros((batch_size, seq_len, min(self.index_topk, compressed_len)), dtype=jnp.int32)
      return empty_indices, (jnp.zeros((batch_size, seq_len, 0), dtype=jnp.float32) if return_scores else None)

    # --- TOP-K ROUTING MATH (Executes in both Prefill and AR) ---
    q = self.q_proj(q_latent).reshape((batch_size, seq_len, self.index_n_heads, self.index_head_dim))
    q = jnp.transpose(q, (0, 2, 1, 3))
    q = self.rotary_emb(q, position_ids, unsqueeze_dim=1)
    weights = self.weights_proj(hidden_states).astype(jnp.float32) * self.weights_scaling

    # The Pallas kernel is a training-only path. AR decode arrives with
    # seq_len==1 and prefill with cache-derived `compressed`; neither shape
    # suits the kernel's fixed grid. It takes precedence over
    # `csa_qk_head_chunk_size`: the kernel keeps the forward intermediate in
    # VMEM instead of chunking it, and chunks its own backward.
    use_kernel = self.config.use_csa_streamindex_kernel and model_mode == MODEL_MODE_TRAIN
    head_chunk_size = getattr(self.config, "csa_qk_head_chunk_size", 0)

    if use_kernel:
      index_scores = self._streamindex_scores_sharded(q, compressed, weights)
    elif head_chunk_size > 0:
      # Student chunks indexer heads (indexer_n_heads); teacher path
      # in calculate_csa_indexer_loss chunks query heads (num_query_heads).
      num_chunks = self.index_n_heads // head_chunk_size
      q_h = q.transpose(1, 0, 2, 3).reshape(num_chunks, head_chunk_size, batch_size, seq_len, self.index_head_dim)
      w_h = weights.transpose(2, 0, 1).reshape(num_chunks, head_chunk_size, batch_size, seq_len)
      compressed_fp32 = compressed.astype(jnp.float32)

      def scan_body_indexer(carry, xs):
        q_c = xs["q"].astype(jnp.float32)
        w_c = xs["w"]

        scores_inner = jnp.einsum(
            "cbsd, bwd -> bcsw",
            q_c,
            compressed_fp32,
            precision=self.config.matmul_precision,
        )
        scores_inner = jax.nn.relu(scores_inner) * self.softmax_scale
        score_chunk = jnp.einsum(
            "bcsw, cbs -> bsw",
            scores_inner,
            w_c,
            precision=self.config.matmul_precision,
        )
        return carry + score_chunk, None

      init_score = jnp.zeros((batch_size, seq_len, compressed_len), dtype=jnp.float32)
      index_scores, _ = jax.lax.scan(jax.checkpoint(scan_body_indexer), init_score, {"q": q_h, "w": w_h})
    else:
      index_scores = csa_streamindex.csa_indexer_scores_jax(
          q,
          compressed,
          weights,
          softmax_scale=self.softmax_scale,
          precision=self.config.matmul_precision,
      )

    k = min(self.index_topk, compressed_len)

    # `future_mask` is supplied by the AR path (derived from cache validity).
    # In prefill/train it is None and is derived from positions here.
    if future_mask is None:
      usable_len = compressed_len * self.compress_rate
      block_positions = position_ids[:, : usable_len : self.compress_rate]
      future_mask = (block_positions[:, None, :] + self.compress_rate) > (position_ids[:, :, None] + 1)

    # Masking is always applied here, never inside the kernel: the kernel cannot
    # see `position_ids` and would have to synthesize them from grid indices,
    # which breaks for packed sequences where positions reset mid-sequence.
    index_scores = jnp.where(future_mask, jnp.full_like(index_scores, -jnp.inf), index_scores)

    combined_invalid = future_mask
    if attention_mask is not None:
      att_m = attention_mask[:, :, :compressed_len]
      index_scores += att_m
      combined_invalid = combined_invalid | (att_m < (DEFAULT_MASK_VALUE / 2))

    top_k_indices = jax.lax.top_k(index_scores, k)[1]
    invalid = jnp.take_along_axis(combined_invalid, top_k_indices, axis=-1)

    final_indices = jnp.where(invalid, jnp.full_like(top_k_indices, -1), top_k_indices)

    return final_indices, (index_scores if return_scores else None)
