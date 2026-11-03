"""Compressed Sparse Attention (CSA) compressor (DeepSeek-V4)."""


from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax.sharding import Mesh

from flax import nnx

from lilax._backend.common.common_types import (
    Array,
    MODEL_MODE_TRAIN,
    MODEL_MODE_AUTOREGRESSIVE,
    DEFAULT_MASK_VALUE,
)

from lilax._backend.layers.compressed_attention.compressor_base import BaseDeepseekCompressor
from lilax._backend.layers.compressed_attention.compressor_cache import (
    prime_prefill_cache_state,
    update_ar_cache_and_get_validity_mask,
)
from lilax._backend.layers.compressed_attention.csa_pooling import (
    compute_csa_prefill_chunk_pooling,
    csa_overlap_pooling,
)
from lilax._backend.layers.compressed_attention.indexer import DeepseekV4Indexer
from lilax._backend.layers.initializers import nd_dense_init, NdInitializer
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant


class DeepseekV4CSACompressor(BaseDeepseekCompressor):
  """Compressed Sparse Attention compressor (DeepSeek-V4 paper §2.3.1).

  Uses overlapping windows to compress local sequence contexts into sparse blocks,
  which are dynamically selected by the Indexer for long-range sparse attention.

  Shape Transformations:
    1. Pool KV (via overlap util): `[batch, seq, emb_dim]` -> `[batch, n_windows, head_dim]`
    2. RoPE: -> `[batch, n_windows, head_dim]`
    3. Expand Output: -> `[batch, n_windows, 1, head_dim]`
    4. Causal & Top-K Masking: -> `[batch, 1, seq, n_windows]`
  """

  def __init__(
      self,
      config: Any,
      compress_ratio: int,
      rotary_embedding: Any,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[Mesh] = None,
  ):
    """Initializes the CSA Compressor.

    Args:
      config: The configuration object for the model containing architecture hyperparameters.
      compress_ratio: The compression ratio (e.g., config.compress_rate_csa) that determines the
        stride size for pooling representations in sparse attention.
      rotary_embedding: A rotary embedding instance used to inject positional information into the
        final compressed representations.
      kernel_init: The initializer used for the kernel weights.
      quant: Optional quantization scheme.
      model_mode: The operational mode (e.g., "train", "prefill").
      rngs: An optional Rngs instance for stochastic initializations or dropout.
    """
    super().__init__(
        config,
        compress_ratio,
        rotary_embedding,
        2,
        kernel_init,
        quant,
        model_mode,
        rngs,
    )

    self.indexer = DeepseekV4Indexer(
        config=config,
        compress_ratio=compress_ratio,
        rotary_embedding=rotary_embedding,
        kernel_init=kernel_init,
        quant=quant,
        rngs=rngs,
        mesh=mesh,
    )

  def __call__(
      self,
      hidden_states: Array,
      q_latent: Array,
      position_ids: Array,
      attention_mask: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      cache: Optional[Any] = None,
      indexer_cache: Optional[Any] = None,
      use_indexer: bool = True,
      return_indexer_scores: bool = False,
      true_length: Optional[Array] = None,
  ) -> Tuple[Array, Array, Optional[Array]]:
    """Forward pass for the CSA compressor.

    Args:
      hidden_states: Input token embeddings.
      q_latent: Latent query representation for index score computation.
      position_ids: Absolute token positions.
      attention_mask: Optional attention mask.
      model_mode: Execution mode (train, prefill, or autoregressive).
      cache: Optional CSA compressor KV cache instance for inference.
      indexer_cache: Optional Indexer KV cache instance for inference.
      use_indexer: Whether to run the indexer to compute sparse attention masks.
      return_indexer_scores: Whether to return indexer scores along with compressed KV and mask.
      true_length: Optional unpadded prompt length per batch row (prefill only), shape `[batch]`.

    Returns:
      compressed_kv: The pooled KV tensors.
      compressed_mask: The sparse attention mask computed by the indexer.
      index_scores (optional): The raw indexer scores if return_indexer_scores is True.
    """
    batch_size, seq_len, _ = hidden_states.shape

    # 1. Run Indexer if use_indexer is True
    if use_indexer:
      top_k_indices, indexer_scores = self.indexer(
          hidden_states,
          q_latent,
          position_ids,
          attention_mask,
          model_mode,
          indexer_cache,
          return_scores=return_indexer_scores,
          true_length=true_length,
      )
    else:
      top_k_indices = None
      indexer_scores = None

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:

      def csa_compressor_fn(buf_kv, buf_gate):
        chunk_kv_reshaped = jnp.swapaxes(buf_kv, 1, 2)
        chunk_gate_reshaped = jnp.swapaxes(buf_gate, 1, 2) + self.position_bias.value

        assert cache.overlap_kv is not None, "cache.overlap_kv cannot be None"
        assert cache.overlap_gate is not None, "cache.overlap_gate cannot be None"
        prior_kv = cache.overlap_kv.get_value()
        prior_gate = cache.overlap_gate.get_value()

        compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
            chunk_kv_reshaped, chunk_gate_reshaped, self.kv_norm, self.head_dim, prior_kv, prior_gate
        )

        block_pos = position_ids - (self.compress_rate - 1)
        compressed = self.rotary_emb(compressed, block_pos, unsqueeze_dim=None)

        return (
            jnp.expand_dims(compressed, 2),
            next_prior_kv,
            next_prior_gate,
        )

      compressed, _ = update_ar_cache_and_get_validity_mask(
          kv=kv,
          gate=gate,
          cache=cache,
          model_mode=model_mode,
          compressor_fn=csa_compressor_fn,
          comp_dim=self.head_dim,
          batch_size=batch_size,
          mask_ndims=None,
      )
      compressed_kv = jnp.expand_dims(compressed, 2)
      compressed_len = compressed_kv.shape[1]

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
          head_dim=self.head_dim,
          dtype=self.dtype,
          cache=cache,
          true_length=true_length,
      )
      compressed_kv = jnp.expand_dims(compressed, 2)

      if cache is not None:
        prime_prefill_cache_state(
            kv=kv,
            gate=gate,
            compressed_kv=compressed_kv,
            seq_len=seq_len,
            usable=usable,
            compress_rate=self.compress_rate,
            cache=cache,
            next_prior_kv=next_prior_kv,
            next_prior_gate=next_prior_gate,
            true_length=true_length,
        )

    if compressed_len == 0:
      empty_mask = jnp.zeros((batch_size, 1, seq_len, 0), dtype=self.dtype)
      return compressed_kv, empty_mask, (indexer_scores if return_indexer_scores else None)

    if not use_indexer or top_k_indices is None:
      compressed_mask = jnp.zeros((batch_size, 1, seq_len, compressed_len), dtype=self.dtype)
      return compressed_kv, compressed_mask, None

    # 3. Apply Dynamic Masking Logic
    k = top_k_indices.shape[-1]
    if k > 0:
      valid = top_k_indices >= 0
      entry_indices = jnp.arange(compressed_len)[None, None, :]
      is_in_topk = jnp.expand_dims(top_k_indices, axis=-1) == entry_indices[None, ...]
      is_valid_and_in_topk = is_in_topk & jnp.expand_dims(valid, axis=-1)

      is_selected = jnp.any(is_valid_and_in_topk, axis=2)
      is_selected = jnp.expand_dims(is_selected, axis=1)

      compressed_mask = jnp.where(is_selected, 0.0, DEFAULT_MASK_VALUE).astype(self.dtype)
    else:
      compressed_mask = jnp.full(
          (batch_size, 1, seq_len, compressed_len),
          DEFAULT_MASK_VALUE,
          dtype=self.dtype,
      )

    return compressed_kv, compressed_mask, (indexer_scores if return_indexer_scores else None)
