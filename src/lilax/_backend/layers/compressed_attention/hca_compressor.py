"""Heavily Compressed Attention (HCA) compressor (DeepSeek-V4)."""


from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

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
from lilax._backend.layers.initializers import nd_dense_init, NdInitializer
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant


class DeepseekV4HCACompressor(BaseDeepseekCompressor):
  """Heavily Compressed Attention compressor.

  Compresses every `compress_rate_hca` source tokens into a single compressed KV entry
  using closed, non-overlapping windows. RoPE is applied to the final compressed token.

  Shape Transformations:
    1. Projections: `[batch, seq, emb_dim]` -> `[batch, seq, head_dim]`
    2. Chunking: -> `[batch, n_windows, compress_rate, head_dim]`
    3. Gating & Sum: -> `[batch, n_windows, head_dim]`
    4. RoPE: -> `[batch, n_windows, head_dim]`
    5. Output Expand: -> `[batch, n_windows, 1, head_dim]`
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
      attention_kernel: str = "dot_product",
  ):
    """Initializes the HCA Compressor.

    Args:
      config: The configuration object for the model containing architecture hyperparameters.
      compress_ratio: The compression ratio (e.g., config.compress_rate_hca) that determines the
        window size for heavily compressed attention.
      rotary_embedding: A rotary embedding instance used to inject positional information into the
        final compressed representations.
      kernel_init: The initializer used for the kernel weights.
      quant: Optional quantization scheme.
      model_mode: The operational mode (e.g., "train", "prefill").
      rngs: An optional Rngs instance for stochastic initializations or dropout.
      attention_kernel: Attention kernel name ("flash", "dot_product").
    """
    super().__init__(
        config,
        compress_ratio,
        rotary_embedding,
        1,
        kernel_init,
        quant,
        model_mode,
        rngs,
    )
    self.attention_kernel = attention_kernel

  def __call__(
      self,
      hidden_states: Array,
      q_normed: Array,
      position_ids: Array,
      model_mode: str,
      cache: Optional[Any] = None,
      true_length: Optional[Array] = None,
  ) -> Tuple[Array, Array]:
    """Forward pass for the HCA compressor.

    Args:
      hidden_states: Input token embeddings. Shape: `[batch, seq_len, emb_dim]`.
      q_normed: Latent query representation (unused in HCA).
      position_ids: Absolute token positions. Shape: `[batch, seq_len]`.
      model_mode: The execution mode (e.g. train, prefill, or autoregressive).
      cache: Optional KV cache instance used during inference.
      true_length: Optional unpadded prompt length per batch row (prefill only), shape `[batch]`.

    Returns:
      compressed_kv: The pooled KV tensors. Shape: `[batch, n_windows, 1, head_dim]`.
      compressed_causal_mask: Causal mask preventing queries from seeing future blocks.
                              Shape: `[batch, 1, seq_len, n_windows]`.
    """
    batch_size, seq_len, _ = hidden_states.shape
    # Project hidden states to KV and Gate components
    # [batch, seq_len, emb_dim] -> [batch, seq_len, head_dim]
    kv = self.kv_proj(hidden_states)
    # [batch, seq_len, emb_dim] -> [batch, seq_len, head_dim]
    gate = self.gate_proj(hidden_states)

    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:

      def hca_compressor_fn(buf_kv, buf_gate):
        gate_weights = jax.nn.softmax(buf_gate + self.position_bias.value[:, None, :], axis=1).astype(buf_kv.dtype)
        compressed = self.kv_norm(jnp.sum(buf_kv * gate_weights, axis=1))
        block_pos = position_ids - (self.compress_rate - 1)
        compressed = self.rotary_emb(compressed, block_pos, unsqueeze_dim=None)
        return jnp.expand_dims(compressed, 2), None, None

      compressed, is_valid = update_ar_cache_and_get_validity_mask(
          kv=kv,
          gate=gate,
          cache=cache,
          model_mode=model_mode,
          compressor_fn=hca_compressor_fn,
          comp_dim=self.head_dim,
          batch_size=batch_size,
          mask_ndims=4,
      )
      compressed_kv = jnp.expand_dims(compressed, 2)  # [B, N, 1, D]
      compressed_mask = jnp.where(is_valid, 0.0, DEFAULT_MASK_VALUE).astype(self.dtype)

      return compressed_kv, compressed_mask

    # --- PREFILL CHUNKING & PRIMING ---
    usable = (seq_len // self.compress_rate) * self.compress_rate
    chunk_kv = kv[:, :usable]
    chunk_gate = gate[:, :usable]

    # Process overlapping windows if there is enough sequence length
    if chunk_kv.shape[1] > 0:
      n_windows = chunk_kv.shape[1] // self.compress_rate

      # Reshape into blocks of size `compress_rate`
      # -> [batch, n_windows, compress_rate, head_dim]
      chunk_kv = chunk_kv.reshape((batch_size, n_windows, self.compress_rate, -1))
      chunk_gate = chunk_gate.reshape((batch_size, n_windows, self.compress_rate, -1)) + self.position_bias.value

      # Apply gating mechanism over each compression window
      # -> [batch, n_windows, head_dim]
      gate_weights = jax.nn.softmax(chunk_gate, axis=2).astype(chunk_kv.dtype)
      compressed = self.kv_norm(jnp.sum(chunk_kv * gate_weights, axis=2))

      # Calculate positions for the compressed blocks
      positions = position_ids[:, : usable : self.compress_rate]

      # Apply Rotary Positional Embeddings to the pooled representations
      # compressed is [batch, n_windows, head_dim]
      compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)
    else:
      # Provide an empty tensor when the sequence is shorter than the compression rate
      compressed = jnp.zeros((batch_size, 0, self.head_dim), dtype=self.dtype)

    # Expand the feature dimension to match the standard KV projection shape
    # -> [batch, n_windows, 1, head_dim]
    compressed_kv = jnp.expand_dims(compressed, axis=2)
    compressed_len = compressed_kv.shape[1]

    # --- PREFILL CACHE PRIMING ---
    if cache is not None:
      prime_prefill_cache_state(
          kv=kv,
          gate=gate,
          compressed_kv=compressed_kv,
          seq_len=seq_len,
          usable=usable,
          compress_rate=self.compress_rate,
          cache=cache,
          true_length=true_length,
      )

    # During decoding (seq_len == 1), dot-product attention requires a zero mask
    if seq_len == 1:
      compressed_mask = jnp.zeros((batch_size, 1, seq_len, compressed_len), dtype=self.dtype)
      return compressed_kv, compressed_mask

    # Skip dense causal mask generation when using static flash attention (HCAStaticMask)
    if self.attention_kernel == "flash":
      return compressed_kv, None

    # Dot-product fallback when no blocks were pooled
    if compressed_len == 0:
      compressed_mask = jnp.zeros((batch_size, 1, seq_len, compressed_len), dtype=self.dtype)
      return compressed_kv, compressed_mask

    # Construct a causal mask preventing early queries from attending to future compressed blocks
    usable_len = compressed_len * self.compress_rate
    block_positions = position_ids[:, : usable_len : self.compress_rate]
    future_mask = (block_positions[:, None, None, :] + self.compress_rate) > (position_ids[:, None, :, None] + 1)
    compressed_causal_mask = jnp.where(future_mask, DEFAULT_MASK_VALUE, 0.0).astype(self.dtype)

    return compressed_kv, compressed_causal_mask
