"""Compressed Attention Layer (DeepSeek-V4)."""


from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
from lilax._backend.utils import model_mode_shapes
from lilax._backend.utils.kernel_tiling import eval_tiles

from flax import nnx

from lilax._backend.common.common_types import (
    Array,
    Config,
    DType,
    MODEL_MODE_TRAIN,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    AttentionType,
    DEFAULT_MASK_VALUE,
)

from lilax._backend.layers.attention_mla import indexer_losses
from lilax._backend.layers.attentions import Attention
from lilax._backend.layers.position_embeddings.deepseek_v4_rotary import DeepSeekV4RotaryEmbedding
from lilax._backend.layers.initializers import nd_dense_init, NdInitializer
from lilax._backend.layers.linears import DenseGeneral, DeepSeekV4GroupedLinear
from lilax._backend.layers.normalizations import RMSNorm
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant
from lilax._backend.inference.kvcache import KVQuant
from lilax._backend.inference import kvcache
from lilax._backend.utils.globals import EPS
from lilax._backend.layers.compressed_attention.csa_compressor import DeepseekV4CSACompressor
from lilax._backend.layers.compressed_attention.flash_kv_padding import compute_hca_padding
from lilax._backend.layers.compressed_attention.hca_compressor import DeepseekV4HCACompressor
from lilax._backend.layers.compressed_attention.segment_masks import compress_segment_ids


class CompressedAttention(Attention):
  """Compressed Attention layer (DeepSeek-V4).

  Wrapper around standard Attention that integrates HCA or CSA compressors based
  on the layer type. It compresses the inputs, concatenates the resulting sparse
  blocks to the standard KV sequence, and injects the compressor's block-masking
  matrix directly into the underlying attention operator.
  """

  def __init__(
      self,
      config: Config,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: Tuple,
      inputs_kv_shape: Tuple,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      float32_qk_product: bool = False,
      float32_logits: bool = False,
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.COMPRESSED,
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,
      # Compression Specific Parameters:
      q_lora_rank: int = 1536,
      compress_ratio: int = 0,
      name: str | None = None,
      rngs: Optional[nnx.Rngs] = None,
      **kwargs,
  ):
    """Inherits all standard Attention hyperparameters and selectively instantiates
    an underlying HCA or CSA compressor based on the provided `compress_ratio`.

    Highlights of DeepSeek-V4 attention integration:
    - Shared-KV: The layer supports decoupling Q and KV heads for heavy compression.
    - MQA: Multi-Query Attention used alongside heavy KV compression.
    - 3 Different Attention Modes: Sliding Window (prefix), HCA (128x), and CSA (4x).
    - Dual RoPE Theta: Uses 10000 for standard uncompressed tokens and 160000 for compressed.

    Args:
      (See maxtext.layers.attentions.Attention for standard attention arguments)
      q_lora_rank: The rank for the LoRA projection in the compressed query.
      compress_ratio: The compression ratio (0, 4, or 128) for the compressor.
    """
    self.q_lora_rank = q_lora_rank
    self.compress_ratio = compress_ratio

    # Determine the correct underlying attention type based on the compress_ratio
    if self.compress_ratio == 0:
      attention_type = AttentionType.LOCAL_SLIDING

    super().__init__(
        config=config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        max_prefill_predict_length=max_prefill_predict_length,
        dropout_rate=dropout_rate,
        kernel_init=kernel_init,
        float32_qk_product=float32_qk_product,
        float32_logits=float32_logits,
        quant=quant,
        kv_quant=kv_quant,
        attention_type=attention_type,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window_size=sliding_window_size,
        use_ragged_attention=use_ragged_attention,
        ragged_block_size=ragged_block_size,
        use_qk_norm=use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        use_bias_in_projections=use_bias_in_projections,
        name=name,
        rngs=rngs,
        **kwargs,
    )

    # DeepSeek-V4 uses a mathematical attention sink (a learnable scalar per-head added to the
    # attention logits prior to softmax, rather than a physical key/value token). We unconditionally
    # initialize it here, overriding the base Attention class which disables it by default.
    self.sinks = nnx.data(
        nnx.Param(
            jnp.zeros((self.num_query_heads,), dtype=self.weight_dtype),
            sharding=(None,),
        )
    )

  def _init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple) -> None:
    """Initializes the compressed projections and Unweighted RMSNorms."""
    # Query Projection Modules
    self.wq_a = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.q_lora_rank,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_attn", "q_lora"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    self.q_norm = RMSNorm(
        num_features=self.q_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    self.q_up_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        with_scale=False,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Key-Value Projection Modules
    self.wkv = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=(self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_attn", "kv_heads", "kv_head_dim"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    self.kv_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Override the base rotary embedding with the correct theta for this layer.
    # CSA / HCA layers use compressed_rope_max_timescale (160000).
    # Sliding window prefix layers use rope_max_timescale (10000).
    is_compressed = self.compress_ratio > 0
    rope_theta = self.config.compressed_rope_max_timescale if is_compressed else self.config.rope_max_timescale
    self.rotary_embedding = DeepSeekV4RotaryEmbedding(
        head_dim=self.config.head_dim,
        partial_rotary_factor=self.config.qk_rope_head_dim / self.config.head_dim,
        rope_theta=rope_theta,
        rope_type=self.config.rope_type if is_compressed else "default",
        rope_factor=self.config.rope_factor,
        beta_fast=self.config.beta_fast,
        beta_slow=self.config.beta_slow,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        truncate=self.config.rope_truncate,
        fprop_dtype=self.dtype,
    )

    if self.compress_ratio > 4:
      self.hca_compressor = DeepseekV4HCACompressor(
          config=self.config,
          compress_ratio=self.compress_ratio,
          rotary_embedding=self.rotary_embedding,
          kernel_init=self.kernel_init,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=self.rngs,
          attention_kernel=self.attention_kernel,
      )
    elif self.compress_ratio == 4:
      self.csa_compressor = DeepseekV4CSACompressor(
          config=self.config,
          compress_ratio=self.compress_ratio,
          rotary_embedding=self.rotary_embedding,
          kernel_init=self.kernel_init,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=self.rngs,
          mesh=self.mesh,
      )

    # Set softmax scaling. DeepSeek-V4 natively uses standard scaling.
    self.softmax_scale = self.head_dim**-0.5

    # Output Projections (Two-Step Grouped Linear)
    in_features_per_group = (self.num_query_heads * self.head_dim) // self.config.o_groups
    o_a_out_features = self.config.o_groups * self.config.o_lora_rank

    self.o_a_proj = DeepSeekV4GroupedLinear(
        in_features_per_group=in_features_per_group,
        out_features=o_a_out_features,
        n_groups=self.config.o_groups,
        kernel_init=self.kernel_init,
        kernel_axes=("o_groups", "q_heads", "o_lora_up_proj"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.o_b_proj = DenseGeneral(
        in_features_shape=o_a_out_features,
        out_features_shape=inputs_q_shape[-1],
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("o_lora_up_proj", "embed_attn"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    if self.model_mode != MODEL_MODE_TRAIN and self.compress_ratio > 0:
      batch_size, _ = model_mode_shapes.get_batch_seq_len_for_mode(self.config, MODEL_MODE_AUTOREGRESSIVE)

      max_prefill_comp = max(1, self.max_prefill_predict_length // self.compress_ratio)
      max_target_comp = max_prefill_comp + max(1, self.max_target_length // self.compress_ratio)

      comp_head_dim = self.head_dim if self.compress_ratio > 4 else 2 * self.head_dim

      self.compressor_cache = kvcache.KVCache(
          max_prefill_length=max_prefill_comp,
          max_target_length=max_target_comp,
          batch=batch_size,
          key_seq_len=max_target_comp,
          value_seq_len=max_target_comp,
          key_heads=1,
          value_heads=1,
          key_head_size=comp_head_dim,
          value_head_size=comp_head_dim,
          dtype=self.dtype,
          model_mode=self.model_mode,
          is_deepseek_v4=True,
          compress_rate=self.compress_ratio,
          rngs=self.rngs,
      )
    else:
      self.compressor_cache = None

    if self.model_mode != MODEL_MODE_TRAIN and self.compress_ratio == 4:
      self.indexer_cache = kvcache.KVCache(
          max_prefill_length=max_prefill_comp,
          max_target_length=max_target_comp,
          batch=batch_size,
          key_seq_len=max_prefill_comp,
          value_seq_len=max_prefill_comp,
          key_heads=1,
          value_heads=1,
          key_head_size=2 * self.config.indexer_head_dim,
          value_head_size=2 * self.config.indexer_head_dim,
          dtype=self.dtype,
          model_mode=self.model_mode,
          is_deepseek_v4=True,
          compress_rate=self.compress_ratio,
          is_indexer=True,
          rngs=self.rngs,
      )
    else:
      self.indexer_cache = None

  @property
  def out_head_dim(self) -> int:
    """Returns the head dimension used prior to the output projection."""
    return self.head_dim

  def _apply_rotary_embedding_v4(
      self, inputs: Array, inputs_positions: Array, unsqueeze_dim: int = -2, reverse: bool = False
  ) -> Array:
    """Applies rotary position embeddings, dispatching keyword arguments safely based on capability."""
    if isinstance(self.rotary_embedding, DeepSeekV4RotaryEmbedding):
      return self.rotary_embedding(inputs, inputs_positions, unsqueeze_dim=unsqueeze_dim, reverse=reverse)
    elif reverse and hasattr(self.rotary_embedding, "reverse"):
      return self.rotary_embedding(inputs, inputs_positions, reverse=True)
    else:
      return self.rotary_embedding(inputs, inputs_positions)

  def compressed_query_projection(self, inputs_q: Array, inputs_positions: Array, model_mode) -> Array:
    """Query projection for Compressed Attention.

    Args:
      inputs_q: The query hidden states. Shape: `[batch, seq_len, emb_dim]`.
      inputs_positions: The token positions, used for Rotary Positional Embeddings (RoPE).
      model_mode: The execution mode (e.g., 'train', 'prefill', 'autoregressive').

    Returns:
      The projected and RoPE-applied query tensor.
      Shape: `[batch, seq_len, num_query_heads, head_dim]`.

    Shape Transformations:
      1. Project `inputs_q` [batch, seq_len, emb_dim] to latent space [batch, seq_len, q_lora_rank].
      2. Normalize latent space via RMSNorm.
      3. Up-project to full head dimension [batch, seq_len, num_query_heads, head_dim].
      4. Apply Unweighted RMSNorm over the `head_dim` axis.
      5. Apply Rotary Positional Embeddings over the entire vector.
      6. Scale by 1/sqrt(head_dim) for numerical stability during attention computation.
    """
    # [batch, seq_len, emb_dim] -> [batch, seq_len, q_lora_rank]
    q_latent = self.wq_a(inputs_q)
    q_normed = self.q_norm(q_latent)

    # [batch, seq_len, q_lora_rank] -> [batch, seq_len, num_query_heads, head_dim]
    q_up = self.wq_b(q_normed)

    q_up_normed = self.q_up_norm(q_up)

    # -> [batch, seq_len, num_query_heads, head_dim]
    q_out = self._apply_rotary_embedding_v4(q_up_normed, inputs_positions, unsqueeze_dim=-2)

    # Scale queries by 1/sqrt(head_dim) prior to attention to prevent softmax saturation
    # -> [batch, seq_len, num_query_heads, head_dim]
    q_out = q_out * self.softmax_scale

    return q_out, q_normed

  def compressed_kv_projection(self, inputs_kv: Array, inputs_positions: Array, model_mode) -> Tuple[Array, Array]:
    """KV projection for Compressed Attention.

    Args:
      inputs_kv: The key/value hidden states. Shape: `[batch, seq_len, emb_dim]`.
      inputs_positions: The token positions, used for Rotary Positional Embeddings (RoPE).
      model_mode: The execution mode (e.g., 'train', 'prefill', 'autoregressive').

    Returns:
      A tuple of (key, value) tensors.
      Shapes: Both are `[batch, seq_len, num_kv_heads, head_dim]`.

    Shape Transformations:
      1. Project `inputs_kv` [batch, seq_len, emb_dim] to full head dimension [batch, seq_len, num_kv_heads, head_dim].
      2. Apply Unweighted RMSNorm over the `head_dim` axis.
      3. Apply Rotary Positional Embeddings over the entire vector.
      4. Note: Compressed caching will append additional slices downstream, but the base
         projections yield symmetrically shaped key and value vectors here.
    """
    # [batch, seq_len, emb_dim] -> [batch, seq_len, num_kv_heads, head_dim]
    kv_up = self.wkv(inputs_kv)

    kv_up_normed = self.kv_norm(kv_up)

    kv_out = self._apply_rotary_embedding_v4(kv_up_normed, inputs_positions, unsqueeze_dim=-2)

    return kv_out, kv_out

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      decoder_segment_ids: Array,
      inputs_positions: Array,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> Array:
    """Forward pass for Compressed Attention.

    Args:
      inputs_q: Query input. Shape: `[batch, q_length, embed_dim]`.
      inputs_kv: KV input. Shape: `[batch, kv_length, embed_dim]`.
      decoder_segment_pos: Segment IDs for masking.
      inputs_positions: Positions for rotary embeddings.
      deterministic: Disables dropout if set to True.
      model_mode: 'train', 'prefill', or 'autoregressive'.

    Returns:
      A tensor of shape `[batch, length, embed_dim]` containing the attended outputs.

    Shape Transformations:
      1. Projections: `[batch, len, emb_dim]` -> Q/K/V: `[batch, len, num_heads, head_dim]`.
      2. Dot Product: Attention over Q, K, V -> `[batch, q_length, num_query_heads, head_dim]`.
      3. Reverse RoPE: Applied in-place on the `head_dim` axis to undo V rotation.
      4. Group Reshape: `[batch, len, q_heads, head_dim]` -> `[batch, len, o_groups, in_features_per_group]`.
      5. Grouped Linear (o_a_proj): -> `[batch, q_length, o_groups, out_features_per_group]`.
      6. Flatten & Dense (o_b_proj): -> `[batch, q_length, emb_dim]`.
    """
    kv_cache = kwargs.get("kv_cache", None)

    q, q_normed = self.compressed_query_projection(inputs_q, inputs_positions, model_mode)
    q = checkpoint_name(q, "query_proj")
    kv, _ = self.compressed_kv_projection(inputs_kv, inputs_positions, model_mode)

    current_kv_cache = kv_cache

    # 1. Update the Local (Sliding Window) KV Cache with the uncompressed tokens
    if model_mode != MODEL_MODE_TRAIN and getattr(self, "KVCache_0", None) is not None:
      current_kv_cache = self.update_kv_caches(
          kv, kv, decoder_segment_ids, model_mode, kwargs.get("previous_chunk", None)
      )

    # Generate compressed representations based on the configured layer type
    compressed_kv = None
    compressed_mask = None
    compressed_segment_mask = None
    decoder_segment_ids_kv = decoder_segment_ids
    compressed_segment_ids = None

    if decoder_segment_ids is not None and self.compress_ratio > 0:
      compressed_segment_ids, decoder_segment_ids_kv, compressed_segment_mask = compress_segment_ids(
          decoder_segment_ids, inputs_kv.shape[1], self.compress_ratio
      )

    # Padded prefill: decode-time compressor state must start at the true prompt end.
    prefill_true_length = None
    if model_mode == MODEL_MODE_PREFILL and decoder_segment_ids is not None:
      prefill_true_length = jnp.sum(decoder_segment_ids != 0, axis=1).astype(jnp.int32)

    # Route to the appropriate compressor depending on the layer's role in the architecture
    if self.compress_ratio > 4:
      compressed_kv, compressed_mask = self.hca_compressor(
          inputs_kv, q_normed, inputs_positions, model_mode, self.compressor_cache, true_length=prefill_true_length
      )
    elif self.compress_ratio == 4:
      use_indexer = self.config.use_indexer
      scaling_factor = self.config.indexer_loss_scaling_factor
      should_compute_loss = use_indexer and scaling_factor > 0.0 and model_mode == MODEL_MODE_TRAIN

      if use_indexer:
        compressed_kv, sparse_compressed_mask, indexer_scores = self.csa_compressor(
            inputs_kv,
            q_normed,
            inputs_positions,
            compressed_segment_mask,
            model_mode,
            self.compressor_cache,
            self.indexer_cache,
            use_indexer=True,
            return_indexer_scores=should_compute_loss,
            true_length=prefill_true_length,
        )
        is_sparse_training = self.config.indexer_sparse_training
        is_dense_warmup = (model_mode == MODEL_MODE_TRAIN) and (scaling_factor > 0.0) and (not is_sparse_training)
        use_sparse_mask = not is_dense_warmup
        compressed_mask = self.get_compressed_mask(
            inputs_positions,
            compressed_kv.shape[1],
            sparse_compressed_mask=sparse_compressed_mask if use_sparse_mask else None,
        )

        if (
            should_compute_loss
            and indexer_scores is not None
            and compressed_kv is not None
            and compressed_kv.shape[1] > 0
        ):
          indexer_loss = self.calculate_csa_indexer_loss(
              indexer_score=indexer_scores,
              query=q,
              compressed_kv=compressed_kv,
              compressed_mask=sparse_compressed_mask,
              segment_mask=compressed_segment_mask,
              position_ids=inputs_positions,
              sparse_loss=is_sparse_training,
              scaling_factor=scaling_factor,
          )
          self.indexer_loss = indexer_losses(indexer_loss)
      else:
        compressed_kv, _, _ = self.csa_compressor(
            inputs_kv,
            q_normed,
            inputs_positions,
            compressed_segment_mask,
            model_mode,
            self.compressor_cache,
            self.indexer_cache,
            use_indexer=False,
            return_indexer_scores=False,
            true_length=prefill_true_length,
        )
        compressed_mask = self.get_compressed_mask(
            inputs_positions,
            compressed_kv.shape[1],
        )

    # Apply segment masking to the compressed blocks
    if compressed_segment_mask is not None and compressed_mask is not None:
      # compressed_segment_mask is [batch, q_len, num_compressed_blocks]
      # compressed_mask is [batch, 1, q_len, num_compressed_blocks]
      compressed_mask = compressed_mask + jnp.expand_dims(
          compressed_segment_mask[:, :, : compressed_mask.shape[-1]], axis=1
      )

    kv = checkpoint_name(kv, "kv_proj")

    pad_kv_total = 0
    unpadded_kv = jnp.concatenate([kv, compressed_kv], axis=1) if compressed_kv is not None else kv

    # Pad total KV length to tile size multiple (config.sa_block_kv) for SPMD sequence divisibility and
    # Tokamax dynamic splash tile boundary alignment. Note: Tokamax kernel inside AttentionOp additionally
    # sets inner block size as min(block_kv, key_len) during kernel invocation.
    if self.attention_kernel == "flash":
      if eval_tiles.is_eval(self.config):
        block_q = self.config.eval_sa_block_q
        block_kv = self.config.eval_sa_block_kv
      else:
        block_q = self.config.sa_block_q
        block_kv = self.config.sa_block_kv
      comp_len = compressed_kv.shape[1] if compressed_kv is not None else 0
      _, pad_kv_total = compute_hca_padding(
          q_len=inputs_q.shape[1],
          comp_len=comp_len,
          block_q=block_q,
          block_kv=block_kv,
          kv_len=kv.shape[1],
      )

      if pad_kv_total > 0:
        if compressed_kv is not None:
          # Prepend padding to the compressed blocks so they remain at the end of the sequence
          compressed_kv = jnp.pad(compressed_kv, ((0, 0), (pad_kv_total, 0), (0, 0), (0, 0)))

          if decoder_segment_ids is not None and compressed_segment_ids is not None:
            comp_seg_padded = jnp.pad(compressed_segment_ids, ((0, 0), (pad_kv_total, 0)), constant_values=-1)
            decoder_segment_ids_kv = jnp.concatenate([decoder_segment_ids, comp_seg_padded], axis=1)
        else:
          # Fallback: Pad at the end if no compressed blocks exist
          kv = jnp.pad(kv, ((0, 0), (0, pad_kv_total), (0, 0), (0, 0)))
          if decoder_segment_ids_kv is not None:
            decoder_segment_ids_kv = jnp.pad(decoder_segment_ids_kv, ((0, 0), (0, pad_kv_total)), constant_values=-1)

    # Prepare the mask shape for the underlying AttentionOp
    if compressed_mask is not None:
      compressed_mask = jnp.expand_dims(compressed_mask, axis=2)

    # Scale queries if a pre-attention scalar is defined
    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      q = q * self.query_pre_attn_scalar

    # Build indexer mask explicitly for tokamax splash kernel (CSA dynamic path)
    indexer_mask = None
    if self.attention_kernel == "flash" and compressed_mask is not None and self.compress_ratio == 4:
      indexer_mask = self.attention_op.generate_attention_mask(
          q,
          unpadded_kv,
          decoder_segment_ids,
          model_mode,
          compressed_mask=compressed_mask,
          pad_kv_total=pad_kv_total,
          decoder_segment_ids_kv=decoder_segment_ids_kv,
      )

      if indexer_mask is not None:
        # Extract single KV head and Query-per-KV head group axes [batch, 1, 1, Q, KV] -> [batch, Q, KV]
        indexer_mask = indexer_mask[:, 0, 0, :, :]

    # Compute Attention
    # -> [batch, q_length, num_query_heads, head_dim]
    attn_out = self.attention_op(
        q,
        kv,
        kv,
        decoder_segment_ids,
        inputs_positions,
        model_mode,
        sinks=self.sinks.value if self.sinks is not None else None,
        compressed_mask=compressed_mask,
        compressed_kv=compressed_kv,
        cached_values=current_kv_cache,
        indexer_mask=indexer_mask,
        decoder_segment_ids_kv=decoder_segment_ids_kv,
        pad_kv_total=pad_kv_total,
        compress_ratio=self.compress_ratio,
    )

    # Reverse RoPE on Values
    attn_out = self._apply_rotary_embedding_v4(attn_out, inputs_positions, unsqueeze_dim=-2, reverse=True)

    attn_out = checkpoint_name(attn_out, "attention_out")

    # Project outputs through Grouped Linear layers
    b, s, h, d = attn_out.shape
    # -> [batch, q_length, o_groups, in_features_per_group]
    grouped_out = attn_out.reshape(b, s, self.config.o_groups, (h * d) // self.config.o_groups)
    # -> [batch, q_length, o_groups, out_features_per_group]
    grouped_out = self.o_a_proj(grouped_out)
    # -> [batch, q_length, o_groups * out_features_per_group]
    grouped_flat = grouped_out.reshape(b, s, -1)
    # -> [batch, q_length, emb_dim]
    final_out = self.o_b_proj(grouped_flat)
    final_out = checkpoint_name(final_out, "out_proj")

    # Return the Tuple expected by the transformer block
    return final_out, current_kv_cache

  def get_compressed_mask(
      self,
      inputs_positions: Array,
      compressed_len: int,
      sparse_compressed_mask: Optional[Array] = None,
  ) -> Array:
    """Builds compressed attention mask. Returns sparse_compressed_mask if provided, else dense causal mask."""
    if sparse_compressed_mask is not None:
      return sparse_compressed_mask
    usable_len = compressed_len * self.compress_ratio
    block_positions = inputs_positions[:, : usable_len : self.compress_ratio]
    is_future = (block_positions[:, None, :] + self.compress_ratio) > (inputs_positions[:, :, None] + 1)
    dense_causal_mask = jnp.where(is_future, DEFAULT_MASK_VALUE, 0.0).astype(self.dtype)
    return dense_causal_mask[:, None, :, :]

  def calculate_csa_indexer_loss(
      self,
      indexer_score: Array,
      query: Array,
      compressed_kv: Array,
      compressed_mask: Array,
      segment_mask: Optional[Array] = None,
      position_ids: Optional[Array] = None,
      sparse_loss: bool = False,
      scaling_factor: float = 1.0,
  ) -> Array:
    """Calculates the indexer KL divergence loss for Compressed Attention (DeepSeek-V4).

    This loss trains the indexer to predict which compressed blocks are important by matching
    the distribution of true attention scores from the main model over compressed KV blocks.

    The target distribution is derived through the following steps:
    1. Compute raw attention scores via Q @ K_comp^T (Q is already pre-scaled by 1/sqrt(head_dim)).
    2. Apply causal block masking and segment masking over compressed blocks.
    3. Softmax across compressed blocks dimension for each head.
    4. Aggregate probabilities by summing across all attention heads.
    5. Apply L1-normalization across the compressed block sequence dimension.

    target_distribution = L1_Normalize(Sum_h(Softmax_w(Q @ K_comp^T + teacher_mask)))

    Reference:
    DeepSeek-V4 (https://arxiv.org/abs/2606.19348):
      - Section 2.3.1 describes the Lightning Indexer and CSA forward architecture.
      - Section 4.2.2 ("Training Setups") describes the 3-stage sparse attention pre-training pipeline:
        Stage 1 (Dense Pre-training, first 1T tokens):
          use_indexer=False, indexer_loss_scaling_factor=0.0
        Stage 2 (Lightning Indexer Warm-up):
          use_indexer=True, indexer_sparse_training=False, indexer_loss_scaling_factor=1.0,
          trainable_parameters_mask=['.*indexer.*']
        Stage 3 (Sparse Pre-training):
          use_indexer=True, indexer_sparse_training=True, indexer_loss_scaling_factor=1.0
    DeepSeek-V3.2 Section 2.1, Eqs. 3–4 (https://arxiv.org/abs/2512.02556) - DeepSeek Sparse Attention (DSA)
      KL divergence distillation loss, adapted here from uncompressed tokens to CSA compressed KV blocks.

    Note:
      Teacher distribution normalizes over compressed blocks only (W), rather than
      the full (S + W) joint keys used during forward attention.

    Args:
      indexer_score: Scores predicted by indexer [batch, q_len, compressed_len].
      query: Query tensor from main model [batch, q_len, heads, dim].
      compressed_kv: Compressed KV tensor from main model [batch, compressed_len, 1, dim].
      compressed_mask: Indexer compressed mask [batch, 1, q_len, compressed_len] or [batch, q_len, compressed_len].
      segment_mask: Segment mask [batch, q_len, compressed_len] or [batch, 1, q_len, compressed_len] or None.
      position_ids: Token position IDs [batch, q_len] or None.
      sparse_loss: Whether to use sparse loss.
      scaling_factor: The scaling factor for the loss.

    Returns:
      The computed scalar KL divergence loss.
    """
    if compressed_kv is None or indexer_score is None:
      return jnp.array(0.0, dtype=jnp.float32)

    batch, q_len, heads, dim = query.shape
    compressed_len = compressed_kv.shape[1]
    if compressed_len == 0:
      return jnp.array(0.0, dtype=jnp.float32)

    # Detach main model components from the computational graph.
    query = jax.lax.stop_gradient(query)
    compressed_kv = jax.lax.stop_gradient(compressed_kv)

    # Construct complete teacher mask: causal block mask + segment packing mask
    # 1. Causal block mask: query at position t can only attend to block w if (w+1)*r <= t+1
    if position_ids is not None:
      usable_len = compressed_len * self.compress_ratio
      block_positions = position_ids[:, : usable_len : self.compress_ratio]
      future_mask = (block_positions[:, None, :] + self.compress_ratio) > (position_ids[:, :, None] + 1)
    else:
      q_pos = jnp.arange(q_len)[:, None]
      block_end_pos = (jnp.arange(compressed_len)[None, :] + 1) * self.compress_ratio
      future_mask = block_end_pos > (q_pos + 1)
      future_mask = jnp.broadcast_to(future_mask[None, :, :], (batch, q_len, compressed_len))

    # Ensure indexer_mask is 2D/3D [batch, q_len, compressed_len]
    if compressed_mask.ndim == 4:
      indexer_mask = compressed_mask[:, 0, :, :]
    else:
      indexer_mask = compressed_mask

    # Combine causal, segment, and sparse masks via boolean disjunction to prevent float32 additive overflow
    is_invalid_block = future_mask
    if segment_mask is not None:
      if segment_mask.ndim == 4:
        c_seg = segment_mask[:, 0, :, :compressed_len]
      elif segment_mask.ndim == 3:
        c_seg = segment_mask[:, :, :compressed_len]
      else:
        c_seg = segment_mask
      is_invalid_block = is_invalid_block | (c_seg < (DEFAULT_MASK_VALUE / 2))

    if sparse_loss:
      is_invalid_block = is_invalid_block | (indexer_mask < (DEFAULT_MASK_VALUE / 2))

    teacher_mask = jnp.where(is_invalid_block, DEFAULT_MASK_VALUE, 0.0)
    c_mask = teacher_mask[:, None, :, :]  # [batch, 1, q_len, compressed_len]

    # Valid tokens mask checks causal, segment, and sparse boundaries after all masks are combined
    valid_tokens_mask = jnp.any(~is_invalid_block, axis=-1)  # [batch, q_len]

    # In CSA, compressed KV is pooled into a single representation per block (num_kv_heads = 1),
    # which is broadcast across all query heads.
    k_vec = compressed_kv[:, :, 0, :] if compressed_kv.ndim == 4 else compressed_kv

    # Student scores: index_scores already contains causal future_mask and segment masking.
    # In sparse training mode, mask non-selected blocks with DEFAULT_MASK_VALUE.
    if sparse_loss:
      student_scores = jnp.where(indexer_mask < (DEFAULT_MASK_VALUE / 2), DEFAULT_MASK_VALUE, indexer_score)
    else:
      student_scores = indexer_score

    safe_student_scores = jnp.where(valid_tokens_mask[:, :, None], student_scores, 0.0)
    log_indexer_probs = jax.nn.log_softmax(safe_student_scores.astype(jnp.float32), axis=-1)
    log_indexer_probs = jnp.where(valid_tokens_mask[:, :, None], log_indexer_probs, 0.0)

    # Query is already scaled by softmax_scale in compressed_query_projection; do not scale again
    head_chunk_size = getattr(self.config, "csa_qk_head_chunk_size", 0)
    if head_chunk_size > 0:
      # Teacher chunks main model query heads (num_query_heads); student path
      # in DeepseekV4Indexer chunks indexer heads (indexer_n_heads).
      num_chunks = heads // head_chunk_size
      # query: [b, t, h, d] -> [h, b, t, d] -> [num_chunks, head_chunk_size, b, t, d]
      q_h = query.transpose(2, 0, 1, 3).reshape(num_chunks, head_chunk_size, batch, q_len, dim)

      def scan_body_heads(carry, xs):
        q_c = xs["q"]  # [h_chunk, b, t, d]
        attn_chunk = jnp.einsum("hbtd, bwd -> bhtw", q_c, k_vec, precision=self.config.matmul_precision)
        attn_chunk = attn_chunk + c_mask

        # Apply NaN shielding for pre-block tokens
        safe_attn = jnp.where(valid_tokens_mask[:, None, :, None], attn_chunk, 0.0)
        # Softmax covers compressed blocks only (W), ignoring the sliding window keys (S).
        probs_chunk = jax.nn.softmax(safe_attn.astype(jnp.float32), axis=-1)
        probs_chunk_sum = jnp.sum(probs_chunk, axis=1)  # [b, t, w]

        return carry + probs_chunk_sum, None

      init_probs = jnp.zeros((batch, q_len, compressed_len), dtype=jnp.float32)
      target_probs, _ = jax.lax.scan(scan_body_heads, init_probs, {"q": q_h})
    else:
      attention_scores = jnp.einsum("bthd, bwd -> bhtw", query, k_vec, precision=self.config.matmul_precision)
      attention_scores = attention_scores + c_mask

      # Apply NaN shielding for pre-block tokens
      safe_scores = jnp.where(valid_tokens_mask[:, None, :, None], attention_scores, 0.0)
      # Softmax covers compressed blocks only (W), ignoring the sliding window keys (S).
      raw_probs = jax.nn.softmax(safe_scores.astype(jnp.float32), axis=-1)
      target_probs = jnp.sum(raw_probs, axis=1)
      # Optimization barrier prevents XLA from fusing downstream operations across the unchunked loss boundary.
      # Redundant in the chunked path above because jax.lax.scan lowers to an explicit HLO while-loop boundary.
      target_probs = jax.lax.optimization_barrier(target_probs)

    # L1 normalize aggregated target distribution across compressed blocks
    target_probs = jnp.where(valid_tokens_mask[:, :, None], target_probs, 0.0)
    target_probs = target_probs / (jnp.sum(target_probs, axis=-1, keepdims=True) + EPS)

    # KL Divergence: KL(attention || indexer)
    log_target_probs = jnp.log(target_probs + EPS)
    kl_element = jnp.where(target_probs > 0.0, target_probs * (log_target_probs - log_indexer_probs), 0.0)
    kl_per_token = jnp.sum(
        jnp.where(valid_tokens_mask[:, :, None], kl_element, 0.0),
        axis=-1,
    )

    # Average loss across valid tokens only (ignoring pre-block tokens t < compress_rate)
    num_valid_tokens = jnp.maximum(jnp.sum(valid_tokens_mask.astype(jnp.float32)), 1.0)
    indexer_loss = (jnp.sum(kl_per_token) / num_valid_tokens) * scaling_factor

    return indexer_loss
