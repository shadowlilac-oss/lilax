# Attention and position settings

Import from `lilax.configuration`. See [attention guide](../../guides/models/attention.md).
Configuration entries below describe keyword fields.

## AttentionParameters

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures attention semantics. Kernel selection belongs to `KernelConfig`.
See [kernel choices and hardware limits](configuration.md#attention-position-and-kernels).
Inherited kernel controls are class variables, not constructor fields.
Positive MLA and CSA head-chunk sizes must divide the query-head count and, with an indexer, its head count.

<details>
<summary>Parameters</summary>

- `attention_type` (`Literal["global", "local_sliding", "chunk", "mla", "full", "compressed", "block_diffusion"]`, optional, defaults to `"global"`). Attention variant.
- `share_kv_projections` (`bool`, optional, defaults to `False`). Shared global key and value projection weights.
- `global_num_kv_heads` (`int`, optional, defaults to `0`). Positive values override global KV head count.
- `hidden_size_per_layer_input` (`NonNegativeInt`, optional, defaults to `0`). Gemma 4 per-layer input embedding width. Positive values add a block to each decoder layer.
- `vocab_size_per_layer_input` (`NonNegativeInt`, optional, defaults to `0`). Gemma 4 per-layer embedding table size. Zero disables the table.
- `num_kv_shared_layers` (`NonNegativeInt`, optional, defaults to `0`). Trailing Gemma 4 layers reusing keys and values from the last non-shared layer of the same attention type. Sliding layers reuse sliding keys and values, and full layers reuse full keys and values.
- `use_double_wide_mlp` (`bool`, optional, defaults to `False`). Doubled MLP width in Gemma 4 layers reusing another layer's keys and values.
- `attention_sink` (`bool`, optional, defaults to `False`). Attention sinks.
- `float32_qk_product` (`bool`, optional, defaults to `False`). Float32 casting of dot-product attention query-key products.
- `float32_logits` (`bool`, optional, defaults to `False`). Float32 casting of dot-product attention logits before softmax.
- `sliding_window_size` (`NonNegativeInt`, optional, defaults to `0`). Local attention window size.
- `chunk_attn_window_size` (`NonNegativeInt`, optional, defaults to `0`). Chunked attention window size.
- `causal_block_size` (`PositiveInt`, optional, defaults to `32`). Token positions per bidirectional block in block-causal attention.
- `attn_logits_soft_cap` (`NonNegativeFloat | None`, optional, defaults to `None`). Attention-logit soft cap. `None` disables it.
- `use_post_attn_norm` (`bool`, optional, defaults to `False`). LayerNorm after attention.
- `use_post_ffw_norm` (`bool`, optional, defaults to `False`). LayerNorm after the feed-forward block.
- `use_qk_clip` (`bool`, optional, defaults to `False`). QK-Clip (MuonClip).
- `qk_clip_threshold` (`float`, optional, defaults to `100.0`). QK-Clip threshold, tau.
- `mla_qk_head_chunk_size` (`int`, optional, defaults to `0`). Heads per MLA query-key group. Zero processes all heads together. A positive value must divide the query-head count and, with an indexer, its head count.
- `csa_qk_head_chunk_size` (`int`, optional, defaults to `0`). Heads per DeepSeek-V4 compressed sparse query-key group. Zero disables grouping. The same divisibility requirements apply.

</details>

## MoBaConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures Mixture of Block Attention.

<details>
<summary>Parameters</summary>

- `moba` (`bool`, optional, defaults to `False`). MoBA selection.
- `moba_chunk_size` (`int`, optional, defaults to `1024`). Chunk size.
- `moba_topk` (`int`, optional, defaults to `8`). Number of selected chunks.

</details>

## MlaAttentionConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures MLA projections and cache behavior. Rank and dimension fields are nonnegative.

<details>
<summary>Parameters</summary>

- `mla_naive_kvcache` (`bool`, optional, defaults to `True`). Naive MLA KV cache.
- `q_lora_rank` (`NonNegativeInt`, optional, defaults to `0`). Query LoRA rank.
- `kv_lora_rank` (`NonNegativeInt`, optional, defaults to `512`). Key and value LoRA rank.
- `qk_nope_head_dim` (`NonNegativeInt`, optional, defaults to `128`). Non-RoPE query and key width.
- `qk_rope_head_dim` (`NonNegativeInt`, optional, defaults to `64`). RoPE query and key width.
- `v_head_dim` (`NonNegativeInt`, optional, defaults to `128`). Value head width.
- `use_sliced_mla_proj` (`bool`, optional, defaults to `False`). Projection weight slicing before multiplication instead of full-weight multiplication and result splitting.
- `use_mla_absorbed_mqa` (`bool`, optional, defaults to `False`). Absorbed-query latent MQA with additional prefill and training computation for a smaller autoregressive KV cache.

</details>

## CompressedAttentionConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures compressed attention. Nonzero per-layer compression ratios must be at least four.

<details>
<summary>Parameters</summary>

- `o_lora_rank` (`NonNegativeInt`, optional, defaults to `0`). Output LoRA rank.
- `o_groups` (`NonNegativeInt`, optional, defaults to `0`). Output groups.
- `compress_ratios` (`list[int]`, optional, defaults to `[]`). Per-layer compression ratios. Zero disables compression for that layer.
- `compressed_rope_max_timescale` (`int`, optional, defaults to `160000`). Positive values override the compressed attention RoPE maximum timescale.
- `use_csa_streamindex_kernel` (`bool`, optional, defaults to `False`). TPU Pallas CSA StreamIndex score kernel.

</details>

## AttentionIndexerConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures sparse indexing for MLA or compressed attention.

<details>
<summary>Parameters</summary>

- `use_indexer` (`bool`, optional, defaults to `False`). Sparse indexer selection.
- `indexer_head_dim` (`NonNegativeInt`, optional, defaults to `128`). Query and key head width.
- `indexer_n_heads` (`NonNegativeInt`, optional, defaults to `64`). Query head count.
- `indexer_topk` (`NonNegativeInt`, optional, defaults to `2048`). Tokens selected per query.
- `indexer_sparse_training` (`bool`, optional, defaults to `False`). Sparse training instead of dense warmup.
- `indexer_loss_scaling_factor` (`float`, optional, defaults to `0.0`). Indexer KL-divergence auxiliary loss weight.
- `indexer_use_approx_top_k` (`bool`, optional, defaults to `False`). Approximate top-k selection on TPU.
- `indexer_approx_top_k_recall` (`float`, optional, defaults to `0.95`). Approximate selection recall target.
- `indexer_mask_exact_topk` (`bool`, optional, defaults to `True`). Exactly `indexer_topk` unmasked entries at boundary ties. `False` uses a score threshold and can admit additional tied entries.

</details>

## Llama4AttentionConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures Llama4 query/key normalization and attention temperature.

<details>
<summary>Parameters</summary>

- `use_qk_norm` (`bool`, optional, defaults to `False`). L2 normalization of query and key vectors after RoPE.
- `temperature_tuning` (`bool`, optional, defaults to `False`). Attention temperature scaling with sequence length.

</details>

## PositionalEmbeddingConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures embedding lookup and positional embeddings.

<details>
<summary>Parameters</summary>

- `use_iota_embed` (`bool`, optional, defaults to `False`). Token embedding computation through one-hot comparisons and a matrix product instead of indexed lookup.
- `use_untrainable_positional_embedding` (`bool`, optional, defaults to `False`). Fixed sinusoidal positions.
- `trainable_position_size` (`int`, optional, defaults to `-1`). Positive values enable GPT-3 style trainable positions.
- `nope_layer_interval` (`int`, optional, defaults to `-1`). Positive values disable RoPE at every N-th layer for Llama4.

</details>

## RopeConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures rotary positional embeddings.

<details>
<summary>Parameters</summary>

- `rope_type` (`RopeType`, optional, defaults to `RopeType.DEFAULT`). RoPE variant.
- `rope_use_scale` (`bool`, optional, defaults to `True`). Llama3.1 style scaling.
- `rope_min_timescale` (`int`, optional, defaults to `1`). Minimum timescale.
- `rope_max_timescale` (`int`, optional, defaults to `10000`). Maximum timescale.
- `rope_linear_scaling_factor` (`float`, optional, defaults to `1.0`). Linear scale for default RoPE.
- `local_rope_max_timescale` (`int`, optional, defaults to `-1`). Positive values override local attention timescale.
- `global_rope_max_timescale` (`int`, optional, defaults to `-1`). Positive values override global attention timescale.
- `global_rope_proportion` (`float`, optional, defaults to `0.25`). Fraction of global head dimensions receiving RoPE.
- `local_rope_proportion` (`float`, optional, defaults to `1.0`). Fraction of local head dimensions receiving RoPE.

</details>

## YarnRopeConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures YaRN scaling. `rope_pairwise=True` requires `rope_interleave=True`.
`rope_dot_pairwise=True` requires `rope_pairwise=True`. Violations raise `ValueError`.

<details>
<summary>Parameters</summary>

- `max_position_embeddings` (`int`, optional, defaults to `163840`). Maximum position count.
- `original_max_position_embeddings` (`int`, optional, defaults to `4096`). Original position count before scaling.
- `rope_factor` (`int`, optional, defaults to `40`). YaRN scale factor.
- `beta_fast` (`int`, optional, defaults to `32`). Lower correction-range boundary for scaled and unscaled frequencies.
- `beta_slow` (`int`, optional, defaults to `1`). Upper correction-range boundary.
- `mscale` (`float`, optional, defaults to `1.0`). Logarithmic attention-scale correction multiplier.
- `rope_interleave` (`bool`, optional, defaults to `True`). Interleaved sine and cosine coordinates instead of concatenation.
- `rope_truncate` (`bool`, optional, defaults to `True`). Floor and ceiling rounding of the correction range.
- `rope_attention_scaling` (`bool`, optional, defaults to `False`). Rotary output scaling, as used by models such as GPT-OSS.
- `rope_pairwise` (`bool`, optional, defaults to `False`). Intact rank-5 `[batch_size, sequence_length, num_heads, half_dim, 2]` pair tensor for interleaved RoPE. `half_dim` counts independent rotation planes and the final dimension contains real and imaginary coordinates.
- `rope_direct_position_freqs` (`bool`, optional, defaults to `False`). Angle computation from position times inverse frequencies instead of a `[max_position_embeddings, half_dim]` lookup table. Positions at or above the maximum use their true angle instead of the clamped final table row.
- `rope_dot_pairwise` (`bool`, optional, defaults to `False`). Rank-4 `[batch, seq, heads, dim]` inputs preserved through a block-skew matrix product on the minor dimension instead of a rank-five reshape.

</details>
