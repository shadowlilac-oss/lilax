# Qwen family settings

Return to the [configuration reference](configuration.md).

Import these configurations from `lilax.configuration`. Keep the settings loaded with the matching model family. See [model configuration](../../guides/models/configuration.md) for loading and updating a focused domain.

These are constructor defaults. A loaded checkpoint can supply different values. Changing architecture dimensions or routing does not convert existing weights.

## Qwen3NextConfig

Import from `lilax.configuration`. Pydantic model whose keyword fields configure Qwen3-Next Gated Delta Net.

<details>
<summary>Parameters</summary>

- `gdn_conv_kernel_dim` (`int`, optional, defaults to `4`). One-dimensional convolution kernel size.
- `gdn_key_head_dim` (`int`, optional, defaults to `128`). Key and query head width.
- `gdn_value_head_dim` (`int`, optional, defaults to `128`). Value head width.
- `gdn_num_key_heads` (`int`, optional, defaults to `16`). Key and query head count.
- `gdn_num_value_heads` (`int`, optional, defaults to `32`). Value head count.
- `gdn_chunk_size` (`int`, optional, defaults to `64`). Parallel scan chunk size.
- `use_qk_norm_in_gdn` (`bool`, optional, defaults to `True`). L2 normalization of queries and keys inside the Gated Delta Rule kernel.
- `gdn_mamba_block_size` (`int`, optional, defaults to `0`). Tokens per vLLM Mamba prefix-cache block in align mode, set by the serving adapter. Zero uses a resident per-request state slot.
- `partial_rotary_factor` (`float`, optional, defaults to `1.0`). Fraction of each head dimension receiving rotary positions.

</details>

## Qwen3_8FlashNextConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures Qwen3.8-Flash-Next. The `flash` QSA kernel requires the target length not to exceed `index_select_budget`.

<details>
<summary>Parameters</summary>

- `hc_count` (`int`, optional, defaults to `4`). Hyperconnection stream count.
- `hc_lowrank` (`int`, optional, defaults to `320`). Hyperconnection bottleneck width.
- `index_n_heads` (`int`, optional, defaults to `4`). Indexer query head count.
- `index_kv_heads` (`int`, optional, defaults to `1`). Indexer key and value head count.
- `index_head_dim` (`int`, optional, defaults to `128`). Indexer query and key width.
- `index_select_block_size` (`int`, optional, defaults to `4`). Indexer block compression size.
- `index_select_budget` (`int`, optional, defaults to `2048`). Indexer token budget.
- `gdn_output_gate_activation` (`str`, optional, defaults to `"sigmoid"`). GDN output gate activation.
- `gdn_chunk_inverse` (`Literal["solve", "block_doubling"]`, optional, defaults to `"block_doubling"`). Matrix products for `block_doubling`, triangular solve for `solve`.
- `qsa_indexer_mask` (`Literal["gather", "reshape"]`, optional, defaults to `"gather"`). Per-token selection for `gather`. `reshape` expands blocks and uses a one-hot contraction for segment IDs.
- `qsa_attention_kernel` (`Literal["dense", "flash"]`, optional, defaults to `"dense"`). Attention score materialization for `dense`, or Tokamax Splash for `flash`.
- `use_ple_layer` (`bool`, optional, defaults to `False`). Per-layer embeddings.
- `ple_layer_id` (`int`, optional, defaults to `1`). Zero-based decoder layer index for the embedding.
- `ple_embed_dim` (`int`, optional, defaults to `2560`). Per-layer embedding width.
- `ple_conv_kernel_size` (`int`, optional, defaults to `4`). PLE convolution kernel size.
- `ngram_size` (`int`, optional, defaults to `3`). PLE n-gram window size.
- `heads_per_ngram` (`int`, optional, defaults to `8`). Hash heads per n-gram.
- `ngram_vocab_size_base` (`int`, optional, defaults to `20_000_000`). Base n-gram vocabulary size.
- `make_ngram_vocab_size_divisible_by` (`int`, optional, defaults to `128`). Per-head vocabulary alignment divisor.
- `split_ngram_parts` (`int`, optional, defaults to `128`). PLE embedding shards.
- `ple_seed` (`int`, optional, defaults to `1234`). Hash seed.
- `ple_eos_token_id` (`int`, optional, defaults to `248044`). Token that resets n-gram hashing boundaries.

</details>

