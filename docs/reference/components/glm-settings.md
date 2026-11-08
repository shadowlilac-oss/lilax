# GLM component settings

Import `Glm5Config` from `lilax.generation.m3_config`. This frozen dataclass describes the advanced GLM component. See [component construction](../../guides/components/m3-components.md) and [checkpoint workflows](../../guides/checkpoints/m3-checkpoints.md).

## Constructor fields

Import from `lilax.generation.m3_config`. Frozen dataclass `Glm5Config`.

Construction returns an immutable configuration and validates the combinations described next.

## Validated combinations

The constructor expands omitted layer and indexer schedules to the layer count. Layer types accept `linear_attention` and `full_attention`. Every indexer type and the MTP indexer type must be `full`.

`remat_policy` accepts `none` or `full`. `router_param_dtype` must be `float32`. `weight_dtype` accepts `float32`, `bfloat16`, or `float64`. Packing is rejected. Routed bias updates and indexer loss scaling must be zero.

`kda_implementation` accepts `xla`, `xla_chunked`, or `mosaic`. The latter two require a non-None linear lower bound. `index_kpool` must be positive and divide `index_topk`. The MLP clamp limit must be positive. MTP layers accept zero or one.

DSA attention accepts `dense_mask`. RoPE dimensions must be zero and `rope_parameters=None`. Attention and KV head counts must match. Linear key and value head counts and dimensions must also match. Invalid combinations raise `ValueError`, except a non-string remat policy raises `TypeError`.

<details>
<summary>Parameters</summary>

- `vocab_size` (`int`, optional, defaults to `154880`). Vocabulary size.
- `model_name` (`str`, optional, defaults to `"glm5.3-flash"`). Model configuration identity.
- `hidden_size` (`int`, optional, defaults to `4096`). Hidden width.
- `num_hidden_layers` (`int`, optional, defaults to `45`). Decoder layer count.
- `first_k_dense_replace` (`int`, optional, defaults to `3`). Initial dense layers before routed experts.
- `layer_types` (`tuple[str, ...] | None`, optional, defaults to `None`). Attention per layer. Omission repeats three linear layers then one full layer.
- `indexer_types` (`tuple[str, ...] | None`, optional, defaults to `None`). Indexers per layer. Omission uses `full` throughout.
- `mtp_indexer_type` (`str`, optional, defaults to `"full"`). The MTP indexer. Only `full` is accepted.
- `linear_num_key_heads` (`int`, optional, defaults to `64`). Linear query and key heads.
- `linear_num_value_heads` (`int`, optional, defaults to `64`). Linear value heads.
- `linear_key_head_dim` (`int`, optional, defaults to `128`). Linear query and key width.
- `linear_value_head_dim` (`int`, optional, defaults to `128`). Linear value width.
- `linear_conv_kernel_dim` (`int`, optional, defaults to `4`). The linear convolution token window.
- `linear_lower_bound` (`float | None`, optional, defaults to `-5.0`). KDA decay lower bound in log space, required by chunked XLA and Mosaic.
- `kda_implementation` (`str`, optional, defaults to `"mosaic"`). KDA implementation, `xla`, `xla_chunked`, or `mosaic`.
- `num_attention_heads` (`int`, optional, defaults to `64`). Full-attention query heads.
- `num_key_value_heads` (`int`, optional, defaults to `64`). Full-attention key and value heads, equal to query heads.
- `q_lora_rank` (`int`, optional, defaults to `1536`). MLA query LoRA rank.
- `kv_lora_rank` (`int`, optional, defaults to `512`). MLA key and value LoRA rank.
- `qk_nope_head_dim` (`int`, optional, defaults to `256`). MLA non-RoPE query and key width.
- `qk_rope_head_dim` (`int`, optional, defaults to `0`). MLA RoPE query and key width.
- `v_head_dim` (`int`, optional, defaults to `256`). MLA value width.
- `rope_parameters` (`Any`, optional, defaults to `None`). RoPE configuration. This NoPE component requires `None`.
- `use_indexer` (`bool`, optional, defaults to `True`). Flag for sparse indexing for MLA or compressed attention.
- `dsa_attention_implementation` (`str`, optional, defaults to `"dense_mask"`). Sparse attention. Only `dense_mask` is accepted.
- `index_n_heads` (`int`, optional, defaults to `32`). Indexer query heads.
- `index_head_dim` (`int`, optional, defaults to `128`). Indexer query and key projection width.
- `index_topk` (`int`, optional, defaults to `2048`). The sparse token budget.
- `index_kpool` (`int`, optional, defaults to `4`). Tokens per compressed pool, which must divide `index_topk`.
- `index_kpool_always_select_tail` (`bool`, optional, defaults to `True`). Flag for inclusion of visible incomplete-tail tokens with selected full pools.
- `indexer_loss_scaling_factor` (`float`, optional, defaults to `0.0`). Indexer auxiliary loss weight, which must be zero.
- `intermediate_size` (`int`, optional, defaults to `12288`). Dense feed-forward width.
- `moe_intermediate_size` (`int`, optional, defaults to `2048`). Routed expert width.
- `n_routed_experts` (`int`, optional, defaults to `288`). Routed expert count.
- `num_experts_per_tok` (`int`, optional, defaults to `8`). Experts selected per token.
- `n_shared_experts` (`int`, optional, defaults to `1`). Shared experts evaluated for each token.
- `n_group` (`int`, optional, defaults to `1`). Routing groups.
- `topk_group` (`int`, optional, defaults to `1`). Eligible routing groups per token.
- `routed_scaling_factor` (`float`, optional, defaults to `2.5`). Routing score scale.
- `norm_topk_prob` (`bool`, optional, defaults to `True`). Flag for normalization of selected expert weights to sum to one.
- `scoring_func` (`str`, optional, defaults to `"sigmoid"`). Router logit transformation before selection.
- `mlp_clamp_limit` (`float`, optional, defaults to `10.0`). A positive expert activation upper clamp.
- `routed_bias_update_rate` (`float`, optional, defaults to `0.0`). Router bias updates. Zero is required.
- `moe_lb_loss_weight` (`float`, optional, defaults to `0.001`). Expert load-balancing loss weight.
- `mhc_expansion_rate` (`int`, optional, defaults to `4`). Parallel hyperconnection streams.
- `mhc_sinkhorn_iters` (`int`, optional, defaults to `20`). Mixing matrix normalization iterations.
- `mhc_eps` (`float`, optional, defaults to `1e-6`). Hyperconnection normalization epsilon.
- `rms_norm_eps` (`float`, optional, defaults to `1e-5`). RMS normalization epsilon.
- `num_nextn_predict_layers` (`int`, optional, defaults to `1`). Auxiliary prediction layers, accepting zero or one.
- `mtp_loss_scaling_factor` (`float`, optional, defaults to `0.1`). MTP auxiliary loss weight.
- `packing` (`bool`, optional, defaults to `False`). Sequence packing. False is required.
- `scan_layers` (`bool`, optional, defaults to `True`). Flag for repeated-layer scanning.
- `dtype` (`str`, optional, defaults to `"bfloat16"`). Activation computation dtype.
- `weight_dtype` (`str`, optional, defaults to `"float32"`). Stored weight dtype.
- `router_param_dtype` (`str`, optional, defaults to `"float32"`). Router dtype. `float32` is required.
- `remat_policy` (`str`, optional, defaults to `"full"`). Activation recomputation.
- `enable_debug_stats` (`bool`, optional, defaults to `False`). Flag for diagnostic statistics.
- `num_vocab_tiling` (`int`, optional, defaults to `1`). Cross-entropy vocabulary tiles.

</details>

## Derived values and helpers

`qk_head_dim` adds the NoPE and RoPE head dimensions. `linear_num_heads` and `linear_head_dim` expose the linear key head count and dimension. `MTP_LAYER_INDEX` is `45`.

### mini_config

```python
Glm5Config.mini_config(**overrides: Any) -> Glm5Config
```

Returns a configuration with five layers, no MTP, and remat `none` unless overridden. Other dimensions remain large.

<details>
<summary>Parameters</summary>

- `**overrides` (`Any`). Overrides for declared configuration fields.

</details>

### from_maxtext_config

```python
Glm5Config.from_maxtext_config(config: Any, **overrides: Any) -> Glm5Config
```

Converts a compatible resolved configuration or replaces an existing `Glm5Config`. Changing layer count rebuilds omitted schedules.
Missing required source attributes fail rather than silently default.

Returns `Glm5Config`.

<details>
<summary>Parameters</summary>

- `config` (`Any`). A compatible resolved configuration or existing `Glm5Config`.
- `**overrides` (`Any`). Configuration field overrides.

</details>

### validate_no_packing

Import from `lilax.generation.m3_config`.

```python
validate_no_packing(decoder_segment_ids: Any) -> None
```

Accepts unpacked segment IDs. Any value above one raises `ValueError`.

Returns `None`.

<details>
<summary>Parameters</summary>

- `decoder_segment_ids` (`Any`). Segment IDs or `None`.

</details>
