# Model configuration

Import these types from `lilax.configuration`. See [configuration guide](../../guides/models/configuration.md).

## MaxTextConfig

Import from `lilax.configuration`. `PreTrainedConfig` subclass with an explicit keyword-only initializer.

`AutoConfig.from_pretrained(...)` and `model.config` expose this model configuration. Runtime-source configuration (`model.runtime.public_config`) uses a different `MaxTextConfig` identity. Resolved configuration (`runtime.config`) is also distinct. See [configuration objects](../execution/runtime.md#configuration-objects).

```python
MaxTextConfig(
	*, model_name: str, vocab_size: int, hidden_size: int, intermediate_size: int,
	num_hidden_layers: int, num_attention_heads: int, num_key_value_heads: int, head_dim: int,
	max_position_embeddings: int = 2048, enable_dropout: bool = True, dropout_rate: float = 0.0,
	moe_intermediate_size: int | None = None,
	architecture_config: ArchitectureConfig | None = None, attention_config: AttentionConfig | None = None,
	position_config: PositionConfig | None = None, moe_config: MoEConfig | None = None,
	qwen_config: QwenConfig | None = None, engram_config: EngramConfig | None = None,
	ming_config: MingConfig | None = None, hyperconnection_config: HyperconnectionConfig | None = None,
	vision_config: VisionConfig | None = None, vision_projector_config: VisionProjectorConfig | None = None,
	audio_config: AudioConfig | None = None, mtp_config: MTPConfig | None = None,
	logits_config: LogitsConfig | None = None, precision_config: PrecisionConfig | None = None,
	quantization_config: QuantizationRecipe | None = None,
	kv_cache_quantization_config: KVCacheQuantizationConfig | None = None,
	sparsity_config: SparsityConfig | None = None, kernel_config: KernelConfig | None = None,
	layout_config: ModelLayoutConfig | None = None, **kwargs,
)
```

Stores effective dimensions and focused model domains. Run and execution keys raise `TypeError`.

`to_dict()` returns JSON-compatible fields, including recipe type metadata. `to_diff_dict()` omits unchanged HF metadata. Configuration persistence follows the upstream [PreTrainedConfig API](https://huggingface.co/docs/transformers/main_classes/configuration).

<details>
<summary>Parameters</summary>

- `model_name` (`str`). Registered model identity.
- `vocab_size` (`int`). Vocabulary size.
- `hidden_size` (`int`). Effective embedding width.
- `intermediate_size` (`int`). Effective MLP intermediate width.
- `num_hidden_layers` (`int`). Decoder layer count.
- `num_attention_heads` (`int`). Query head count.
- `num_key_value_heads` (`int`). Key and value head count.
- `head_dim` (`int`). Head width.
- `max_position_embeddings` (`int`, optional, defaults to `2048`). Maximum supported positions.
- `enable_dropout` (`bool`, optional, defaults to `True`). Training dropout.
- `dropout_rate` (`float`, optional, defaults to `0.0`). Dropout probability.
- `moe_intermediate_size` (`int | None`, optional, defaults to `None`). Routed expert intermediate width.
- `architecture_config` (`ArchitectureConfig | None`, optional, defaults to `None`). Architecture settings.
- `attention_config` (`AttentionConfig | None`, optional, defaults to `None`). Attention semantics.
- `position_config` (`PositionConfig | None`, optional, defaults to `None`). Position settings.
- `moe_config` (`MoEConfig | None`, optional, defaults to `None`). Expert routing.
- `qwen_config` (`QwenConfig | None`, optional, defaults to `None`). Qwen family settings.
- `engram_config` (`EngramConfig | None`, optional, defaults to `None`). Engram settings.
- `ming_config` (`MingConfig | None`, optional, defaults to `None`). Ming settings.
- `hyperconnection_config` (`HyperconnectionConfig | None`, optional, defaults to `None`). Hyperconnections.
- `vision_config` (`VisionConfig | None`, optional, defaults to `None`). Vision tower.
- `vision_projector_config` (`VisionProjectorConfig | None`, optional, defaults to `None`). Vision-to-text projection.
- `audio_config` (`AudioConfig | None`, optional, defaults to `None`). Audio encoder.
- `mtp_config` (`MTPConfig | None`, optional, defaults to `None`). Multi-token prediction.
- `logits_config` (`LogitsConfig | None`, optional, defaults to `None`). Output logits and loss.
- `precision_config` (`PrecisionConfig | None`, optional, defaults to `None`). Compute and storage precision.
- `quantization_config` (`QuantizationRecipe | None`, optional, defaults to `None`). Compute or QAT recipe.
- `kv_cache_quantization_config` (`KVCacheQuantizationConfig | None`, optional, defaults to `None`). Cache quantization.
- `sparsity_config` (`SparsityConfig | None`, optional, defaults to `None`). Weight sparsity.
- `kernel_config` (`KernelConfig | None`, optional, defaults to `None`). Attention and expert kernels.
- `layout_config` (`ModelLayoutConfig | None`, optional, defaults to `None`). Layer layout.
- `**kwargs` (`Any`). Additional HF metadata.

</details>

## Domain composition

| Model field | Public type | Responsibility |
| --- | --- | --- |
| `architecture_config` | `ArchitectureConfig` | Decoder block, activations, normalization, projection layout |
| `attention_config` | `AttentionConfig` | Attention semantics and family variants |
| `position_config` | `PositionConfig` | Position embeddings, RoPE, YaRN, and multimodal positions |
| `moe_config` | `MoEConfig` | Expert routing and DeepSeek expert configuration |
| `qwen_config` | `QwenConfig` | Qwen next-generation family settings |
| `engram_config` | `EngramConfig` | Engram settings |
| `ming_config` | `MingConfig` | Ming settings |
| `hyperconnection_config` | `HyperconnectionConfig` | Manifold-constrained hyperconnections |
| `vision_config` | `VisionConfig` | Vision tower and freezing |
| `vision_projector_config` | `VisionProjectorConfig` | Vision-to-text projection |
| `audio_config` | `AudioConfig` | Audio encoder and freezing |
| `mtp_config` | `MTPConfig` | Multi-token prediction |
| `logits_config` | `LogitsConfig` | Logit computation and loss behavior |
| `precision_config` | `PrecisionConfig` | Activation, gradient, and weight types |
| `quantization_config` | `QuantizationRecipe` | [Compute and QAT recipes](quantization.md) |
| `kv_cache_quantization_config` | `KVCacheQuantizationConfig` | Cache quantization |
| `sparsity_config` | `SparsityConfig` | Weight sparsity |
| `kernel_config` | `KernelConfig` | Attention and expert kernels |
| `layout_config` | `ModelLayoutConfig` | Layer layout |

Loaded configurations retain family-specific settings. A default domain object is not a substitute for a different model family's loaded configuration.

Focused fields are listed under [expert routing](moe.md), [Qwen](qwen.md), [vision and audio](media-models.md), [logits and prediction](model-loss.md), [Engram](engram.md), [Ming](ming.md), and [hyperconnections](hyperconnections.md). See [family configuration tasks](../../guides/models/family-settings.md) for usage.

## Architecture and scaling

### ArchitectureConfig

Import from `lilax.configuration`. Pydantic model whose keyword fields select decoder structure.

<details>
<summary>Parameters</summary>

- `decoder_block` (`DecoderBlockType`, optional, defaults to `DecoderBlockType.LLAMA2`). Family decoder implementation. See [enum values](configuration-enums.md).
- `dense_init_scale` (`float`, optional, defaults to `1.0`). Dense initialization scale.
- `attention_output_dim` (`int`, optional, defaults to `-1`). Positive values override attention output width.
- `global_head_dim` (`int`, optional, defaults to `0`). Global attention query and key width.
- `mlp_activations` (`list[str]`, optional, defaults to `["silu", "linear"]`). MLP activation functions.
- `mlp_activations_limit` (`float`, optional, defaults to `-1.0`). MLP activation upper bound. `-1.0` disables clipping.
- `normalization_layer_epsilon` (`float`, optional, defaults to `1e-5`). Normalization epsilon.
- `fused_qkv` (`bool`, optional, defaults to `False`). Fused Q, K, and V projections where supported.
- `attention_bias` (`bool`, optional, defaults to `False`). Learnable Q, K, and V biases.
- `fused_mlp` (`bool`, optional, defaults to `False`). Fused MLP layers where supported.
- `qk_norm_with_scale` (`bool`, optional, defaults to `True`). Learned query/key normalization scales.
- `v_norm_with_scale` (`bool`, optional, defaults to `True`). Learned value normalization scale.

</details>

### ModelLayoutConfig

Import from `lilax.configuration`. Pydantic model with a keyword field selecting layer layout.

<details>
<summary>Parameters</summary>

- `inhomogeneous_layer_cycle_interval` (`int`, optional, defaults to `1`). Layer cycle interval.

</details>

### ModelScalingConfig

Import from `lilax.configuration`. Pydantic model whose keyword fields describe base dimensions and scale.

<details>
<summary>Parameters</summary>

- `global_parameter_scale` (`int`, optional, defaults to `1`). Global dimension scale.
- `base_emb_dim` (`int`, optional, defaults to `2048`). Base embedding width.
- `base_num_query_heads` (`int`, optional, defaults to `16`). Base query head count.
- `base_num_kv_heads` (`int`, optional, defaults to `16`). Base key/value head count.
- `base_mlp_dim` (`int`, optional, defaults to `7168`). Base feed-forward width.
- `base_num_decoder_layers` (`int`, optional, defaults to `16`). Base decoder layer count.
- `base_moe_mlp_dim` (`int`, optional, defaults to `-1`). Base expert intermediate width.

</details>

### ModelScalingConfig.apply

```python
scaling.apply(config)
```

Returns a configuration with scaled effective dimensions. Scales that are not positive powers of two raise `ValueError`.

<details>
<summary>Parameters</summary>

- `config` (`MaxTextConfig`). Model configuration to scale.

</details>

## Attention, position, and kernels

### AttentionConfig

Import from `lilax.configuration`. Pydantic model composing attention semantics through keyword fields. See [attention fields](attention.md).

<details>
<summary>Parameters</summary>

- `general` (`AttentionParameters | None`, optional, defaults to `None`). General attention.
- `moba` (`MoBaConfig | None`, optional, defaults to `None`). Mixture of Block Attention.
- `mla` (`MlaAttentionConfig | None`, optional, defaults to `None`). MLA projections and caching.
- `compressed` (`CompressedAttentionConfig | None`, optional, defaults to `None`). Compressed attention.
- `indexer` (`AttentionIndexerConfig | None`, optional, defaults to `None`). Sparse indexer.
- `llama4` (`Llama4AttentionConfig | None`, optional, defaults to `None`). Llama4 attention.

</details>

### PositionConfig

Import from `lilax.configuration`. Pydantic model composing positional encodings through keyword fields.

<details>
<summary>Parameters</summary>

- `embedding` (`PositionalEmbeddingConfig | None`, optional, defaults to `None`). Position embeddings.
- `rope` (`RopeConfig | None`, optional, defaults to `None`). RoPE.
- `yarn` (`YarnRopeConfig | None`, optional, defaults to `None`). YaRN scaling.
- `use_mrope` (`bool`, optional, defaults to `False`). Qwen3-Omni multidimensional RoPE.
- `mrope_section` (`list[int]`, optional, defaults to `[24, 20, 20]`). Temporal, height, and width dimensions.
- `position_id_per_seconds` (`int`, optional, defaults to `25`). Temporal position granularity in tokens per second.

</details>

### KernelConfig

Import from `lilax.configuration`. Pydantic model selecting attention and expert implementations through keyword fields.

`KernelConfig.attention` is a string. The attention dispatcher recognizes the following values. Unknown names raise `ValueError` when dispatched.

| Value | Effect and restrictions |
| --- | --- |
| `"autoselected"` | Uses dot-product attention for CPU, autoregressive decoding, or query lengths below 128. Otherwise selects the hardware's flash path. |
| `"dot_product"` | Computes attention with explicit query-key and attention-value products. Available on CPU, GPU, and TPU. |
| `"flash"` | Uses Splash on TPU and the GPU flash implementation on GPU. Explicit TPU flash rejects autoregressive decode. The GPU decode path uses dot-product attention. |
| `"cudnn_flash_te"` | Uses cuDNN through NVIDIA Transformer Engine on GPU. Rejects autoregressive decode. |
| `"cudnn_flash_jax"` | Uses JAX's cuDNN fused attention on GPU, including its autoregressive path. |
| `"paged"` | Selects the dot-product dispatch path. This name alone does not enable the vLLM ragged paged attention integration. |
| `"vllm_rpa"`, `"vllm_batched_rpa"` | Select the vLLM TPU ragged paged attention integration for inference. Require `vllm-tpu` and serving cache metadata. The batched variant sets `USE_BATCHED_RPA_KERNEL="1"`. |

`use_ragged_attention` selects TPU or GPU ragged attention during autoregressive decoding before ordinary kernel dispatch. MoBA requires `"dot_product"`. DeepSeek4 requires `"dot_product"` or `"flash"` with `use_tokamax_splash=True`. Block-diffusion attention accepts `"dot_product"` on other hardware, but `"flash"` and `"autoselected"` require TPU. Kernel selection does not change the loaded model's attention semantics.

Kernel-selection names overridden as class variables in `AttentionParameters` are not constructor fields. Use `KernelConfig` for those controls.

<details>
<summary>Parameters</summary>

- `attention` (`str`, optional, defaults to `"autoselected"`). Implementation selected from the table above.
- `use_ragged_attention` (`bool`, optional, defaults to `False`). Ragged attention kernels.
- `enable_padding_causal_mask` (`bool`, optional, defaults to `True`). Retained flag. Current attention execution does not read it, so changing it does not affect masking.
- `use_tokamax_splash` (`bool`, optional, defaults to `False`). Tokamax Splash.
- `use_jax_splash` (`bool`, optional, defaults to `False`). JAX Splash.
- `splash` (`SplashAttentionConfig | None`, optional, defaults to `None`). Quantization and segment masking.
- `moe` (`MoEKernelsConfig | None`, optional, defaults to `None`). Expert kernels.

</details>

### SplashAttentionConfig

Import from `lilax.configuration`. Pydantic model configuring Splash quantization and masking through keyword fields.

<details>
<summary>Parameters</summary>

- `experimental_sa_quant_q_fp8` (`bool`, optional, defaults to `False`). Experimental FP8 query quantization.
- `experimental_sa_quant_k_fp8` (`bool`, optional, defaults to `False`). Experimental FP8 key quantization.
- `sa_ignore_segment_ids` (`bool`, optional, defaults to `False`). Segment ID exclusion in Splash attention.

</details>

## Family composition

`QwenConfig` is the union of `Qwen3NextConfig` and `Qwen3_8FlashNextConfig`.
See [vision and audio keyword fields](media-models.md) for tower selection, dimensions, and freezing.
`VisionEncoderBlockType.NONE` selects no vision encoder block.

### MoEConfig

Import from `lilax.configuration`. Pydantic model composing general and DeepSeek expert settings through keyword fields.

<details>
<summary>Parameters</summary>

- `general` (`MoEParameters | None`, optional, defaults to `None`). General expert routing.
- `deepseek` (`DeepSeekMoEConfig | None`, optional, defaults to `None`). DeepSeek settings.

</details>

## Enums

The [enum reference](configuration-enums.md) lists all names and values by type.

Public enum types are `DType`, `MatmulPrecision`, `QuantizationType`, `KvQuantAxis`, `DecoderBlockType`, `RopeType`, `ProfilerType`, `VisionEncoderBlockType`, and `TEGroupedGemmQuantizationType`. Use members matching the selected model and hardware. Serialization stores their values.

`DType` members are `BFLOAT16`, `FLOAT32`, `FLOAT16`, `FLOAT8_E4M3FN`, and `FLOAT8_E5M2`. Their values are the lowercase dtype names. `MatmulPrecision` members are `DEFAULT`, `HIGH`, `HIGHEST`, `BFLOAT16`, and `FLOAT32`, with lowercase values.

`KvQuantAxis` values are `""`, `"dkv"`, and `"heads_and_dkv"`. `RopeType` values are `"default"`, `"llama3.1"`, and `"yarn"`. `ProfilerType` values are `""`, `"xplane"`, and `"nsys"`. `TEGroupedGemmQuantizationType` values are `""`, `"te_no_quant"`, and `"te_mxfp8"`.
