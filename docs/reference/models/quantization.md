# Precision and quantization

Import from `lilax.configuration`. See [quantization guide](../../guides/models/quantization.md).

## PrecisionConfig

Import from `lilax.configuration`. Pydantic model whose keyword fields configure compute and stored-weight precision.

<details>
<summary>Parameters</summary>

- `dtype` (`DType`, optional, defaults to `DType.BFLOAT16`). Activation dtype.
- `grad_dtype` (`DType`, optional, defaults to `DType.FLOAT32`). Gradient dtype.
- `weight_dtype` (`DType`, optional, defaults to `DType.FLOAT32`). Weight dtype.
- `matmul_precision` (`MatmulPrecision`, optional, defaults to `MatmulPrecision.DEFAULT`). Matrix multiplication precision.
- `activations_in_float32` (`bool`, optional, defaults to `False`). Float32 activation conversion before the nonlinearity.
- `dtype_mm` (`str`, optional, defaults to `"float32"`). Vision encoder dtype.
- `unquantized_modules` (`list[str]`, optional, defaults to `[]`). Module names or patterns retaining activation dtype instead of FP8.
- `weight_block_size` (`int | list[int] | None`, optional, defaults to `None`). Weight scaling block dimensions. `None` uses per-tensor scales.

</details>

## Compute recipes

`QuantizationRecipe` is the union of the recipes below and `QwixQATConfig`.

| Recipe | Arguments and defaults | Selects |
| --- | --- | --- |
| `ComputeRecipeConfig` | `quantization=QuantizationType.NONE` | No quantization by default; Flax FP8 ops for `fp8` or `nanoo_fp8`, or the selected Transformer Engine recipe. |
| `AqtInt8Config` | `quantization_local_shard_count=-1` | AQT `int8`. |
| `AqtMixedPrecisionConfig` | `quant_cfg_path=""`; must be nonempty when used | AQT `intmp`, with settings read from the JSON file at `quant_cfg_path`. |
| `JaxMXConfig` | Required `quantization` is `jax_mxfp8` or `jax_mxfp4`; `jax_mx_gradient_mode=""` also accepts `mxfp8`, `mxfp4` | JAX block-scaled MX computation, for train mode only. |
| `ServeFP8WeightConfig` | `act_quantization_calibration_method="absmax"` | `serve_fp8_weight` through the serving FP8-weight backend. |
| `QwixRecipeConfig` | `quantization=QuantizationType.NONE`, `quantize_router_proj=True`, `qwix_bwd_weight_grad_tile_count=1` | Qwix enabled with the selected mode; the default mode is empty. |
| `QwixFP8FullConfig` | Full forward and backward FP8 settings below | `fp8_full` through Qwix model interception. Requires the batch-split schedule to be disabled. |
| `BatchSplitQwixConfig` | `weight_quantization_calibration_method="absmax"`, `act_quantization_calibration_method="absmax"`, `bwd_quantization_calibration_method="absmax"` | `fp8_full` with Qwix enabled. With the batch-split schedule, uses explicitly wired Qwix dot and einsum operations. |
| `BatchSplitManualConfig` | `use_qwix_quantization=False` | `fp8_full` with manual quantization enabled for the batch-split implementation. |

`ComputeRecipeConfig.quantization` accepts `""`, `"fp8"`, `"nanoo_fp8"`, `"te_no_quant"`, `"te_fp8_delayedscaling"`, `"te_fp8_currentscaling"`, `"te_mxfp8"`, `"te_nvfp4"`, and `"te_nvfp4_no_rht"`. `QuantizationType.NONE` has value `""`, not `"none"`. Other values fail Pydantic validation.

The batch-split recipes do not enable `use_batch_split_schedule`. Construction raises `ValueError` unless that schedule is already selected. A nonempty `QwixRecipeConfig` or `QwixFP8FullConfig` recipe instead raises `ValueError` with that schedule enabled. `QwixFP8FullConfig` and `BatchSplitQwixConfig` select the same mode and backend flags, with different calibration fields and schedule requirements.

Passing a `quantization_config` object outside the supported recipe classes to the model factory raises `TypeError` during projection to runtime settings. The `MaxTextConfig` constructor validates recipe types through Pydantic instead. Using `AqtMixedPrecisionConfig` with an empty `quant_cfg_path` raises `ValueError`. A nonempty path is opened and parsed as JSON when the backend configures quantization, so file-access and JSON errors can occur then.

`QwixRecipeConfig.quantization` accepts `""`, `"int4"`, `"int8"`, `"fp4"`, `"fp8_e5m2"`, `"fp8_e4m3"`, `"fp8"`, `"fp8_gpu"`, and `"fp8_nanoo"`. Its backward weight-gradient tile count must be positive.

`JaxMXConfig` requires a JAX build providing the selected scaled-dot mode. `jax_mxfp4` and gradient mode `mxfp4`
require JAX 0.12.0 or newer. An unavailable mode raises `ValueError`. The NVIDIA `26.10-py3` environment pins
JAX `0.11.2.dev20260904`; use a compatible JAX environment for MXFP4.
JAX 0.12.0 executes MXFP4 natively on AMD gfx950 GPUs and uses a dequantize-and-dot fallback elsewhere.
See the [JAX 0.12.0 release notes](https://docs.jax.dev/en/latest/changelog.html#jax-0-12-0-october-19-2026).

### ComputeRecipeConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects no quantization, Flax FP8, or a Transformer Engine recipe. Invalid modes fail Pydantic validation.

<details>
<summary>Parameters</summary>

- `quantization` (`Literal[QuantizationType.NONE, QuantizationType.FP8, QuantizationType.NANOO_FP8, QuantizationType.TE_NO_QUANT, QuantizationType.TE_FP8_DS, QuantizationType.TE_FP8_CS, QuantizationType.TE_MXFP8, QuantizationType.TE_NVFP4, QuantizationType.TE_NVFP4_NO_RHT]`, optional, defaults to `QuantizationType.NONE`). Compute recipe. Accepted string values are listed above.

</details>

### AqtInt8Config

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects AQT int8 computation.

<details>
<summary>Parameters</summary>

- `quantization_local_shard_count` (`int`, optional, defaults to `-1`). Local shard count for AQT quantization.

</details>

### AqtMixedPrecisionConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects AQT mixed precision. Empty paths raise `ValueError`. The backend opens and parses the JSON file during quantization setup.

<details>
<summary>Parameters</summary>

- `quant_cfg_path` (`str`, optional, defaults to `""`). Required nonempty JSON configuration path when used.

</details>

### JaxMXConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects JAX block-scaled MX computation for train mode only.

<details>
<summary>Parameters</summary>

- `quantization` (`Literal[QuantizationType.JAX_MXFP8, QuantizationType.JAX_MXFP4]`). MX mode, `jax_mxfp8` or `jax_mxfp4`.
- `jax_mx_gradient_mode` (`Literal["", "mxfp8", "mxfp4"]`, optional, defaults to `""`). Gradient quantization mode.

</details>

### ServeFP8WeightConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects the serving FP8-weight backend.

<details>
<summary>Parameters</summary>

- `act_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Activation calibration.

</details>

### QwixRecipeConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects Qwix model interception. Nonempty modes cannot use the batch-split schedule.

<details>
<summary>Parameters</summary>

- `quantization` (`Literal[QuantizationType.NONE, QuantizationType.INT4, QuantizationType.INT8, QuantizationType.FP4, QuantizationType.FP8_E5M2, QuantizationType.FP8_E4M3, QuantizationType.FP8, QuantizationType.FP8_GPU, QuantizationType.FP8_NANO_V2]`, optional, defaults to `QuantizationType.NONE`). Qwix compute mode. Accepted string values are listed above.
- `quantize_router_proj` (`bool`, optional, defaults to `True`). Gate projection quantization.
- `qwix_bwd_weight_grad_tile_count` (`PositiveInt`, optional, defaults to `1`). Backward weight-gradient tile count.

</details>

### QwixFP8FullConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects full forward and backward FP8 through Qwix interception. Requires the batch-split schedule disabled.
MTP quantization requires positive MTP layer count. Output-head quantization requires an untied head, one vocabulary tile, and `LogitsConfig.logits_dot_in_fp32=False`. Conflicting settings raise `ValueError`.

<details>
<summary>Parameters</summary>

- `quantize_mtp` (`bool`, optional, defaults to `False`). MTP block quantization.
- `quantize_router_proj` (`bool`, optional, defaults to `True`). MoE gate projection quantization.
- `quantize_logits_proj` (`bool`, optional, defaults to `False`). Output head quantization. Requires [`logits_dot_in_fp32=False`](model-loss.md#logitsconfig).
- `logits_proj_quant_calibration_method` (`str`, optional, defaults to `""`). Output-head weight and activation calibration. Empty inherits the corresponding general methods.
- `logits_proj_bwd_quant_calibration_method` (`str`, optional, defaults to `""`). Output-head gradient calibration. Empty inherits the general backward method.
- `sa_qproj_bwd_bf16` (`bool`, optional, defaults to `False`). Unquantized bfloat16 self-attention query-projection backward products.
- `weight_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Weight calibration.
- `act_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Activation calibration.
- `bwd_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Gradient calibration.
- `drhs_grad_quantization_calibration_method` (`str | None`, optional, defaults to `None`). Output cotangent calibration in weight-gradient products. `None` inherits backward calibration.

</details>

### BatchSplitQwixConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects full FP8 with Qwix operations explicitly wired into batch-split execution. Requires the schedule enabled.

<details>
<summary>Parameters</summary>

- `weight_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Weight calibration.
- `act_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Activation calibration.
- `bwd_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Backward calibration.

</details>

### BatchSplitManualConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects manual full FP8 batch-split execution. Requires the schedule enabled.

<details>
<summary>Parameters</summary>

- `use_qwix_quantization` (`bool`, optional, defaults to `False`). Qwix backend selection flag.

</details>

## Qwix QAT

ODML rejects `tile_size`. Qt rejects nondefault per-channel disabling, fixed ranges, or `strict=False`. QAT cannot be combined with an existing Qwix precision recipe. Conflicting saved and requested QAT configurations raise `ValueError`.

### QwixQATConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures Qwix quantization-aware training.

<details>
<summary>Parameters</summary>

- `rules` (`tuple[QwixQATRule, ...]`). Quantization rules.
- `provider` (`Literal["odml", "qt"]`, optional, defaults to `"odml"`). Qwix provider.
- `disable_per_channel_weights` (`bool`, optional, defaults to `False`). Per-channel weight quantization disabling.
- `fixed_range_for_inputs` (`tuple[float, float] | None`, optional, defaults to `None`). Fixed input range.
- `fixed_range_for_outputs` (`tuple[float, float] | None`, optional, defaults to `None`). Fixed output range.
- `strict` (`bool`, optional, defaults to `True`). Strict quantization application.

</details>

### QwixQATRule

Import from `lilax.configuration`. Pydantic model with keyword fields.

Selects operations and their quantization settings.

<details>
<summary>Parameters</summary>

- `module_path` (`str`, optional, defaults to `".*"`). Module path pattern.
- `op_names` (`tuple[str, ...]`, optional, defaults to `()`). Operation names.
- `weight_qtype` (`str | None`, optional, defaults to `None`). Weight quantization type.
- `act_qtype` (`str | None`, optional, defaults to `None`). Activation quantization type.
- `act_static_scale` (`bool | None`, optional, defaults to `None`). Static activation scale selection.
- `weight_calibration_method` (`str`, optional, defaults to `"absmax"`). Weight calibration.
- `act_calibration_method` (`str | None`, optional, defaults to `None`). Activation calibration.
- `act_batch_axes` (`tuple[int, ...]`, optional, defaults to `(0,)`). Activation batch axes.
- `tile_size` (`int | float | None`, optional, defaults to `None`). Quantization tile size. Unsupported by ODML.

</details>

## KVCacheQuantizationConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures KV-cache quantization.

<details>
<summary>Parameters</summary>

- `quantize_kvcache` (`bool`, optional, defaults to `False`). Cache quantization.
- `kv_quant_axis` (`KvQuantAxis`, optional, defaults to `KvQuantAxis.HEADS_AND_DKV`). Cache scaling axes.
- `kv_quant_dtype` (`Literal["int8", "int4"]`, optional, defaults to `"int8"`). Cache storage type.

</details>

## SparsityConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures N:M weight sparsity.

<details>
<summary>Parameters</summary>

- `weight_sparsity_n` (`int | None`, optional, defaults to `None`). N in the sparsity pattern.
- `weight_sparsity_m` (`int | None`, optional, defaults to `None`). M in the sparsity pattern.
- `weight_sparsity_update_step` (`int`, optional, defaults to `10`). Sparsity update interval.
- `weight_sparsity_start_step` (`int`, optional, defaults to `50`). First sparsity step.

</details>
