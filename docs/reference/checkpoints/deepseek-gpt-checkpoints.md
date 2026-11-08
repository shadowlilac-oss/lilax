# DeepSeek and GPT checkpoint conversion

Import from `lilax.checkpoints.conversion`.

## DeepSeek

Config-driven conversion accepts resolved configuration (`runtime.config`). See [configuration objects](../execution/runtime.md#configuration-objects).

File converters spawn CPU workers, save Orbax weights, and return `None`. Weight functions return parameter dictionaries. `model_params=None` selects the built-in family architecture mapping; an explicit mapping replaces it. The scanned converter rejects an unknown size without an explicit mapping with `NotImplementedError`. `enable_mtp` includes supported multi-token prediction weights. It does not add MTP to a checkpoint that lacks them.

The scanned and unscanned functions share these built-in `model_size` identities: `deepseek2-16b`, `deepseek3-671b`, `kimi-k2-1t`, `kimi-k2-thinking`, `kimi-k2.5-text`, and `kimi-k2.6-text`. These are converter layouts, not a declaration of Hub loading, training, or export support. An explicit `model_params` mapping bypasses the built-in size lookup.

### convert_deepseek

```python
convert_deepseek(base_model_path: str, maxtext_model_path: str, model_size: str,
	simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True,
	enable_mtp: bool = False, weight_dtype: str = "float16", model_params: dict | None = None) -> None
```

Writes scanned Orbax weights in a CPU subprocess.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). HF source weights.
- `maxtext_model_path` (`str`). Orbax output.
- `model_size` (`str`). A listed identity unless architecture is explicit.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Host devices.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `enable_mtp` (`bool`, optional, defaults to `False`). Flag for inclusion of supported source MTP weights.
- `weight_dtype` (`str`, optional, defaults to `"float16"`). Converted weight dtype.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override for the built-in lookup.

</details>

### convert_deepseek_unscanned

```python
convert_deepseek_unscanned(base_model_path: str, maxtext_model_path: str, model_size: str,
	simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True,
	model_params: dict | None = None) -> None
```

Writes unscanned Orbax weights in a CPU subprocess.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `maxtext_model_path` (`str`). Output storage.
- `model_size` (`str`). A built-in identity.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Host devices.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override for the built-in lookup.

</details>

### deepseek_to_jax_weights

```python
deepseek_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None,
	enable_mtp: bool = False, weight_dtype: str = "float16", scanned: bool = True) -> dict
```

Returns converted parameters without saving.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `model_size` (`str`). A built-in identity.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override.
- `enable_mtp` (`bool`, optional, defaults to `False`). Flag for inclusion of supported MTP weights.
- `weight_dtype` (`str`, optional, defaults to `"float16"`). Converted dtype.
- `scanned` (`bool`, optional, defaults to `True`). Scanned layout.

</details>

### deepseek_unscanned_to_jax_weights

```python
deepseek_unscanned_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None) -> dict
```

Returns unscanned parameters without saving.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `model_size` (`str`). A built-in identity.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override.

</details>

## GPT-OSS and PaxML

Both scanned and unscanned GPT-OSS functions accept `gpt-oss-20b` and `gpt-oss-120b` with `model_params=None`. An explicit `model_params` mapping replaces the built-in lookup.

### convert_gpt_oss

```python
convert_gpt_oss(base_model_path: str, maxtext_model_path: str, model_size: str,
	simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True,
	model_params: dict | None = None) -> None
```

Saves scanned weights in a CPU worker.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `maxtext_model_path` (`str`). Output storage.
- `model_size` (`str`). A GPT-OSS identity above.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Host devices.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override.

</details>

### convert_gpt_oss_unscanned

```python
convert_gpt_oss_unscanned(base_model_path: str, maxtext_model_path: str, model_size: str,
	simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True,
	model_params: dict | None = None) -> None
```

Saves unscanned weights in a CPU worker.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `maxtext_model_path` (`str`). Output storage.
- `model_size` (`str`). A GPT-OSS identity above.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Host devices.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override.

</details>

### gpt_oss_to_jax_weights

```python
gpt_oss_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None) -> dict
```

Returns scanned parameters without saving.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `model_size` (`str`). A GPT-OSS identity.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override.

</details>

### gpt_oss_unscanned_to_jax_weights

```python
gpt_oss_unscanned_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None) -> dict
```

Returns unscanned parameters without saving.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `model_size` (`str`). A GPT-OSS identity.
- `model_params` (`dict | None`, optional, defaults to `None`). Architecture mapping override.

</details>

### convert_gpt3_from_paxml_with_defaults

```python
convert_gpt3_from_paxml_with_defaults(paxml_ckpt_path, maxtext_model_name, base_output_directory, run_name)
```

Resolves conversion settings, converts parameters and optimizer state, and saves a training checkpoint.
Initializes JAX, reads cloud storage, and may exit after preemption. Does not return a model wrapper.

<details>
<summary>Parameters</summary>

- `paxml_ckpt_path` (`str`). Source path beginning with `gs://`.
- `maxtext_model_name` (`str`). The target model layout.
- `base_output_directory` (`str`). Output storage.
- `run_name` (`str`). The output run directory.

</details>

### convert_gpt3_from_paxml

```python
convert_gpt3_from_paxml(paxml_ckpt_path, config)
```

Converts parameters and optimizer state and saves a training checkpoint. Non-GCS input raises `ValueError`.
Initializes JAX, reads cloud storage, and may exit after preemption. Does not return a model wrapper.

<details>
<summary>Parameters</summary>

- `paxml_ckpt_path` (`str`). The `gs://` source.
- `config` (`object`). Resolved execution conversion settings.

</details>

## Dequantization

See [conversion workflow](../../guides/checkpoints/family-checkpoints.md). Tensor functions below require actual PyTorch tensor inputs.

### dequantize_deepseek_checkpoint

```python
dequantize_deepseek_checkpoint(input_path: str, output_path: str, cache_file_num: int = 2)
```

Reads and rewrites safetensors shards and their index. Identical input and output paths raise `ValueError`.
Returns `None`.

<details>
<summary>Parameters</summary>

- `input_path` (`str`). Indexed source shards.
- `output_path` (`str`). A separate output directory.
- `cache_file_num` (`int`, optional, defaults to `2`). Limit on retained source shard files.

</details>

### dequantize_mxfp4_checkpoint

```python
dequantize_mxfp4_checkpoint(input_path: str, output_path: str, target_dtype: torch.dtype, cache_size: int)
```

Rewrites MXFP4 safetensors shards and their index using CUDA-loaded tensors. Changes the default PyTorch dtype.
The output directory must exist.

Returns `None`.

<details>
<summary>Parameters</summary>

- `input_path` (`str`). Source shards.
- `output_path` (`str`). Output storage.
- `target_dtype` (`torch.dtype`). Output and default dtype.
- `cache_size` (`int`). Limit on cached source shard files.

</details>

### dequantize_fp8

```python
dequantize_fp8(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor
```

Applies block scales to a two-dimensional FP8 matrix.

Returns bfloat16 weights.

<details>
<summary>Parameters</summary>

- `x` (`torch.Tensor`). The two-dimensional weight matrix.
- `s` (`torch.Tensor`). The two-dimensional scale grid.
- `block_size` (`int`, optional, defaults to `128`). The scale block width and height.

</details>

### unpack_fp4

```python
unpack_fp4(packed: torch.Tensor) -> torch.Tensor
```

Returns two unpacked FP4 values per byte through the E2M1 value table.

<details>
<summary>Parameters</summary>

- `packed` (`torch.Tensor`). Packed bytes.

</details>

### dequantize_deepseek_mxfp4

```python
dequantize_deepseek_mxfp4(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor
```

Unpacks MXFP4 values and applies scales. Nondivisible weight and scale grids raise `ValueError`.
Returns bfloat16 weights.

<details>
<summary>Parameters</summary>

- `quantized` (`torch.Tensor`). Packed or already expanded values.
- `scales` (`torch.Tensor`). The matching scale grid.

</details>

### e2m1_to_fp

```python
e2m1_to_fp(x: torch.Tensor)
```

Returns floating-point values decoded from E2M1 codes.

<details>
<summary>Parameters</summary>

- `x` (`torch.Tensor`). E2M1 codes.

</details>

### dequantize_mxfp4

```python
dequantize_mxfp4(blocks_2x_e2m1: torch.Tensor, scales_e8m0: torch.Tensor, data_type: torch.dtype)
```

Returns unpacked, scaled values cast to `data_type`.

<details>
<summary>Parameters</summary>

- `blocks_2x_e2m1` (`torch.Tensor`). Two packed E2M1 values per byte.
- `scales_e8m0` (`torch.Tensor`). Block scales.
- `data_type` (`torch.dtype`). Output dtype.

</details>

### dequantize_pack_quantized_int4

```python
dequantize_pack_quantized_int4(packed, scale, out_shape)
```

Returns bfloat16 weights. Input width must be divisible by the INT4 group size or raises `ValueError`.

<details>
<summary>Parameters</summary>

- `packed` (`torch.Tensor`). Packed PyTorch INT4 weights.
- `scale` (`torch.Tensor`). Matching PyTorch group scales.
- `out_shape` (`tuple[int, int]`). Output and input feature dimensions.

</details>
