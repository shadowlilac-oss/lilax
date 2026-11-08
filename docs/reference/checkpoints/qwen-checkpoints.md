# Qwen checkpoint conversion

Import these functions from `lilax.checkpoints.conversion`. Paths identify checkpoint files on disk. File converters return `None`, spawn a CPU subprocess, and save Orbax weights. Subprocess and file errors propagate.

## File converters

`convert_qwen3_moe` accepts `qwen3-30b-a3b`, `qwen3-235b-a22b`, and `qwen3-480b-a35b`. An unknown size raises `ValueError`. The Next converters use their own Next architecture mapping. The unscanned converter rejects unknown sizes with `NotImplementedError`; the scanned converter raises `ValueError` in its worker. Zero layer and expert limits select the whole configured model.

`simulated_cpu_devices_count` selects the worker's host device count. `use_ocdbt` and `use_zarr3` select checkpoint storage options.

The Next built-in identity is `qwen3-next-80b-a3b`. Scanned conversion with `convert_qwen3_next` currently raises `KeyError` for this identity. Use `convert_qwen3_next_unscanned` for unscanned conversion. This is a limitation of scanned checkpoint conversion.

### convert_qwen3_moe

```python
convert_qwen3_moe(base_model_path: str, maxtext_model_path: str, model_size: str,
	simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True)
```

Writes scanned Orbax weights in a CPU subprocess.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). The HF checkpoint directory.
- `maxtext_model_path` (`str`). The Orbax destination.
- `model_size` (`str`). A supported MoE identity listed above.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Worker host devices.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT storage.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3 storage.

</details>

### convert_qwen3_next

```python
convert_qwen3_next(base_model_path: str, maxtext_model_path: str, model_size: str,
	simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True,
	num_layers_to_convert: int = 0, num_experts_to_convert: int = 0)
```

Starts scanned conversion in a CPU subprocess. The built-in identity currently fails with `KeyError` as described above.

Returns `None` after a successful conversion writes Orbax weights.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). The HF source.
- `maxtext_model_path` (`str`). The Orbax destination.
- `model_size` (`str`). Model identity, `qwen3-next-80b-a3b`.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Worker host devices.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `num_layers_to_convert` (`int`, optional, defaults to `0`). Limit on layers. Zero selects all.
- `num_experts_to_convert` (`int`, optional, defaults to `0`). Limit on experts. Zero selects all.

</details>

### convert_qwen3_next_unscanned

```python
convert_qwen3_next_unscanned(base_model_path: str, maxtext_model_path: str, model_size: str,
	simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True)
```

Writes unscanned Orbax weights in a CPU subprocess.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). The HF source.
- `maxtext_model_path` (`str`). The Orbax destination.
- `model_size` (`str`). Model identity, `qwen3-next-80b-a3b`.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Worker host devices.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.

</details>

## Parameter dictionaries

These functions expect complete family-specific architecture mappings, not a root model configuration object. Prefer a file converter when you want the built-in mapping. See [conversion workflow](../../guides/checkpoints/family-checkpoints.md).

### qwen3_moe_to_maxtext_weights

```python
qwen3_moe_to_maxtext_weights(base_model_path: str, model_params: dict) -> dict
```

Reads HF tensors.

Returns a scanned parameter dictionary.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). HF tensors.
- `model_params` (`dict`). Architecture mapping with required keys `num_hidden_layers`, `num_experts`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, and `moe_intermediate_size`.

</details>

### qwen3_next_to_maxtext_weights

```python
qwen3_next_to_maxtext_weights(base_model_path: str, model_params: dict, args) -> dict
```

Returns a scanned parameter dictionary. The built-in Next architecture mapping raises `KeyError` with this routine.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). HF tensors.
- `model_params` (`dict`). Next architecture mapping with `num_layers`, `emb_dim`, `num_q_heads`, `num_kv_heads`, `num_experts`, `head_dim`, `moe_intermediate_size`, `inhomogeneous_layer_cycle_interval`, `gdn_num_value_heads`, `gdn_num_key_heads`, `gdn_key_head_dim`, `gdn_value_head_dim`, and `gdn_conv_kernel_dim`.
- `args` (`argparse.Namespace`). Conversion limits with `num_layers_to_convert` and `num_experts_to_convert` attributes.

</details>

### qwen3_next_unscanned_to_maxtext_weights

```python
qwen3_next_unscanned_to_maxtext_weights(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process)
```

Returns an unscanned parameter dictionary.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). HF tensors.
- `model_size` (`str`). The conversion layout.
- `model_params` (`dict`). Complete Next architecture dimensions.
- `mem_info` (`psutil.Process`). Process used for memory logging.

</details>

### qwen3_next_unscanned_to_jax_weights

```python
qwen3_next_unscanned_to_jax_weights(base_model_path: str, model_size: str)
```

Selects the built-in architecture without saving.

Returns unscanned weights.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). HF tensors.
- `model_size` (`str`). The built-in Next architecture.

</details>
