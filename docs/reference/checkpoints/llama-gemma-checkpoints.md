# Llama and Gemma checkpoint conversion

Import from `lilax.checkpoints.conversion`. File converters return `None` and write checkpoint files. The conversion uses host memory and CPU subprocesses.

## Llama, Mistral, and Mixtral

`huggingface_checkpoint` selects Hugging Face safetensors input rather than original PyTorch shards. The scanned Llama4 conversion requires it to be true. Unknown family sizes raise `NotImplementedError` in the general, Llama4, and inference-only file converters. The inference-only converter reads original `.pth` shards.

Built-in `model_size` identities differ by converter:

| Converter | Accepted identities |
| --- | --- |
| `convert_llama_or_mistral`, `llama_or_mistral_to_jax_weights` | `llama2-7b`, `llama2-13b`, `llama2-70b`, `llama3-8b`, `llama3-70b`, `llama3.1-8b`, `llama3.1-70b`, `llama3.1-405b`, `llama3.3-70b`, `llama4-17b-16e`, `llama4-17b-128e`, `mistral-7b`, `mixtral-8x7b`, `mixtral-8x22b` |
| `convert_llama4_unscanned` | `llama4-17b-16e`, `llama4-17b-128e` |
| `convert_llama_for_inference` | `llama2-7b`, `llama2-13b`, `llama2-70b`, `llama3-8b`, `llama3-70b`, `llama3.1-8b`, `llama3.1-70b`, `llama3.1-405b`, `llama3.3-70b` |

These functions do not accept a custom architecture mapping. Converter identities do not establish ordinary model loading, training, or export support.

`lora_input_adapters_path` is a local directory containing adapter subdirectories with `adapter_config.json` and `adapter_model.bin`. A `gs://` adapter source raises `NotImplementedError`. Converted adapters are saved under `loras/<adapter-name>` in the destination.

### convert_llama_or_mistral

```python
convert_llama_or_mistral(
	base_model_path: str, maxtext_model_path: str, model_size: str,
	lora_input_adapters_path: str | None = None, huggingface_checkpoint: bool = False,
	use_ocdbt: bool = True, use_zarr3: bool = True, simulated_cpu_devices_count: int = 16,
)
```

Writes scanned family weights and optional adapters through a CPU subprocess.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `maxtext_model_path` (`str`). Output checkpoint storage.
- `model_size` (`str`). An identity in the registry above.
- `lora_input_adapters_path` (`str | None`, optional, defaults to `None`). Local adapter subdirectories described above.
- `huggingface_checkpoint` (`bool`, optional, defaults to `False`). Safetensors rather than original shards. Scanned Llama4 requires true.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Worker host devices.

</details>

### convert_llama4_unscanned

```python
convert_llama4_unscanned(base_model_path: str, maxtext_model_path: str, model_size: str,
	huggingface_checkpoint: bool = False, use_ocdbt: bool = True, use_zarr3: bool = True,
	simulated_cpu_devices_count: int = 16) -> None
```

Writes unscanned Llama4 weights through a CPU subprocess.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `maxtext_model_path` (`str`). Output checkpoint storage.
- `model_size` (`str`). One of the two Llama4 identities above.
- `huggingface_checkpoint` (`bool`, optional, defaults to `False`). HF safetensors input.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Worker host devices.

</details>

### convert_llama_for_inference

```python
convert_llama_for_inference(base_model_path, maxtext_model_path, model_size)
```

Reads original `.pth` shards and writes inference weights through a CPU subprocess.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Original shards.
- `maxtext_model_path` (`str`). Output checkpoint storage.
- `model_size` (`str`). An inference identity listed above.

</details>

### llama_or_mistral_to_jax_weights

```python
llama_or_mistral_to_jax_weights(base_model_path: str, model_size: str, huggingface_ckpt: bool)
```

Returns a parameter dictionary without saving.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `model_size` (`str`). The general converter's identity.
- `huggingface_ckpt` (`bool`). HF safetensors rather than original shards.

</details>

### llama4_unscanned_to_jax_weights

```python
llama4_unscanned_to_jax_weights(base_model_path: str, model_size: str, huggingface_ckpt: bool)
```

Returns an unscanned parameter dictionary without saving. Indexes the general converter's full mapping without the file converter's Llama4-only guard.
Its HF branch requires `rope_type="llama3.1"`, supplied by the built-in mapping only for the two Llama4 identities. Use those identities for Llama4 conversion.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Source weights.
- `model_size` (`str`). A built-in conversion identity.
- `huggingface_ckpt` (`bool`). HF input.

</details>

### llama_lora_to_jax_weights

```python
llama_lora_to_jax_weights(lora_config: dict, model_size: str)
```

Returns adapter weights, not a PEFT model.

<details>
<summary>Parameters</summary>

- `lora_config` (`dict`). `lora_model_path`, rank, and target settings.
- `model_size` (`str`). A built-in base architecture.

</details>

## Export an Orbax state

Orbax export accepts resolved configuration (`runtime.config`). See [configuration objects](../execution/runtime.md#configuration-objects).

### llama_state_to_hf

```python
llama_state_to_hf(training_state, model_size)
```

Returns an HF state dictionary from compatible training state.

<details>
<summary>Parameters</summary>

- `training_state` (`TrainState`). A compatible parameter-bearing training state.
- `model_size` (`str`). The supported layout.

</details>

### convert_llama_orbax_to_hf

```python
convert_llama_orbax_to_hf(hf_model_path, config)
```

Restores a checkpoint, constructs the matching HF model, and calls `save_pretrained`.

Returns `None`.
Root HF-style configuration does not replace the resolved execution configuration.
See [model.save_pretrained](../models/models.md) for the everyday public workflow.

<details>
<summary>Parameters</summary>

- `hf_model_path` (`str`). The HF output directory.
- `config` (`object`). Resolved checkpoint and model settings.

</details>

## Gemma

All three converters return `None`. Unsupported sizes raise `NotImplementedError`.
Sizes identify conversion layouts, not general model support. See [conversion workflow](../../guides/checkpoints/family-checkpoints.md).

### convert_gemma

```python
convert_gemma(base_model_path: str, maxtext_model_path: str, model_size: str) -> None
```

Writes Gemma checkpoint files through a CPU worker.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Original Flax/Orbax Gemma checkpoint directory.
- `maxtext_model_path` (`str`). Destination storage.
- `model_size` (`str`). Model size, `2b`, `7b`, or `9b`.

</details>

### convert_gemma2

```python
convert_gemma2(base_model_path: str, maxtext_model_path: str, model_size: str) -> None
```

Writes Gemma2 checkpoint files through a CPU worker.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Original Flax/Orbax Gemma2 checkpoint directory.
- `maxtext_model_path` (`str`). Destination storage.
- `model_size` (`str`). Model size, `2b`, `9b`, or `27b`.

</details>

### convert_gemma3

```python
convert_gemma3(base_model_path: str, maxtext_model_path: str, model_size: str) -> None
```

Writes Gemma3 checkpoint files through a CPU worker.

Returns `None`.

<details>
<summary>Parameters</summary>

- `base_model_path` (`str`). Original Flax/Orbax Gemma3 checkpoint directory.
- `maxtext_model_path` (`str`). Destination storage.
- `model_size` (`str`). Model size, `4b`, `12b`, or `27b`.

</details>
