# Checkpoint inspection and transforms

Import from `lilax.checkpoints.management`. See [utility guide](../../guides/checkpoints/checkpoint-utilities.md).

Resolved configuration comes from `runtime.config`. See [configuration objects](../execution/runtime.md#configuration-objects). `TrainStateNNX` and `TrainState` name the existing NNX and Flax training-state objects.

## Inspect and compare

Inspection's optional output path receives the printed structure. These functions do not return model objects.

### inspect_hf

```python
inspect_hf(path, format="safetensors", check_dtype=False, output_file="")
```

Prints checkpoint structure. Absent files or unsupported formats raise `ValueError`.

Returns a name-to-description dictionary.

<details>
<summary>Parameters</summary>

- `path` (`str | os.PathLike`). Local checkpoint files.
- `format` (`str`, optional, defaults to `"safetensors"`). Checkpoint format, `safetensors` or `pth`.
- `check_dtype` (`bool`, optional, defaults to `False`). Flag for dtype descriptions.
- `output_file` (`str`, optional, defaults to `""`). Output path for the printed structure when nonempty.

</details>

### inspect_orbax

```python
inspect_orbax(path, check_dtype=False, output_file="")
```

Reads and prints Orbax metadata, filtered to parameters when a parameter root exists.

Returns a name-to-description dictionary.

<details>
<summary>Parameters</summary>

- `path` (`str | os.PathLike`). The Orbax checkpoint.
- `check_dtype` (`bool`, optional, defaults to `False`). Flag for dtype descriptions.
- `output_file` (`str`, optional, defaults to `""`). Output path for the printed structure when nonempty.

</details>

### inspect_maxtext

```python
inspect_maxtext(config, check_dtype=False, output_file="")
```

Launches a CPU worker to describe the abstract architecture.

Returns descriptions.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved architecture settings.
- `check_dtype` (`bool`, optional, defaults to `False`). Flag for dtype descriptions.
- `output_file` (`str`, optional, defaults to `""`). Output path for the printed structure when nonempty.

</details>

### compare_hf_ckpt

```python
compare_hf_ckpt(candidate_path, reference_path="", config=None, max_workers=12, rtol=1e-2, atol=1e-2)
```

Runs a CPU comparison. Explicit references read local safetensors. Otherwise supported resolved configuration selects Hub reference weights.
Missing configuration, unsupported reference model identities, and other CPU comparison failures raise `RuntimeError`.

Returns `True` when keys, shapes, and tensor values match within the supplied tolerances, otherwise `False`.

<details>
<summary>Parameters</summary>

- `candidate_path` (`str | os.PathLike`). Candidate safetensors.
- `reference_path` (`str`, optional, defaults to `""`). Local reference weights.
- `config` (`object | None`, optional, defaults to `None`). Resolved configuration for supported Hub reference weights when no path is supplied.
- `max_workers` (`int`, optional, defaults to `12`). Limit on loading workers.
- `rtol` (`float`, optional, defaults to `1e-2`). Relative tolerance.
- `atol` (`float`, optional, defaults to `1e-2`). Absolute tolerance.

</details>

## Config-driven transforms

These operations take a resolved execution configuration. The root HF-style model configuration does not contain the required resolved checkpoint, mesh, and storage fields. A model runtime can supply `runtime.config` after `runtime.resolve_config()`; the caller must configure compatible source and destination checkpoint settings before invoking a transform.

### generate_decode_checkpoint

```python
generate_decode_checkpoint(config)
```

Reads full training state from `load_full_state_path`, optionally unrolls scanned layers, and saves parameter-only weights at `<checkpoint_dir>/0/items`.

Returns `True`, not the output path. `runtime.initialize(overrides=...)` accepts source and destination settings. `checkpoint_dir` is derived from `base_output_directory` and `run_name`, so a direct override is replaced. For Orbax training checkpoints, `source_checkpoint_layout` is `"orbax"`, `load_parameters_path` must be empty, and `enable_checkpointing` must be true. See [full-state extraction](../../guides/checkpoints/checkpoint-utilities.md#extract-decode-weights-from-full-training-state).

The operation deletes the entire existing destination checkpoint directory before writing. A missing destination or missing optimizer state fails with an assertion. With LoRA input directories, base weights go to `<checkpoint_dir>/base/0/items` and adapters to `<checkpoint_dir>/loras/<adapter_id>/0/items`.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved full-state input, destination, storage, and model settings.

</details>

### quantize_checkpoint

```python
quantize_checkpoint(config, rng=None)
```

Selects unsafe RBG and applies layerwise quantization. Full training-state inputs fail with an assertion.

Returns `None`.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved parameter input and quantization settings.
- `rng` (`jax.Array | None`, optional, defaults to `None`). A JAX key. Omission uses the utility's fixed key.

</details>

### load_and_quantize_checkpoint

```python
load_and_quantize_checkpoint(config)
```

Selects unsafe RBG. Full training-state inputs fail with an assertion.

Returns the inference engine's loaded parameter tree.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved parameter input and quantization settings.

</details>

### reshard_checkpoint

```python
reshard_checkpoint(config, simulated_cpu_devices_count=16)
```

Runs in a CPU subprocess, reads configured source parameters, and writes target mesh layout.

Returns the saved path.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved checkpoint and target mesh settings.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Worker host devices.

</details>

### checkpoint_loop

```python
checkpoint_loop(config)
```

Initializes model and optimizer state, replaces optimizer moments with parameter-derived values, and saves at configured cadence.
Optional restoration runs after saves. Configured page-cache removal invokes a privileged host command.
`lilax.checkpoints.standalone_checkpointer.run(config)` selects the same operation.

Returns the state.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved storage workload, model, cadence, and optional restore settings.

</details>

### add_entropy_to_checkpoint

```python
add_entropy_to_checkpoint(state)
```

Replaces optimizer first and second moments with cosine and sine functions of parameters.

Returns the changed state.
This is a storage workload utility, not an optimizer initialization recommendation.

<details>
<summary>Parameters</summary>

- `state` (`TrainStateNNX | TrainState`). Compatible parameter and optimizer state.

</details>

## Direct conversion setup

Import from `lilax.checkpoints.conversion`. Neither converter accepts a Hub ID as its sole argument.
See [convert_checkpoint](../training/checkpoints.md) for routine path-based conversion.

### convert_to_huggingface

```python
convert_to_huggingface(config, *, override_model_architecture=False, hf_model_path=None)
```

Writes HF files to `config.base_output_directory`, or `tmp/<run_name>` when that directory setting is empty.
Full training state is rejected. Projection layout restrictions apply.

Returns the output-directory string.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved conversion settings.
- `override_model_architecture` (`bool`, optional, defaults to `False`). Architecture override behavior.
- `hf_model_path` (`str | None`, optional, defaults to `None`). Tokenizer and multimodal processor asset source override. A missing or empty value uses the model's registered HF repository.

</details>

### convert_to_maxtext

```python
convert_to_maxtext(config, *, lazy_load_tensors=True, eager_load_method="safetensors", hf_model_path=None,
	revision=None, save_dtype="bfloat16", simulated_cpu_devices_count=16)
```

Imports a registered model into a parameter checkpoint. Unknown model identities raise `ValueError`.

Returns the saved checkpoint path string ending in `0/items`.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved model and checkpoint settings.
- `lazy_load_tensors` (`bool`, optional, defaults to `True`). Lazy source tensor loading.
- `eager_load_method` (`str`, optional, defaults to `"safetensors"`). Eager loading format.
- `hf_model_path` (`str | None`, optional, defaults to `None`). Source identity override.
- `revision` (`str | None`, optional, defaults to `None`). The source revision.
- `save_dtype` (`str`, optional, defaults to `"bfloat16"`). Saved parameter dtype.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Worker host devices.

</details>

### stitch_and_save_checkpoints

```python
stitch_and_save_checkpoints(config: Any, vision_checkpoint_path: str, llm_checkpoint_path: str,
	output_checkpoint_path: str)
```

Restores matching vision and language subtrees, retains initialized unmatched weights, and writes the stitched checkpoint.
Missing vision encoder, decoder, or projector selections fail with an assertion. A terminal output `items` component is removed before saving.
Returns `None`.

See [Omni checkpoint loading](../components/omni-checkpoints.md) for the public runtime producer. A complete pretrained Wan pipeline is not a producer for these utilities.

<details>
<summary>Parameters</summary>

- `config` (`Any`). Resolved Omni component selections and execution settings.
- `vision_checkpoint_path` (`str`). Vision parameters.
- `llm_checkpoint_path` (`str`). Language parameters.
- `output_checkpoint_path` (`str`). Stitched output storage.

</details>
