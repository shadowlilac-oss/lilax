# Experimental Omni checkpoint workflow

Import from `lilax.generation.omni`. See [Omni evaluation guide](../../guides/components/omni-checkpoints.md).

Initialization and decoding accept resolved configuration (`runtime.config`). Import `Runtime` from `lilax.runtime`. See [configuration objects](../execution/runtime.md#configuration-objects).

## load_omni_config

```python
load_omni_config(checkpoint_path, config_path=None, *, devices=None)
```

Resolves the stitched model configuration and prepares single-sample batch settings using the selected devices.

Returns a runtime.

<details>
<summary>Parameters</summary>

- `checkpoint_path` (`str`). The parameter checkpoint path.
- `config_path` (`str | os.PathLike | None`, optional, defaults to `None`). A configuration file. Omission selects bundled Gemma3/Qwen3 stitched settings.
- `devices` (`Sequence[jax.Device] | None`, optional, defaults to `None`). JAX devices. Omission uses available devices.

</details>

## run_evaluation

```python
run_evaluation(checkpoint_path, runtime, num_samples=5, description="Omni Model", max_new_tokens=128)
```

Restores weights, loads ChartQA, and logs questions, reference answers, and generated responses.
Dataset-loading failure is logged and returns early. This operation does not compute an accuracy score.

Returns `None`.

<details>
<summary>Parameters</summary>

- `checkpoint_path` (`str`). The parameter checkpoint path.
- `runtime` (`Runtime`). The runtime produced by `load_omni_config`.
- `num_samples` (`int`, optional, defaults to `5`). Limit on randomly selected validation examples.
- `description` (`str`, optional, defaults to `"Omni Model"`). Label for logged results.
- `max_new_tokens` (`int`, optional, defaults to `128`). Limit on completion length.

</details>

## initialize_model_and_weights

```python
initialize_model_and_weights(config, checkpoint_path, mesh)
```

Constructs the component model and restores its parameter tree.

Returns `(model, params)`.

<details>
<summary>Parameters</summary>

- `config` (`object`). The resolved configuration supplied by this workflow's runtime.
- `checkpoint_path` (`str`). The parameter checkpoint path.
- `mesh` (`jax.sharding.Mesh`). The JAX mesh for initialization and restoration.

</details>

## decode_omni_sample

```python
decode_omni_sample(model, params, config, mesh, tokenizer, prompt_str, pil_image, max_new_tokens=128)
```

Generates one image-conditioned completion. Expanded prompt and vision tokens exceeding the prefill length raise `ValueError`.

Returns the decoded completion string.

<details>
<summary>Parameters</summary>

- `model` (`flax.linen.Module`). The model produced by initialization.
- `params` (`dict`). Its restored parameter tree.
- `config` (`object`). The matching resolved configuration.
- `mesh` (`jax.sharding.Mesh`). The JAX execution mesh.
- `tokenizer` (`transformers.PreTrainedTokenizerBase`). The matching tokenizer with encode and decode operations.
- `prompt_str` (`str`). The prompt text.
- `pil_image` (`PIL.Image.Image`). A PIL image supporting conversion to RGB.
- `max_new_tokens` (`int`, optional, defaults to `128`). Limit on generated tokens within the configured target capacity.

</details>
