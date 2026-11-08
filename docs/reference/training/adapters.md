# LoRA adapters

Import from `lilax.peft`. See [LoRA guide](../../guides/training/lora.md).

## LoraConfig

Import from `lilax.peft`. Keyword-only dataclass configuring low-rank adapters.

```python
LoraConfig(
	*, r: int, lora_alpha: float, target_modules: str = "",
	lora_weight_qtype: str | None = None, lora_tile_size: int | None = None,
)
```

`r` is the low-rank adapter dimension and `lora_alpha` controls adapter scaling. `target_modules` is a regular expression over Lilax module paths. Empty selects a pattern by model-name prefix, including family-specific attention and feed-forward projections. An unknown prefix uses a generic attention-and-MLP expression and logs a warning. A custom expression is used as supplied, without the automatic scanned-layer path expansion applied to defaults. See the [attention-only Qwen3 example](../../guides/training/lora.md#configure-and-attach-the-adapter).

`lora_weight_qtype` selects the Qwix weight quantization type for QLoRA. `None` leaves weights unquantized by this option. `lora_tile_size` sets the block size along the innermost contraction axis for block-wise quantization. `None` leaves that subchannel tiling unset. Supported quantization types and shape constraints follow the installed Qwix version.

<details>
<summary>Parameters</summary>

- `r` (`int`). Low-rank dimension.
- `lora_alpha` (`float`). Adapter scaling.
- `target_modules` (`str`, optional, defaults to `""`). Module-path regular expression. Empty selects the family-specific default described above.
- `lora_weight_qtype` (`str | None`, optional, defaults to `None`). Qwix weight quantization selection.
- `lora_tile_size` (`int | None`, optional, defaults to `None`). Innermost contraction-axis quantization block size.

</details>

## get_peft_model

```python
get_peft_model(model, peft_config: LoraConfig) -> PeftModel
```

Attaches an adapter to the supplied model's runtime. It materializes immediately when the base is already materialized, otherwise construction remains deferred.

Returns a `PeftModel`.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Base model.
- `peft_config` (`LoraConfig`). Adapter configuration.

</details>

## PeftModel persistence

Import `PeftModel` from `lilax.peft`. Model class extending `MaxTextModel` with adapter persistence.

Both persistence methods accept `"maxtext"` and `"huggingface"`. Other formats raise `ValueError`.

Hugging Face saves write `adapter_model.safetensors` and `adapter_config.json` with PEFT `LORA` and `CAUSAL_LM` metadata and Hugging Face target-module names. With `format="huggingface"`, `model_id` accepts a local adapter directory or a Hub adapter repository ID. `revision` and `token` control Hub file access. Restore occurs during materialization, immediately if the base model is already materialized.

Hugging Face adapter conversion requires registered HF configuration and parameter mappings for the base model's identity. Unsupported identities can raise `KeyError`. A matching adapter file format does not add support for an arbitrary base architecture or target module.

Native adapter saves write an adapter-only Orbax checkpoint at step `0` with adapter configuration metadata. Use a fresh adapter destination.

Hugging Face restore requires LORA metadata with rank and alpha matching the supplied configuration. Empty weights, unsupported dtypes, unknown mappings, duplicate mappings, missing A/B factors, and missing model adapter parameters raise `ValueError`.

LoRA applied to a Qwix QAT model or QAT checkpoint raises `ValueError`. Adapter methods do not provide model merging, arbitrary PEFT techniques, or the complete upstream PEFT API.

### PeftModel.from_pretrained

```python
PeftModel.from_pretrained(model, model_id, peft_config: LoraConfig, *, format="maxtext", revision=None, token=None)
```

Restores adapter weights on a supplied base model, subject to the persistence restrictions above.

Returns a `PeftModel`.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Base model.
- `model_id` (`str | os.PathLike`). Native checkpoint path, or local directory or Hub repository for HF format.
- `peft_config` (`LoraConfig`). Adapter configuration matching saved rank and alpha.
- `format` (`str`, optional, defaults to `"maxtext"`). Adapter checkpoint format.
- `revision` (`str | None`, optional, defaults to `None`). HF adapter revision.
- `token` (`str | bool | None`, optional, defaults to `None`). HF adapter access token.

</details>

### PeftModel.save_pretrained

```python
model.save_pretrained(save_directory, *, format="huggingface")
```

Writes adapter parameters using the selected format and the persistence rules above.
Native saves require a fresh destination.

Returns `None`.

<details>
<summary>Parameters</summary>

- `save_directory` (`str | os.PathLike`). Adapter output directory.
- `format` (`str`, optional, defaults to `"huggingface"`). Adapter checkpoint format.

</details>
