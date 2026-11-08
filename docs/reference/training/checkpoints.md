# Checkpoints and conversion

Import configuration classes and `convert_checkpoint` from `lilax.checkpoints`. See [save and resume guide](../../guides/training/checkpoints.md).

## Model saving

```python
model.save_pretrained(save_directory, *, format="maxtext", export_config=None)
```

Saves weight-only checkpoints. `"huggingface"` requests portable export. Unsupported formats raise `ValueError`.

The default `format="maxtext"` save writes Orbax weights under `save_directory/_lilax_model`. Repeated saves replace that child only when `maxtext_easy_config_metadata.json` declares Lilax ownership. An existing undeclared child causes the save to fail without forced replacement. Saves update `config.json` and model metadata while preserving unrelated files, checkpoints, and logs in the destination. Processing-class saves update their named files there.

Public model loading and conversion accept the destination root and still load legacy flat native checkpoints. After successful native and Hugging Face saves to the same local directory, public loading and automatic conversion select the latest saved format. External Orbax clients read native weights at `save_directory/_lilax_model`, including the retained native weights after Hugging Face export. A later native save can replace that child.

HF export writes a full temporary native weight copy under `~/tmp/lilax/` on the home filesystem, then removes the temporary directory when export exits. Resolving the HF configuration can download a repository snapshot unless a local or cached source satisfies the request.

Unsupported export identities raise `ValueError`. Qwix QAT export and padded GPT-OSS expert output weights raise `NotImplementedError`. Projection layout restrictions also apply.

Returns `None` for native saving. Successful Hugging Face export returns its output-directory string.

<details>
<summary>Parameters</summary>

- `save_directory` (`str | os.PathLike`). Destination root.
- `format` (`str`, optional, defaults to `"maxtext"`). Native checkpoint or `"huggingface"` portable export.
- `export_config` (`HuggingFaceExportConfig | None`, optional, defaults to `None`). HF export options. `None` uses default options.

</details>

### HuggingFaceExportConfig

Import from `lilax.checkpoints`. Pydantic model with keyword fields.

Selects the HF configuration snapshot for export. The model's registered repository supplies it when no path is given.

<details>
<summary>Parameters</summary>

- `hf_model_path` (`str | Path | None`, optional, defaults to `None`). HF model configuration source.
- `revision` (`str | None`, optional, defaults to `None`). Snapshot revision.
- `token` (`str | bool | None`, optional, defaults to `None`). Snapshot access authentication.
- `cache_dir` (`str | Path | None`, optional, defaults to `None`). Snapshot cache.
- `local_files_only` (`bool`, optional, defaults to `False`). Local cached files only.
- `override_model_architecture` (`bool`, optional, defaults to `False`). HF architecture override during export.

</details>

## TrainingCheckpointConfig

Import from `lilax.checkpoints`. Pydantic model with keyword fields.

Pass this object as `args.checkpoint_config`. LoRA-only selection is supported by compatible post-training workflows. `Trainer` and the `"maxtext"` GRPO engine reject it.

<details>
<summary>Parameters</summary>

- `enabled` (`bool`, optional, defaults to `True`). Checkpoint saving.
- `async_checkpointing` (`bool`, optional, defaults to `True`). Asynchronous saving.
- `max_num_checkpoints_to_keep` (`int | None`, optional, defaults to `None`). Latest checkpoints retained. `None` sets no limit.
- `save_checkpoint_on_start` (`bool`, optional, defaults to `True`). Checkpoint save at training start.
- `save_checkpoint_on_completion` (`bool`, optional, defaults to `True`). Checkpoint save at completion.
- `enable_continuous_checkpointing` (`bool`, optional, defaults to `False`). Orbax continuous save policy instead of a fixed interval.
- `enable_autocheckpoint` (`bool`, optional, defaults to `False`). Additional preemption saves with fixed-step policy when continuous checkpointing is disabled.
- `save_only_lora_params` (`bool | None`, optional, defaults to `None`). Adapter-only saving. `None` retains workflow selection.
- `restore_only_lora_params` (`bool | None`, optional, defaults to `None`). Adapter-only restoration. `None` retains workflow selection.

</details>

## Resume selection

`trainer.train(resume_from_checkpoint=True)` selects the latest checkpoint under `<output_dir>/<run_name>/checkpoints/`. Saved steps occupy numeric subdirectories. Reusing `output_dir` alone does not identify the previous run. An empty `run_name` resolves to `JOBSET_NAME` when set, otherwise `<model_name>_<YYYY-MM-DD-HH-MM>`. Resuming with `True` requires the same resolved run name. No available checkpoint raises `ValueError`.

A nonempty string selects the `checkpoints/` root itself, not `output_dir` or the run directory. A final numeric path component selects a specific step within its parent directory. `False` and `None` start without explicit resume selection.

`checkpoint_step` selects a nonnegative integer step. Boolean or negative values raise `ValueError`. Combining it with a resume request also raises `ValueError`.

## convert_checkpoint

```python
convert_checkpoint(source, destination, *, fused_qkv=None, fused_mlp=None, config: CheckpointConversionConfig | None = None)
```

Imports supported HF weights or changes supported saved weight-only projection layouts.

Returns the destination `Path`.

HF import resolves a local snapshot or downloads a supported Hub repository. This function has no `revision`, `token`, `cache_dir`, or `model_name` argument. Hub downloads use the default revision and cache with ambient authentication. Repositories whose identity is ambiguous cannot be disambiguated through this function. A local snapshot selects already downloaded files.

HF import creates a full converted checkpoint under `conversion_output_dir` or the destination parent, then copies the weights to the destination. During that copy, both converted copies coexist in addition to the source snapshot. The temporary staging directory is removed when import exits.

Existing destinations, destinations inside the source, incompatible projection layouts, and full training state inputs are rejected. QAT projection conversion raises `NotImplementedError`. Stored GPT-OSS prefused weights without shard provenance cannot be split. A supported model import does not imply all layout conversions are supported.

For direct file operations, see [inspection and transforms](../checkpoints/checkpoint-utilities.md), [storage managers](../checkpoints/checkpoint-storage.md), [family conversions](../../guides/checkpoints/family-checkpoints.md), and [GLM component checkpoints](../checkpoints/m3-checkpoints.md).

<details>
<summary>Parameters</summary>

- `source` (`str | Path`). Supported source checkpoint or Hub repository ID.
- `destination` (`str | Path`). New destination outside the source.
- `fused_qkv` (`bool | None`, optional, defaults to `None`). Requested query/key/value projection layout.
- `fused_mlp` (`bool | None`, optional, defaults to `None`). Requested MLP projection layout.
- `config` (`CheckpointConversionConfig | None`, optional, defaults to `None`). HF import worker settings.

</details>

### CheckpointConversionConfig

Import from `lilax.checkpoints`. Pydantic model with keyword fields.

Configures the HF import worker. These settings do not apply to saved native projection-layout changes.
Multimodal import rejects lazy tensor loading.

<details>
<summary>Parameters</summary>

- `conversion_output_dir` (`str | Path | None`, optional, defaults to `None`). Temporary converted-checkpoint parent. `None` uses the destination parent.
- `lazy_load_tensors` (`bool`, optional, defaults to `True`). Source tensor loading on demand instead of eager loading of all state.
- `eager_load_method` (`str`, optional, defaults to `"safetensors"`). Eager loader, `safetensors` or `transformers`.
- `save_dtype` (`str`, optional, defaults to `"bfloat16"`). Converted weight storage dtype.
- `simulated_cpu_devices_count` (`int`, optional, defaults to `16`). Simulated CPU devices for worker checkpoint sharding.

</details>
