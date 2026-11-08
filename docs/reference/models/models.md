# Models and Hub loading

Import the factories and model capability helpers from `lilax`. See [loading guide](../../guides/models/loading.md). `ArrayLike` refers to `jax.typing.ArrayLike`.

## AutoConfig.from_pretrained

```python
AutoConfig.from_pretrained(
	pretrained_model_name_or_path: str | Path, *, revision: str | None = None,
	token: str | bool | None = None, cache_dir: str | Path | None = None,
	local_files_only: bool = False, model_name: str | None = None,
) -> MaxTextConfig
```

Returns a `MaxTextConfig` without creating model weights. `model_name` disambiguates a registered model identity. Invalid identities, conflicting snapshot metadata, and malformed saved metadata raise `ValueError`.

<a id="environment-overrides"></a>

**Environment overrides.**

When loading Hub configuration or older saved source configuration, Lilax applies `M_<KEY>` variables to known source keys, with the key uppercased. For example, `M_RUN_NAME` overrides `run_name`. These values are marked explicit. An environment variable colliding with an explicit loader override raises `ValueError`, even when the values agree. The resolved `model_name` is an explicit loader override, so `M_MODEL_NAME` collides. A collision with a model YAML key also raises `ValueError` unless `override_model_config` is enabled in the source configuration.

Environment values are converted to the original scalar type. Boolean values accept `y`, `yes`, `t`, `true`, `1`, `n`, `no`, `f`, `false`, and `0`, case-insensitively. Invalid scalar values raise `ValueError`. Non-scalar or `None` source values raise `TypeError` for an `M_<KEY>` override. Unknown keys have no effect. After this merge, `HF_TOKEN` fills `hf_access_token` only if that field is empty. A nonempty configured token takes precedence over this fallback.

A saved model configuration containing both `model_name` and `hidden_size` loads directly as `MaxTextConfig` and bypasses this environment merge. The `token` factory argument controls snapshot access independently of the configuration's `hf_access_token` field. Hub authentication defaults are described in the [official environment reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hf_token).

<details>
<summary>Parameters</summary>

- `pretrained_model_name_or_path` (`str | Path`). Supported repository ID, registered identity, or local configuration directory.
- `revision` (`str | None`, optional, defaults to `None`). Snapshot revision.
- `token` (`str | bool | None`, optional, defaults to `None`). Snapshot authentication.
- `cache_dir` (`str | Path | None`, optional, defaults to `None`). Snapshot cache directory.
- `local_files_only` (`bool`, optional, defaults to `False`). Local cached files only.
- `model_name` (`str | None`, optional, defaults to `None`). Disambiguating registered identity.

</details>

## AutoModelForCausalLM

Import from `lilax`. Factory class with static loading methods.

### from_config

```python
AutoModelForCausalLM.from_config(config, *, quantization_config=None, seed=0)
```

Returns a `MaxTextModel` with new weights deferred until materialization.

<details>
<summary>Parameters</summary>

- `config` (`MaxTextConfig`). Model configuration from `AutoConfig.from_pretrained`.
- `quantization_config` (`QuantizationRecipe | None`, optional, defaults to `None`). Model recipe override.
- `seed` (`int`, optional, defaults to `0`). Weight initialization seed.

</details>

### from_pretrained

```python
AutoModelForCausalLM.from_pretrained(
	pretrained_model_name_or_path, *, config=None, revision=None, token=None, cache_dir=None,
	local_files_only=False, quantization_config=None, fuse_qkv=None, fuse_mlp=None, precision_config=None,
)
```

`config` accepts model configuration from `AutoConfig.from_pretrained`. `quantization_config` and `precision_config` select the corresponding model settings. `fuse_qkv` and `fuse_mlp` select projection layout. Snapshot access options have the same role as in `AutoConfig`.

The model factory has no `model_name`, `device_map`, or `trust_remote_code` argument. Unsupported Hub import identities raise `ValueError`. A GCS checkpoint requires an explicit `config`.

On first materialization, Hub weights are converted in a CPU subprocess into a full additional weight copy at `<root>/lilax-models/<sha256>/0/items`. The root uses runtime `base_output_directory`, then `cache_dir`, then `HF_HUB_CACHE`, in that priority order. A trainer's `output_dir` supplies `base_output_directory`. An existing runtime `load_parameters_path` takes precedence. Reuse requires matching configuration, snapshot path, revision, weight dtype, simulated CPU-device count, and conversion settings. Missing or incompatible saved conversion metadata raises an error. Multihost conversion requires a shared runtime `base_output_directory`, for example through a shared trainer `output_dir`, otherwise it raises `ValueError`. `cache_dir` alone does not satisfy this requirement. See [first-use conversion](../../guides/models/loading.md#allow-space-for-first-use-conversion) for disk planning.

Returns a `MaxTextModel` with weight restoration deferred until materialization.

<details>
<summary>Parameters</summary>

- `pretrained_model_name_or_path` (`str | Path`). Supported Hub repository or local checkpoint path.
- `config` (`MaxTextConfig | None`, optional, defaults to `None`). Model configuration from `AutoConfig.from_pretrained`. Required for GCS checkpoints.
- `revision` (`str | None`, optional, defaults to `None`). Snapshot revision.
- `token` (`str | bool | None`, optional, defaults to `None`). Snapshot access token.
- `cache_dir` (`str | Path | None`, optional, defaults to `None`). Snapshot cache and conversion fallback root.
- `local_files_only` (`bool`, optional, defaults to `False`). Local cached files only.
- `quantization_config` (`QuantizationRecipe | None`, optional, defaults to `None`). Compute or QAT recipe override.
- `fuse_qkv` (`bool | None`, optional, defaults to `None`). Projection layout override.
- `fuse_mlp` (`bool | None`, optional, defaults to `None`). MLP layout override.
- `precision_config` (`PrecisionConfig | None`, optional, defaults to `None`). Precision override.

</details>

## MaxTextModel

Import from `lilax`. Model wrapper class returned by the factories.

The factory result exposes `config`, `generation_config`, `runtime`, and `is_materialized`.

`model.config` is the public model configuration. Runtime setup uses runtime-source configuration (`model.runtime.public_config`). Resolution produces resolved configuration (`runtime.config`). See [configuration objects](../execution/runtime.md#configuration-objects) for their accepted uses.

`existing_graph_config` returns construction settings for an already materialized model, otherwise `None`. `existing_graph_qat_config` returns that graph's QAT selection. `pending_qat_config` reports the QAT selection without forcing materialization.

`generate` and `generate_batch` are described in [generation reference](../inference/generation.md). `save_pretrained(save_directory, *, format="maxtext", export_config=None)` is described in [checkpoint reference](../training/checkpoints.md).

### prepare_for_full_state_restore

```python
model.prepare_for_full_state_restore(qat_config, *, runtime)
```

Prepares a validated QAT selection before materialization. A wrong type raises `TypeError`.
Conflicting checkpoint, requested, runtime, or graph selections raise `ValueError`.
The Trainer normally owns this operation during resume.

Returns the model.

<details>
<summary>Parameters</summary>

- `qat_config` (`QwixQATConfig`). Validated QAT selection.
- `runtime` (`Runtime`). Original model runtime. Another runtime raises `ValueError`.

</details>

### materialize

```python
model.materialize(*, runtime)
```

Constructs or restores weights once. A different runtime raises `ValueError`.

Returns the model.

<details>
<summary>Parameters</summary>

- `runtime` (`Runtime`). Original model runtime.

</details>

### __call__

```python
model(
	input_ids, *, position_ids=None, attention_mask=None, segment_ids=None, inputs_embeds=None,
	images=None, image_masks=None, videos=None, video_masks=None, video_grid_thw=None, audio_values=None,
	decoder_target_tokens=None, decoder_target_mask=None, enable_dropout=True,
)
```

Ordinarily returns a JAX logits array with shape `(batch, sequence_length, vocab_size)`. There is no Transformers `ModelOutput` wrapper or `.logits` attribute. Position IDs default to sequence positions. Missing segment IDs derive from the attention mask, or ones when no mask is supplied. Media arguments require matching model support. Evaluation calls normally select `enable_dropout=False`.

The public call uses train model mode even inside an evaluation runtime context. It returns `None` for vocabulary tiling with `num_vocab_tiling > 1`, or indexer warmup with `use_indexer=True`, positive `indexer_loss_scaling_factor`, and `indexer_sparse_training=False`. With `vllm_rpa` or `vllm_batched_rpa` attention, it returns `(hidden_state, kv_caches)`, with `expert_indices` appended when present. These configurations do not provide an ordinary logits array through this call.

<details>
<summary>Parameters</summary>

- `input_ids` (`ArrayLike`). Input batch, converted to a JAX array.
- `position_ids` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Token positions. Missing values use sequence positions.
- `attention_mask` (`ArrayLike | None`, optional, defaults to `None`). Token validity mask. Zero marks masked positions.
- `segment_ids` (`ArrayLike | None`, optional, defaults to `None`). Sequence segments. Missing values derive from the attention mask or ones. A supplied attention mask zeros masked segments.
- `inputs_embeds` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Decoder input embeddings forwarded to the model.
- `images` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Matching-family image values.
- `image_masks` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Image validity.
- `videos` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Matching-family video values.
- `video_masks` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Video validity.
- `video_grid_thw` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Temporal, height, and width grid metadata.
- `audio_values` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Matching-family audio features.
- `decoder_target_tokens` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Target tokens forwarded to model loss paths.
- `decoder_target_mask` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Target validity forwarded to model loss paths.
- `enable_dropout` (`bool`, optional, defaults to `True`). Dropout selection. Evaluation normally passes `False`.

</details>

## ModelCapabilities and registries

### ModelCapabilities

Import from `lilax`. Frozen dataclass returned by `get_model_capabilities`.

<details>
<summary>Parameters</summary>

- `model_name` (`str`). Registered model identity.
- `native` (`bool`). Native construction capability.
- `hf_import` (`bool`). HF weight-import capability.
- `hf_export` (`bool`). HF weight-export capability.
- `hf_repo_id` (`str | None`). Registered Hub repository, when available.

</details>

### get_model_capabilities

```python
get_model_capabilities(model_name: str) -> ModelCapabilities
```

Returns a frozen record with `model_name`, `native`, `hf_import`, `hf_export`, and `hf_repo_id`.
Unknown names raise `ValueError`.

<details>
<summary>Parameters</summary>

- `model_name` (`str`). Registered identity whose capabilities are requested.

</details>

### Registries

Import these exports from `lilax`. Membership in a repository map alone does not establish import or export support.

| Export | Type | Meaning |
| --- | --- | --- |
| `HF_IMPORT_MODELS` | `frozenset[str]` | Identities with HF import capability. |
| `HF_EXPORT_MODELS` | `frozenset[str]` | Identities with HF export capability. |
| `NATIVE_MODELS` | `frozenset[str]` | Native construction identities. |
| `HF_REPOS` | `dict[str, str]` | Registered identities mapped to repository IDs. |
| `MODEL_CONFIG_NAMES` | `dict[str, str]` | Aliases mapped to configuration identities. |
| `__version__` | `str` | Lilax package version. |

## Supported model identities

The following table mirrors `HF_REPOS`, `HF_IMPORT_MODELS`, and `HF_EXPORT_MODELS`. These are static registry capabilities. Runtime combinations can impose further restrictions. A Hub mapping alone does not imply weight-import support.

| Model name | Hub repository | HF import | HF export |
| --- | --- | --- | --- |
| `gemma2-2b` | `google/gemma-2-2b` | Yes | Yes |
| `gemma2-9b` | `google/gemma-2-9b` | Yes | Yes |
| `gemma2-27b` | `google/gemma-2-27b` | Yes | Yes |
| `gemma3-4b` | `google/gemma-3-4b-it` | Yes | Yes |
| `gemma3-12b` | `google/gemma-3-12b-it` | Yes | Yes |
| `gemma3-27b` | `google/gemma-3-27b-it` | Yes | Yes |
| `gemma4-26b` | `google/gemma-4-26b-a4b-it` | Yes | Yes |
| `gemma4-31b` | `google/gemma-4-31b-it` | Yes | Yes |
| `gemma4-e2b` | `google/gemma-4-E2B-it` | Yes | Yes |
| `gemma4-e4b` | `google/gemma-4-E4B-it` | Yes | Yes |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | Yes | Yes |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | Yes | Yes |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B-Instruct` | Yes | Yes |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | Yes | Yes |
| `qwen3-4b` | `Qwen/Qwen3-4B` | Yes | Yes |
| `qwen3-4b-thinking-2507` | `Qwen/Qwen3-4B-Thinking-2507` | Yes | Yes |
| `qwen3-8b` | `Qwen/Qwen3-8B` | Yes | Yes |
| `qwen3-14b` | `Qwen/Qwen3-14B` | Yes | Yes |
| `qwen3-32b` | `Qwen/Qwen3-32B` | Yes | Yes |
| `qwen3-vl-2b` | `Qwen/Qwen3-VL-2B-Instruct` | Yes | Yes |
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | Yes | Yes |
| `qwen3-vl-30b-a3b` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Yes | Yes |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B` | Yes | Yes |
| `llama3.1-8b-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` | Yes | Yes |
| `llama3.1-70b-Instruct` | `meta-llama/Llama-3.1-70B-Instruct` | Yes | No |
| `llama3.1-70b` | `meta-llama/Llama-3.1-70B` | Yes | Yes |
| `llama3.1-405b` | `meta-llama/Llama-3.1-405B` | Yes | Yes |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Yes | Yes |
| `qwen3-30b-a3b-base` | `Qwen/Qwen3-30B-A3B-Base` | Yes | Yes |
| `qwen3-235b-a22b` | `Qwen/Qwen3-235B-A22B-Thinking-2507` | Yes | Yes |
| `qwen3-480b-a35b` | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | No | No |
| `deepseek2-16b` | `deepseek-ai/DeepSeek-V2-Lite` | Yes | Yes |
| `deepseek3-671b` | `deepseek-ai/DeepSeek-V3` | Yes | Yes |
| `deepseek3.2-671b` | `deepseek-ai/DeepSeek-V3.2` | Yes | Yes |
| `deepseek4-284b` | `deepseek-ai/DeepSeek-V4-Flash` | Yes | Yes |
| `deepseek4-1600b` | `deepseek-ai/DeepSeek-V4-Pro` | Yes | Yes |
| `gpt-oss-20b` | `unsloth/gpt-oss-20b-BF16` | Yes | Yes |
| `gpt-oss-120b` | `unsloth/gpt-oss-120b-BF16` | Yes | Yes |
| `qwen3-omni-30b-a3b` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Yes | No |
| `qwen3-next-80b-a3b` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | Yes | Yes |
| `qwen3.8-flash-next` | `Qwen/Qwen3.8-Flash-Next` | Yes | Yes |
| `qwen3.8-flash-next-mini` | `Qwen/Qwen3.8-Flash-Next` | Yes | Yes |
| `qwen3.5-397b-a17b` | `Qwen/Qwen3.5-397B-A17B` | Yes | Yes |
| `qwen3.5-397b-a17b-fp8` | `Qwen/Qwen3.5-397B-A17B-FP8` | Yes | Yes |
| `qwen3.5-35b-a3b` | `Qwen/Qwen3.5-35B-A3B` | Yes | Yes |
| `qwen3.5-35b-a3b-fp8` | `Qwen/Qwen3.5-35B-A3B-FP8` | Yes | Yes |
| `qwen3.5-35b-fp8` | `Qwen/Qwen3.5-35B-A3B-FP8` | Yes | Yes |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | Yes | Yes |
| `qwen3.8-2.4t-a95b` | `Qwen/Qwen3.8-2.4T-A95B` | Yes | Yes |
| `mixtral-8x7b` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Yes | Yes |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | No | No |
| `mixtral-8x22b` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | Yes | Yes |
| `olmo3-7b` | `allenai/Olmo-3-7B-Instruct` | Yes | No |
| `olmo3-7b-pt` | `allenai/Olmo-3-1025-7B` | Yes | No |
| `olmo3-32b` | `allenai/Olmo-3-32B-Think` | Yes | No |
| `weaver-mini` | `Qwen/Qwen3-VL-4B-Instruct` | Yes | No |
| `weaver-max` | `Qwen/Qwen3-VL-8B-Instruct` | Yes | No |
| `glm5.3-flash` | `zai-org/GLM-5.3-Flash` | No | No |
| `ming-3.0-audio-tiny` | `inclusionAI/Ming-3.0-audio-tiny` | Yes | No |
| `hy4-preview` | `tencent/Hy4-preview` | Yes | Yes |

All table identities are in `NATIVE_MODELS` except `llama3.1-8b-Instruct`, `llama3.1-70b-Instruct`, and `qwen3.5-35b-fp8`. `MODEL_CONFIG_NAMES` maps those aliases to `llama3.1-8b`, `llama3.1-70b`, and `qwen3.5-35b-a3b-fp8`, respectively.

The remaining `NATIVE_MODELS` identities have no registered Hub repository and no HF import or export capability:

- `default`, `deepseek-custom`, `deepseek2-236b`, `deepseek3-671b-2dfsdp`, `deepseek3-671b-batchsplit`, `deepseek3-671b-lineage`, `deepseek3-test`, `deepseek3-tiny`, `deepseek4-tiny`.
- `envy-switch-base`, `envy-switch-large`, `envy-switch-xxl`, `envy-test`, `gemma-2b`, `gemma-7b`.
- `gpt3-175b`, `gpt3-22b`, `gpt3-52k`, `gpt3-6b`, `kimi-k2-1t`.
- `llama2-13b`, `llama2-70b`, `llama2-7b`, `llama3-405b`, `llama3-70b`, `llama3-8b`, `llama3.3-70b`, `llama4-17b-128e`, `llama4-17b-16e`.
- `qwen3-1.7b-base`, `qwen3-1.7b`, `qwen3-14b-base`, `qwen3-4b-base`, `qwen3-8b-base`, `qwen3-custom-30b-a3b`, `weaver-mini-diffuser`, `weaver`.

These identities support construction through a local one-key `config.json` and `AutoModelForCausalLM.from_config`, as shown in [construct a native-only model](../../guides/models/loading.md#construct-a-native-only-model). Passing a bare identity to `AutoConfig.from_pretrained` attempts a Hub snapshot lookup, even when no Hub repository is registered.

Repository IDs `Qwen/Qwen3-VL-4B-Instruct` and `Qwen/Qwen3.8-Flash-Next` each match two distinct identities. `AutoConfig.from_pretrained(..., model_name=...)` disambiguates them. The resulting configuration is passed as `config` to `AutoModelForCausalLM.from_pretrained`. That model factory has no `model_name` argument. The two Qwen FP8 aliases normalize to one configuration identity, so their shared repository is not ambiguous.

## Snapshot utilities

Import from `lilax.modeling.hub`.

### resolve_model_name

```python
resolve_model_name(
	pretrained_model_name_or_path: str | Path, *, model_name: str | None = None,
	snapshot: str | Path | None = None,
) -> str
```

Checks local configuration metadata when available. Unknown, ambiguous, or conflicting identities raise `ValueError`.

Returns a registered identity.

<details>
<summary>Parameters</summary>

- `pretrained_model_name_or_path` (`str | Path`). Model source to identify.
- `model_name` (`str | None`, optional, defaults to `None`). Explicit identity.
- `snapshot` (`str | Path | None`, optional, defaults to `None`). Local snapshot supplying metadata.

</details>

### resolve_model_snapshot

```python
resolve_model_snapshot(
	pretrained_model_name_or_path: str | Path, *, revision: str | None = None,
	token: str | bool | None = None, cache_dir: str | Path | None = None,
	local_files_only: bool = False, allow_patterns: list[str] | None = None,
) -> Path
```

Downloads snapshot files without loading weights. An existing local directory is returned directly.
Otherwise calls the Hub snapshot downloader.

Returns the snapshot `Path`.

<details>
<summary>Parameters</summary>

- `pretrained_model_name_or_path` (`str | Path`). Registered identity, repository, or local directory.
- `revision` (`str | None`, optional, defaults to `None`). Snapshot revision.
- `token` (`str | bool | None`, optional, defaults to `None`). Snapshot authentication.
- `cache_dir` (`str | Path | None`, optional, defaults to `None`). Snapshot cache.
- `local_files_only` (`bool`, optional, defaults to `False`). Local cached files only.
- `allow_patterns` (`list[str] | None`, optional, defaults to `None`). Download filename filters.

</details>
