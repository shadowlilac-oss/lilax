# M3 construction and configuration

M3 names the family of Flax NNX model components exposed through `lilax.generation.m3`, with their own configuration, scanning, and checkpoint functions. Component construction returns an NNX object. `AutoModelForCausalLM` returns a public model object with loading and generation methods. These return types and configuration types are distinct.

Resolved configuration (`runtime.config`) and runtime-source configuration (`model.runtime.public_config`) are distinct from the root HF-style `lilax.MaxTextConfig`. See [configuration objects](../execution/runtime.md#configuration-objects).

## Model factory

### create_model

Import from `lilax.generation.m3`.

```python
create_model(config, mesh, **kwargs)
```

Supports `glm5.3-flash`, `qwen3-0.6b`, `qwen3-vl-2b`, `qwen3.8-flash-next`, and `qwen3.8-flash-next-mini`. Unknown names raise `ValueError`.
Returns an NNX component.

For GLM, `rngs` is required. `place_on_devices`, a `remat_policy` keyword override, unsupported runtime quantization, and extra keywords are rejected. GLM configuration remat policies are `"none"` and `"full"`.

<details>
<summary>Parameters</summary>

- `config` (`object | Glm5Config`). The supported component's resolved configuration, including `model_name`. `Glm5Config` applies to the GLM component.
- `mesh` (`jax.sharding.Mesh`). The JAX placement mesh.
- `**kwargs` (`dict[str, Any]`). Component construction arguments, including required GLM `rngs`.

</details>

### maybe_apply_scan

```python
maybe_apply_scan(model: nnx.Module, config) -> nnx.Module
```

Optionally replaces repeated-layer containers in place.

Returns the component after scanning.

<details>
<summary>Parameters</summary>

- `model` (`nnx.Module`). The component to scan.
- `config` (`object`). Resolved configuration with `scan_layers` and, when enabled, `param_scan_axis`.

</details>

### resolve_runtime_config

```python
resolve_runtime_config(public_config, *, overrides=None,
	train_dataset_provided=False, eval_dataset_provided=False) -> object
```

Resolves a configuration produced through the model's runtime. Passing the root `lilax.MaxTextConfig` directly raises `TypeError`.
Returns resolved configuration.

<details>
<summary>Parameters</summary>

- `public_config` (`object`). Runtime-source configuration (`model.runtime.public_config`).
- `overrides` (`dict[str, Any] | None`, optional, defaults to `None`). Execution setting overrides.
- `train_dataset_provided` (`bool`, optional, defaults to `False`). Flag for an explicitly supplied training dataset.
- `eval_dataset_provided` (`bool`, optional, defaults to `False`). Flag for an explicitly supplied evaluation dataset.

</details>

## Configuration types

Import from `lilax.generation.m3_config`.

### BaseConfig

Import from `lilax.generation.m3_config`. Pydantic model.

Composes component domains. It supports `get`, `keys`, `values`, `items`, `get_keys`, indexing, membership, and iteration. Extra keys are allowed. These configurations are separate from HF model and trainer configurations.

<details>
<summary>Parameters</summary>

- `device` (`DeviceConfig`, optional, defaults to `DeviceConfig()`). Device settings. Each instance receives a fresh value.
- `checkpoint` (`CheckpointConfig`, optional, defaults to `CheckpointConfig()`). Storage settings. Each instance receives a fresh value.
- `trainer` (`TrainerConfig`, optional, defaults to `TrainerConfig()`). Training preferences. Each instance receives a fresh value.
- `data` (`DataConfig`, optional, defaults to `DataConfig()`). Data settings. Each instance receives a fresh value.
- `transformer` (`Optional[TransformerConfig]`, optional, defaults to `None`). Transformer settings.

</details>

### from_dict

```python
BaseConfig.from_dict(raw: dict[str, Any]) -> BaseConfig
DiffusionConfig.from_dict(raw: dict[str, Any]) -> DiffusionConfig
```

Returns typed component configuration. `BaseConfig.from_dict` constructs all five domains, including a transformer domain.
`DiffusionConfig` shares mapping behavior and adds autoencoder, scheduler, text encoder, diffusion transformer, and pipeline domains.
Their declared fields are in [diffusion settings](diffusion-settings.md). Extra keys are allowed.

<details>
<summary>Parameters</summary>

- `raw` (`dict[str, Any]`). Nested domains or flat component settings.

</details>

### load_yaml_config

```python
load_yaml_config(file_path: str, search_paths: Sequence[str] | None = None) -> dict[str, Any]
```

Reads and merges configuration mappings. Missing input files raise `FileNotFoundError`.
Returns the merged dictionary.

<details>
<summary>Parameters</summary>

- `file_path` (`str`). The YAML file.
- `search_paths` (`Sequence[str] | None`, optional, defaults to `None`). Defaults search directories. Omission derives directories from the input path.

</details>

### process_config

```python
process_config(raw: dict[str, Any] | BaseConfig, overrides: Optional[dict[str, Any]] = None, **kwargs) -> BaseConfig
```

Resolves interpolation, dtypes, batch dimensions, and directories, and initializes JAX.
Returns the selected component configuration.

<details>
<summary>Parameters</summary>

- `raw` (`dict[str, Any] | BaseConfig`). Component settings.
- `overrides` (`Optional[dict[str, Any]]`, optional, defaults to `None`). Input setting overrides.
- `**kwargs` (`dict[str, Any]`). Additional overrides after the mapping overrides.

</details>

### Glm5Config

Import from `lilax.generation.m3_config`. Frozen dataclass.

Its [field and helper reference](glm-settings.md) documents `mini_config` and `from_maxtext_config`.
`mini_config` selects five layers, no MTP, and remat `"none"`. Other dimensions remain large.

See [Glm5Config fields](glm-settings.md#constructor-fields) for constructor parameters.

### Glm5Sharding

Import from `lilax.generation.m3_config`. Frozen dataclass.

```python
Glm5Sharding(
	fsdp_axis: str = "fsdp", dp_axis: str = "data", ep_axis: str = "expert", tp_axis: str = "tensor",
	use_tp: bool = False, ep_in_batch: bool = False, ep_size: int = 1,
)
```

Describes GLM sharding. Calling it maps logical axes. `constrain(array, tensor_name, axes)` returns constrained arrays.

See [M3 guide](../../guides/components/m3-components.md).

Field lookup is split into [execution and data](m3-execution-settings.md), [training](m3-training-settings.md), [GLM architecture](glm-settings.md), and [diffusion domains](diffusion-settings.md). Use the [settings guide](../../guides/components/m3-settings.md) to compose those domains.

<details>
<summary>Parameters</summary>

- `fsdp_axis` (`str`, optional, defaults to `"fsdp"`). Parameter sharding axis name.
- `dp_axis` (`str`, optional, defaults to `"data"`). Data axis name.
- `ep_axis` (`str`, optional, defaults to `"expert"`). Expert axis name.
- `tp_axis` (`str`, optional, defaults to `"tensor"`). Tensor axis name.
- `use_tp` (`bool`, optional, defaults to `False`). Flag for tensor parallelism.
- `ep_in_batch` (`bool`, optional, defaults to `False`). Flag for expert parallelism in batch mapping.
- `ep_size` (`int`, optional, defaults to `1`). Expert axis size.

</details>

### from_mesh

```python
Glm5Sharding.from_mesh(mesh: jax.sharding.Mesh) -> Glm5Sharding
```

Requires `data`, `fsdp`, `tensor`, and `expert` axes. Nonstandard axes cannot have size above one.
Returns the corresponding sharding configuration.

<details>
<summary>Parameters</summary>

- `mesh` (`jax.sharding.Mesh`). The mesh axes and sizes.

</details>

### Sharding call

```python
sharding(tensor_name: str, axes: Sequence[str | None], tensor_type: TensorType = TensorType.WEIGHT) -> PartitionSpec
```

Returns mapped partition information. The default call maps weights. `sharding.constrain` maps activations.

<details>
<summary>Parameters</summary>

- `tensor_name` (`str`). Tensor name.
- `axes` (`Sequence[str | None]`). Logical dimension names.
- `tensor_type` (`TensorType`, optional, defaults to `TensorType.WEIGHT`). Weight or activation mapping.

</details>

### constrain

```python
sharding.constrain(x: jax.Array, tensor_name: str, axes: Sequence[str | None]) -> jax.Array
```

Returns the array constrained using activation mapping.

<details>
<summary>Parameters</summary>

- `x` (`jax.Array`). The array.
- `tensor_name` (`str`). Tensor name.
- `axes` (`Sequence[str | None]`). Logical dimension names.

</details>
