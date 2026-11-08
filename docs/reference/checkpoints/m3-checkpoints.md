# M3 checkpoint operations

Import from `lilax.checkpoints.m3`.

## GLM state

### save_orbax_checkpoint

```python
save_orbax_checkpoint(model: Glm5Model, orbax_dir: str) -> None
```

Saves all NNX state and waits for completion.

Returns `None`. `Glm5Model` denotes the constructed GLM component.

<details>
<summary>Parameters</summary>

- `model` (`Glm5Model`). The initialized NNX model state.
- `orbax_dir` (`str`). The checkpoint destination.

</details>

### restore_orbax_checkpoint

```python
restore_orbax_checkpoint(model: Any, orbax_dir: str, *, mesh: jax.sharding.Mesh | None) -> None
```

Mutates the initialized model. A non-None mesh raises `ValueError`. Sharding must already belong to the model.
Saved metadata, shapes, dtypes, and model sharding must match. Missing metadata or incompatible leaves raise `ValueError`.
Unsupported sharding metadata raises `TypeError`.

Returns `None`.

<details>
<summary>Parameters</summary>

- `model` (`Any`). The initialized GLM component to update.
- `orbax_dir` (`str`). The saved checkpoint.
- `mesh` (`jax.sharding.Mesh | None`). Required mesh argument, which must be `None`.

</details>

### load_hf_state_into_model

```python
load_hf_state_into_model(model: Glm5Model, hf_tensors: Mapping[str, Any], *, ignore_mtp_layer: bool) -> None
```

Loads HF tensor names into the initialized GLM component, skipping visual tensors.
Checks MTP selection, duplicate normalized keys, shapes, unmapped keys, and unpopulated parameters.
Invalid mappings raise `ValueError`.

Returns `None`.

<details>
<summary>Parameters</summary>

- `model` (`Glm5Model`). The initialized GLM component.
- `hf_tensors` (`Mapping[str, Any]`). Mapping of HF parameter names to tensors.
- `ignore_mtp_layer` (`bool`). MTP exclusion flag, which must agree with the model configuration.

</details>

### convert_hf_checkpoint_to_orbax

```python
convert_hf_checkpoint_to_orbax(
	hf_checkpoint_dir: str, orbax_output_dir: str, config: Glm5Config, *, ignore_mtp_layer: bool,
	slice_to_config_layers: bool, num_workers: int, max_inflight_bytes: int,
) -> None
```

Streams safetensors into an Orbax checkpoint. Missing shards, missing weights, incompatible shapes, and a nonempty output directory fail conversion.
Worker count and byte limit must be positive integers excluding booleans. The MTP flag must agree with configuration.
Returns `None`.

<details>
<summary>Parameters</summary>

- `hf_checkpoint_dir` (`str`). The safetensors source directory.
- `orbax_output_dir` (`str`). An absent or empty output directory.
- `config` (`Glm5Config`). Target architecture settings.
- `ignore_mtp_layer` (`bool`). Flag for exclusion of MTP weights.
- `slice_to_config_layers` (`bool`). Flag for restriction of source weights to configured layers.
- `num_workers` (`int`). Limit on concurrent conversion workers.
- `max_inflight_bytes` (`int`). Limit on in-flight tensor bytes.

</details>

## Layer subsets

### extract_mini_slice_keys

```python
extract_mini_slice_keys(weight_map: Mapping[str, str], num_hidden_layers: int, *, include_mtp_layer: int | None) -> dict[str, str]
```

Excludes visual weights. Layer count must be positive. Missing common or layer weights raise `ValueError`.
Returns the selected name-to-shard mapping.

<details>
<summary>Parameters</summary>

- `weight_map` (`Mapping[str, str]`). Mapping of HF tensor names to source shards.
- `num_hidden_layers` (`int`). The positive decoder layer count.
- `include_mtp_layer` (`int | None`). Required MTP layer selection, `45` or `None` to exclude MTP.

</details>

### write_mini_slice_checkpoint

```python
write_mini_slice_checkpoint(src_hf_dir: str, dst_hf_dir: str, *, num_hidden_layers: int, include_mtp_layer: int | None) -> dict[str, str]
```

Requires a safetensors index. Writes selected tensors and an index, copies available tokenizer files, and updates layer and MTP counts in `config.json`.
Trims explicit layer schedules.

Returns the rewritten weight map.

<details>
<summary>Parameters</summary>

- `src_hf_dir` (`str`). The indexed source checkpoint.
- `dst_hf_dir` (`str`). The output directory.
- `num_hidden_layers` (`int`). The positive decoder layer count.
- `include_mtp_layer` (`int | None`). MTP layer selection, `45` or `None` to exclude MTP.

</details>

## Layer-name transforms

These functions return transformed trees and do not write checkpoints. See [checkpoint workflow](../../guides/checkpoints/m3-checkpoints.md).

### to_legacy_layer_names

```python
to_legacy_layer_names(tree)
```

Returns a tree with indexed decoder layer mappings translated into `layers_<index>` names.

<details>
<summary>Parameters</summary>

- `tree` (`Mapping | list | tuple`). The parameter tree to translate.

</details>

### from_legacy_layer_names

```python
from_legacy_layer_names(tree, model_state=None)
```

Returns a tree with indexed mappings restored to the expected structure.

<details>
<summary>Parameters</summary>

- `tree` (`Mapping | list | tuple`). The legacy parameter tree.
- `model_state` (`Mapping | nnx.State | None`, optional, defaults to `None`). A target state to match its layer nesting.

</details>

### match_checkpoint_layer_names

```python
match_checkpoint_layer_names(target, checkpoint_tree)
```

Returns the target state translated to match a legacy checkpoint layout, or the unchanged target for other layouts.

<details>
<summary>Parameters</summary>

- `target` (`Mapping | nnx.State`). The target model state.
- `checkpoint_tree` (`Mapping | nnx.State`). The layout to match.

</details>
