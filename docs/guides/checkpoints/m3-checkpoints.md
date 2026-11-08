# Save and convert GLM component checkpoints

Use these functions with the [GLM component workflow](../components/m3-components.md). They save component state, separate from a Trainer checkpoint.

## Save and restore a component

The component's initialized architecture and dtypes must match the checkpoint.

```python
from lilax.checkpoints.m3 import restore_orbax_checkpoint, save_orbax_checkpoint

save_orbax_checkpoint(model, "/checkpoints/glm-component")
restore_orbax_checkpoint(model, "/checkpoints/glm-component", mesh=None)
```

Here `model` is the GLM component created in the linked guide. Configure sharding when constructing the component. The restore function rejects a non-None `mesh` argument.

Both calls return `None`. Restore updates the supplied component in place.

## Select a layer subset from Hugging Face files

The following writes a two-layer subset to the destination. It can overwrite files there, so use a separate directory. The destination contains a consolidated safetensors file, an index, available tokenizer files, and an updated configuration.

```python
from lilax.checkpoints.m3 import write_mini_slice_checkpoint

weight_map = write_mini_slice_checkpoint(
	"/checkpoints/glm-hf", "/checkpoints/glm-two-layer-hf", num_hidden_layers=2, include_mtp_layer=None,
)
print(next(iter(weight_map.items())))
```

This excludes the MTP layer. Keeping MTP requires its source layer index, `45`. The source checkpoint must contain the selected layers and common embedding, normalization, and head weights.

The returned mapping associates each selected tensor name with `"model.safetensors"`. The layer count illustrates subset selection, not a training recommendation.

## Stream a conversion into Orbax

Use the full `Glm5Config` that matches your source checkpoint, or a configuration matching a selected layer subset. Here `source_directory` contains your HF checkpoint, `destination_directory` is your empty output directory, `workers` is your selected worker count, and `byte_limit` is your machine's inflight memory budget. This writes an Orbax checkpoint and refuses a nonempty destination.

```python
from lilax.checkpoints.m3 import convert_hf_checkpoint_to_orbax

convert_hf_checkpoint_to_orbax(
	source_directory, destination_directory, config,
	ignore_mtp_layer=config.num_nextn_predict_layers == 0,
	slice_to_config_layers=True, num_workers=workers, max_inflight_bytes=byte_limit,
)
```

The byte budget must fit the converter's largest work unit. The converter returns `None`. The reference describes [in-memory loading and layer-name transforms](../../reference/checkpoints/m3-checkpoints.md).
