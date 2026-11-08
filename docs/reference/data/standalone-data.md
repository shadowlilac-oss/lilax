# Standalone data loading

These functions need resolved dataset and execution settings. Root HF model configuration alone is insufficient. They allocate model and optimizer state even though they do not train. Prefer [dataset iteration](../../guides/data/datasets.md) when you only need to inspect records.

The resolved configuration is `model.runtime.resolve_config().config` for a model whose runtime has the required dataset and training settings. See [configuration objects](../execution/runtime.md#configuration-objects).

## run

Import from `lilax.data_pipeline.standalone_dataloader`.

```python
run(config)
```

Validates configuration, sets the TensorFlow logging environment, and loads batches without training steps.
TFDS and C4 MLPerf datasets also set `TFDS_DATA_DIR`.

Returns the initialized training state.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with dataset and training execution settings.

</details>

## data_load_loop

Import from `lilax.data_pipeline.standalone_dataloader` or `lilax.data_pipeline.api`.

```python
data_load_loop(config, state=None)
```

Sets up model, optimizer, and loader. Loads a first batch, then batches through the configured step count.
Blocks for completion and prints durations on process zero.

Returns the initialized state.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with dataset and training execution settings.
- `state` (`Any`, optional, defaults to `None`). Accepted but ignored. Replaced by setup output before use.

</details>
