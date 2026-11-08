# Inspect and manage checkpoint files

Use [Trainer checkpointing](../training/checkpoints.md) for ordinary save and resume. The utilities here work directly with checkpoint structures.

## Inspect local files

```python
from lilax.checkpoints.management import inspect_hf, inspect_orbax

hf_structure = inspect_hf("/checkpoints/qwen3-hf", check_dtype=True)
orbax_structure = inspect_orbax("/checkpoints/qwen3-orbax/0/items", check_dtype=True)
print(next(iter(hf_structure.items())))
```

Both calls print parameter names and return mappings of shape and optional dtype descriptions. Add `output_file` to save that description.

## Compare two Hugging Face checkpoints

```python
from lilax.checkpoints.management import compare_hf_ckpt

matches = compare_hf_ckpt("/checkpoints/candidate-hf", reference_path="/checkpoints/reference-hf")
print(matches)
```

This launches a CPU worker and returns a boolean indicating whether the tensors match within the function's tolerance defaults. It does not load either checkpoint into a Lilax model.

## Save a parameter tree

Supply a compatible parameter tree as `parameters`, for example the dictionary from [family weight conversion](family-checkpoints.md#convert-weights-without-saving). The following writes that tree to a parameter checkpoint. It can overwrite the destination.

```python
from lilax.checkpoints.management import load_checkpoint_metadata, save_params_to_path

save_params_to_path("/checkpoints/parameter-export", parameters)
metadata = load_checkpoint_metadata("/checkpoints/parameter-export")
```

The caller supplies a compatible parameter tree. Metadata loading returns an empty dictionary when metadata cannot be read.

## Use a checkpoint manager

Manager construction creates or accesses the directory and applies the retention policy. The returned manager remains owned by the caller.

```python
from lilax.checkpoints.management import (
	all_steps, create_orbax_checkpoint_manager, latest_step, wait_until_finished,
)

manager = create_orbax_checkpoint_manager(
	"/checkpoints/training", enable_checkpointing=True, use_async=False, save_interval_steps=100,
)
wait_until_finished(manager)
print(latest_step(manager), all_steps(manager))
```

The printed values are the latest saved step or `None`, and the list of saved steps. The interval illustrates the API. See [storage operations](../../reference/checkpoints/checkpoint-storage.md).

## Extract decode weights from full training state

Use the model configuration and optimizer settings that produced the full training checkpoint. The source must contain optimizer state. This example takes the existing trainer and a source step's `items` directory, without separate LoRA input adapters.

The conversion deletes the entire destination checkpoint directory if it exists. Use a separate output run outside the source directory. With the settings below, the deleted destination is `./decode-export/weights/checkpoints/`.

```python
from lilax.checkpoints.management import generate_decode_checkpoint

def export_decode_weights(trainer, source_items_directory):
	runtime = trainer.model.runtime
	runtime.initialize(overrides={
		"load_full_state_path": source_items_directory, "load_parameters_path": "",
		"source_checkpoint_layout": "orbax",
		"base_output_directory": "./decode-export", "run_name": "weights", "enable_checkpointing": True,
	}, train_dataset_provided=trainer.train_dataset is not None, eval_dataset_provided=trainer.eval_dataset is not None)
	completed = generate_decode_checkpoint(runtime.config)
	return completed, runtime.config.checkpoint_dir + "0/items"
```

Call `export_decode_weights(trainer, "/absolute/path/to/training/checkpoints/100/items")` with your existing trainer and actual checkpoint step. The returned pair is `True` and `./decode-export/weights/checkpoints/0/items`. The output contains parameter-only decode weights, not resumable optimizer state or Hugging Face safetensors.

`checkpoint_dir` is derived from `base_output_directory` and `run_name` during initialization. Setting `checkpoint_dir` alone does not select the destination. Clear `load_parameters_path` and select the Orbax layout when loading full training state. The example uses the training runtime to retain matching architecture and optimizer settings; it does not construct a compatible runtime from checkpoint files alone.

See [runtime-source checkpoint fields](../../reference/execution/runtime-source-fields.md#checkpoint-sources-and-destinations) for source selection, destination derivation, and standalone-loop controls.

Quantization, resharding, and the standalone save loop also use a resolved execution configuration. Their [focused reference](../../reference/checkpoints/checkpoint-utilities.md) states the setup and output contracts. The save-loop utility changes optimizer moments to exercise checkpoint storage and can drop the host page cache when explicitly configured.
