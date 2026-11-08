# Checkpoint storage operations

Import from `lilax.checkpoints.management`. These operations consume parameter trees, abstract state, and checkpoint managers. See [utility examples](../../guides/checkpoints/checkpoint-utilities.md).

`CheckpointManager` accepts Orbax v1 `training.Checkpointer`, `EmergencyCheckpointManager`, and `EmergencyReplicatorCheckpointManager`. Resolved configuration comes from `runtime.config`. See [configuration objects](../execution/runtime.md#configuration-objects).

## Create a manager

```python
create_orbax_checkpoint_manager(
	checkpoint_dir: str, enable_checkpointing: bool, use_async: bool, save_interval_steps: int,
	dataset_type: None | str = None, orbax_logger: Any = None, use_ocdbt: bool = True, use_zarr3: bool = True,
	enable_continuous_checkpointing: bool = False, max_num_checkpoints_to_keep: int = 10,
	checkpoint_storage_concurrent_gb: int = 96, enable_autocheckpoint: bool = False,
	todelete_subdir: str | None = None, todelete_full_path: str | None = None,
	ocdbt_target_data_file_size_bytes: int | None = None, colocated_python_checkpointing: bool = False,
)
```

Disabled checkpointing returns `None`. Enabled construction creates or validates the directory and returns an Orbax training checkpointer with save and preservation policies. `dataset_type` is deprecated. Storage settings control format, concurrency, and deletion destinations.

<details>
<summary>Parameters</summary>

- `checkpoint_dir` (`str`). Storage.
- `enable_checkpointing` (`bool`). Flag for manager construction.
- `use_async` (`bool`). Asynchronous saves.
- `save_interval_steps` (`int`). Save cadence.
- `dataset_type` (`None | str`, optional, defaults to `None`). Deprecated argument.
- `orbax_logger` (`Any`, optional, defaults to `None`). Checkpoint logging.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `enable_continuous_checkpointing` (`bool`, optional, defaults to `False`). Continuous preservation.
- `max_num_checkpoints_to_keep` (`int`, optional, defaults to `10`). Limit on retained checkpoints.
- `checkpoint_storage_concurrent_gb` (`int`, optional, defaults to `96`). Limit on storage concurrency in GB.
- `enable_autocheckpoint` (`bool`, optional, defaults to `False`). Flag for automatic checkpoints.
- `todelete_subdir` (`str | None`, optional, defaults to `None`). The deletion subdirectory.
- `todelete_full_path` (`str | None`, optional, defaults to `None`). The deletion destination.
- `ocdbt_target_data_file_size_bytes` (`int | None`, optional, defaults to `None`). Target OCDBT file bytes.
- `colocated_python_checkpointing` (`bool`, optional, defaults to `False`). Colocated Python persistence.

</details>

### create_emergency_checkpoint_manager

```python
create_emergency_checkpoint_manager(local_checkpoint_dir: str, persistent_checkpoint_dir: str,
	global_mesh: jax.sharding.Mesh, abstract_state: Any, local_save_interval_steps: int,
	persistent_save_interval_steps: int, orbax_logger: Any = None)
```

Configures distributed process IDs and creates the persistent directory. GPU local directories are process-specific.
Returns an Orbax emergency manager.

<details>
<summary>Parameters</summary>

- `local_checkpoint_dir` (`str`). Local storage.
- `persistent_checkpoint_dir` (`str`). Durable storage.
- `global_mesh` (`jax.sharding.Mesh`). Device placement.
- `abstract_state` (`Any`). The target state structure.
- `local_save_interval_steps` (`int`). Local cadence.
- `persistent_save_interval_steps` (`int`). Durable cadence.
- `orbax_logger` (`Any`, optional, defaults to `None`). Checkpoint logging.

</details>

### create_replicator_checkpoint_manager

```python
create_replicator_checkpoint_manager(local_checkpoint_dir: str, save_interval_steps: int,
	global_mesh: jax.sharding.Mesh, colocated_python_checkpointing: bool = False)
```

Configures distributed process IDs.

Returns an Orbax replicator manager.

<details>
<summary>Parameters</summary>

- `local_checkpoint_dir` (`str`). Local storage.
- `save_interval_steps` (`int`). Cadence.
- `global_mesh` (`jax.sharding.Mesh`). The device mesh.
- `colocated_python_checkpointing` (`bool`, optional, defaults to `False`). Colocated persistence.

</details>

## Inspect and wait

### latest_step

```python
latest_step(checkpoint_manager)
```

Returns the latest integer step or `None`.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager`). A compatible Orbax manager.

</details>

### all_steps

```python
all_steps(checkpoint_manager)
```

Returns saved steps.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager`). A compatible manager.

</details>

### wait_until_finished

```python
wait_until_finished(checkpoint_manager)
```

Waits for pending saves.

Returns `None`.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager`). A compatible manager.

</details>

### is_saving_in_progress

```python
is_saving_in_progress(checkpoint_manager) -> bool
```

Returns save progress, always false for emergency manager variants.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager`). A compatible manager.

</details>

### reached_preemption

```python
reached_preemption(checkpoint_manager, step: int) -> bool
```

Returns the manager or distributed preemption signal.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager`). A compatible manager.
- `step` (`int`). The synchronization step.

</details>

### load_checkpoint_metadata

```python
load_checkpoint_metadata(checkpoint_dir_path: str) -> Any
```

Returns custom metadata, or an empty dictionary after logging an unreadable metadata error.

<details>
<summary>Parameters</summary>

- `checkpoint_dir_path` (`str`). The checkpoint.

</details>

## Save and restore parameter trees

### save_params_to_path

```python
save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True)
```

Writes an `items` checkpoint containing parameters with overwrite enabled. Empty destinations fail with an assertion.
Returns `None`.

<details>
<summary>Parameters</summary>

- `checkpoint_dir` (`str`). The nonempty destination.
- `params` (`dict | nnx.State`). The parameter tree.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.

</details>

### load_params_from_path

```python
load_params_from_path(load_parameters_from_path, abstract_unboxed_params, checkpoint_storage_concurrent_gb,
	use_ocdbt=True, use_zarr3=True, enable_single_replica_ckpt_restoring: bool = False,
	convert_flat_moe_weights: bool = False, colocated_python_checkpointing: bool = False)
```

Returns restored parameters matching the abstract target. The path must be nonempty. Shape mismatches fail.
Optional flat MoE conversion changes compatible stored layouts during restore.

<details>
<summary>Parameters</summary>

- `load_parameters_from_path` (`str`). The source checkpoint.
- `abstract_unboxed_params` (`dict | nnx.State`). Target shapes and placement.
- `checkpoint_storage_concurrent_gb` (`int`). Limit on storage concurrency.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `enable_single_replica_ckpt_restoring` (`bool`, optional, defaults to `False`). Single-replica restoration.
- `convert_flat_moe_weights` (`bool`, optional, defaults to `False`). Flag for conversion of compatible flat expert weights.
- `colocated_python_checkpointing` (`bool`, optional, defaults to `False`). Colocated persistence.

</details>

## Training-state saves

With `return_result=True`, both save functions return a receipt and require `action_id` and `completed_step`, or raise `ValueError`.
These are receipt records used by [training engines](../training/engine-extensions.md).
The functions require compatible execution state and configuration. HF model configuration alone does not supply them.

### save_checkpoint

```python
save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False,
	training_graph=None, *, action_id=None, completed_step=None, return_result=False)
```

Optionally preserves data-iterator state and configuration metadata.

Returns a save boolean by default.
Existing checkpoint files return false.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager`). The save manager.
- `step` (`int`). The saved step.
- `state` (`TrainState | TrainStateNNX | dict | nnx.State | flax.struct.PyTreeNode`). Compatible training state.
- `config` (`object | None`, optional, defaults to `None`). Resolved execution settings and metadata.
- `data_iterator` (`MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None`, optional, defaults to `None`). Restorable iterator state.
- `force` (`bool`, optional, defaults to `False`). Flag for a forced save attempt.
- `training_graph` (`dict | None`, optional, defaults to `None`). Graph metadata.
- `action_id` (`str | None`, optional, defaults to `None`). A receipt action.
- `completed_step` (`int | None`, optional, defaults to `None`). Receipt progress.
- `return_result` (`bool`, optional, defaults to `False`). Receipt output.

</details>

### maybe_save_checkpoint

```python
maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None, training_graph=None,
	*, action_id=None, completed_step=None, return_result=False, force=False)
```

Applies cadence and duplicate-save rules and handles preemption after pending saves finish. Default return is `None`.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager | None`). Save manager, or `None` to disable persistence.
- `state` (`TrainState | TrainStateNNX | dict | nnx.State`). Compatible training state.
- `config` (`object`). Resolved cadence and execution settings.
- `data_iterator` (`MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None`). Iterator state.
- `step` (`int | None`, optional, defaults to `None`). Save step override for the state-derived step.
- `training_graph` (`dict | None`, optional, defaults to `None`). Graph metadata.
- `action_id` (`str | None`, optional, defaults to `None`). A receipt action.
- `completed_step` (`int | None`, optional, defaults to `None`). Receipt progress.
- `return_result` (`bool`, optional, defaults to `False`). Receipt output.
- `force` (`bool`, optional, defaults to `False`). Flag for a forced save attempt.

</details>

### checkpoint_exception_guard

```python
checkpoint_exception_guard(config, checkpoint_manager, handler_fn=None)
```

Returns a context manager for elastic checkpoint handling. Unhandled errors propagate.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved elastic execution settings.
- `checkpoint_manager` (`CheckpointManager`). The active manager.
- `handler_fn` (`Callable[[Exception], Any] | None`, optional, defaults to `None`). Callback for caught errors after elastic handling.

</details>

## Training-state restore

```python
load_state_if_possible(
	checkpoint_manager: CheckpointManager | None,
	data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
	load_parameters_from_path: str, load_full_state_from_path: str, checkpoint_storage_concurrent_gb: int,
	abstract_unboxed_pre_state: TrainState | nnx.State,
	enable_single_replica_ckpt_restoring: bool | None = False, dataset_type: str | None = "synthetic", step: int = -1,
	use_ocdbt=True, use_zarr3=True, enable_orbax_v1=False,
	checkpoint_conversion_fn=None, source_checkpoint_layout="orbax",
	expansion_factor_real_data: int = -1, maxtext_config: Any | None = None,
	snapshotter: ocp.training.pathways.Snapshotter | None = None, selected_source: RestoreSource | None = None,
)
```

Returns a pair of restored state and separately loaded parameters. Manager restore usually returns the state under `items`; parameter-only restore fills the second value. The abstract state determines shapes and placement. An absent source can return `(None, None)`. Missing requested steps or a changed selected source raise `ValueError`. A frozen snapshot selection raises `RuntimeError`. `enable_orbax_v1` is deprecated. Optional source-selection records belong to the training lifecycle; no separate backend record construction is required for ordinary resume.

<details>
<summary>Parameters</summary>

- `checkpoint_manager` (`CheckpointManager | None`). A compatible manager or `None`.
- `data_iterator` (`MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None`). Restorable iterator state or `None`.
- `load_parameters_from_path` (`str`). Parameter-only restoration.
- `load_full_state_from_path` (`str`). Full-state restoration.
- `checkpoint_storage_concurrent_gb` (`int`). Limit on restore storage concurrency.
- `abstract_unboxed_pre_state` (`TrainState | nnx.State`). The target abstract state and placement.
- `enable_single_replica_ckpt_restoring` (`bool | None`, optional, defaults to `False`). Single-replica restoration.
- `dataset_type` (`str | None`, optional, defaults to `"synthetic"`). Iterator handling.
- `step` (`int`, optional, defaults to `-1`). A restore step.
- `use_ocdbt` (`bool`, optional, defaults to `True`). OCDBT.
- `use_zarr3` (`bool`, optional, defaults to `True`). Zarr3.
- `enable_orbax_v1` (`bool`, optional, defaults to `False`). Deprecated argument.
- `checkpoint_conversion_fn` (`Callable | None`, optional, defaults to `None`). Conversion function for restored state.
- `source_checkpoint_layout` (`str`, optional, defaults to `"orbax"`). Source layout.
- `expansion_factor_real_data` (`int`, optional, defaults to `-1`). Iterator expansion.
- `maxtext_config` (`Any | None`, optional, defaults to `None`). Compatible execution settings.
- `snapshotter` (`ocp.training.pathways.Snapshotter | None`, optional, defaults to `None`). The training lifecycle snapshotter.
- `selected_source` (`RestoreSource | None`, optional, defaults to `None`). The frozen lifecycle restore selection.

</details>

## Engine manager

### EngineCheckpointState

Import from `lilax.checkpoints.management`. Dataclass.

```python
EngineCheckpointState(model: nnx.Module, optimizer: nnx.optimizer.Optimizer | None = None)
```

Stores an NNX model and optional optimizer.

<details>
<summary>Parameters</summary>

- `model` (`nnx.Module`). Model state.
- `optimizer` (`nnx.optimizer.Optimizer | None`, optional, defaults to `None`). Optimizer state.

</details>

### EngineCheckpointManager

Import from `lilax.checkpoints.management`. Class.

```python
EngineCheckpointManager(checkpoint_dir: str, config: object)
```

Constructs persistence from resolved execution storage settings. An empty directory string disables persistence.
The configuration is `runtime.config`, not HF model configuration.

<details>
<summary>Parameters</summary>

- `checkpoint_dir` (`str`). Checkpoint storage path, or `""` to disable persistence.
- `config` (`object`). Resolved storage settings.

</details>

### get_latest_step

```python
manager.get_latest_step() -> int | None
```

Returns the latest saved step, or `None` when persistence is disabled or absent.

### wait_until_finished

```python
manager.wait_until_finished() -> None
```

Waits for pending saves.

Returns `None`.

### close

```python
manager.close() -> None
```

Closes the enabled manager.

Returns `None`.

### save_checkpoint

```python
manager.save_checkpoint(step: int, checkpoint_state: EngineCheckpointState,
	custom_metadata: Any = None, force: bool = False) -> bool
```

Returns a save boolean. Skips an already saved latest step and disabled persistence.

<details>
<summary>Parameters</summary>

- `step` (`int`). The save step.
- `checkpoint_state` (`EngineCheckpointState`). Model and optimizer state.
- `custom_metadata` (`Any`, optional, defaults to `None`). Checkpoint metadata.
- `force` (`bool`, optional, defaults to `False`). Flag for a forced save attempt.

</details>

### restore_checkpoint

```python
manager.restore_checkpoint(checkpoint_state: EngineCheckpointState,
	step: int | None = None) -> tuple[int | None, EngineCheckpointState, Any]
```

Restores the latest or selected step and updates supplied NNX objects.

Returns `(step, state, metadata)`.
Disabled or absent persistence returns original state with no step. A caught restore failure logs the error and returns `(None, None, None)`.
`EngineCheckpointRestoreError` is the exported runtime error type. This manager does not promise to raise it for every restore failure.

<details>
<summary>Parameters</summary>

- `checkpoint_state` (`EngineCheckpointState`). The mutable restoration target.
- `step` (`int | None`, optional, defaults to `None`). Restore step, or `None` for the latest checkpoint.

</details>
