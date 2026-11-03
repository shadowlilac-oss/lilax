"""Select a checkpoint restore source and read its training graph metadata."""

from etils import epath
from lilax._backend.common.checkpoints import checkpointing, emergency_checkpointing, snapshotting
from lilax._backend.common.checkpoints.restore_source_types import RestoreSource, _normalize_checkpoint_root
from orbax.checkpoint import v1 as ocp


def select_restore_source(
    checkpoint_manager, load_parameters_from_path, load_full_state_from_path, *, step=-1,
    source_checkpoint_layout="orbax", snapshotter=None,
):
  if snapshotting.should_restore_snapshot(snapshotter, checkpoint_manager):
    return RestoreSource("snapshot", step=snapshotting.latest_snapshot_step(snapshotter))
  if checkpoint_manager is not None:
    if step >= 0 and step not in checkpointing.all_steps(checkpoint_manager):
      raise ValueError(f"Checkpoint step {step} was not found in the checkpoint manager.")
    selected_step = checkpointing.latest_step(checkpoint_manager) if step < 0 else step
    if selected_step is not None:
      if isinstance(checkpoint_manager, (emergency_checkpointing.CheckpointManager, emergency_checkpointing.ReplicatorCheckpointManager)):
        return RestoreSource("manager", step=selected_step, manager=checkpoint_manager)
      metadata = checkpoint_manager.checkpointables_metadata(selected_step)
      return RestoreSource("manager", path=str(metadata.path), step=selected_step, manager=checkpoint_manager)
  parameters_path = _normalize_checkpoint_root(load_parameters_from_path) if load_parameters_from_path else ""
  full_state_path = _normalize_checkpoint_root(load_full_state_from_path) if load_full_state_from_path else ""
  if source_checkpoint_layout == "safetensors_dynamic":
    return RestoreSource("dynamic", path=parameters_path or full_state_path, layout=source_checkpoint_layout)
  if parameters_path:
    return RestoreSource("parameters", path=parameters_path, layout=source_checkpoint_layout)
  if full_state_path:
    return RestoreSource("full_state", path=full_state_path, layout=source_checkpoint_layout)
  return RestoreSource("none")


def read_training_graph_metadata(source: RestoreSource):
  if source.kind == "manager" and source.path is not None:
    metadata = ocp.checkpointables_metadata(epath.Path(source.path))
  elif source.kind == "full_state" and source.layout == "orbax":
    metadata = ocp.checkpointables_metadata(epath.Path(source.path))
  else:
    return None
  custom_metadata = metadata.custom_metadata or {}
  if "maxtext_easy_training_graph" not in custom_metadata:
    return None
  record = custom_metadata["maxtext_easy_training_graph"]
  if record is None:
    raise ValueError("Invalid null maxtext_easy_training_graph checkpoint metadata.")
  return record
