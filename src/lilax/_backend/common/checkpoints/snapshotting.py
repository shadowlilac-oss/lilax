"""In-memory snapshots of the train state for elastic training."""

from absl import logging
from flax import nnx
from grain import experimental as grain_experimental
import jax
from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.checkpoints import full_state_restore
from lilax._backend.common import train_state_nnx
from lilax._backend.input_pipeline import multihost_dataloading, synthetic_data_processing
from lilax._backend.distributed import elastic
from lilax._backend.observability import max_logging
from orbax.checkpoint import v1 as ocp

Snapshotter = ocp.training.pathways.Snapshotter


def create_snapshotter(config) -> Snapshotter | None:
  """Returns a new snapshotter in snapshot mode, and None otherwise."""
  if not elastic.elastic_snapshot(config):
    return None
  if config.shard_optimizer_over_data:
    max_logging.log(
        "WARNING: elastic_backup_kind=snapshot with shard_optimizer_over_data=True: the optimizer state is sharded"
        " across slices, so a snapshot cannot be restored after a slice is lost and the checkpoint is used instead."
    )
  return Snapshotter(replica_axis_index=config.mesh_axes.index("data"))


def reset_snapshotter(snapshotter: Snapshotter) -> Snapshotter:
  """Returns a new snapshotter holding the latest completed snapshot."""
  new_snapshotter = Snapshotter(replica_axis_index=snapshotter.replica_axis_index)
  # pylint: disable=protected-access
  with snapshotter._lock:
    new_snapshotter._latest_snapshot = snapshotter._latest_snapshot
    snapshotter._latest_snapshot = None
  # pylint: enable=protected-access
  return new_snapshotter


def latest_snapshot_step(snapshotter: Snapshotter | None) -> int | None:
  """Returns the step of the latest completed snapshot, if any."""
  if snapshotter is None:
    return None
  if latest := snapshotter.latest:
    return latest.step
  return None


def wait_for_snapshot(snapshotter: Snapshotter, step: int) -> bool:
  """Waits for the queue to drain and checks whether step was saved."""
  snapshotter.join()
  return latest_snapshot_step(snapshotter) == step


def _elastic_iterator(
    data_iterator,
) -> grain_experimental.ElasticIterator | multihost_dataloading.RemoteIteratorWrapper | None:
  """Returns the elastic iterator, or None for synthetic data."""
  if data_iterator is None or isinstance(
      data_iterator,
      (synthetic_data_processing.SyntheticDataIterator, synthetic_data_processing.PlaceHolderDataIterator),
  ):
    return None
  if isinstance(data_iterator, multihost_dataloading.RemoteIteratorWrapper) and data_iterator.elastic:
    return data_iterator
  if isinstance(data_iterator, multihost_dataloading.MultiHostDataLoadIterator):
    for candidate in (data_iterator.local_iterator, getattr(data_iterator, "dataloader", None)):
      if isinstance(candidate, grain_experimental.ElasticIterator):
        return candidate
  raise ValueError(
      "Snapshot mode requires a MultiHostDataLoadIterator or RemoteIteratorWrapper wrapping grain's ElasticIterator"
      f" (or synthetic data), got {type(data_iterator).__name__}."
  )


def _mesh_of(pytree):
  """Returns the mesh of the first leaf with a NamedSharding, or None."""
  for leaf in jax.tree.leaves(pytree):
    if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) and isinstance(leaf.sharding, jax.sharding.NamedSharding):
      return leaf.sharding.mesh
  return None


def _replicate_single_device_sharded_arrays(pytree):
  """Replicates single-device sharded arrays across the whole mesh."""
  mesh = _mesh_of(pytree)
  if mesh is None:
    return pytree
  replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _replicate(x):
    if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)) and isinstance(x.sharding, jax.sharding.SingleDeviceSharding):
      if isinstance(x, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=replicated_sharding)
      return jax.device_put(x, replicated_sharding)
    return x

  return jax.tree.map(_replicate, pytree)


def _restore_original_shardings(restored, target):
  """Moves restored arrays back onto the target shardings and memory kinds."""

  def _put(restored_leaf, target_leaf):
    if not isinstance(restored_leaf, jax.Array) or not isinstance(target_leaf, (jax.Array, jax.ShapeDtypeStruct)):
      return restored_leaf
    sharding = target_leaf.sharding
    if sharding is None:
      return restored_leaf
    if restored_leaf.sharding == sharding and restored_leaf.sharding.memory_kind == sharding.memory_kind:
      return restored_leaf
    if jax.dtypes.issubdtype(restored_leaf.dtype, jax.dtypes.prng_key):
      key_data = jax.device_put(jax.random.key_data(restored_leaf), sharding)
      return jax.random.wrap_key_data(key_data, dtype=restored_leaf.dtype)
    return jax.device_put(restored_leaf, sharding)

  return jax.tree.map(_put, restored, target)


def save_snapshot(snapshotter: Snapshotter, state: nnx.State, step: int, data_iterator=None) -> None:
  """Starts copying the train state and iterator state to pinned host memory."""
  iterator = _elastic_iterator(data_iterator)
  if iterator is None:
    iterator_state = None
  else:
    iterator_state = iterator.get_state()
  snapshot = _replicate_single_device_sharded_arrays(
      {"items": train_state_nnx.to_checkpoint_dict(state), "data_iterator": iterator_state}
  )
  max_logging.log(f"Saving in-memory snapshot at step {step}...")
  snapshotter.save(step, snapshot)


def should_save_snapshot(config, step: int, checkpoint_manager) -> bool:
  """Returns whether a snapshot is due and no checkpoint save is running."""
  if step % config.elastic_snapshot_interval != 0:
    return False
  if checkpoint_manager is not None and checkpointing.is_saving_in_progress(checkpoint_manager):
    max_logging.log(f"Skipping the in-memory snapshot at step {step} because a checkpoint save is in progress.")
    return False
  return True


def maybe_save_snapshot(snapshotter: Snapshotter | None, config, state, step: int, checkpoint_manager, data_iterator):
  """Snapshots after step when due, doing nothing outside snapshot mode."""
  if snapshotter is not None and should_save_snapshot(config, step, checkpoint_manager):
    save_snapshot(snapshotter, state, step, data_iterator)


def maybe_back_up_and_scale_up(
    snapshotter: Snapshotter | None, config, state, step: int, checkpoint_manager, data_iterator
) -> None:
  """Backs up the step before interrupting an attempt for scale-up."""
  if not elastic.is_scale_up_event(config):
    return
  if snapshotter is None and checkpoint_manager is None:
    logging.log_first_n(
        logging.WARNING,
        "New slices are available, but there is no snapshot or checkpoint to resume from, so not scaling up.",
        1,
    )
    return
  if snapshotter is None:
    max_logging.log(f"New slices are available. Checkpointing step {step} before scaling up.")
    checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator)
  else:
    max_logging.log(f"New slices are available. Snapshotting step {step} before scaling up.")
    if checkpoint_manager is not None:
      checkpointing.wait_until_finished(checkpoint_manager)
    snapshotter.join()
    if latest_snapshot_step(snapshotter) != step:
      save_snapshot(snapshotter, state, step, data_iterator)
      if not wait_for_snapshot(snapshotter, step):
        max_logging.log(
            f"WARNING: the in-memory snapshot of step {step} did not complete before the scale-up. The next attempt"
            f" resumes from the latest snapshot (step {latest_snapshot_step(snapshotter)}) or checkpoint."
        )
  elastic.maybe_elastic_scale_up(config, checkpoint_manager)


def should_restore_snapshot(snapshotter: Snapshotter | None, checkpoint_manager) -> bool:
  """Returns whether the snapshot takes priority over the checkpoint."""
  snapshot_step = latest_snapshot_step(snapshotter)
  if snapshot_step is None:
    return False
  if checkpoint_manager is None:
    return True
  checkpoint_step = checkpointing.latest_step(checkpoint_manager)
  if checkpoint_step is not None and checkpoint_step > snapshot_step:
    max_logging.log(
        f"Restoring the checkpoint at step {checkpoint_step} rather than the in-memory snapshot at step"
        f" {snapshot_step}, which is older."
    )
    return False
  return True


def _load_snapshot(snapshotter: Snapshotter, abstract_state: nnx.State, config, abstract_iterator_state):
  """Restores the snapshot onto the abstract state's shardings."""
  target_items = train_state_nnx.to_checkpoint_dict(abstract_state)
  target = {
      "items": _replicate_single_device_sharded_arrays(target_items),
      "data_iterator": abstract_iterator_state,
  }
  restored = snapshotter.load(target)
  items = _restore_original_shardings(restored["items"], target_items)
  state = full_state_restore.restored_linen_to_nnx(items, abstract_state, config=config)
  return state, restored["data_iterator"]


def load_snapshot(
    snapshotter: Snapshotter | None, abstract_state: nnx.State, config, data_iterator=None
) -> nnx.State | None:
  """Restores the train state and iterator state from the latest snapshot, if possible."""
  if snapshotter is None:
    return None
  step = latest_snapshot_step(snapshotter)
  if step is None:
    return None
  iterator = _elastic_iterator(data_iterator)
  if iterator is None:
    abstract_iterator_state = None
  else:
    abstract_iterator_state = iterator.get_state()
  max_logging.log(f"Restoring the in-memory snapshot at step {step}...")
  try:
    state, iterator_state = _load_snapshot(snapshotter, abstract_state, config, abstract_iterator_state)
  except (jax.errors.JaxRuntimeError, RuntimeError) as e:
    if isinstance(e, jax.errors.JaxRuntimeError) and elastic.is_slice_down_error(e):
      raise
    max_logging.log(f"Restoring the in-memory snapshot at step {step} failed: {e}")
    return None
  if iterator is not None:
    iterator.set_state(iterator_state)
    max_logging.log(f"Restored the data iterator state: {iterator_state}.")
  max_logging.log(f"Restored the in-memory snapshot at step {step}.")
  return state
