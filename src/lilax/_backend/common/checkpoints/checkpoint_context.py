"""Maps checkpoint configuration onto Orbax contexts and training policies."""

import datetime

from orbax.checkpoint import pathways as ocp_pathways
from orbax.checkpoint import v1 as ocp

_BYTES_PER_GB = 10**9
_SINGLE_REPLICA_BROADCAST_MEMORY_LIMIT_BYTES = 1024 * 1024 * 1000


def build_save_decision_policy(
    *,
    save_interval_steps: int | None = None,
    enable_continuous_checkpointing: bool = False,
    enable_autocheckpoint: bool = False,
) -> ocp.training.save_decision_policies.SaveDecisionPolicy:
  """Builds the checkpoint save decision policy."""
  policies = ocp.training.save_decision_policies
  if enable_continuous_checkpointing:
    return policies.ContinuousCheckpointingPolicy()
  if enable_autocheckpoint:
    if save_interval_steps is not None:
      return policies.AnySavePolicy(
          [policies.PreemptionCheckpointingPolicy(), policies.FixedIntervalPolicy(save_interval_steps)]
      )
    return policies.PreemptionCheckpointingPolicy()
  if save_interval_steps is None:
    raise ValueError("save_interval_steps must be provided for fixed interval checkpointing.")
  return policies.FixedIntervalPolicy(interval=save_interval_steps)


def build_preservation_policy(*, max_to_keep: int) -> ocp.training.preservation_policies.PreservationPolicy:
  """Builds the policy retaining the latest checkpoints."""
  return ocp.training.preservation_policies.LatestN(max_to_keep)


def build_context(
    *,
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    ocdbt_target_data_file_size_bytes: int | None = None,
    checkpoint_storage_concurrent_gb: int | None = None,
    enable_continuous_checkpointing: bool = False,
    todelete_full_path: str | None = None,
    todelete_subdir: str | None = None,
    enable_single_replica_ckpt_restoring: bool = False,
    replica_axis_index: int = 0,
    colocated_python_checkpointing: bool = False,
    partial_load: bool = False,
    checkpoint_layout: ocp.options.CheckpointLayout | None = None,
) -> ocp.Context:
  """Builds an Orbax context from checkpoint options."""
  ctx = ocp.Context()
  ctx.array.saving.use_ocdbt = use_ocdbt
  ctx.array.saving.use_zarr3 = use_zarr3
  if ocdbt_target_data_file_size_bytes is not None:
    ctx.array.saving.ocdbt_target_data_file_size = ocdbt_target_data_file_size_bytes
    ctx.array.saving.storage_options.chunk_byte_size = ocdbt_target_data_file_size_bytes
  if checkpoint_storage_concurrent_gb is not None:
    concurrent_bytes = checkpoint_storage_concurrent_gb * _BYTES_PER_GB
    ctx.memory.write_concurrent_bytes = concurrent_bytes
    ctx.memory.read_concurrent_bytes = concurrent_bytes
  if enable_continuous_checkpointing:
    ctx.asynchronous.timeout_secs = int(datetime.timedelta(minutes=60).total_seconds())
  if todelete_full_path is not None:
    ctx.deletion.gcs_deletion_options.todelete_full_path = todelete_full_path
  if todelete_subdir is not None:
    raise ValueError("Renaming to subdirectory before deleting (todelete_subdir) is now unsupported by Orbax v1.")
  if enable_single_replica_ckpt_restoring:
    ctx.array.loading.use_load_and_broadcast = True
    ctx.array.loading.load_and_broadcast_options.replica_axis_index = replica_axis_index
    ctx.array.loading.load_and_broadcast_options.broadcast_memory_limit_bytes = (
        _SINGLE_REPLICA_BROADCAST_MEMORY_LIMIT_BYTES
    )
  if colocated_python_checkpointing:
    ctx.pathways.checkpointing_impl = ocp_pathways.CheckpointingImpl.from_options(use_colocated_python=True)
  else:
    ctx.pathways.checkpointing_impl = ocp_pathways.CheckpointingImpl.NO_DISPATCHER
  if partial_load:
    ctx.pytree.loading.partial_load = True
  if checkpoint_layout is not None:
    ctx.checkpoint_layout = checkpoint_layout
  return ctx
