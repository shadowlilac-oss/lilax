"""Create and operate Orbax checkpoint managers."""

import collections
import contextlib
import os
import time
from typing import Any

from etils import epath
from flax import nnx, struct
from flax.nnx import variablelib
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
from lilax._backend.checkpoint_conversion.utils import load_dynamic
from lilax._backend.common.checkpoints import checkpoint_context, emergency_checkpointing, grain_utility, snapshotting
from lilax._backend.common.checkpoints.flat_moe import _flat_moe_convert, _flat_moe_restore_plan
from lilax._backend.common.checkpoints.full_state_restore import _load_full_state_from_path, restored_linen_to_nnx
from lilax._backend.common.checkpoints.linen_compatibility import (
    _alias_legacy_collections, _deep_merge_dicts, _log_unconsumed_checkpoint_weights, _lookup_path,
    _raise_on_weight_mismatch, _raise_weight_problems, _unalias_legacy_collections, _weight_mismatches,
)
from lilax._backend.common.checkpoints.lora_filter import _filter_lora_trainable_state
from lilax._backend.common.checkpoints.quantized_restore import _augment_want_with_scales, maybe_dequantize_restored_params
from lilax._backend.common.checkpoints.restore_source_types import RestoreSource, _normalize_checkpoint_root
from lilax._backend.common import train_state_nnx
from lilax._backend.input_pipeline import multihost_dataloading, synthetic_data_processing
from lilax._backend.trainers.diloco import checkpointing as diloco_checkpoint_utils
from lilax._backend.utils import elastic_utils, exceptions, gcs_utils
from lilax._backend.utils import globals as maxtext_globals
from lilax._backend.utils import max_logging
from orbax.checkpoint import logging as ocp_logging
from orbax.checkpoint import v1 as ocp

load_safetensors_dynamic_state = load_dynamic.load_safetensors_dynamic_state
PlaceHolderDataIterator = synthetic_data_processing.PlaceHolderDataIterator
MultiHostDataLoadIterator = multihost_dataloading.MultiHostDataLoadIterator
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = maxtext_globals.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
EmergencyCheckpointManager = emergency_checkpointing.CheckpointManager
EmergencyReplicatorCheckpointManager = emergency_checkpointing.ReplicatorCheckpointManager
create_orbax_emergency_checkpoint_manager = emergency_checkpointing.create_emergency_checkpoint_manager
create_orbax_emergency_replicator_checkpoint_manager = emergency_checkpointing.create_replicator_checkpoint_manager
CheckpointManager = ocp.training.Checkpointer | EmergencyCheckpointManager | EmergencyReplicatorCheckpointManager


def _abstract_params(abstract_unboxed_pre_state):
  if isinstance(abstract_unboxed_pre_state, nnx.State):
    model_state = abstract_unboxed_pre_state["model"] if "model" in abstract_unboxed_pre_state else abstract_unboxed_pre_state
    return model_state.filter(lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat)))
  elif hasattr(abstract_unboxed_pre_state, "model") and isinstance(abstract_unboxed_pre_state.model, nnx.Module):
    state = nnx.state(abstract_unboxed_pre_state.model)
    return state.filter(lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat)))
  return abstract_unboxed_pre_state.params


def _bare_weights(tree):
  return tree["params"] if isinstance(tree, dict) and len(tree) == 1 and "params" in tree else tree


def create_orbax_checkpoint_manager(
    checkpoint_dir: str, enable_checkpointing: bool, use_async: bool, save_interval_steps: int,
    dataset_type: None | str = None, orbax_logger: Any = None, use_ocdbt: bool = True, use_zarr3: bool = True,
    enable_continuous_checkpointing: bool = False, max_num_checkpoints_to_keep: int = 10,
    checkpoint_storage_concurrent_gb: int = 96, enable_autocheckpoint: bool = False,
    todelete_subdir: str | None = None, todelete_full_path: str | None = None,
    ocdbt_target_data_file_size_bytes: int | None = None, colocated_python_checkpointing: bool = False,
):
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  if dataset_type is not None:
    max_logging.warning(
        "Specifying dataset_type upon checkpointer creation is deprecated and"
        " will be removed soon, this is now handled dynamically by Orbax Checkpointer."
    )
  max_logging.log(f"Creating checkpointer with ocdbt={use_ocdbt} and zarr3={use_zarr3}")
  validated_path = gcs_utils.mkdir_and_check_permissions(checkpoint_dir)
  if ocdbt_target_data_file_size_bytes is None:
    ocdbt_target_data_file_size_bytes = DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, ocdbt_target_data_file_size_bytes=ocdbt_target_data_file_size_bytes,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      enable_continuous_checkpointing=enable_continuous_checkpointing, todelete_full_path=todelete_full_path,
      todelete_subdir=todelete_subdir, partial_load=True, colocated_python_checkpointing=colocated_python_checkpointing,
  )
  manager = ocp.training.Checkpointer(
      validated_path, context=context,
      save_decision_policy=checkpoint_context.build_save_decision_policy(
          save_interval_steps=save_interval_steps, enable_continuous_checkpointing=enable_continuous_checkpointing,
          enable_autocheckpoint=enable_autocheckpoint,
      ),
      preservation_policy=checkpoint_context.build_preservation_policy(max_to_keep=max_num_checkpoints_to_keep),
      logger=orbax_logger,
  )
  manager.use_async = use_async
  max_logging.log("Checkpoint manager created!")
  return manager


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


def latest_step(checkpoint_manager):
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return checkpoint_manager.latest_step()
  latest = checkpoint_manager.latest
  return latest.step if latest is not None else None


def all_steps(checkpoint_manager):
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return checkpoint_manager.all_steps()
  return [checkpoint.step for checkpoint in checkpoint_manager.checkpoints]


def wait_until_finished(checkpoint_manager):
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    checkpoint_manager.wait_until_finished()
  else:
    checkpoint_manager.wait()


def is_saving_in_progress(checkpoint_manager) -> bool:
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return False
  return checkpoint_manager.is_saving_in_progress()


def reached_preemption(checkpoint_manager, step: int) -> bool:
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return checkpoint_manager.reached_preemption(step)
  return multihost_utils.reached_preemption_sync_point(step)


def load_state_if_possible(
    checkpoint_manager: CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
    load_parameters_from_path: str, load_full_state_from_path: str, checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState | nnx.State,
    enable_single_replica_ckpt_restoring: bool | None = False, dataset_type: str | None = "synthetic",
    step: int = -1, use_ocdbt=True, use_zarr3=True, enable_orbax_v1=False, checkpoint_conversion_fn=None,
    source_checkpoint_layout="orbax", expansion_factor_real_data: int = -1, maxtext_config: Any | None = None,
    snapshotter: ocp.training.pathways.Snapshotter | None = None,
    selected_source: RestoreSource | None = None,
):
  if enable_orbax_v1:
    max_logging.warning("enable_orbax_v1 is deprecated and will be removed, as Orbax v1 is now the default checkpointing API.")
  if load_parameters_from_path:
    load_parameters_from_path = _normalize_checkpoint_root(load_parameters_from_path)
  if load_full_state_from_path:
    load_full_state_from_path = _normalize_checkpoint_root(load_full_state_from_path)
  if selected_source is not None:
    if selected_source.kind == "snapshot":
      raise RuntimeError("A frozen snapshot cannot be restored by the latest-only snapshot loader.")
    if selected_source.kind == "manager" and checkpoint_manager is not selected_source.manager:
      raise ValueError("The selected checkpoint manager has changed.")
    load_parameters_from_path = selected_source.path if selected_source.kind in ("parameters", "dynamic") else ""
    load_full_state_from_path = selected_source.path if selected_source.kind == "full_state" else ""
    source_checkpoint_layout = selected_source.layout or "orbax"
  is_nnx = isinstance(abstract_unboxed_pre_state, (nnx.State, train_state_nnx.TrainStateNNX))
  if selected_source is None and snapshotting.should_restore_snapshot(snapshotter, checkpoint_manager):
    snapshot_state = snapshotting.load_snapshot(snapshotter, abstract_unboxed_pre_state, maxtext_config, data_iterator)
    if snapshot_state is not None:
      if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
        return (snapshot_state, None)
      return ({"items": snapshot_state}, None)
    max_logging.log("In-memory snapshot restore failed; falling back to checkpoint restore.")
  if checkpoint_manager is not None and (selected_source is None or selected_source.kind == "manager"):
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")
    if selected_source is None:
      if step >= 0 and step not in all_steps(checkpoint_manager):
        raise ValueError(f"Checkpoint step {step} was not found in the checkpoint manager.")
      step = latest_step(checkpoint_manager) if step < 0 else step
    else:
      step = selected_source.step
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")
      is_diloco = bool(maxtext_config and getattr(maxtext_config, "enable_diloco", False))
      if is_diloco:
        restore_target = diloco_checkpoint_utils.to_diloco_checkpoint_dict(abstract_unboxed_pre_state, config=maxtext_config)
      elif is_nnx:
        restore_target = train_state_nnx.to_checkpoint_dict(abstract_unboxed_pre_state)
      else:
        restore_target = abstract_unboxed_pre_state
      if maxtext_config and getattr(getattr(maxtext_config, "lora", None), "enable_lora", False):
        restore_target = _filter_lora_trainable_state(restore_target, preserve_optimizer=is_nnx and not is_diloco)
      if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
        restored = emergency_checkpointing.restore(checkpoint_manager, step, restore_target)
        if is_diloco:
          restored = diloco_checkpoint_utils.from_diloco_checkpoint_dict(restored, abstract_unboxed_pre_state, config=maxtext_config)
        elif is_nnx:
          restored = restored_linen_to_nnx(restored, abstract_unboxed_pre_state, config=maxtext_config)
        return (restored, None)
      assert isinstance(checkpoint_manager, ocp.training.Checkpointer)
      if selected_source is not None:
        selected_metadata = checkpoint_manager.checkpointables_metadata(step)
        if str(selected_metadata.path) != selected_source.path:
          raise ValueError("The selected checkpoint path has changed.")
      abstract_checkpointables = {"items": restore_target}
      if (
          dataset_type == "grain" and data_iterator and not isinstance(data_iterator, PlaceHolderDataIterator)
          and (checkpoint_manager.directory / str(step) / "iter").exists()
      ):
        abstract_checkpointables["iter"] = grain_utility.for_restore(
            checkpoint_manager, step, data_iterator, expansion_factor_real_data
        )
      restored = checkpoint_manager.load_checkpointables(
          step if selected_source is None else selected_metadata, abstract_checkpointables
      )
      if is_diloco:
        restored_items = diloco_checkpoint_utils.from_diloco_checkpoint_dict(
            restored["items"], abstract_unboxed_pre_state, config=maxtext_config
        )
        restored = {"items": restored_items}
      elif is_nnx:
        restored_items = restored_linen_to_nnx(restored["items"], abstract_unboxed_pre_state, config=maxtext_config)
        restored = {"items": restored_items}
      return (restored, None)
  if source_checkpoint_layout == "safetensors_dynamic":
    path = load_parameters_from_path or load_full_state_from_path
    max_logging.log(f"Dynamic On-the-Fly Formatting: Loading SafeTensors from {path}")
    params = _abstract_params(abstract_unboxed_pre_state)
    restored, restored_params = load_safetensors_dynamic_state(path, params, maxtext_config)
    _raise_on_weight_mismatch(_bare_weights(params.to_pure_dict() if is_nnx else params), _bare_weights(restored_params))
    if is_nnx:
      nnx.replace_by_pure_dict(params, restored_params["params"])
      return restored, params
    return restored, restored_params
  elif load_parameters_from_path != "":
    params = _abstract_params(abstract_unboxed_pre_state)
    restored_params = load_params_from_path(
        load_parameters_from_path, params, checkpoint_storage_concurrent_gb, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3,
        enable_single_replica_ckpt_restoring=bool(enable_single_replica_ckpt_restoring),
        convert_flat_moe_weights=bool(getattr(maxtext_config, "moe_flat_fsdp_weights", False)),
        colocated_python_checkpointing=bool(getattr(maxtext_config, "colocated_python_checkpointing", False)),
    )
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
    restored_state = _load_full_state_from_path(
        path=load_full_state_from_path, abstract_unboxed_pre_state=abstract_unboxed_pre_state,
        checkpoint_conversion_fn=checkpoint_conversion_fn, source_checkpoint_layout=source_checkpoint_layout,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3,
        enable_single_replica_ckpt_restoring=bool(enable_single_replica_ckpt_restoring), maxtext_config=maxtext_config,
    )
    return {"items": restored_state}, None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:
  if not config.enable_checkpoint_cloud_logger:
    return None
  max_logging.log("Setting up checkpoint logger...")
  cloud_logger_cls = getattr(ocp_logging, "CloudLogger", None)
  cloud_logger_options_cls = getattr(ocp_logging, "CloudLoggerOptions", None)
  if cloud_logger_cls is None or cloud_logger_options_cls is None:
    max_logging.warning(
        "enable_checkpoint_cloud_logger is set but Orbax CloudLogger is unavailable"
        " (is google-cloud-logging installed?). Checkpoint logs are disabled."
    )
    return None
  logger_name = f"goodput_{config.run_name}"
  try:
    orbax_cloud_logger = cloud_logger_cls(options=cloud_logger_options_cls(job_name=config.run_name, logger_name=logger_name))
  except Exception as error:
    max_logging.warning(f"Failed to initialize Orbax CloudLogger: {error}. Checkpoint logs are disabled.")
    return None
  max_logging.log(f"Successfully set up checkpoint cloud logger: {logger_name}")
  return orbax_cloud_logger


def _set_path(tree, path, value):
  node = tree
  for key in path[:-1]:
    if not isinstance(node, dict) or key not in node:
      return False
    node = node[key]
  if not isinstance(node, dict) or path[-1] not in node:
    return False
  node[path[-1]] = value
  return True


def load_params_from_path(
    load_parameters_from_path, abstract_unboxed_params, checkpoint_storage_concurrent_gb, use_ocdbt=True,
    use_zarr3=True, enable_single_replica_ckpt_restoring: bool = False, convert_flat_moe_weights: bool = False,
    colocated_python_checkpointing: bool = False,
):
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")
  path = epath.Path(_normalize_checkpoint_root(load_parameters_from_path))
  is_nnx = isinstance(abstract_unboxed_params, nnx.State)
  want = abstract_unboxed_params.to_pure_dict() if is_nnx else abstract_unboxed_params
  restore_key = os.path.basename(str(load_parameters_from_path).rstrip("/"))
  if restore_key not in ("model_params", "model"):
    restore_key = "params"
  checkpointable_name = "items" if (path / "items").exists() else None
  metadata_context = (
      checkpoint_context.build_context(colocated_python_checkpointing=True)
      if colocated_python_checkpointing else contextlib.nullcontext()
  )
  try:
    with metadata_context:
      stored = ocp.metadata(path, checkpointable_name=checkpointable_name).metadata
  except Exception as error:
    max_logging.log(f"Skipping pre-load shape check, checkpoint metadata unreadable: {error}")
    stored = None
  flat_moe_plan = {}
  if convert_flat_moe_weights and isinstance(stored, dict):
    stored_root = stored.get(restore_key)
    stored_params = (
        stored_root.get("params")
        if is_nnx and isinstance(stored_root, dict) and isinstance(stored_root.get("params"), dict)
        else stored_root
    )
    flat_moe_plan = _flat_moe_restore_plan(want, stored_params)
    for weight_path, (_, stored_sds) in flat_moe_plan.items():
      _set_path(want, weight_path, stored_sds)
    if flat_moe_plan:
      max_logging.log(
          f"moe_flat_fsdp_weights: {len(flat_moe_plan)} routed-expert weights will be restored in the checkpoint's"
          f" 3D layout and converted to the flat layout: {['/'.join(map(str, path)) for path in flat_moe_plan]}"
      )
  aliased_collections = {}
  if restore_key in ("model_params", "model"):
    params_collection = want
  elif is_nnx and restore_key == "params":
    stored_params_meta = stored.get("params") if isinstance(stored, dict) and isinstance(stored.get("params"), dict) else {}
    stored_collections = set(stored_params_meta.keys()) if stored_params_meta else None
    collection_leaves = collections.defaultdict(dict)
    for path_tuple, leaf in nnx.to_flat_state(abstract_unboxed_params):
      if isinstance(leaf, nnx.Variable):
        col = variablelib.variable_name_from_type(leaf.type, allow_register=True)
        value = leaf.get_value() if hasattr(leaf, "get_value") else leaf
      else:
        col = "params"
        value = leaf
      collection_leaves[col][tuple(path_tuple)] = value
    params_collection = {col: nnx.traversals.unflatten_mapping(values) for col, values in collection_leaves.items()}
    if "params" not in params_collection:
      params_collection["params"] = want
    for weight_path, (_, stored_sds) in flat_moe_plan.items():
      for col_tree in params_collection.values():
        _set_path(col_tree, weight_path, stored_sds)
    if "params" in params_collection:
      params_collection["params"] = _augment_want_with_scales(params_collection["params"], stored, is_nnx=True, restore_key="params")
    params_collection, aliased_collections = _alias_legacy_collections(params_collection, stored_params_meta)
    params_collection = {
        col: tree for col, tree in params_collection.items()
        if stored_collections is None or col in stored_collections or col == "params"
    }
  else:
    params_collection = _augment_want_with_scales(want, stored, is_nnx=False, restore_key=restore_key)
    if isinstance(stored, dict):
      params_collection, aliased_collections = _alias_legacy_collections(params_collection, stored.get(restore_key))
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True, enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
      colocated_python_checkpointing=colocated_python_checkpointing,
  )
  audit_stored = None
  with context:
    if isinstance(stored, dict):
      stored_collection = stored.get(restore_key)
      audit_stored = stored_collection
      if restore_key == "params" and is_nnx and isinstance(stored_collection, dict):
        audit_stored = {}
        for col_tree in stored_collection.values():
          if isinstance(col_tree, dict):
            audit_stored = _deep_merge_dicts(audit_stored, col_tree)
        merged_stored = {}
        for col_name in params_collection.keys():
          if col_name in stored_collection and isinstance(stored_collection[col_name], dict):
            merged_stored = _deep_merge_dicts(merged_stored, stored_collection[col_name])
        stored_collection = merged_stored
      _raise_weight_problems(_weight_mismatches(want, stored_collection, check_missing=False))
    restored = ocp.load(path, {restore_key: params_collection}, checkpointable_name=checkpointable_name)
  restored_collection = restored[restore_key]
  if not is_nnx and aliased_collections:
    restored_collection = _unalias_legacy_collections(restored_collection, aliased_collections)
  if restore_key in ("model_params", "model"):
    restored_weights = restored_collection
  elif is_nnx:
    restored_weights = {}
    if isinstance(restored_collection, dict):
      for col_key in sorted(restored_collection, key=lambda key: (key != "params", key)):
        col_val = restored_collection[col_key]
        if isinstance(col_val, dict):
          restored_weights = _deep_merge_dicts(restored_weights, col_val)
        else:
          restored_weights[col_key] = col_val
    else:
      restored_weights = restored_collection
  else:
    restored_weights = restored_collection
  for weight_path, (flat_sds, stored_sds) in flat_moe_plan.items():
    array = _lookup_path(restored_weights, weight_path)
    if not isinstance(array, jax.Array):
      continue
    start = time.time()
    converted = _flat_moe_convert(array, flat_sds)
    assert _set_path(restored_weights, weight_path, converted)
    _set_path(want, weight_path, flat_sds)
    max_logging.log(
        f"moe_flat_fsdp_weights: converted {'/'.join(map(str, weight_path))} {tuple(stored_sds.shape)} ->"
        f" {tuple(converted.shape)} in {time.time() - start:.1f}s"
    )
  restored_weights = maybe_dequantize_restored_params(restored_weights, want)
  _raise_on_weight_mismatch(want, restored_weights)
  _log_unconsumed_checkpoint_weights(want, audit_stored)
  if is_nnx:
    nnx.replace_by_pure_dict(abstract_unboxed_params, restored_weights)
    return abstract_unboxed_params
  return restored_weights


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  assert checkpoint_dir, "checkpoint_dir is not defined."
  max_logging.log(f"Saving params checkpoint with use_ocdbt={use_ocdbt} and" f" use_zarr3={use_zarr3}")
  context = checkpoint_context.build_context(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
  with context:
    ocp.save(checkpoint_dir, {"params": params}, checkpointable_name="items", overwrite=True)
  max_logging.log(f"Params checkpoint saved at: {checkpoint_dir}")


def load_checkpoint_metadata(checkpoint_dir_path: str) -> Any:
  checkpoint_dir = epath.Path(_normalize_checkpoint_root(checkpoint_dir_path))
  try:
    metadata = ocp.checkpointables_metadata(checkpoint_dir)
    return metadata.custom_metadata or {}
  except Exception as error:
    max_logging.log(f"Warning: Failed to load checkpoint metadata: {error}")
    return {}


def _uses_local_checkpoint_period(config):
  return config.enable_emergency_checkpoint or config.enable_multi_tier_checkpointing


def _should_save_checkpoint_at_step(checkpoint_manager, step, config, force):
  if force:
    return True
  if step == 0 and not config.save_checkpoint_on_start:
    return False
  if config.enable_continuous_checkpointing:
    base_checkpoint_due = bool(checkpoint_manager.should_save(step))
  else:
    base_checkpoint_due = step % config.checkpoint_period == 0
  local_checkpoint_due = _uses_local_checkpoint_period(config) and step % config.local_checkpoint_period == 0
  autocheckpoint_due = config.enable_autocheckpoint and reached_preemption(checkpoint_manager, step)
  return base_checkpoint_due or local_checkpoint_due or autocheckpoint_due


def _handle_post_checkpoint_preemption(checkpoint_manager, step, force_ckpt_save):
  is_preempted = reached_preemption(checkpoint_manager, step)
  if force_ckpt_save or is_preempted:
    wait_until_finished(checkpoint_manager)
  if is_preempted:
    raise exceptions.StopTraining("Job received termination signal (SIGTERM).")


@contextlib.contextmanager
def checkpoint_exception_guard(config, checkpoint_manager, handler_fn=None):
  try:
    yield
    elastic_utils.maybe_elastic_scale_up(config, checkpoint_manager)
  except Exception as error:
    elastic_utils.maybe_bubble_elastic_exception(config, error)
    if handler_fn:
      handler_fn(error)
    else:
      raise


def maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None, training_graph=None):
  if checkpoint_manager is None:
    return
  if step is not None:
    actual_step = int(step)
  else:
    actual_step = int(state.step if config.enable_diloco else state.optimizer.step) - 1
  force_ckpt_save = step is None and actual_step != -1 and (actual_step % config.checkpoint_period != 0)
  if not _should_save_checkpoint_at_step(checkpoint_manager, actual_step, config, force_ckpt_save):
    _handle_post_checkpoint_preemption(checkpoint_manager, actual_step, force_ckpt_save)
    return
  if latest_step(checkpoint_manager) == actual_step or actual_step in all_steps(checkpoint_manager):
    max_logging.log(f"Checkpoint for step {actual_step} already exists, skipping save.")
    return

  def _checkpoint_error_handler(err):
    raise RuntimeError(f"Checkpointing failed. {str(err)}") from err

  with checkpoint_exception_guard(config, checkpoint_manager, _checkpoint_error_handler):
    checkpoint_saved = save_checkpoint(
        checkpoint_manager, actual_step, state, config, data_iterator, force_ckpt_save, training_graph=training_graph
    )
  if checkpoint_saved:
    print_save_message(actual_step, config.async_checkpointing)
  _handle_post_checkpoint_preemption(checkpoint_manager, actual_step, force_ckpt_save)


def save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False, training_graph=None):
  if not isinstance(state, (dict, nnx.State, train_state.TrainState, struct.PyTreeNode)):
    if isinstance(state, train_state_nnx.TrainStateNNX):
      state = nnx.state(state)
    elif not isinstance(state, (dict, nnx.State)):
      state = {}
  is_nnx = isinstance(state, nnx.State) and not (config and getattr(config, "enable_diloco", False))
  if config and getattr(config, "enable_diloco", False):
    state = diloco_checkpoint_utils.to_diloco_checkpoint_dict(state, config)
  elif config:
    if isinstance(state, nnx.State):
      state = train_state_nnx.to_checkpoint_dict(state)
  if config and getattr(config, "enable_checkpointing", False):
    if (
        force
        or (step % config.checkpoint_period == 0 and not getattr(config, "enable_continuous_checkpointing", False))
        or (_uses_local_checkpoint_period(config) and step % config.local_checkpoint_period == 0)
        or (getattr(config, "enable_autocheckpoint", False) and reached_preemption(checkpoint_manager, step))
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    filtered = _filter_lora_trainable_state(state, preserve_optimizer=is_nnx)
    if filtered:
      state = filtered
  if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
    return emergency_checkpointing.save(checkpoint_manager, step, state, config, force)
  custom_metadata = {}
  if training_graph is not None:
    custom_metadata["maxtext_easy_training_graph"] = training_graph
  if config:
    if hasattr(config, "scan_layers"):
      custom_metadata["scan_layers"] = config.scan_layers
    if hasattr(config, "lora") and config.lora and getattr(config.lora, "lora_rank", 0) > 0:
      custom_metadata["lora"] = config.lora.model_dump()
  checkpointables = {"items": state}
  if config and getattr(config, "dataset_type", None) == "grain" and not isinstance(data_iterator, PlaceHolderDataIterator):
    checkpointables["iter"] = grain_utility.for_save(data_iterator, config.expansion_factor_real_data)
  try:
    if getattr(checkpoint_manager, "use_async", False):
      response = checkpoint_manager.save_checkpointables_async(
          step, checkpointables, force=force, custom_metadata=custom_metadata
      )
      return response is not None
    return checkpoint_manager.save_checkpointables(step, checkpointables, force=force, custom_metadata=custom_metadata)
  except FileExistsError as error:
    max_logging.log(f"Checkpoint for step {step} already exists, skipping save. ({error})")
    return False
