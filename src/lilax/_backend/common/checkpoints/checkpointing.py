"""Create and operate Orbax checkpoint managers."""

import collections
import contextlib
import dataclasses
import importlib
import os
import time
from typing import Any

from etils import epath
from flax import nnx, struct
from flax.nnx import variablelib
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from lilax._backend.checkpoint_conversion.utils import load_dynamic
from lilax._backend.common.checkpoints import checkpoint_context, emergency_checkpointing, grain_utility, snapshotting
from lilax._backend.common import train_state_nnx
from lilax._backend.input_pipeline import multihost_dataloading, synthetic_data_processing
from lilax._backend.layers.quantization.weights import dequantize_weight
from lilax._backend.trainers.diloco.utils import spmd_diloco_checkpointing as diloco_checkpoint_utils
from lilax._backend.utils import elastic_utils, exceptions, gcs_utils
from lilax._backend.utils import globals as maxtext_globals
from lilax._backend.utils import max_logging
from lilax._backend.utils import sharding as maxtext_sharding
from orbax.checkpoint import logging as ocp_logging
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.arrays import sharding as sharding_utils

load_safetensors_dynamic_state = load_dynamic.load_safetensors_dynamic_state
PlaceHolderDataIterator = synthetic_data_processing.PlaceHolderDataIterator
MultiHostDataLoadIterator = multihost_dataloading.MultiHostDataLoadIterator
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = maxtext_globals.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
EmergencyCheckpointManager = emergency_checkpointing.CheckpointManager
EmergencyReplicatorCheckpointManager = emergency_checkpointing.ReplicatorCheckpointManager
create_orbax_emergency_checkpoint_manager = emergency_checkpointing.create_emergency_checkpoint_manager
create_orbax_emergency_replicator_checkpoint_manager = emergency_checkpointing.create_replicator_checkpoint_manager
CheckpointManager = ocp.training.Checkpointer | EmergencyCheckpointManager | EmergencyReplicatorCheckpointManager


def _deep_merge_dicts(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
  out = dict(target)
  for key, value in source.items():
    if key in out and isinstance(out[key], dict) and isinstance(value, dict):
      out[key] = _deep_merge_dicts(out[key], value)
    else:
      out[key] = value
  return out


def _weight_mismatches(want, have, path=(), check_missing: bool = True, is_quantized_param: bool = False):
  if isinstance(want, dict):
    out = []
    is_quant = is_quantized_param or any(key in want for key in ("qvalue", "qarray"))
    for key, value in want.items():
      nested = have.get(key) if isinstance(have, dict) else None
      if check_missing or nested is not None:
        out.extend(_weight_mismatches(value, nested, path + (key,), check_missing=check_missing, is_quantized_param=is_quant))
    return out
  name = "/".join(str(part) for part in path)
  is_missing = have is None or (check_missing and isinstance(have, jax.ShapeDtypeStruct))
  if is_missing and ("lora_a" in name or "lora_b" in name or "rngs" in path or "rng" in path):
    return []
  if is_missing and is_quantized_param:
    return []
  if is_missing:
    return [(name, f"missing (model expects {getattr(want, 'shape', '?')} {getattr(want, 'dtype', '?')})")] if check_missing else []
  want_shape, got_shape = getattr(want, "shape", None), getattr(have, "shape", None)
  if want_shape is not None and got_shape is None:
    return [(name, f"structural mismatch: model expects a tensor but checkpoint provides {type(have).__name__}")]
  if want_shape is not None and got_shape is not None and tuple(want_shape) != tuple(got_shape):
    return [(name, f"shape {tuple(got_shape)} but the model expects {tuple(want_shape)}")]
  return []


def _lookup_path(tree, path):
  node = tree
  for key in path:
    if not isinstance(node, dict) or key not in node:
      return None
    node = node[key]
  return node


def _alias_legacy_collections(params_collection, stored_collection_meta):
  if not isinstance(stored_collection_meta, dict) or not stored_collection_meta:
    return params_collection, {}
  stored_params = stored_collection_meta.get("params")
  if not isinstance(stored_params, dict):
    return params_collection, {}
  patched = dict(params_collection)
  params_flat = dict(nnx.traversals.flatten_mapping(patched.get("params", {})))
  aliased = {}
  for col, tree in params_collection.items():
    if col == "params" or col in stored_collection_meta or not isinstance(tree, dict):
      continue
    tree_flat = dict(nnx.traversals.flatten_mapping(tree))
    moved = []
    for path, leaf in tree_flat.items():
      if _lookup_path(stored_params, path) is None:
        continue
      params_flat[path] = leaf
      moved.append(path)
    if moved:
      aliased[col] = moved
      moved_set = set(moved)
      remaining = {path: value for path, value in tree_flat.items() if path not in moved_set}
      if remaining:
        patched[col] = nnx.traversals.unflatten_mapping(remaining)
      else:
        patched.pop(col, None)
      max_logging.log(
          f"Checkpoint predates the '{col}' collection; restoring its {len(moved)} weight(s) "
          "from the 'params' collection instead of skipping them."
      )
  if aliased:
    patched["params"] = nnx.traversals.unflatten_mapping(params_flat)
  return patched, aliased


def _unalias_legacy_collections(restored_collection, aliased):
  if not aliased or not isinstance(restored_collection, dict):
    return restored_collection
  out = dict(restored_collection)
  params_flat = dict(nnx.traversals.flatten_mapping(out.get("params", {})))
  for col, paths in aliased.items():
    col_flat = dict(nnx.traversals.flatten_mapping(out.get(col, {})))
    for path in paths:
      if path in params_flat:
        col_flat[path] = params_flat.pop(path)
    if col_flat:
      out[col] = nnx.traversals.unflatten_mapping(col_flat)
  out["params"] = nnx.traversals.unflatten_mapping(params_flat)
  return out


def _log_unconsumed_checkpoint_weights(want, stored_collection):
  if not isinstance(want, dict) or not isinstance(stored_collection, dict):
    return
  try:
    want_paths = set(nnx.traversals.flatten_mapping(want).keys())
    stored_paths = set(nnx.traversals.flatten_mapping(stored_collection).keys())
  except Exception:
    return
  unconsumed = sorted("/".join(str(part) for part in path) for path in stored_paths - want_paths)
  if not unconsumed:
    return
  preview = ", ".join(unconsumed[:5])
  max_logging.log(
      f"WARNING: {len(unconsumed)} checkpoint weight(s) were not requested by the model and "
      f"will be ignored (the model will use its initialised values). First few: {preview}"
  )


def _expected_and_restored_params(abstract_nnx_state, restored_linen):
  want = nnx.split_state(abstract_nnx_state, nnx.Param, ...)[0].to_pure_dict().get("model", {})
  have = restored_linen.get("params", {}).get("params", {})
  return want, have


def _raise_weight_problems(problems):
  if not problems:
    return
  lines = "\n".join(f"  - '{path}': {reason}" for path, reason in problems)
  raise ValueError(
      "Checkpoint does not match the model:\n"
      f"{lines}\n"
      "Verify the checkpoint matches the model architecture (emb_dim, mlp_dim, num layers, scan_layers)."
  )


def _is_custom_projector_problem(path: str, want: dict) -> bool:
  parts = [part for part in path.replace(".", "/").split("/") if part and part != "params"]
  if len(parts) >= 2 and parts[0] == "vision_encoder":
    proj_name = parts[1]
    proj_dict = want.get("vision_encoder", {}).get(proj_name, {})
    if isinstance(proj_dict, dict) and any("custom_linear" in key for key in proj_dict.keys()):
      max_logging.warning(
          f"===Warning: weight mismatch found in custom vision projector: {proj_name}.\n"
          f"Path: {path}\n"
          "This custom vision projector will be initialized with random weights.==="
      )
      return True
  return False


def _raise_on_weight_mismatch(want, have, config=None):
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    want = _filter_lora_trainable_state(want)
  problems = _weight_mismatches(want, have)
  problems = [(path, reason) for path, reason in problems if not _is_custom_projector_problem(path, want)]
  _raise_weight_problems(problems)


def _linen_items_to_nnx(restored_linen, abstract_nnx_state):
  linen_state, aux_state, ephemeral = train_state_nnx.split_for_checkpoint(abstract_nnx_state)
  weights = train_state_nnx.from_linen_checkpoint_dict(restored_linen)
  if "model" in weights:
    nnx.replace_by_pure_dict(linen_state, {"model": weights["model"]})
  if "optimizer" in weights:
    nnx.replace_by_pure_dict(linen_state, {"optimizer": weights["optimizer"]})
  nnx_aux = restored_linen.get("nnx_aux")
  if nnx_aux:
    train_state_nnx.apply_checkpoint_aux(aux_state, nnx_aux)
  elif isinstance(restored_linen.get("params"), dict):
    custom_collections = {key: value for key, value in restored_linen["params"].items() if key != "params" and isinstance(value, dict)}
    if custom_collections:
      merged_custom = {}
      for col_data in custom_collections.values():
        merged_custom = _deep_merge_dicts(merged_custom, col_data)
      train_state_nnx.apply_checkpoint_aux(aux_state, {"model": merged_custom})
  return nnx.merge_state(linen_state, aux_state, ephemeral)


def _load_linen_checkpoint_into_nnx(
    path, abstract_nnx_state, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3,
    enable_single_replica_ckpt_restoring: bool = False, config=None,
):
  max_logging.log(f"Restoring Linen-layout checkpoint into NNX state at {path}")
  if config and getattr(config, "enable_diloco", False):
    return diloco_checkpoint_utils.restore_diloco_checkpoint(
        path, abstract_nnx_state, checkpoint_storage_concurrent_gb, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, config=config
    )
  linen_abstract = train_state_nnx.to_checkpoint_dict(abstract_nnx_state)
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    linen_abstract = _filter_lora_trainable_state(linen_abstract, preserve_optimizer=True)
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True, enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
      colocated_python_checkpointing=bool(getattr(config, "colocated_python_checkpointing", False)),
  )
  root = epath.Path(_normalize_checkpoint_root(path))
  with context:
    checkpointable_name = "items" if (root / "items").exists() else None
    restored = ocp.load(root, linen_abstract, checkpointable_name=checkpointable_name)
  return restored_linen_to_nnx(restored, abstract_nnx_state, config=config)


def restored_linen_to_nnx(restored_linen, abstract_nnx_state, config=None):
  _raise_on_weight_mismatch(*_expected_and_restored_params(abstract_nnx_state, restored_linen), config=config)
  return _linen_items_to_nnx(restored_linen, abstract_nnx_state)


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


def _resolve_conversion_fn(checkpoint_conversion_fn):
  if checkpoint_conversion_fn is None:
    raise ValueError(
        "source_checkpoint_layout='safetensors' needs `checkpoint_conversion_fn` to map the "
        "checkpoint's weights onto the model's, e.g. checkpoint_conversion_fn=my_pkg.my_module.my_fn."
    )
  if callable(checkpoint_conversion_fn):
    return checkpoint_conversion_fn
  module_name, _, fn_name = str(checkpoint_conversion_fn).rpartition(".")
  if not module_name:
    raise ValueError(f"`checkpoint_conversion_fn` must be a dotted path to a function, got {checkpoint_conversion_fn!r}.")
  try:
    fn = getattr(importlib.import_module(module_name), fn_name, None)
  except ImportError as error:
    raise ValueError(f"Could not import `checkpoint_conversion_fn` {checkpoint_conversion_fn!r}: {error}") from error
  if not callable(fn):
    raise ValueError(f"`checkpoint_conversion_fn` {checkpoint_conversion_fn!r} is not a function.")
  return fn


def _load_full_state_from_path(
    path, abstract_unboxed_pre_state, checkpoint_conversion_fn, source_checkpoint_layout,
    checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3, enable_single_replica_ckpt_restoring: bool = False,
    maxtext_config=None,
):
  if source_checkpoint_layout == "orbax":
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      return _load_linen_checkpoint_into_nnx(
          path, abstract_unboxed_pre_state, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3,
          enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring, config=maxtext_config,
      )
    context = checkpoint_context.build_context(
        use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.ORBAX,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
        colocated_python_checkpointing=bool(getattr(maxtext_config, "colocated_python_checkpointing", False)),
    )
    with context:
      return ocp.load(path, abstract_unboxed_pre_state)
  if source_checkpoint_layout == "safetensors":
    if enable_single_replica_ckpt_restoring:
      max_logging.warning("enable_single_replica_ckpt_restoring is not supported for safetensors layout.")
    conversion_fn = _resolve_conversion_fn(checkpoint_conversion_fn)
    context = checkpoint_context.build_context(
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS,
    )
    with context:
      metadata = ocp.metadata(path)
      simple_abstract_state = metadata.metadata
      shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

      def combine_sharding(sds, shardings):
        return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

      sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
      pre_transformed_state = ocp.load(path, sharded_abstract_state)
    state = conversion_fn(pre_transformed_state)
    if isinstance(abstract_unboxed_pre_state, nnx.State) and not isinstance(state, nnx.State):
      state = restored_linen_to_nnx(state, abstract_unboxed_pre_state, config=maxtext_config)
    return state
  raise ocp.errors.InvalidLayoutError(f"Unknown checkpoint layout: {source_checkpoint_layout}")


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


def _normalize_checkpoint_root(path_str):
  path_str = str(path_str).rstrip("/")
  if path_str == "items":
    return "."
  return path_str.removesuffix("/items")


@dataclasses.dataclass(frozen=True)
class RestoreSource:
  kind: str
  path: str | None = None
  step: int | None = None
  layout: str | None = None
  manager: Any = dataclasses.field(default=None, repr=False, compare=False)


def select_restore_source(
    checkpoint_manager, load_parameters_from_path, load_full_state_from_path, *, step=-1,
    source_checkpoint_layout="orbax", snapshotter=None,
):
  if snapshotting.should_restore_snapshot(snapshotter, checkpoint_manager):
    return RestoreSource("snapshot", step=snapshotting.latest_snapshot_step(snapshotter))
  if checkpoint_manager is not None:
    if step >= 0 and step not in all_steps(checkpoint_manager):
      raise ValueError(f"Checkpoint step {step} was not found in the checkpoint manager.")
    selected_step = latest_step(checkpoint_manager) if step < 0 else step
    if selected_step is not None:
      if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
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


def _scale_sharding(weight_sharding, scale_shape):
  mesh = getattr(weight_sharding, "mesh", None)
  if mesh is None:
    return None
  spec = getattr(weight_sharding, "spec", None)
  scale_shape = tuple(scale_shape)
  if spec is None or len(spec) != len(scale_shape):
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  try:
    return jax.sharding.NamedSharding(mesh, maxtext_sharding.adjust_pspec_for_indivisible_shapes(spec, scale_shape, mesh))
  except Exception:
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _augment_target_with_scales(want_node: Any, meta_node: Any) -> Any:
  if not isinstance(want_node, dict) or not isinstance(meta_node, dict):
    return want_node
  augmented = {}
  for key, value in want_node.items():
    scale_key = f"{key}_scale"
    if isinstance(value, dict):
      augmented[key] = _augment_target_with_scales(value, meta_node.get(key))
    elif scale_key not in want_node and scale_key in meta_node:
      meta_scale = meta_node[scale_key]
      meta_param = meta_node.get(key)
      param_dtype = getattr(meta_param, "dtype", getattr(value, "dtype", jnp.bfloat16))
      weight_sharding = getattr(value, "sharding", None)
      augmented[key] = jax.ShapeDtypeStruct(shape=value.shape, dtype=param_dtype, sharding=weight_sharding)
      augmented[scale_key] = jax.ShapeDtypeStruct(
          shape=meta_scale.shape, dtype=meta_scale.dtype, sharding=_scale_sharding(weight_sharding, meta_scale.shape)
      )
    else:
      augmented[key] = value
  return augmented


def _augment_want_with_scales(want: Any, stored: Any, is_nnx: bool, restore_key: str) -> Any:
  if not isinstance(stored, dict) or not isinstance(want, dict):
    return want
  meta_weights = stored.get(restore_key)
  if restore_key == "params" and is_nnx and isinstance(meta_weights, dict):
    meta_weights = meta_weights.get("params")
  if not is_nnx and "params" in want and isinstance(meta_weights, dict) and "params" not in meta_weights:
    return {"params": _augment_target_with_scales(want["params"], meta_weights)}
  return _augment_target_with_scales(want, meta_weights)


def maybe_dequantize_restored_params(restored_weights: Any, expected_param_pytree: Any) -> Any:
  if not isinstance(restored_weights, dict):
    return restored_weights
  expected_dict = expected_param_pytree if isinstance(expected_param_pytree, dict) else {}
  scales_to_drop = {
      f"{key}_scale" for key in restored_weights if f"{key}_scale" in restored_weights and f"{key}_scale" not in expected_dict
  }
  out = {}
  for key, value in restored_weights.items():
    if key in scales_to_drop:
      continue
    scale_key = f"{key}_scale"
    if scale_key in scales_to_drop:
      target_param = expected_dict.get(key)
      target_dtype = getattr(target_param, "dtype", jnp.bfloat16)
      out[key] = dequantize_weight(value, restored_weights[scale_key], compute_dtype=target_dtype)
    elif isinstance(value, dict):
      out[key] = maybe_dequantize_restored_params(value, expected_dict.get(key))
    else:
      out[key] = value
  return out


_FLAT_MOE_LEAVES = ("wi_0", "wi_1", "wo")


def _flat_moe_restore_plan(want, stored_weights):
  plan = {}

  def _walk(want_node, stored_node, path):
    if isinstance(want_node, dict):
      for key, value in want_node.items():
        _walk(value, stored_node.get(key) if isinstance(stored_node, dict) else None, path + (key,))
      return
    if not path or path[-1] not in _FLAT_MOE_LEAVES or stored_node is None:
      return
    want_shape = tuple(getattr(want_node, "shape", ()) or ())
    stored_shape = tuple(getattr(stored_node, "shape", ()) or ())
    if not want_shape or not stored_shape or want_shape == stored_shape or len(want_shape) != len(stored_shape) - 1:
      return
    if len(stored_shape) == 4:
      experts, layers, rows, cols = stored_shape
      flat_shape = (experts * rows, layers, cols)
    elif len(stored_shape) == 3:
      experts, rows, cols = stored_shape
      flat_shape = (experts * rows, cols)
    else:
      return
    if want_shape != flat_shape:
      return
    sharding = getattr(want_node, "sharding", None)
    mesh = getattr(sharding, "mesh", None)
    spec = getattr(sharding, "spec", None)
    if mesh is None or spec is None:
      return
    dim0 = spec[0] if len(spec) > 0 else None
    dim0_axes = () if dim0 is None else ((dim0,) if isinstance(dim0, str) else tuple(dim0))
    expert_axis = dim0_axes[0] if len(dim0_axes) > 1 else None
    rest = dim0_axes[1:] if len(dim0_axes) > 1 else dim0_axes
    rest_spec = None if not rest else (rest[0] if len(rest) == 1 else rest)
    tail = tuple(spec)[1:] + (None,) * max(0, len(flat_shape) - len(spec))
    if len(stored_shape) == 4:
      stored_spec = jax.sharding.PartitionSpec(expert_axis, tail[0], rest_spec, tail[1])
    else:
      stored_spec = jax.sharding.PartitionSpec(expert_axis, rest_spec, tail[0])
    stored_sds = jax.ShapeDtypeStruct(stored_shape, want_node.dtype, sharding=jax.sharding.NamedSharding(mesh, stored_spec))
    plan[path] = (want_node, stored_sds)

  _walk(want, stored_weights, ())
  return plan


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


def _flat_moe_convert(x, target):
  out_sharding = target.sharding
  mesh = out_sharding.mesh
  spec = tuple(out_sharding.spec) + (None,) * (len(target.shape) - len(out_sharding.spec))
  if x.ndim == 4:
    experts, layer_count, rows, cols = x.shape
    layer_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(spec[0], spec[2]))

    def _layer_fn(array, index):
      return jax.lax.dynamic_index_in_dim(array, index, axis=1, keepdims=False).reshape(experts * rows, cols).astype(target.dtype)

    layer_fn = jax.jit(_layer_fn, out_shardings=layer_sharding)
    layers = [layer_fn(x, jnp.int32(index)) for index in range(layer_count)]
    x.delete()
    stacked = jax.jit(lambda *arrays: jnp.stack(arrays, axis=1), out_shardings=out_sharding)(*layers)
    for layer in layers:
      layer.delete()
    return stacked
  experts, rows, cols = x.shape
  result = jax.jit(lambda array: array.reshape(experts * rows, cols).astype(target.dtype), out_shardings=out_sharding)(x)
  x.delete()
  return result


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


def _filter_lora_trainable_state(state, preserve_optimizer=False):
  def _lora_filter(path, val):
    path_str = "/".join(str(part) for part in path).lower()
    return isinstance(val, nnx.LoRAParam) or "lora" in path_str or "step" in path_str

  if isinstance(state, (nnx.State, nnx.Module)):
    return nnx.state(state, _lora_filter)

  def _filter_dict(val, path=()):
    if path == ("nnx_aux",) or (preserve_optimizer and path == ("opt_state",)):
      return val
    if isinstance(val, dict):
      res = {}
      for key, value in val.items():
        curr_path = path + (str(key),)
        filtered = _filter_dict(value, curr_path)
        if filtered is not None:
          res[key] = filtered
      return res if res else None
    path_str = "/".join(path).lower()
    if "lora" in path_str or "step" in path_str:
      return val
    return None

  return _filter_dict(state)


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
