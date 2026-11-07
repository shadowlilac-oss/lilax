"""Initialize a sharded train state and restore it from checkpoints or parameters."""

from flax import nnx, traverse_util
from flax.core.spmd import logical_axis_rules as axis_rules
import jax
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.trainers.diloco import diloco
from lilax._backend.trainers.diloco import sync as diloco_utils
from lilax._backend.distributed import state_abstraction


def setup_training_state(
    data_iterator, config, mesh, checkpoint_manager, init_state_fn, snapshotter=None, step=-1, selected_source=None,
    init_state_args=(),
):
  return setup_initial_state(
      data_iterator, config, mesh, checkpoint_manager, init_state_fn, True, snapshotter=snapshotter, step=step,
      selected_source=selected_source, init_state_args=init_state_args,
  )


def setup_initial_state(
    data_iterator, config, mesh, checkpoint_manager, init_state_fn, is_training=True, snapshotter=None, step=-1,
    selected_source=None, init_state_args=(),
):
  unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = state_abstraction.get_abstract_state(
      config, mesh, init_state_fn, is_training
  )

  with axis_rules(config.logical_axis_rules):
    native = lineage_adapter.is_native(config)
    restored, raw_params = _load_checkpoint(
        data_iterator, config, mesh, checkpoint_manager, unboxed_abstract_state, native, snapshotter, step,
        selected_source,
    )
    was_restored = bool(restored is not None or raw_params is not None)
    init_state_partial = init_state_fn
    init_state_partial.__name__ = "initialize_state"
    state = jax.jit(
        lambda *args: nnx.state(init_state_partial(*args)), in_shardings=None, out_shardings=state_mesh_shardings
    )(*init_state_args)
    if raw_params:
      _restore_raw_params(state, raw_params, native, config, mesh)
    if restored:
      state = _overlay_restored_state(state, restored, checkpoint_manager, config)

  if config.enable_diloco:
    state = diloco_utils.setup_diloco_initial_state(
        state=state, config=config, mesh=mesh, state_mesh_shardings=state_mesh_shardings, restored=restored
    )
  return state, state_mesh_annotations, state_mesh_shardings, data_iterator, was_restored


def _load_checkpoint(
    data_iterator, config, mesh, checkpoint_manager, unboxed_abstract_state, native, snapshotter, step, selected_source
):
  if native and selected_source is not None and selected_source.kind == "parameters":
    restored, raw_params = None, None
  else:
    restored, raw_params = checkpointing.load_state_if_possible(
        checkpoint_manager, data_iterator, "" if native else config.load_parameters_path, config.load_full_state_path,
        config.checkpoint_storage_concurrent_gb, unboxed_abstract_state, config.enable_single_replica_ckpt_restoring,
        config.dataset_type, step=step, use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3, enable_orbax_v1=config.enable_orbax_v1,
        checkpoint_conversion_fn=config.checkpoint_conversion_fn, source_checkpoint_layout=config.source_checkpoint_layout,
        expansion_factor_real_data=config.expansion_factor_real_data, maxtext_config=config, snapshotter=snapshotter,
        selected_source=selected_source,
    )
  if (native and restored is None and raw_params is None and config.load_parameters_path
      and (selected_source is None or selected_source.kind == "parameters")):
    raw_params = checkpointing.load_params_from_path(
        selected_source.path if selected_source is not None else config.load_parameters_path,
        lineage_adapter.maxtext_restore_target(config, mesh),
        config.checkpoint_storage_concurrent_gb, use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
        enable_single_replica_ckpt_restoring=bool(config.enable_single_replica_ckpt_restoring),
        colocated_python_checkpointing=bool(config.enable_single_controller and config.colocated_python_checkpointing),
    )
  return restored, raw_params


def _restore_raw_params(state, raw_params, native, config, mesh):
  if native:
    lineage_adapter.restore_params(state["model"], raw_params, config, mesh)
    return
  target_model = (
      state["model"] if (isinstance(state, (nnx.State, dict)) and "model" in state) else getattr(state, "model", state)
  )
  raw_model_params = (
      raw_params["model"] if (isinstance(raw_params, (nnx.State, dict)) and "model" in raw_params) else raw_params
  )
  if hasattr(raw_model_params, "to_pure_dict"):
    raw_model_params = raw_model_params.to_pure_dict()
  if isinstance(raw_model_params, dict) and "params" in raw_model_params:
    raw_model_params = raw_model_params["params"]
  target_pure = target_model.to_pure_dict() if hasattr(target_model, "to_pure_dict") else target_model
  sharded_aligned = _reshard_aligned(target_pure, raw_model_params)
  nnx.update(target_model, sharded_aligned)


def _reshard_aligned(target, raw):
  target_flat = traverse_util.flatten_dict(target)
  raw_flat = traverse_util.flatten_dict(raw)
  res_flat = {}
  for k, target_val in target_flat.items():
    if k in raw_flat and not isinstance(raw_flat[k], jax.ShapeDtypeStruct):
      raw_val = raw_flat[k]
      if hasattr(target_val, "sharding") and target_val.sharding is not None:
        res_flat[k] = jax.device_put(raw_val, target_val.sharding)
      else:
        res_flat[k] = raw_val
    else:
      res_flat[k] = target_val
  return traverse_util.unflatten_dict(res_flat)


def _overlay_restored_state(state, restored, checkpoint_manager, config):
  is_emergency = isinstance(
      checkpoint_manager,
      (emergency_checkpoint_manager.CheckpointManager,
       emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager),
  )
  overlay = restored if is_emergency else restored["items"]
  if config.enable_diloco and isinstance(overlay, diloco.DiLoCoTrainState):
    return overlay
  overlay_pure_dict = overlay.to_pure_dict() if hasattr(overlay, "to_pure_dict") else overlay
  merged = _merge_restored_overlay(overlay_pure_dict, state.to_pure_dict())
  nnx.replace_by_pure_dict(state, merged)
  return state


def _has_shape_dtype_struct(tree):
  return any(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree_util.tree_leaves(tree))


def _merge_restored_overlay(ckpt_node, init_node):
  if _has_shape_dtype_struct(ckpt_node):
    if isinstance(ckpt_node, dict) and isinstance(init_node, dict):
      res = {}
      for k in init_node:
        if k in ckpt_node:
          res[k] = _merge_restored_overlay(ckpt_node[k], init_node[k])
        else:
          res[k] = init_node[k]
      return res
    else:
      return init_node
  return ckpt_node
