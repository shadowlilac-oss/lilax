"""Set up a Linen decode state from a checkpoint or random initialization."""

from flax.core.spmd import logical_axis_rules as axis_rules
import jax
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

from lilax._backend.common import state_initialization
from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.utils import max_logging, state_abstraction
from lilax._backend.pytree import partitioning


def setup_decode_state(config, mesh, checkpoint_manager, init_state_fn):
  """Set up a Linen decode state from a checkpoint or random initialization."""
  if not config.load_parameters_path:
    max_logging.log("No decode checkpoint specified - generating random weights.")
    shaped_state, state_mesh_annotations, state_mesh_shardings = state_abstraction.get_abstract_state(
        config, mesh, init_state_fn, False
    )
    with axis_rules(config.logical_axis_rules):
      restored, raw_params = checkpointing.load_state_if_possible(
          checkpoint_manager, None, config.load_parameters_path, config.load_full_state_path,
          config.checkpoint_storage_concurrent_gb, shaped_state, config.enable_single_replica_ckpt_restoring,
          config.dataset_type, use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3, enable_orbax_v1=config.enable_orbax_v1,
          checkpoint_conversion_fn=config.checkpoint_conversion_fn, source_checkpoint_layout=config.source_checkpoint_layout,
          expansion_factor_real_data=config.expansion_factor_real_data, maxtext_config=config,
      )
      if restored:
        if isinstance(
            checkpoint_manager,
            (emergency_checkpoint_manager.CheckpointManager,
             emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager),
        ):
          state = restored
        else:
          state = restored["items"]
      else:
        state = jax.jit(init_state_fn, in_shardings=None, out_shardings=state_mesh_shardings)()
        if config.weight_sparsity_n and config.weight_sparsity_m and raw_params:
          params = jax.tree_util.tree_map(
              lambda raw, initial: initial if isinstance(raw, jax.ShapeDtypeStruct) else raw, raw_params, state.params
          )
          state = state.replace(params=params)
        elif raw_params:
          state = state.replace(params=raw_params)
  else:
    max_logging.log(f"Loading decode params from {config.load_parameters_path}")
    shaped_state, state_mesh_annotations, _ = state_abstraction.get_abstract_state(config, mesh, init_state_fn, False)
    with axis_rules(config.logical_axis_rules):
      params = checkpointing.load_params_from_path(
          config.load_parameters_path, shaped_state.params, config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt, config.checkpoint_storage_use_zarr3,
      )
    state = state_initialization.init_decode_state(None, params)

  state = partitioning.unbox_logicallypartioned(state)
  return state, state_mesh_annotations
