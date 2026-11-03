"""Checkpointing utilities for the training engine."""

import dataclasses
import os
from typing import Any

from absl import logging
from flax import nnx
import jax
from lilax._backend.configs import pyconfig
import orbax.checkpoint as ocp
import orbax.checkpoint.pathways as ocp_pathways
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.serialization import type_handler_registry


@dataclasses.dataclass
class CheckpointState:
  """Model and optimizer state saved between optimizer steps."""

  model: nnx.Module
  optimizer: nnx.optimizer.Optimizer | None = None


class CheckpointRestoreError(RuntimeError):
  """A checkpoint exists but cannot be restored by this engine."""


_PATHWAYS_PERSISTENCE_REGISTERED = False


def _maybe_register_pathways_persistence() -> None:
  global _PATHWAYS_PERSISTENCE_REGISTERED
  if _PATHWAYS_PERSISTENCE_REGISTERED:
    return
  if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") != "1":
    return

  _PATHWAYS_PERSISTENCE_REGISTERED = True
  ocp_pathways.register_type_handlers(
      use_single_replica_array_handler=False,
      checkpointing_impl=ocp_pathways.CheckpointingImpl.PERSISTENCE,
      array_metadata_store=array_metadata_store_lib.Store(),
  )
  handler = type_handler_registry.get_type_handler(jax.Array)
  handler_name = type(handler).__name__
  store = getattr(handler, "_array_metadata_store", None)
  if handler_name in ("CloudPathwaysArrayHandler", "PathwaysPersistenceArrayHandler"):
    if store is None:
      logging.error("Registered %s but array metadata store is None; saving may fail on typed PRNG keys.", handler_name)
    else:
      logging.info(
          "Registered Pathways persistence array handler (%s, store=%s); TPUs will write directly to storage.",
          handler_name,
          type(store).__name__,
      )
  else:
    logging.warning(
        "Pathways persistence registration fell back: jax.Array is handled by %s, not a persistence handler.",
        handler_name,
    )


class CheckpointManager:
  """Orbax checkpoint manager for the training engine."""

  def __init__(self, checkpoint_dir: str, config: pyconfig.HyperParameters) -> None:
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if checkpoint_dir:
      _maybe_register_pathways_persistence()

      def _pytree_handler() -> ocp.PyTreeCheckpointHandler:
        return ocp.PyTreeCheckpointHandler(
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
            save_device_host_concurrent_gb=config.checkpoint_storage_device_host_concurrent_gb,
        )

      self._checkpoint_manager = ocp.CheckpointManager(
          directory=checkpoint_dir,
          options=ocp.CheckpointManagerOptions(
              save_interval_steps=config.checkpoint_period,
              max_to_keep=config.max_num_checkpoints_to_keep,
              enable_async_checkpointing=config.async_checkpointing,
          ),
          item_handlers={"model_params": _pytree_handler(), "optimizer_state": _pytree_handler()},
      )

  def get_latest_step(self) -> int | None:
    if self._checkpoint_manager:
      return self._checkpoint_manager.latest_step()
    return None

  def wait_until_finished(self) -> None:
    if self._checkpoint_manager:
      self._checkpoint_manager.wait_until_finished()

  def save_checkpoint(
      self, step: int, checkpoint_state: CheckpointState, custom_metadata: Any = None, force: bool = False
  ) -> bool:
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping save.")
      return False
    if self.get_latest_step() == step:
      logging.info("Checkpoint already saved at step %d, skipping save.", step)
      return False

    params = nnx.state(checkpoint_state.model)
    jax.block_until_ready(params)
    model_cp_args = ocp.args.PyTreeSave(item=params, save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params))
    save_args = {"model_params": model_cp_args}
    if checkpoint_state.optimizer:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      jax.block_until_ready(optimizer_state)
      optimizer_cp_args = ocp.args.PyTreeSave(
          item=optimizer_state, save_args=jax.tree.map(lambda _: ocp.SaveArgs(), optimizer_state)
      )
      save_args["optimizer_state"] = optimizer_cp_args

    return self._checkpoint_manager.save(
        step=step, args=ocp.args.Composite(**save_args), custom_metadata=custom_metadata, force=force
    )

  def restore_checkpoint(
      self, checkpoint_state: CheckpointState, step: int | None = None
  ) -> tuple[int | None, CheckpointState, Any]:
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, checkpoint_state, None
    if step is None:
      step = self.get_latest_step()
      if step is None:
        logging.info("No checkpoint found, skipping restore.")
        return None, checkpoint_state, None

    metadata = self._checkpoint_manager.metadata(step)
    restore_args: dict[str, Any] = {}
    abstract_params = nnx.state(checkpoint_state.model)
    restore_args["model_params"] = ocp.args.PyTreeRestore(
        item=abstract_params, restore_args=ocp.checkpoint_utils.construct_restore_args(target=abstract_params)
    )
    if checkpoint_state.optimizer is not None and "optimizer_state" in metadata.item_metadata:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      restore_args["optimizer_state"] = ocp.args.PyTreeRestore(
          item=optimizer_state,
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              target=nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
          ),
      )

    custom_metadata = None
    if metadata and hasattr(metadata, "custom_metadata"):
      custom_metadata = metadata.custom_metadata
    try:
      restored_items = self._checkpoint_manager.restore(step=step, args=ocp.args.Composite(**restore_args))
    except Exception as error:
      logging.exception("Failed to restore checkpoint: %s", error)
      return None, None, None

    if "model_params" in restored_items:
      nnx.update(checkpoint_state.model, restored_items["model_params"])
    if checkpoint_state.optimizer is not None and "optimizer_state" in restored_items:
      nnx.update(checkpoint_state.optimizer, restored_items["optimizer_state"])
    return step, checkpoint_state, custom_metadata

  def close(self) -> None:
    if self._checkpoint_manager:
      self._checkpoint_manager.close()
