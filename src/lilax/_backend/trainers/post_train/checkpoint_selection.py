"""Selected-step restoration for native Tunix trainers."""

from tunix.sft import checkpoint_manager, profiler
from tunix.sft.checkpoint_options import create_checkpointing_options
from orbax.checkpoint import v1 as ocp

from lilax._backend.common.checkpoints.save_receipt import save_receipt
from lilax._backend.common.checkpoints.checkpoint_context import build_preservation_policy, build_save_decision_policy


def training_checkpoint_options(config):
  """Use the existing save and retention policies for the Tunix manager."""
  return create_checkpointing_options(
      save_decision_policy=build_save_decision_policy(
          save_interval_steps=config.checkpoint_period,
          enable_continuous_checkpointing=config.enable_continuous_checkpointing,
          enable_autocheckpoint=config.enable_autocheckpoint,
      ),
      preservation_policy=build_preservation_policy(max_to_keep=config.max_num_checkpoints_to_keep),
      enable_async_checkpointing=config.async_checkpointing,
  )


class TrainingCheckpointManager(checkpoint_manager.CheckpointManager):
  def __init__(self, root_directory=None, options=None):
    super().__init__(root_directory=root_directory, options=options)
    self.training_graph = None
    self.save_only_lora_params = None
    self.restore_only_lora_params = None
    self.lifecycle = None
    self.save_receipts = []
    self.allow_saves = True
    self._closed = False

  def close(self):
    if self._closed:
      return
    try:
      super().close()
    finally:
      self.allow_saves = False
      self._closed = True

  def _save_checkpointables(self, step, checkpointables, force, custom_metadata):
    if self.lifecycle is None:
      return super()._save_checkpointables(step, checkpointables, force, custom_metadata)
    action = self.lifecycle._save_action(self.lifecycle.native_trainer.train_steps)
    metadata = dict(custom_metadata or {}) | {
        "lilax_save_action": action,
    }
    response = None
    if self._checkpointer is not None:
      try:
        if self._options.enable_async_checkpointing:
          response = self._checkpointer.save_checkpointables_async(
              step, checkpointables, force=force, custom_metadata=metadata,
          )
        else:
          response = self._checkpointer.save_checkpointables(
              step, checkpointables, force=force, custom_metadata=metadata,
          )
      except ocp.training.errors.StepAlreadyExistsError:
        response = False
    receipt = save_receipt(action["action_id"], action["completed_step"], step, response)
    self.save_receipts.append(receipt)
    return receipt.outcome != "no_save"

  def wait_until_finished(self):
    if self._checkpointer is not None:
      self._checkpointer.wait()

  def save(self, step, model, optimizer=None, save_only_lora_params=False, force=False, custom_metadata=None):
    if not self.allow_saves:
      return False
    if self.save_only_lora_params is not None:
      save_only_lora_params = self.save_only_lora_params
    metadata = dict(custom_metadata or {})
    if self.training_graph is not None:
      metadata["maxtext_easy_training_graph"] = self.training_graph
    return super().save(step, model, optimizer, save_only_lora_params, force, metadata)

  def maybe_restore(self, model, optimizer=None, step=None, restore_only_lora_params=False):
    if self.restore_only_lora_params is not None:
      restore_only_lora_params = self.restore_only_lora_params
    checkpoint_step, metadata = super().maybe_restore(model, optimizer, step, restore_only_lora_params)
    action = (metadata or {}).get("lilax_save_action")
    if action is not None:
      if (not isinstance(action, dict) or set(action) != {"action_id", "completed_step"}
          or not isinstance(action["action_id"], str) or not action["action_id"]
          or type(action["completed_step"]) is not int or action["completed_step"] < 0):
        raise ValueError("Invalid lilax_save_action metadata in the selected training checkpoint.")
      return action["completed_step"], metadata
    return checkpoint_step, metadata


def select_training_checkpoint(root, options, step=None, manager=None):
  manager = manager or TrainingCheckpointManager(root_directory=root, options=options)
  selected_step = manager.latest_step() if step is None else step
  marker = None
  if selected_step is not None:
    if manager._checkpointer is None:
      raise ValueError("A checkpoint directory is required to restore a selected step.")
    metadata = manager._checkpointer.checkpointables_metadata(selected_step)
    if metadata is None:
      raise ValueError(f"Checkpoint step {selected_step} does not exist.")
    custom_metadata = metadata.custom_metadata or {}
    if "maxtext_easy_training_graph" in custom_metadata:
      marker = custom_metadata["maxtext_easy_training_graph"]
      if marker is None:
        raise ValueError("Invalid null training graph metadata.")
  return manager, selected_step, marker


def restore_trainer_step(trainer, checkpoint_root, checkpoint_step, manager=None):
  if not checkpoint_root and checkpoint_step is not None:
    raise ValueError("A checkpoint directory is required to restore a selected step.")
  trainer.checkpoint_manager.close()
  trainer.config.checkpoint_root_directory = checkpoint_root
  trainer.checkpoint_manager = manager or checkpoint_manager.CheckpointManager(
      root_directory=checkpoint_root, options=trainer.config.checkpointing_options,
  )
  if checkpoint_step is not None:
    trainer._train_steps, trainer._restored_custom_metadata = trainer.checkpoint_manager.maybe_restore(
        trainer.model, trainer.optimizer, step=checkpoint_step, restore_only_lora_params=trainer._lora_enabled,
    )
  accumulation_steps = trainer.config.get_with_default("gradient_accumulation_steps", 1)
  trainer._iter_steps = trainer._train_steps * accumulation_steps
  max_step = None if trainer.config.max_steps is None else trainer.config.max_steps * accumulation_steps
  trainer._prof = profiler.Profiler(
      initial_step=trainer._iter_steps, max_step=max_step, profiler_options=trainer.config.profiler_options,
  )
