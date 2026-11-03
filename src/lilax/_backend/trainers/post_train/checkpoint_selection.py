"""Selected-step restoration for native Tunix trainers."""

from tunix.sft import checkpoint_manager, profiler


class TrainingCheckpointManager(checkpoint_manager.CheckpointManager):
  def __init__(self, root_directory=None, options=None):
    super().__init__(root_directory=root_directory, options=options)
    self.training_graph = None
    self.save_only_lora_params = None
    self.restore_only_lora_params = None

  def save(self, step, model, optimizer=None, save_only_lora_params=False, force=False, custom_metadata=None):
    if self.save_only_lora_params is not None:
      save_only_lora_params = self.save_only_lora_params
    metadata = dict(custom_metadata or {})
    if self.training_graph is not None:
      metadata["maxtext_easy_training_graph"] = self.training_graph
    return super().save(step, model, optimizer, save_only_lora_params, force, metadata)

  def maybe_restore(self, model, optimizer=None, step=None, restore_only_lora_params=False):
    if self.restore_only_lora_params is not None:
      restore_only_lora_params = self.restore_only_lora_params
    return super().maybe_restore(model, optimizer, step, restore_only_lora_params)


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
