"""Managed post-training lifecycle."""

from lilax.data_pipeline.api import create_data_iterator
from lilax._backend.trainers.post_train.checkpoint_selection import TrainingCheckpointManager

from .._base import _PostTrainer
from .evaluation import _evaluate_tunix


class _ManagedPostTrainer(_PostTrainer):
  def _checkpoint_manager(self, tunix_config):
    return TrainingCheckpointManager(self.config.checkpoint_dir, tunix_config.checkpointing_options)

  def evaluate(self, eval_dataset=None):
    if self.native_trainer is None:
      raise RuntimeError("Training must complete before evaluation.")
    if eval_dataset is not None:
      if self.config.dataset_type != "hf" or self.config.eval_interval <= 0:
        raise ValueError("Supplying an evaluation dataset requires dataset_type='hf' and a positive eval_interval.")
      _, eval_iterator = create_data_iterator(
          self.config, self.model.runtime.mesh, train_dataset=self.train_dataset, eval_dataset=eval_dataset,
          processing_class=self.processing_class,
      )
    else:
      eval_iterator = self.native_trainer.data_hooks.eval_data_iterator
    return _evaluate_tunix(self.native_trainer, self.config, self.model.runtime.mesh, eval_iterator)

  def save_model(self, output_dir=None):
    if self.native_trainer is None:
      raise RuntimeError("Training must complete before saving a checkpoint.")
    trainer = self.native_trainer
    manager = TrainingCheckpointManager(
        root_directory=self.config.checkpoint_dir if output_dir is None else str(output_dir),
        options=trainer.config.checkpointing_options,
    )
    manager.training_graph = trainer.checkpoint_manager.training_graph
    manager.save_only_lora_params = trainer.checkpoint_manager.save_only_lora_params
    try:
      latest_step = manager.latest_step()
      if latest_step is None or latest_step < trainer.train_steps:
        manager.save(
            trainer.train_steps, trainer.model, trainer.optimizer, save_only_lora_params=trainer._lora_enabled,
            force=True, custom_metadata=trainer.custom_checkpoint_metadata(),
        )
    finally:
      manager.close()

  def save_state(self):
    self.save_model()
