"""Managed post-training lifecycle."""

from contextlib import closing

from lilax.data_pipeline.trainer import validate_eval_dataset
from lilax._backend.trainers.post_train.checkpoint_selection import TrainingCheckpointManager

from ..lifecycle._base import _PostTrainer
from .evaluation import _evaluate_tunix


class _ManagedPostTrainer(_PostTrainer):
  def _checkpoint_manager(self, tunix_config):
    root = self.config.checkpoint_dir if self.config.enable_checkpointing else None
    return TrainingCheckpointManager(root, tunix_config.checkpointing_options)

  def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    dataset = self.eval_dataset if eval_dataset is None else eval_dataset
    validate_eval_dataset(dataset)
    if dataset is None:
      raise ValueError("Trainer.evaluate requires an evaluation dataset.")
    if isinstance(dataset, dict):
      metrics = {}
      for name, source in dataset.items():
        metrics.update(self.evaluate(source, ignore_keys, f"{metric_key_prefix}_{name}"))
      return metrics
    _, eval_iterator = self._data_iterators(eval_dataset=dataset, evaluation_only=True)
    with closing(eval_iterator):
      return self._evaluate_iterator(eval_iterator, metric_key_prefix)

  def _evaluate_iterator(self, eval_iterator, metric_key_prefix):
    if self.native_trainer is None:
      tunix_config = self._tunix_config(self.config, report_to=[])
      manager = self._checkpoint_manager(tunix_config)
      with closing(manager):
        manager, _ = self._prepare_training_checkpoint(None, manager, resume=False)
        options = self._backend_options()
        options.pop("lifecycle")
        self.native_trainer, _ = self._setup_native(
            self.config, model=self.model.native_model, mesh=self.model.runtime.mesh, train_iterator=iter(()),
            eval_iterator=eval_iterator, checkpoint_manager=manager, report_to=[], **options,
        )
    metrics = _evaluate_tunix(self.native_trainer, self.config, self.model.runtime.mesh, eval_iterator)
    return self._evaluation_metrics(metrics, metric_key_prefix)

  def save_model(self, output_dir=None):
    self.model.save_pretrained(self.args.output_dir if output_dir is None else output_dir)
    if self.processing_class is not None:
      self.processing_class.save_pretrained(self.args.output_dir if output_dir is None else output_dir)

  def save_state(self):
    if self.native_trainer is None:
      raise RuntimeError("Training must complete before saving a checkpoint.")
    if not self.config.enable_checkpointing:
      raise ValueError("Checkpointing is disabled in the training configuration.")
    trainer = self.native_trainer
    manager = TrainingCheckpointManager(
        root_directory=self.config.checkpoint_dir,
        options=trainer.config.checkpointing_options,
    )
    manager.training_graph = trainer.checkpoint_manager.training_graph
    manager.save_only_lora_params = trainer.checkpoint_manager.save_only_lora_params
    manager.lifecycle = self
    self.control.should_save = False
    try:
      latest_step = manager.latest_step()
      if latest_step is None or latest_step < trainer.train_steps:
        manager.save(
            trainer.train_steps, trainer.model, trainer.optimizer, save_only_lora_params=trainer._lora_enabled,
            force=True, custom_metadata=trainer.custom_checkpoint_metadata(),
        )
      for receipt in manager.save_receipts:
        self._resolve_save(receipt, manager._checkpointer)
    finally:
      manager.close()

