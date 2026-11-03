"""Supervised and preference training through the native Tunix paths."""

from contextlib import nullcontext

from flax.core.spmd import logical_axis_rules
import jax
from orbax import checkpoint as ocp
from tunix.sft import metrics_logger

from lilax.data import create_data_iterator
from lilax._backend.configs import pyconfig
from lilax._backend.configs.runtime import resolve_initialized_config
from lilax._backend.common.profiler import xprof_server
from lilax._backend.trainers.post_train.dpo import train_dpo
from lilax._backend.trainers.post_train.checkpoint_selection import TrainingCheckpointManager
from lilax._backend.trainers.post_train.distillation import distillation_utils, save_top_k_teacher_logits, train_distill
from lilax._backend.trainers.post_train.sft import train_sft, train_sft_native
from lilax._backend.utils import maxtext_utils
from lilax.configuration import MaxTextConfig
from lilax.models import MaxTextModel
from lilax.peft import PeftModel
from lilax.runtime import Runtime

from ._base import _PostTrainer
from ._post_training_arguments import _PostTrainingArguments, SFTConfig, DPOConfig, DistillationConfig
from ._post_training_distillation import _DistillationStudentModel, DistillationTrainer, run_distillation
from ._post_training_shared import _check_post_training_data_options, _evaluate_tunix
from .pre_training import Trainer as _NativeTrainer
from .training_args import MMAP_FIELDS, TrainingArguments
from .reporting import MetricReporter



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


class SFTTrainer(_ManagedPostTrainer):
  _native_train = staticmethod(train_sft.train)
  _tunix_config = staticmethod(train_sft.get_tunix_config)

  def __init__(
      self, model, args=None, train_dataset=None, eval_dataset=None, processing_class=None,
      data_collator=None, optimizers=(None, None), callbacks=None, *, lifecycle="tunix",
  ):
    if lifecycle not in ("tunix", "native"):
      raise ValueError("lifecycle must be 'tunix' or 'native'.")
    if lifecycle == "native" and args is not None and (
        getattr(args, "save_only_lora_params", None) is not None
        or getattr(args, "restore_only_lora_params", None) is not None
    ):
      raise ValueError("LoRA checkpoint selection is only supported by the Tunix lifecycle.")
    overrides = {"use_sft": True}
    runtime = model.runtime
    source = runtime.public_config.get_source_inputs()
    if source is not None and source[0] is None:
      defaults = {
          "use_tunix_gradient_accumulation": True, "sft_train_on_completion_only": True,
          "packing": True, "learning_rate": 2e-5,
      }
      overrides.update({key: value for key, value in defaults.items() if key not in runtime.public_config._explicit_keys})
    if train_dataset is not None or eval_dataset is not None:
      overrides["dataset_type"] = "hf"
    if args is not None:
      overrides.update(args._overrides())
    if lifecycle == "native":
      overrides["use_tunix_gradient_accumulation"] = False
    data_overrides = runtime._overrides | overrides
    explicit_data_fields = runtime.public_config._explicit_keys | data_overrides.keys()
    unsupported = sorted(set(MMAP_FIELDS) & explicit_data_fields)
    data_config = runtime.public_config.get_source_values() | data_overrides
    if data_config["grain_file_type"] in ("mmap", "mmap_npy") and (
        data_config["dataset_type"] == "grain" or "grain_file_type" in explicit_data_fields
    ):
      unsupported.append("grain_file_type")
    if unsupported:
      raise ValueError(f"SFT does not support these mmap data settings: {sorted(unsupported)}")
    self._training_overrides = overrides
    graph_config = model.existing_graph_config
    if graph_config is not None:
      self._check_precision_overrides(graph_config, overrides)
      config = resolve_initialized_config(
          runtime.public_config, runtime.config, overrides=runtime._overrides | overrides,
          train_dataset_provided=train_dataset is not None, eval_dataset_provided=eval_dataset is not None,
      )
    else:
      runtime.initialize(
          overrides=overrides, train_dataset_provided=train_dataset is not None,
          eval_dataset_provided=eval_dataset is not None,
      )
      self._check_precision_options(runtime.config, overrides)
      config = runtime.config
    _check_post_training_data_options(config, runtime, args)
    super().__init__(
        model, args, train_dataset, eval_dataset, processing_class, data_collator, optimizers, callbacks,
        resolved_config=config, defer_materialization=True,
    )
    self.lifecycle = lifecycle
    self._native_sft_trainer = None

  def train(self, resume_from_checkpoint=None):
    if self.lifecycle == "native":
      lora = self.config.lora if isinstance(self.model, PeftModel) else None
      args = train_sft_native.NativeSFTArguments(self.args, lora, self._training_overrides)
      self._native_sft_trainer = _NativeTrainer(
          self.model, args=args, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
          processing_class=self.processing_class,
      )
      config = args.resolve(
          self.model.runtime, train_dataset_provided=self.train_dataset is not None,
          eval_dataset_provided=self.eval_dataset is not None,
      )
      self._native_sft_trainer.config = (
          self._with_model_config(config, self.model) if self.model.is_materialized else config
      )
      state = train_sft_native.train(self._native_sft_trainer, resume_from_checkpoint)
      self.config = self._native_sft_trainer.config
      self.native_trainer = self._native_sft_trainer
      return state
    trainer = super().train(resume_from_checkpoint)
    self.model.native_model = trainer.model
    return trainer

  def evaluate(self, eval_dataset=None):
    if self._native_sft_trainer is not None:
      return self._native_sft_trainer.evaluate(eval_dataset)
    return super().evaluate(eval_dataset)

  def save_model(self, output_dir=None):
    if self._native_sft_trainer is not None:
      return self._native_sft_trainer.save_model(output_dir)
    return super().save_model(output_dir)

  def save_state(self):
    if self._native_sft_trainer is not None:
      return self._native_sft_trainer.save_state()
    return super().save_state()


class DPOTrainer(_ManagedPostTrainer):
  _native_train = staticmethod(train_dpo.train)
  _tunix_config = staticmethod(train_dpo.get_tunix_config)

  def __init__(self, model, args=None, *trainer_args, **kwargs):
    if args is not None:
      unsupported = [name for name in MMAP_FIELDS if name in args._overrides()]
      if args._overrides().get("grain_file_type") in ("mmap", "mmap_npy"):
        unsupported.append("grain_file_type")
      if unsupported:
        raise ValueError(f"Native DPO does not support these data overrides: {sorted(unsupported)}")
    super().__init__(model, args, *trainer_args, defer_materialization=True, **kwargs)
    _check_post_training_data_options(self.config, model.runtime, args)


