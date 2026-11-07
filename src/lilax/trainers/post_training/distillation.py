"""Distillation training through the native Tunix path."""

from contextlib import ExitStack, closing, nullcontext
import copy
import os
from typing import Literal

import jax
import pathwaysutils

from lilax._backend.common.profiler import xprof_server
from lilax._backend.trainers.post_train.distillation import distillation_utils, save_top_k_teacher_logits, train_distill
from lilax._backend.trainers.post_train.checkpoint_selection import training_checkpoint_options
from lilax._backend.configs.inert import MaxTextConfig
from lilax.modeling.models import MaxTextModel
from lilax.runtime import Runtime
from lilax.data import TeacherLogitsDataset
from lilax.data_pipeline.trainer import create_data_iterators, validate_eval_dataset
from lilax.training import LearnToInitConfig

from ..lifecycle._base import _PostTrainer
from ..lifecycle.checkpoint_request import checkpoint_request
from .arguments import _PostTrainingArguments
from .evaluation import _evaluate_tunix


class DistillationConfig(_PostTrainingArguments):
  alpha: float = 0.5
  temperature: float = 1.0
  beta: float = 0.0
  feature_loss_type: Literal["cosine", "l2"] = "cosine"
  layer_indices: list[int] | None = None
  alpha_end: float | None = None
  alpha_schedule: Literal["constant", "linear", "cosine"] = "constant"
  temperature_end: float | None = None
  temperature_schedule: Literal["constant", "linear", "cosine"] = "constant"
  beta_end: float | None = None
  beta_schedule: Literal["constant", "linear", "cosine"] = "constant"
  learn_to_init: LearnToInitConfig | None = None

  def _overrides(self):
    values = super()._overrides() | {"distill_" + name: getattr(self, name) for name in (
        "alpha", "temperature", "beta", "feature_loss_type", "layer_indices", "alpha_end", "alpha_schedule",
        "temperature_end", "temperature_schedule", "beta_end", "beta_schedule",
    )}
    if self.learn_to_init is not None:
      values.update(self.learn_to_init._source_values())
    return values

class _DistillationStudentModel(MaxTextModel):
  def materialize(self, *, runtime):
    if runtime is not self.runtime:
      raise ValueError("A model must be materialized with its original runtime.")
    if self.is_materialized:
      return self
    config = self.construction_config
    qat_config = self.pending_qat_config
    if qat_config is not None and config.quantization and config.use_qwix_quantization:
      raise ValueError("Qwix QAT cannot be combined with an existing Qwix precision recipe.")
    with runtime.context():
      model = train_distill.get_maxtext_model(config, runtime.mesh)
    if qat_config is not None:
      model, config = self._apply_qat(model, config, qat_config)
    self._native_model, self.construction_config = model, config
    return self


class DistillationTrainer(_PostTrainer):
  def __init__(self, model, teacher_model, *, args: DistillationConfig, data_collator=None, train_dataset=None,
               eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None)):
    if args is not None and not isinstance(args, DistillationConfig):
      raise TypeError("DistillationTrainer requires DistillationConfig arguments.")
    super().__init__(model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                     processing_class=processing_class, callbacks=callbacks, data_collator=data_collator,
                     optimizers=optimizers, defer_materialization=True)
    if teacher_model is not None:
      teacher_model.runtime.resolve_config()
      if teacher_model.runtime.mesh is None:
        teacher_model.runtime.mesh = model.runtime.mesh
        teacher_model.runtime.devices = model.runtime.devices
      elif model.runtime.mesh != teacher_model.runtime.mesh:
        raise ValueError("Student and teacher must use the same mesh.")
      teacher_overrides = {name: getattr(self.config, name) for name in (
          "per_device_batch_size", "max_target_length", "gradient_accumulation_steps",
      )}
      if teacher_model.existing_graph_config is not None:
        changed = [name for name, value in teacher_overrides.items()
                   if value != getattr(teacher_model.existing_graph_config, name)]
        if changed:
          raise ValueError(f"Distillation batch settings differ from the materialized teacher: {changed}.")
      teacher_model.runtime.initialize(overrides=teacher_overrides, train_dataset_provided=train_dataset is not None,
                                       eval_dataset_provided=eval_dataset is not None)
    self.teacher_model = teacher_model
    self.teacher_config = teacher_model.runtime.config if teacher_model is not None else self.config
    self.offline_data_dir = train_dataset.files if isinstance(train_dataset, TeacherLogitsDataset) else None
    if teacher_model is None and self.offline_data_dir is None:
      raise ValueError("Offline distillation requires a TeacherLogitsDataset when teacher_model is None.")
    self.config.get_keys()["teacher_config"] = self.teacher_config
    model.runtime.config.get_keys()["teacher_config"] = self.teacher_config

  @classmethod
  def _from_source_config(cls, config, args=None, *, config_file=None, source_overrides=None, **kwargs):
    """Construct the student and teacher from a native distillation configuration."""
    if args is not None and not isinstance(args, DistillationConfig):
      raise TypeError("DistillationTrainer requires DistillationConfig arguments.")
    get_source_inputs = getattr(config, "get_source_inputs", None)
    source = get_source_inputs() if get_source_inputs is not None else None
    if source is None and config_file is None:
      raise ValueError("Distillation configuration source is unavailable; provide config_file.")
    root, caller_overrides = source if source is not None else (None, {})
    if config_file is not None:
      root = config_file
    caller_overrides = caller_overrides | dict(source_overrides or {})
    global_config = MaxTextConfig(**(caller_overrides | {"base_config": root}))
    student_overrides = dict(global_config.student_overrides or {})
    student_config = MaxTextConfig(**(caller_overrides | student_overrides | {"base_config": root}))
    teacher_overrides = dict(global_config.teacher_overrides or {})
    teacher_overrides.setdefault("distill_beta", global_config.distill_beta)
    offline_data_dir = global_config.offline_data_dir
    is_offline = bool(offline_data_dir)
    if not is_offline and not teacher_overrides.get("load_parameters_path"):
      raise ValueError("Teacher model path is missing from teacher_overrides.load_parameters_path.")
    teacher_config = MaxTextConfig(**(teacher_overrides | {"base_config": root}))
    student_runtime = Runtime(student_config).initialize()
    teacher_runtime = Runtime(teacher_config).resolve_config()
    teacher_runtime.mesh = student_runtime.mesh
    with student_runtime.context():
      teacher = None
      if not is_offline:
        teacher_native = train_distill.get_maxtext_model(teacher_runtime.config, student_runtime.mesh)
        teacher_native.eval()
        teacher = MaxTextModel(teacher_native, teacher_runtime)
      student_runtime.config.get_keys()["teacher_config"] = teacher_runtime.config
      student = _DistillationStudentModel(None, student_runtime)
    if args is not None:
      cls._check_precision_overrides(student.construction_config, args._overrides())
      resolved_config = args._resolve(
          student_runtime, train_dataset_provided=kwargs.get("train_dataset") is not None,
          eval_dataset_provided=kwargs.get("eval_dataset") is not None,
      )
    else:
      resolved_config = student_runtime.config
    trainer = cls.__new__(cls)
    _PostTrainer.__init__(trainer, student,
        args=args or DistillationConfig(output_dir=resolved_config.base_output_directory),
        resolved_config=resolved_config, defer_materialization=True, **kwargs)
    trainer.teacher_model = teacher
    trainer.teacher_config = teacher_runtime.config
    trainer.offline_data_dir = global_config.offline_data_dir
    return trainer

  def generate_teacher_logits(
      self, *, dataset=None, top_k=128,
      optional_keys=("inputs_position", "inputs_segmentation", "targets_segmentation", "targets"),
      gcs_upload_path=None, local_tmp_dir="/tmp", steps_per_file=50, upload_workers=4, serialization_workers=16,
  ):
    """Generate offline distillation records using the supplied teacher."""
    if self.teacher_model is None:
      raise ValueError("Teacher logits require a teacher model.")
    train_iterator, _ = create_data_iterators(
        self.config, self.model.runtime.mesh, train_dataset=self.train_dataset if dataset is None else dataset,
        processing_class=self.processing_class, data_collator=self.data_collator,
    )
    with closing(train_iterator):
      save_top_k_teacher_logits.generate_and_save_data(
          self.config, self.teacher_model.native_model, self.model.runtime.mesh, train_iterator,
          top_k=top_k, optional_keys=optional_keys, gcs_upload_path=gcs_upload_path, local_tmp_dir=local_tmp_dir,
          steps_per_file=steps_per_file, upload_workers=upload_workers, serialization_workers=serialization_workers,
      )

  def train(self, resume_from_checkpoint=None, *, checkpoint_step=None):
    request = checkpoint_request(resume_from_checkpoint, checkpoint_step)
    options = training_checkpoint_options(self.config)
    with ExitStack() as resources:
      manager = distillation_utils.MaxTextCheckpointManager(
          None, (request.directory or self.config.checkpoint_dir) if self.config.enable_checkpointing else None,
          self.config, options,
      )
      resources.callback(manager.close)
      manager, checkpoint_step = self._prepare_training_checkpoint(request.step, manager, resume=request.resume)
      args = self.args
      offline_data_dir = self.offline_data_dir
      is_offline = self.teacher_model is None
      config, data_config = self.config, self.config
      if is_offline:
        config, data_config = copy.deepcopy(self.config), copy.deepcopy(self.config)
        data_config.get_keys().update(grain_train_files=offline_data_dir, eval_interval=0, is_offline_distillation=True,
                                      dataset_shuffle_buffer_size=0, dataset_shuffle_seed=0)
      train_iterator, eval_iterator = self._data_iterators()
      for iterator in (train_iterator, eval_iterator):
        if iterator is not None and hasattr(iterator, "close"):
          resources.callback(iterator.close)
      resources.callback(manager.wait_until_finished)
      xprof = self.model.runtime._profile.xprof if self.model.runtime._profile is not None else None
      server = (xprof_server(self.config.tensorboard_dir, xprof.port, jax.process_index())
                if xprof is not None and xprof.launch_server else nullcontext())
      with self._reporting_context(), self.model.runtime.context(), server:
        self.native_trainer, self._eval_iterator = train_distill.train_distill(
            config, self.teacher_config, is_offline=is_offline,
            offline_data_dir=offline_data_dir, student_model=self.model.native_model,
            teacher_model=self.teacher_model.native_model if self.teacher_model is not None else None,
            mesh=self.model.runtime.mesh,
            train_iterator=train_iterator, eval_iterator=eval_iterator, return_trainer=True, checkpoint_step=checkpoint_step,
            report_to=[],
            checkpoint_manager=manager,
            optimizer=self.optimizers[0], learning_rate_schedule=self.optimizers[1],
            processing_class=self.processing_class, lifecycle=self,
        )
      self.model.native_model = self.native_trainer.model.student_model
      self.state.global_step = self.native_trainer.train_steps
      return self._train_output()

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
    temporary_trainer = self.native_trainer is None
    with ExitStack() as resources:
      _, eval_iterator = self._data_iterators(eval_dataset=dataset, evaluation_only=True)
      if eval_iterator is None:
        raise ValueError("No evaluation iterator is configured.")
      resources.callback(eval_iterator.close)
      if temporary_trainer:
        self.model.materialize(runtime=self.model.runtime)
        self.native_trainer, self._eval_iterator = train_distill._train_distill_impl(
            self.config, self.teacher_config, is_offline=True,
            student_model=self.model.native_model, teacher_model=None,
            mesh=self.model.runtime.mesh, train_iterator=iter(()), eval_iterator=eval_iterator,
            optimizer=self.optimizers[0], learning_rate_schedule=self.optimizers[1], processing_class=self.processing_class,
            report_to=[], evaluation_only=True,
        )
        resources.callback(self.native_trainer.checkpoint_manager.close)
      try:
        metrics = _evaluate_tunix(
            self.native_trainer, self.config, self.model.runtime.mesh,
            distillation_utils.MaxTextToTunixIterator(eval_iterator),
        )
      finally:
        if temporary_trainer:
          self.native_trainer = None
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
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=trainer.checkpoint_manager._iterator,
        root_directory=self.config.checkpoint_dir,
        student_config=self.config, options=trainer.config.checkpointing_options,
    )
    manager.training_graph = trainer.checkpoint_manager.training_graph
    manager.save_only_lora_params = trainer.checkpoint_manager.save_only_lora_params
    manager.lifecycle = self
    self.control.should_save = False
    try:
      step = trainer.train_steps + (1 if self.config.learn_to_init_mode else 0)
      latest_step = manager.latest_step()
      if latest_step is None or latest_step < step:
        manager.save(
            step, trainer.model, optimizer=trainer.optimizer, save_only_lora_params=trainer._lora_enabled,
            force=True, custom_metadata=trainer.custom_checkpoint_metadata(),
        )
      for receipt in manager.save_receipts:
        self._resolve_save(receipt, manager._checkpointer)
    finally:
      manager.close()

def _run_distillation(*, config_path, overrides, resume_from_checkpoint):
  """Run distillation from a root configuration and caller overrides."""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  config =MaxTextConfig(**(dict(overrides) | {"base_config": config_path}))
  trainer = DistillationTrainer._from_source_config(config)
  return trainer.train(resume_from_checkpoint=resume_from_checkpoint)

