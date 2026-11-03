"""Distillation training through the native Tunix path."""

from contextlib import nullcontext

import jax
from orbax import checkpoint as ocp

from lilax.data_pipeline.api import create_data_iterator
from lilax._backend.configs import pyconfig
from lilax._backend.common.profiler import xprof_server
from lilax._backend.trainers.post_train.distillation import distillation_utils, save_top_k_teacher_logits, train_distill
from lilax.configuration import MaxTextConfig
from lilax.modeling.models import MaxTextModel
from lilax.runtime import Runtime

from .._base import _PostTrainer
from ..reporting import MetricReporter
from .arguments import _PostTrainingArguments
from .evaluation import _evaluate_tunix


class DistillationConfig(_PostTrainingArguments):
  def __init__(self, *args, is_offline=False, offline_data_dir=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.is_offline = is_offline
    self.offline_data_dir = offline_data_dir

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
  def __init__(self, model, teacher_model, args=None, *, teacher_config=None, **kwargs):
    if args is not None and not isinstance(args, DistillationConfig):
      raise TypeError("DistillationTrainer requires DistillationConfig arguments.")
    super().__init__(model, args=args, defer_materialization=True, **kwargs)
    if teacher_model is not None and model.runtime.mesh != teacher_model.runtime.mesh:
      raise ValueError("Student and teacher must use the same mesh.")
    self.teacher_model = teacher_model
    self.teacher_config = teacher_model.runtime.config if teacher_model is not None else teacher_config
    self.offline_data_dir = self.config.offline_data_dir

  @classmethod
  def from_config(cls, config, args=None, *, config_file=None, source_overrides=None, **kwargs):
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
    offline_data_dir = (args.offline_data_dir if args is not None else None) or global_config.offline_data_dir
    is_offline = (args.is_offline if args is not None else False) or bool(offline_data_dir)
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
      resolved_config = args.resolve(
          student_runtime, train_dataset_provided=kwargs.get("train_dataset") is not None,
          eval_dataset_provided=kwargs.get("eval_dataset") is not None,
      )
    else:
      resolved_config = student_runtime.config
    trainer = cls(
        student, teacher, args=args, teacher_config=teacher_runtime.config, resolved_config=resolved_config, **kwargs,
    )
    trainer.offline_data_dir = global_config.offline_data_dir
    return trainer

  def generate_teacher_logits(
      self, *, train_iterator=None, top_k=128,
      optional_keys=("inputs_position", "inputs_segmentation", "targets_segmentation", "targets"),
      gcs_upload_path=None, local_tmp_dir="/tmp", steps_per_file=50, upload_workers=4, serialization_workers=16,
  ):
    """Generate offline distillation records using the supplied teacher."""
    if self.teacher_model is None:
      raise ValueError("Teacher logits require a teacher model.")
    if train_iterator is None:
      train_iterator, _ = self._data_iterators()
    if train_iterator is None:
      train_iterator, _ = create_data_iterator(self.config, self.model.runtime.mesh)
    save_top_k_teacher_logits.generate_and_save_data(
        self.config, self.teacher_model.native_model, self.model.runtime.mesh, train_iterator,
        top_k=top_k, optional_keys=optional_keys, gcs_upload_path=gcs_upload_path, local_tmp_dir=local_tmp_dir,
        steps_per_file=steps_per_file, upload_workers=upload_workers, serialization_workers=serialization_workers,
    )

  def train(self, resume_from_checkpoint=None):
    if resume_from_checkpoint is None or resume_from_checkpoint is True:
      checkpoint_step = None
    elif (
        isinstance(resume_from_checkpoint, int) and not isinstance(resume_from_checkpoint, bool)
        and resume_from_checkpoint >= 0
    ):
      checkpoint_step = resume_from_checkpoint
    else:
      raise ValueError("resume_from_checkpoint must be None, True, or a nonnegative checkpoint step.")
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=self.config.checkpoint_period, max_to_keep=self.config.max_num_checkpoints_to_keep,
        enable_async_checkpointing=self.config.async_checkpointing, create=True,
    )
    manager = distillation_utils.MaxTextCheckpointManager(None, self.config.checkpoint_dir, self.config, options)
    manager, checkpoint_step = self._prepare_training_checkpoint(checkpoint_step, manager)
    args = self.args if self.args is not None else DistillationConfig()
    offline_data_dir = args.offline_data_dir or self.offline_data_dir
    is_offline = args.is_offline or bool(offline_data_dir)
    if is_offline:
      config = self.config._pydantic_config.model_copy(update={"is_offline_distillation": True}, deep=True)
      self.config = pyconfig.HyperParameters(config)
    train_iterator, eval_iterator = self._data_iterators()
    server = xprof_server(self.config.tensorboard_dir, args.xprof_port, jax.process_index()) if args.launch_xprof else nullcontext()
    with server:
      self.native_trainer, self._eval_iterator = train_distill.train_distill(
          self.config, self.teacher_config, is_offline=is_offline,
          offline_data_dir=offline_data_dir, student_model=self.model.native_model,
          teacher_model=self.teacher_model.native_model if self.teacher_model is not None else None,
          mesh=self.model.runtime.mesh,
          train_iterator=train_iterator, eval_iterator=eval_iterator, return_trainer=True, checkpoint_step=checkpoint_step,
          report_to=args.report_to, reporter_factory=MetricReporter,
          checkpoint_manager=manager,
      )
    self.model.native_model = self.native_trainer.model.student_model
    return self.native_trainer

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
      eval_iterator = self._eval_iterator
    if eval_iterator is None:
      raise ValueError("No evaluation iterator is configured.")
    if hasattr(eval_iterator, "reset"):
      eval_iterator.reset()
    return _evaluate_tunix(
        self.native_trainer, self.config, self.model.runtime.mesh,
        distillation_utils.MaxTextToTunixIterator(eval_iterator),
    )

  def save_model(self, output_dir=None):
    if self.native_trainer is None:
      raise RuntimeError("Training must complete before saving a checkpoint.")
    trainer = self.native_trainer
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=trainer.checkpoint_manager._iterator,
        root_directory=self.config.checkpoint_dir if output_dir is None else str(output_dir),
        student_config=self.config, options=trainer.config.checkpointing_options,
    )
    manager.training_graph = trainer.checkpoint_manager.training_graph
    manager.save_only_lora_params = trainer.checkpoint_manager.save_only_lora_params
    try:
      step = trainer.train_steps + (1 if self.config.learn_to_init_mode else 0)
      latest_step = manager.latest_step()
      if latest_step is None or latest_step < step:
        manager.save(
            step, trainer.model, optimizer=trainer.optimizer, save_only_lora_params=trainer._lora_enabled,
            force=True, custom_metadata=trainer.custom_checkpoint_metadata(),
        )
        manager.wait_until_finished()
    finally:
      manager.close()

  def save_state(self):
    self.save_model()


def run_distillation(*, config_path, overrides, resume_from_checkpoint):
  """Run distillation from a root configuration and caller overrides."""
  config = MaxTextConfig(**(dict(overrides) | {"base_config": config_path}))
  trainer = DistillationTrainer.from_config(config)
  return trainer.train(resume_from_checkpoint=resume_from_checkpoint)

