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
from .pre_training import Trainer as _NativeTrainer
from .training_args import MMAP_FIELDS, TrainingArguments
from .reporting import MetricReporter


def _check_post_training_data_options(config, runtime, args):
  explicit = runtime.public_config._explicit_keys | runtime._overrides.keys()
  if args is not None:
    explicit = explicit | args._overrides().keys()
  unsupported = []
  if config.dataset_type not in ("grain", "tfds") and "colocated_python_data_input" in explicit:
    unsupported.append("colocated_python_data_input")
  padding_ignored = (
      config.dataset_type in ("synthetic", "c4_mlperf")
      or (config.dataset_type in ("grain", "tfds") and config.colocated_python_data_input)
      or (config.dataset_type == "hf" and config.use_sft and config.use_multimodal)
  )
  if padding_ignored:
    unsupported.extend(name for name in ("generate_padding_batch_train", "generate_padding_batch_eval") if name in explicit)
  if unsupported:
    raise ValueError(f"Post-training data pipeline does not use these settings: {sorted(unsupported)}")


def _evaluate_tunix(trainer, config, mesh, eval_iterator):
  if eval_iterator is None:
    raise ValueError("No evaluation iterator is configured.")
  if hasattr(eval_iterator, "reset"):
    eval_iterator.reset()
  previous_logger, previous_hooks = trainer.metrics_logger, trainer.training_hooks
  data_hooks = trainer.data_hooks
  previous_iterator = data_hooks.eval_data_iterator if data_hooks is not None else None
  trainer.metrics_logger = metrics_logger.MetricsLogger()
  if data_hooks is not None:
    schedule = maxtext_utils.create_learning_rate_schedule(config)
    trainer.training_hooks = type(previous_hooks)(config, mesh, schedule, data_hooks.goodput_recorder)
    data_hooks.eval_data_iterator = eval_iterator
  try:
    with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
      _, eval_step = trainer.jit_train_and_eval_step()
      trainer._run_eval(eval_iterator, eval_step)
    logger = trainer.metrics_logger
    if not logger.metric_exists(trainer.metrics_prefix, "loss", "eval"):
      raise ValueError("The evaluation iterator is empty.")
    return {
        "eval_loss": float(logger.get_metric(trainer.metrics_prefix, "loss", "eval")),
        "eval_perplexity": float(logger.get_metric(trainer.metrics_prefix, "perplexity", "eval")),
    }
  finally:
    if data_hooks is not None:
      trainer.training_hooks.metric_logger.flush_metrics_and_cleanup()
      data_hooks.eval_data_iterator = previous_iterator
    trainer.metrics_logger, trainer.training_hooks = previous_logger, previous_hooks


class _PostTrainingArguments(TrainingArguments):
  def __init__(self, *args, save_only_lora_params=None, restore_only_lora_params=None, **kwargs):
    super().__init__(*args, **kwargs)
    for name, value in (("save_only_lora_params", save_only_lora_params),
                        ("restore_only_lora_params", restore_only_lora_params)):
      if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be None or a boolean.")
    self.save_only_lora_params = save_only_lora_params
    self.restore_only_lora_params = restore_only_lora_params


class SFTConfig(_PostTrainingArguments):
  def __init__(
      self, *args, max_length=None, packing=None, completion_only_loss=None, train_data_columns=None,
      eval_data_columns=None, tokenize_train_data=None, tokenize_eval_data=None, num_epoch=None,
      enable_data_shuffling=None, data_shuffle_seed=None, chat_template=None, chat_template_path=None,
      formatting_func_path=None, formatting_func_kwargs=None, max_segments_per_seq=None, grain_worker_count=None,
      hf_path=None, hf_name=None, hf_data_dir=None, hf_train_files=None, hf_eval_split=None,
      hf_eval_files=None, hf_access_token=None, dataset_path=None, dataset_name=None, eval_dataset_name=None,
      train_split=None, eval_split=None, grain_train_mixture_config_path=None, grain_use_elastic_iterator=None,
      grain_packing_type=None, grain_per_worker_buffer_size=None, grain_worker_count_eval=None,
      grain_per_worker_buffer_size_eval=None, grain_num_threads=None, grain_num_threads_eval=None,
      grain_prefetch_buffer_size=None, grain_prefetch_buffer_size_eval=None, grain_data_source_max_workers=None,
      grain_shuffle_buffer_size=None, olmo_index_path=None, olmo_path_remap_from=None, olmo_path_remap_to=None,
      olmo_apply_ngram_filter=None, train_image_column=None, eval_image_column=None, default_prompt=None,
      image_placeholder=None, video_placeholder=None, max_num_images_per_example=None,
      use_tunix_gradient_accumulation=None, quantization=None, use_qwix_quantization=None,
      use_manual_quantization=None, jax_mx_gradient_mode=None, quant_cfg_path=None,
      quantization_local_shard_count=None, qwix_bwd_weight_grad_tile_count=None,
      weight_quantization_calibration_method=None, act_quantization_calibration_method=None,
      bwd_quantization_calibration_method=None, drhs_grad_quantization_calibration_method=None,
      quantize_mtp=None, quantize_router_proj=None, quantize_logits_proj=None,
      logits_proj_quant_calibration_method=None, logits_proj_bwd_quant_calibration_method=None,
      sa_qproj_bwd_bf16=None, replicate_quant_scale=None, **kwargs,
  ):
    super().__init__(
        *args, quantization=quantization, use_qwix_quantization=use_qwix_quantization,
        use_manual_quantization=use_manual_quantization, jax_mx_gradient_mode=jax_mx_gradient_mode,
        quant_cfg_path=quant_cfg_path, quantization_local_shard_count=quantization_local_shard_count,
        qwix_bwd_weight_grad_tile_count=qwix_bwd_weight_grad_tile_count,
        weight_quantization_calibration_method=weight_quantization_calibration_method,
        act_quantization_calibration_method=act_quantization_calibration_method,
        bwd_quantization_calibration_method=bwd_quantization_calibration_method,
        drhs_grad_quantization_calibration_method=drhs_grad_quantization_calibration_method,
        quantize_mtp=quantize_mtp, quantize_router_proj=quantize_router_proj,
        quantize_logits_proj=quantize_logits_proj, logits_proj_quant_calibration_method=logits_proj_quant_calibration_method,
        logits_proj_bwd_quant_calibration_method=logits_proj_bwd_quant_calibration_method,
        sa_qproj_bwd_bf16=sa_qproj_bwd_bf16, replicate_quant_scale=replicate_quant_scale, **kwargs,
    )
    self.max_length = max_length
    self.packing = packing
    self.completion_only_loss = completion_only_loss
    self.train_data_columns = train_data_columns
    self.eval_data_columns = eval_data_columns
    self.tokenize_train_data = tokenize_train_data
    self.tokenize_eval_data = tokenize_eval_data
    self.num_epoch = num_epoch
    self.enable_data_shuffling = enable_data_shuffling
    self.data_shuffle_seed = data_shuffle_seed
    self.chat_template = chat_template
    self.chat_template_path = chat_template_path
    self.formatting_func_path = formatting_func_path
    self.formatting_func_kwargs = formatting_func_kwargs
    self.max_segments_per_seq = max_segments_per_seq
    self.grain_worker_count = grain_worker_count
    self.hf_path = hf_path
    self.hf_name = hf_name
    self.hf_data_dir = hf_data_dir
    self.hf_train_files = hf_train_files
    self.hf_eval_split = hf_eval_split
    self.hf_eval_files = hf_eval_files
    self.hf_access_token = hf_access_token
    self.dataset_path = dataset_path
    self.dataset_name = dataset_name
    self.eval_dataset_name = eval_dataset_name
    self.train_split = train_split
    self.eval_split = eval_split
    self.grain_train_mixture_config_path = grain_train_mixture_config_path
    self.grain_use_elastic_iterator = grain_use_elastic_iterator
    self.grain_packing_type = grain_packing_type
    self.grain_per_worker_buffer_size = grain_per_worker_buffer_size
    self.grain_worker_count_eval = grain_worker_count_eval
    self.grain_per_worker_buffer_size_eval = grain_per_worker_buffer_size_eval
    self.grain_num_threads = grain_num_threads
    self.grain_num_threads_eval = grain_num_threads_eval
    self.grain_prefetch_buffer_size = grain_prefetch_buffer_size
    self.grain_prefetch_buffer_size_eval = grain_prefetch_buffer_size_eval
    self.grain_data_source_max_workers = grain_data_source_max_workers
    self.grain_shuffle_buffer_size = grain_shuffle_buffer_size
    self.olmo_index_path = olmo_index_path
    self.olmo_path_remap_from = olmo_path_remap_from
    self.olmo_path_remap_to = olmo_path_remap_to
    self.olmo_apply_ngram_filter = olmo_apply_ngram_filter
    self.train_image_column = train_image_column
    self.eval_image_column = eval_image_column
    self.default_prompt = default_prompt
    self.image_placeholder = image_placeholder
    self.video_placeholder = video_placeholder
    self.max_num_images_per_example = max_num_images_per_example
    self.use_tunix_gradient_accumulation = use_tunix_gradient_accumulation

  def _overrides(self):
    return super()._overrides() | {
        key: value for key, value in (
            ("max_target_length", self.max_length), ("packing", self.packing),
            ("sft_train_on_completion_only", self.completion_only_loss),
            ("train_data_columns", self.train_data_columns), ("eval_data_columns", self.eval_data_columns),
            ("tokenize_train_data", self.tokenize_train_data), ("tokenize_eval_data", self.tokenize_eval_data),
            ("num_epoch", self.num_epoch), ("enable_data_shuffling", self.enable_data_shuffling),
            ("data_shuffle_seed", self.data_shuffle_seed), ("chat_template", self.chat_template),
            ("chat_template_path", self.chat_template_path), ("formatting_func_path", self.formatting_func_path),
            ("formatting_func_kwargs", self.formatting_func_kwargs), ("max_segments_per_seq", self.max_segments_per_seq),
            ("grain_worker_count", self.grain_worker_count),
            ("hf_path", self.hf_path), ("hf_name", self.hf_name),
            ("hf_data_dir", self.hf_data_dir), ("hf_train_files", self.hf_train_files),
            ("hf_eval_split", self.hf_eval_split), ("hf_eval_files", self.hf_eval_files),
            ("hf_access_token", self.hf_access_token), ("dataset_path", self.dataset_path),
            ("dataset_name", self.dataset_name), ("eval_dataset_name", self.eval_dataset_name),
            ("train_split", self.train_split), ("eval_split", self.eval_split),
            ("grain_train_mixture_config_path", self.grain_train_mixture_config_path),
            ("grain_use_elastic_iterator", self.grain_use_elastic_iterator),
            ("grain_packing_type", self.grain_packing_type),
            ("grain_per_worker_buffer_size", self.grain_per_worker_buffer_size),
            ("grain_worker_count_eval", self.grain_worker_count_eval),
            ("grain_per_worker_buffer_size_eval", self.grain_per_worker_buffer_size_eval),
            ("grain_num_threads", self.grain_num_threads), ("grain_num_threads_eval", self.grain_num_threads_eval),
            ("grain_prefetch_buffer_size", self.grain_prefetch_buffer_size),
            ("grain_prefetch_buffer_size_eval", self.grain_prefetch_buffer_size_eval),
            ("grain_data_source_max_workers", self.grain_data_source_max_workers),
            ("grain_shuffle_buffer_size", self.grain_shuffle_buffer_size),
            ("olmo_index_path", self.olmo_index_path), ("olmo_path_remap_from", self.olmo_path_remap_from),
            ("olmo_path_remap_to", self.olmo_path_remap_to), ("olmo_apply_ngram_filter", self.olmo_apply_ngram_filter),
            ("train_image_column", self.train_image_column), ("eval_image_column", self.eval_image_column),
            ("default_prompt", self.default_prompt), ("image_placeholder", self.image_placeholder),
            ("video_placeholder", self.video_placeholder),
            ("max_num_images_per_example", self.max_num_images_per_example),
            ("use_tunix_gradient_accumulation", self.use_tunix_gradient_accumulation),
        ) if value is not None
    }


class DPOConfig(_PostTrainingArguments):
  pass


class DistillationConfig(_PostTrainingArguments):
  def __init__(self, *args, is_offline=False, offline_data_dir=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.is_offline = is_offline
    self.offline_data_dir = offline_data_dir


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
