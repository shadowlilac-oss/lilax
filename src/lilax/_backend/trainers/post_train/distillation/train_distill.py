"""Native MaxText and Tunix distillation training."""

import inspect
import itertools
import logging
import re
from typing import Any, Callable

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint
from tunix.sft import metrics_logger, peft_trainer, profiler

from lilax._backend.common.goodput import (
    GoodputEvent, RECORD_JOB_END_TIME, RECORD_JOB_START_TIME, create_goodput_recorder,
    maybe_monitor_goodput, maybe_record_goodput, record_goodput,
)
from lilax._backend.input_pipeline import input_pipeline_interface, tokenizer
from lilax._backend.layers.learn_to_init_layer import apply_lti_model_update
from lilax._backend.optimizers import optimizers
from lilax._backend.trainers.post_train.distillation import distillation_utils, lti_utils
from lilax._backend.trainers.post_train.hooks import GoodputTrainingHooks
from lilax._backend.utils import device_mesh, max_logging, maxtext_utils, model_creation_utils


class ReportingMetricsLogger(metrics_logger.MetricsLogger):
  def __init__(self):
    super().__init__()
    self.reporter = None
    self.pending = {}

  def log(self, metrics_prefix, metric_name, scalar_value, mode, step):
    super().log(metrics_prefix, metric_name, scalar_value, mode, step)
    if self.reporter is not None and jax.process_index() == 0:
      self.pending[f"{metrics_prefix}/{mode}/{metric_name}"] = scalar_value

  def report(self, step, mode):
    if self.reporter is not None and self.pending:
      self.reporter.log({"scalar": self.pending}, step, str(mode))
      self.pending = {}


def get_distillation_optimizer(config, max_train_steps):
  """Create the MaxText optimizer with an injectable learning rate."""
  if config.opt_type == "muon":
    raise ValueError("Muon optimizer is not currently supported in distillation mode.")
  schedule = optax.schedules.warmup_cosine_decay_schedule(
      init_value=0.0, peak_value=config.learning_rate,
      warmup_steps=int(config.warmup_steps_fraction * max_train_steps), decay_steps=max_train_steps,
      end_value=config.learning_rate_final_fraction * config.learning_rate,
  )

  def optimizer_factory(learning_rate):
    optimizer = optimizers.get_optimizer(config, learning_rate, model=None)
    if config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(optax.clip_by_global_norm(max_norm=config.gradient_clipping_threshold), optimizer)
    return optimizer

  return optax.inject_hyperparams(optimizer_factory)(learning_rate=schedule)


def create_forward_fn(config) -> Callable[..., distillation_utils.DistillationForwardOutput]:
  """Bind the native model forward pass to its configuration."""
  def model_forward_fn(model, input_tokens, positions, attention_mask, decoder_segment_ids=None, cache=None, **kwargs):
    del attention_mask, cache
    logits = model(
        decoder_input_tokens=input_tokens, decoder_positions=positions, decoder_segment_ids=decoder_segment_ids,
        enable_dropout=config.enable_dropout, decoder_target_tokens=kwargs.get("decoder_target_tokens", None),
        decoder_target_mask=kwargs.get("decoder_target_mask", None),
    )
    out_projection_activations = None
    if config.distill_beta > 0.0:
      out_projection_activations = maxtext_utils.get_intermediate_value(model, "out_projection_activations", clear=True)
    moe_lb_loss = None
    if config.num_experts > 1 and config.load_balance_loss_weight > 0.0:
      intermediate_outputs = nnx.pop(model, nnx.Intermediate)
      total_moe_lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_lb_loss")
      if total_moe_lb_losses:
        moe_lb_loss = jnp.mean(jnp.concatenate(total_moe_lb_losses))
    return distillation_utils.DistillationForwardOutput(
        logits=logits, out_projection_activations=out_projection_activations, moe_lb_loss=moe_lb_loss
    )
  return model_forward_fn


def _log_config_details(config, label: str) -> None:
  kv_heads = getattr(config, "num_kv_heads", config.num_query_heads)
  max_logging.log(f"--- {label} Configuration ---")
  max_logging.log(f"  Model Name:      {config.model_name}")
  max_logging.log(f"  Dimensions:      {config.num_decoder_layers} Layers, {config.emb_dim} Emb Dim, {config.head_dim} Head Dim")
  max_logging.log(f"  Attention Heads: {config.num_query_heads} Query, {kv_heads} KV")
  max_logging.log(f"  Vocab Size:      {config.vocab_size}")
  max_logging.log(f"  Checkpoint:      {config.load_parameters_path}")


class ModelBundle(nnx.Module):
  """Teacher, student, and distillation step state."""

  def __init__(self, teacher_model: nnx.Module | None, student_model: nnx.Module):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.training_step = nnx.Variable(jnp.zeros((), dtype=jnp.int32))

  def __call__(self, *args, **kwargs):
    raise NotImplementedError("Use `call_student` or `call_teacher` explicitly.")

  def call_student(self, *args, **kwargs):
    return self.student_model(*args, **kwargs)

  def call_teacher(self, *args, **kwargs):
    return jax.lax.stop_gradient(self.teacher_model(*args, **kwargs))


class MaxTextDistillationTrainer(peft_trainer.PeftTrainer):
  """Tunix trainer with student-only optimization and native batch fields."""

  def __init__(
      self, model, strategy: distillation_utils.DistillationStrategy, optimizer, training_config, student_config,
      teacher_config, is_offline: bool = False, student_freeze_param_filter: Callable[[Any], bool] | None = None,
      **kwargs,
  ):
    super().__init__(model=model, optimizer=optax.set_to_zero(), training_config=training_config, **kwargs)
    self.strategy = strategy
    self.checkpoint_manager = None
    self._tflops_combined, self._tflops_student, self._tflops_teacher = (
        distillation_utils.calculate_distillation_tflops_per_device(student_config, teacher_config, is_offline=is_offline)
    )
    max_logging.log(
        f"Per-step per-device TFLOPs — combined: {self._tflops_combined:.2f}, "
        f"student (fwd+bwd): {self._tflops_student:.2f}, teacher (fwd-only): {self._tflops_teacher:.2f}"
    )
    if training_config.gradient_accumulation_steps is not None and training_config.gradient_accumulation_steps > 1:
      optimizer = optax.MultiSteps(optimizer, training_config.gradient_accumulation_steps)
    base_wrt = nnx.LoRAParam if getattr(self, "_lora_enabled", False) else nnx.Param
    if student_freeze_param_filter:
      def wrt_filter(path, value):
        if not isinstance(value, base_wrt):
          return False
        freeze = student_freeze_param_filter(path)
        logging.info("Student model freezing info: Parameter %s; freeze=%s", path, freeze)
        return not freeze
      self.wrt_filter = wrt_filter
    else:
      self.wrt_filter = base_wrt
    self.optimizer = nnx.Optimizer(model.student_model, optimizer, wrt=self.wrt_filter)
    self._tunix_expects_grad_norm = False
    try:
      source = inspect.getsource(peft_trainer.PeftTrainer._train_step)
      self._tunix_expects_grad_norm = "grad_norm" in source
    except (TypeError, OSError):
      pass

  def _train_step(self, model, optimizer, grad_accumulator, inputs, is_update_step=True, **kwargs):
    batch = self.gen_model_input_fn(inputs)
    student = model.student_model
    teacher = model.teacher_model
    current_step = model.training_step[...]
    if "teacher_output" in batch:
      teacher_output = batch["teacher_output"]
    else:
      teacher_output = self.strategy.teacher_forward_fn(
          model=teacher, input_tokens=batch["input_tokens"], positions=batch["positions"],
          attention_mask=batch.get("attention_mask"), decoder_segment_ids=batch.get("decoder_segment_ids"),
          decoder_target_tokens=batch.get("targets", None), decoder_target_mask=batch.get("targets_segmentation", None),
          cache=None,
      )
    teacher_output = jax.tree.map(jax.lax.stop_gradient, teacher_output)
    student_graphdef, diff_params, rest = nnx.split(student, self.wrt_filter, ...)

    def loss_wrapper_pure(diff_params, rest):
      local_student = nnx.merge(student_graphdef, diff_params, rest, copy=True)
      student_output = self.strategy.student_forward_fn(
          model=local_student, input_tokens=batch["input_tokens"], positions=batch["positions"],
          attention_mask=batch.get("attention_mask"), decoder_segment_ids=batch.get("decoder_segment_ids"),
          decoder_target_tokens=batch.get("targets", None), decoder_target_mask=batch.get("targets_segmentation", None),
          cache=None,
      )
      labels = self.strategy.create_labels(batch["targets"], targets_segmentation=batch.get("targets_segmentation", None))
      loss, aux = self.strategy.compute_loss(student_output, teacher_output, labels, step=current_step)
      _, _, new_rest = nnx.split(local_student, self.wrt_filter, ...)
      return loss, (aux, new_rest)

    grad_fn = jax.value_and_grad(loss_wrapper_pure, argnums=0, has_aux=True)
    (loss, (aux, new_rest)), grads = grad_fn(diff_params, rest)
    nnx.update(student, new_rest)
    optimizer.update(student, grads)
    model.training_step.set_value(current_step + 1)
    if getattr(self, "_tunix_expects_grad_norm", True):
      return loss, aux, optax.global_norm(grads)
    return loss, aux

  def _eval_step(self, model, inputs):
    inputs = self.gen_model_input_fn(inputs)
    student_output = self.strategy.student_forward_fn(
        model=model.student_model, input_tokens=inputs["input_tokens"], positions=inputs["positions"],
        attention_mask=inputs.get("attention_mask"), decoder_segment_ids=inputs.get("decoder_segment_ids"), cache=None,
    )
    labels = self.strategy.create_labels(inputs["targets"], targets_segmentation=inputs.get("targets_segmentation", None))
    return self.strategy.compute_eval_loss(student_output, labels)

  def _log_metrics(self, loss, step=None, additional_metrics=None, **kwargs):
    super()._log_metrics(loss=loss, step=step, additional_metrics=additional_metrics, **kwargs)
    step_time_delta = kwargs.get("step_time_delta")
    tflops_metrics = {
        "perf/per_device_tflops": self._tflops_combined, "perf/per_device_tflops_student": self._tflops_student,
        "perf/per_device_tflops_teacher": self._tflops_teacher,
    }
    tflops_per_sec = None
    if step_time_delta is not None and step_time_delta > 0:
      tflops_per_sec = self._tflops_combined / step_time_delta
      tflops_metrics.update({
          "perf/per_device_tflops_per_sec": tflops_per_sec,
          "perf/per_device_tflops_per_sec_student": self._tflops_student / step_time_delta,
          "perf/per_device_tflops_per_sec_teacher": self._tflops_teacher / step_time_delta,
      })
    for name, value in tflops_metrics.items():
      self.metrics_logger.log(self.metrics_prefix, name, value, self._mode, step)
    if isinstance(self.metrics_logger, ReportingMetricsLogger):
      self.metrics_logger.report(step, self._mode)
    if tflops_per_sec is not None and self._mode == metrics_logger.Mode.TRAIN:
      max_logging.log(
          f"step {step} | step_time={step_time_delta:.2f}s | TFLOPs/s/device: {tflops_per_sec:.2f} "
          f"(student={self._tflops_student / step_time_delta:.2f}, teacher={self._tflops_teacher / step_time_delta:.2f})"
      )

  def _prepare_inputs(self, input_data: distillation_utils.MaxTextTrainingInput):
    if hasattr(input_data, "microbatches"):
      input_data.microbatches = [self._prepare_inputs(batch) for batch in input_data.microbatches]
      return input_data
    return distillation_utils.MaxTextTrainingInput(
        input_tokens=input_data.input_tokens, input_mask=input_data.input_mask, positions=input_data.positions,
        decoder_segment_ids=input_data.decoder_segment_ids, targets=input_data.targets,
        targets_position=input_data.targets_position, targets_segmentation=input_data.targets_segmentation,
        top_k_logits=input_data.top_k_logits, top_k_indices=input_data.top_k_indices,
    )

  def _post_process_train_step(self, aux: dict[str, tuple[jax.Array, jax.Array]]) -> None:
    if self._buffered_train_metrics is None:
      return
    for name, value in aux.items():
      if name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[name] = ([], distillation_utils.weighted_mean)
      self._buffered_train_metrics.additional_metrics[name][0].append(value)

    def scalar(value):
      total, count = value
      count = float(count)
      return float(total) / count if count > 0 else 0.0

    headline_keys = (
        distillation_utils.METRIC_TOTAL_LOSS, distillation_utils.METRIC_HARD_LOSS,
        distillation_utils.METRIC_SOFT_LOSS, distillation_utils.METRIC_KL_DIV_T1,
        distillation_utils.METRIC_STUDENT_PERPLEXITY, distillation_utils.METRIC_TEACHER_PERPLEXITY,
    )
    optional_keys = (
        distillation_utils.METRIC_OUT_PROJ_FEATURE_LOSS, distillation_utils.METRIC_MOE_LB_LOSS,
        distillation_utils.METRIC_TEACHER_MOE_LB_LOSS,
    )
    parts = []
    for key in headline_keys:
      if key in aux:
        parts.append(f"{key.split('/', 1)[1]}={scalar(aux[key]):.4g}")
    for key in optional_keys:
      if key in aux:
        value = scalar(aux[key])
        if value != 0:
          parts.append(f"{key.split('/', 1)[1]}={value:.4g}")
    if parts:
      max_logging.log("Distillation metrics | " + " ".join(parts))

  def setup_checkpoint_manager_and_restore(self, raw_train_iter, config, checkpoint_step=None, checkpoint_manager=None):
    """Restore the model and raw input iterator before wrapping it."""
    is_grain_dataset = config.dataset_type == "grain"
    has_save_method = hasattr(raw_train_iter, "save")
    enable_checkpointing = raw_train_iter is not None and (is_grain_dataset or has_save_method)
    iterator_to_manage = raw_train_iter if enable_checkpointing else None
    if enable_checkpointing:
      max_logging.log("Input Pipeline Checkpointing: ENABLED")
      max_logging.log(f"Details: dataset_type='{config.dataset_type}', has_save={has_save_method}")
    else:
      max_logging.log("Input Pipeline Checkpointing: DISABLED")
      if raw_train_iter is None:
        max_logging.log("Reason: train_iter is None")
      else:
        max_logging.log(
            f"Reason: Iterator '{type(raw_train_iter).__name__}' is not recognized as Grain "
            f"(dataset_type='{config.dataset_type}', has_save={has_save_method})"
        )
    if self.checkpoint_manager is not None:
      self.checkpoint_manager.close()
    self.checkpoint_manager = checkpoint_manager or distillation_utils.MaxTextCheckpointManager(
        raw_iterator=iterator_to_manage, root_directory=config.checkpoint_dir, student_config=config,
        options=self.config.checkpointing_options,
    )
    self.checkpoint_manager._iterator = iterator_to_manage
    if checkpoint_step is None and checkpoint_manager is None:
      checkpoint_step = self.checkpoint_manager.latest_step()
    if checkpoint_step is None:
      self._train_steps, self._restored_custom_metadata = 0, {}
    else:
      self._train_steps, self._restored_custom_metadata = self.checkpoint_manager.maybe_restore(
          self.model, self.optimizer, restore_only_lora_params=getattr(self, "_lora_enabled", False),
          step=checkpoint_step,
      )
    grad_accum_steps = self.config.get_with_default("gradient_accumulation_steps", 1)
    self._iter_steps = self._train_steps * grad_accum_steps
    if enable_checkpointing and checkpoint_step is not None:
      restored_iter = self.checkpoint_manager.restore_iterator(step=checkpoint_step)
      if restored_iter is not None:
        max_logging.log("Restored input pipeline state to match model step.")
        return restored_iter
    return raw_train_iter


def get_maxtext_model(config, mesh: jax.sharding.Mesh) -> nnx.Module:
  """Load a MaxText model on the specified mesh."""
  max_logging.log(f"Initializing model: {config.model_name}...")
  return model_creation_utils.from_pretrained(config, mesh=mesh)


def build_training_components(student_config, teacher_config, is_offline: bool = False, offline_data_dir: str | None = None):
  """Build the distillation strategy, optimizer, and Tunix training config."""
  tok = tokenizer.build_tokenizer(
      tokenizer_path=student_config.tokenizer_path, tokenizer_type=student_config.tokenizer_type,
      add_bos=student_config.add_bos, add_eos=student_config.add_eos, hf_access_token=student_config.hf_access_token,
  )
  pad_id = tok.pad_id if tok.pad_id is not None else 0
  student_forward_fn = create_forward_fn(student_config)
  teacher_forward_fn = create_forward_fn(teacher_config)
  strategy = distillation_utils.CombinedDistillationStrategy(
      student_forward_fn=student_forward_fn, teacher_forward_fn=teacher_forward_fn, pad_id=pad_id,
      temperature=student_config.distill_temperature, alpha=student_config.distill_alpha,
      beta_feature=student_config.distill_beta, layer_indices=student_config.distill_layer_indices,
      feature_loss_type=student_config.distill_feature_loss_type, vocab_size=student_config.vocab_size,
      alpha_end=student_config.distill_alpha_end, alpha_schedule=student_config.distill_alpha_schedule,
      temperature_end=student_config.distill_temperature_end,
      temperature_schedule=student_config.distill_temperature_schedule, beta_end=student_config.distill_beta_end,
      beta_schedule=student_config.distill_beta_schedule, max_steps=student_config.steps,
  )
  optimizer = get_distillation_optimizer(student_config, student_config.steps)
  checkpointing_options = checkpoint.CheckpointManagerOptions(
      save_interval_steps=student_config.checkpoint_period, max_to_keep=student_config.max_num_checkpoints_to_keep,
      enable_async_checkpointing=student_config.async_checkpointing, create=True,
  )
  profiler_options = None
  if student_config.profiler == "xplane":
    profiler_options = profiler.ProfilerOptions(
        log_dir=student_config.tensorboard_dir, skip_first_n_steps=student_config.skip_first_n_steps_for_profiler,
        profiler_steps=student_config.profiler_steps, set_profile_options=False,
        enable_continuous_profiling=student_config.enable_continuous_profiling,
    )
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=student_config.tensorboard_dir, flush_every_n_steps=student_config.log_period
  )
  train_config = peft_trainer.TrainingConfig(
      max_steps=student_config.steps, eval_every_n_steps=student_config.eval_interval,
      metrics_logging_options=metrics_logging_options, profiler_options=profiler_options,
      checkpoint_root_directory=None, checkpointing_options=checkpointing_options,
      gradient_accumulation_steps=student_config.gradient_accumulation_steps,
      data_sharding_axis=tuple(student_config.data_sharding),
  )
  return strategy, optimizer, train_config


def train_distill(
    student_config, teacher_config, is_offline: bool = False, offline_data_dir: str | None = None, *,
    student_model: nnx.Module | None = None, teacher_model: nnx.Module | None = None,
    mesh: jax.sharding.Mesh | None = None, train_iterator: Any = None, eval_iterator: Any = None,
    return_trainer: bool = False, checkpoint_step: int | None = None, report_to=None, reporter_factory=None,
    checkpoint_manager=None,
):
  """Run online or offline teacher-student distillation."""
  if student_config.vocab_size != teacher_config.vocab_size:
    raise ValueError(
        f"Vocab size mismatch! Student: {student_config.vocab_size}, Teacher: {teacher_config.vocab_size}. "
        "Distillation requires matching vocabularies."
    )
  for batch_field in ("per_device_batch_size", "max_target_length", "gradient_accumulation_steps"):
    student_value = getattr(student_config, batch_field)
    teacher_value = getattr(teacher_config, batch_field)
    if student_value != teacher_value:
      raise ValueError(
          f"Distillation batch shape mismatch on '{batch_field}': "
          f"student={student_value} vs teacher={teacher_value}. The teacher consumes batches from the "
          f"student-driven input pipeline, so these must agree. Set '{batch_field}' at "
          f"the YAML top level (not inside *_overrides) so both configs inherit it."
      )
  goodput_recorder = create_goodput_recorder(student_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  job_completed_gracefully = False
  try:
    with maybe_monitor_goodput(student_config):
      trainer, raw_eval_iter = _train_distill_impl(
          student_config, teacher_config, is_offline, offline_data_dir, goodput_recorder,
          student_model=student_model, teacher_model=teacher_model, mesh=mesh,
          train_iterator=train_iterator, eval_iterator=eval_iterator, checkpoint_step=checkpoint_step,
          report_to=report_to, reporter_factory=reporter_factory,
          checkpoint_manager=checkpoint_manager,
      )
      job_completed_gracefully = True
  finally:
    if job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
  if return_trainer:
    return trainer, raw_eval_iter


def _train_distill_impl(
    student_config, teacher_config, is_offline: bool = False, offline_data_dir: str | None = None,
    goodput_recorder: Any = None, *, student_model: nnx.Module | None = None,
    teacher_model: nnx.Module | None = None, mesh: jax.sharding.Mesh | None = None,
    train_iterator: Any = None, eval_iterator: Any = None, checkpoint_step: int | None = None,
    report_to=None, reporter_factory=None, checkpoint_manager=None,
):
  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    if mesh is None:
      devices = jax.devices()
      devices_array = device_mesh.create_device_mesh(student_config, devices)
      mesh = jax.sharding.Mesh(devices_array, student_config.mesh_axes)
    max_logging.log("Applying logical axis rules for model initialization and training...")
    with jax.set_mesh(mesh), logical_axis_rules(student_config.logical_axis_rules):
      if is_offline:
        max_logging.log("Offline Distillation: Skipping Teacher Model loading.")
        teacher_model = None
      elif teacher_model is None:
        max_logging.log(f"Loading Teacher from {teacher_config.load_parameters_path}...")
        _log_config_details(teacher_config, "Teacher")
        teacher_model = get_maxtext_model(teacher_config, mesh)
      if teacher_model is not None:
        teacher_model.eval()
      student_config.get_keys()["teacher_config"] = teacher_config
      if student_model is None:
        max_logging.log(f"Loading Student from {student_config.load_parameters_path}...")
        _log_config_details(student_config, "Student")
        student_model = get_maxtext_model(student_config, mesh)
      student_params_to_update = getattr(student_config, "student_params_to_update", []) or []
      student_param_update_templates = [re.compile(template) for template in student_params_to_update]

      def student_freeze_param_fn(path) -> bool:
        path_str = "/".join(str(part) for part in path)
        return not any(regex.search(path_str) for regex in student_param_update_templates)

      if teacher_model:
        lti_utils.prepare_student_weights(
            student_model, teacher_model,
            teacher_weights_copy_map=getattr(student_config, "distill_weights_copy_map", {}),
            student_weights_share_map=getattr(student_config, "distill_student_weights_share_map", {}),
        )
      student_model.train()
      model_bundle = ModelBundle(teacher_model, student_model)

  with jax.set_mesh(mesh), logical_axis_rules(student_config.logical_axis_rules):
    with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
      strategy, optimizer, train_config = build_training_components(
          student_config, teacher_config, is_offline, offline_data_dir
      )
      trainer = MaxTextDistillationTrainer(
          model=model_bundle, strategy=strategy, optimizer=optimizer, training_config=train_config,
          student_config=student_config, teacher_config=teacher_config, is_offline=is_offline,
          student_freeze_param_filter=student_freeze_param_fn if student_params_to_update else None,
          metrics_logger=ReportingMetricsLogger() if report_to is not None else None,
      )
      trainer.is_managed_externally = True
      trainer._has_aux = True
      if is_offline:
        max_logging.log("Initializing Data Iterators via MaxText pipeline...")
        student_config.get_keys()["grain_train_files"] = offline_data_dir
        student_config.get_keys()["eval_interval"] = 0
        student_config.get_keys()["is_offline_distillation"] = True
        student_config.get_keys()["dataset_shuffle_buffer_size"] = 0
        student_config.get_keys()["dataset_shuffle_seed"] = 0
      if train_iterator is None:
        raw_train_iter, raw_eval_iter = input_pipeline_interface.create_data_iterator(student_config, mesh)
      else:
        raw_train_iter, raw_eval_iter = train_iterator, None
      if eval_iterator is not None:
        raw_eval_iter = eval_iterator
      raw_train_iter = trainer.setup_checkpoint_manager_and_restore(
          raw_train_iter, student_config, checkpoint_step, checkpoint_manager,
      )
      model_bundle.training_step.set_value(jnp.array(trainer._train_steps, dtype=jnp.int32))

      def custom_gen_model_input_fn(batch):
        inputs_dict = {
            "input_tokens": batch.input_tokens, "positions": batch.positions, "attention_mask": batch.input_mask,
            "decoder_segment_ids": batch.decoder_segment_ids, "targets": batch.targets,
            "targets_position": batch.targets_position, "targets_segmentation": batch.targets_segmentation,
            "cache": None,
        }
        if getattr(batch, "top_k_logits", None) is None:
          return inputs_dict
        inputs_dict["teacher_output"] = distillation_utils.DistillationForwardOutput(
            logits=batch.top_k_logits, out_projection_activations=None, top_k_indices=batch.top_k_indices
        )
        return inputs_dict

      trainer = trainer.with_gen_model_input_fn(custom_gen_model_input_fn)
      trainer.with_training_hooks(GoodputTrainingHooks(student_config, goodput_recorder))
      train_iter: Any = distillation_utils.MaxTextToTunixIterator(raw_train_iter)
      if isinstance(student_config.steps, int) and isinstance(trainer._train_steps, int):
        grad_accum = student_config.gradient_accumulation_steps if isinstance(
            student_config.gradient_accumulation_steps, int
        ) else 1
        remaining_microsteps = max(0, (student_config.steps - trainer._train_steps) * grad_accum)
        train_iter = itertools.islice(train_iter, remaining_microsteps)
      eval_iter = None
      if raw_eval_iter is not None:
        max_logging.log("Evaluation iterator successfully initialized.")
        eval_iter = distillation_utils.MaxTextToTunixIterator(raw_eval_iter)
      elif student_config.eval_interval > 0:
        max_logging.log("Warning: eval_interval > 0 but create_data_iterator returned None for eval_iter.")
    max_logging.log("Starting Distillation Training...")
    reporter = None
    try:
      if report_to:
        reporter = reporter_factory(student_config, report_to, tensorboard=True)
        trainer.metrics_logger.reporter = reporter
      trainer.train(train_iter, eval_iter)
    finally:
      if reporter is not None:
        reporter.close()

  if student_config.learn_to_init_mode:
    max_logging.log("Learn-to-init mode enabled. Finalizing student model...")
    apply_lti_model_update(trainer.model.student_model, student_config)
  if student_config.save_checkpoint_on_completion:
    should_save = student_config.steps % student_config.checkpoint_period
    if should_save:
      max_logging.log(f"Saving final checkpoint to {student_config.checkpoint_dir}...")
      try:
        saved = trainer.checkpoint_manager.save(
            trainer.train_steps + (1 if student_config.learn_to_init_mode else 0), trainer.model,
            optimizer=trainer.optimizer, save_only_lora_params=getattr(trainer, "_lora_enabled", False), force=True,
        )
        if saved:
          trainer.checkpoint_manager.wait_until_finished()
          max_logging.log("Final checkpoint saved.")
      except Exception as error:
        max_logging.log(f"Warning: Failed to save final checkpoint: {error}")
    else:
      max_logging.log("Waiting for automatic periodic checkpoint to finish...")
      trainer.checkpoint_manager.wait_until_finished()
  trainer.close()
  max_logging.log("Distillation Complete.")
  return trainer, raw_eval_iter
