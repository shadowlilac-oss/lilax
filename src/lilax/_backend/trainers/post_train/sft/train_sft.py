from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.core.spmd import logical_axis_rules
from flax.nnx import tracers
from orbax import checkpoint as ocp
from tunix.sft import diffusion as tunix_diffusion_sft
from tunix.sft import metrics_logger, peft_trainer, profiler

from lilax._backend.common.data_loader import DataLoader
from lilax._backend.common.goodput import GoodputEvent, RECORD_JOB_END_TIME, maybe_record_goodput, record_goodput
from lilax._backend.integration.tunix import diffusion_sft as maxtext_diffusion_sft
from lilax._backend.optimizers import optimizers
from lilax._backend.trainers.post_train.sft import hooks
from lilax._backend.trainers.post_train.checkpoint_selection import restore_trainer_step
from lilax._backend.trainers.pre_train.train import loss_fn
from lilax._backend.utils import lora_utils, max_logging, maxtext_utils, model_creation_utils


class MaxTextPeftTrainer(peft_trainer.PeftTrainer):
  def create_train_step_fn(self):
    loss_fn_ref = self.loss_fn
    has_aux = self._has_aux
    gen_fn = self.gen_model_input_fn
    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    nnx.pop(self.model, nnx.Intermediate)
    graphdef, _, _ = nnx.split(self.model, wrt, ...)
    uses_gradient_accumulation = not (
        self.config.get_with_default("gradient_accumulation_steps", 1) == 1 and self.config.max_seq_token_per_tpu is None
    )

    def train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, grad_accumulator: Any = None,
        inputs: Any = None, is_update_step: Any = True,
    ):
      inputs = gen_fn(inputs)
      nnx.pop(model, nnx.Intermediate)
      _, diff_params, rest = nnx.split(model, wrt, ...)

      def loss_wrapper(diff_params, rest, **inputs_kw):
        local_model = nnx.merge(graphdef, diff_params, rest)
        for _, variable in nnx.iter_graph(local_model):
          if isinstance(variable, nnx.Variable) and hasattr(variable, "_trace_state"):
            if not variable._trace_state.is_valid():
              object.__setattr__(variable, "_trace_state", tracers.TraceState())
        out = loss_fn_ref(local_model, **inputs_kw)
        _, _, new_rest = nnx.split(local_model, wrt, ...)
        if has_aux:
          loss, aux = out
          return loss, (aux, new_rest)
        return out, (None, new_rest)

      grad_fn = jax.value_and_grad(loss_wrapper, argnums=0, has_aux=True)
      (out_val, (aux, new_rest)), grads = grad_fn(diff_params, rest, **inputs)
      for _, variable in nnx.iter_graph(model):
        if isinstance(variable, nnx.Variable) and hasattr(variable, "_trace_state"):
          if not variable._trace_state.is_valid():
            object.__setattr__(variable, "_trace_state", tracers.TraceState())
      nnx.update(model, new_rest)

      if not uses_gradient_accumulation:
        if isinstance(grads, dict) and not isinstance(grads, nnx.State):
          grads = nnx.State(grads)
        optimizer.update(model, grads)
        grad_norm = optax.global_norm(jax.tree_util.tree_map(lambda value: value.astype(jnp.float32), grads))
      else:
        grad_accumulator.add(grads)

        def apply_updates(model, optimizer, grad_accumulator):
          acc_grads = grad_accumulator.get()
          norm = optax.global_norm(jax.tree_util.tree_map(lambda value: value.astype(jnp.float32), acc_grads))
          if isinstance(acc_grads, dict) and not isinstance(acc_grads, nnx.State):
            acc_grads = nnx.State(acc_grads)
          optimizer.update(model, acc_grads)
          grad_accumulator.reset()
          return norm

        def skip_updates(model, optimizer, grad_accumulator):
          return jnp.array(0.0, dtype=jnp.float32)

        grad_norm = nnx.cond(is_update_step, apply_updates, skip_updates, model, optimizer, grad_accumulator)
      return out_val, aux if has_aux else None, grad_norm

    return train_step


def get_tunix_config(mt_config, report_to=None):
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=mt_config.checkpoint_period, enable_async_checkpointing=mt_config.async_checkpointing
  )
  metrics_logging_options = (metrics_logger.MetricsLoggerOptions(log_dir=mt_config.tensorboard_dir)
                             if report_to is None else None)
  profiler_options = None
  if mt_config.profiler:
    set_profile_options = True
    platform_version = jax.extend.backend.get_backend().platform_version.strip()
    if platform_version.startswith("Pathways"):
      max_logging.log("Pathways backend detected. Disabling setting profile options.")
      set_profile_options = False
    profiler_options = profiler.ProfilerOptions(
        log_dir=mt_config.tensorboard_dir, skip_first_n_steps=mt_config.skip_first_n_steps_for_profiler,
        profiler_steps=mt_config.profiler_steps, set_profile_options=set_profile_options,
        enable_continuous_profiling=mt_config.enable_continuous_profiling,
    )
  return peft_trainer.TrainingConfig(
      eval_every_n_steps=mt_config.eval_interval, max_steps=mt_config.steps,
      gradient_accumulation_steps=mt_config.gradient_accumulation_steps if mt_config.gradient_accumulation_steps > 1 else None,
      checkpoint_root_directory=mt_config.checkpoint_dir, checkpointing_options=checkpointing_options,
      metrics_logging_options=metrics_logging_options, profiler_options=profiler_options,
      data_sharding_axis=tuple(mt_config.data_sharding),
  )


def use_maxtext_loss_function(trainer, mt_config):
  def make_loss_func(is_train):
    def loss_func(model, inputs, inputs_position, inputs_segmentation, targets, targets_position, targets_segmentation):
      data = {
          "inputs": inputs, "inputs_position": inputs_position, "inputs_segmentation": inputs_segmentation,
          "targets": targets, "targets_position": targets_position, "targets_segmentation": targets_segmentation,
      }
      return loss_fn(model, mt_config, data, dropout_rng=None, params=None, is_train=is_train)
    return loss_func

  trainer = trainer.with_loss_fn(make_loss_func(is_train=True), has_aux=True)
  trainer.eval_loss_fn = make_loss_func(is_train=False)
  return trainer


def configure_training_objective(trainer, mt_config):
  if getattr(mt_config, "training_objective", "causal_lm") != "block_diffusion":
    return use_maxtext_loss_function(trainer, mt_config)
  max_logging.log("Configuring Tunix target-aligned block-diffusion SFT adapter.")
  return tunix_diffusion_sft.configure_diffusion_sft(
      trainer, maxtext_diffusion_sft.create_batch_adapter(mt_config),
      maxtext_diffusion_sft.create_target_aligned_logits_fn(mt_config),
  )


def _create_trainer(model, optimizer, tunix_config, mt_config):
  if getattr(mt_config, "training_objective", "causal_lm") == "block_diffusion":
    return peft_trainer.PeftTrainer(model, optimizer, tunix_config)
  return MaxTextPeftTrainer(model, optimizer, tunix_config)


def validate_config(config):
  if config.optimizer_memory_host_offload:
    raise ValueError(
        "optimizer_memory_host_offload=True is not supported on the post-training "
        "SFT path because the underlying Tunix PeftTrainer does not support "
        "host offloading of the optimizer state."
    )
  if getattr(config, "training_objective", "causal_lm") == "block_diffusion":
    unsupported_auxiliary_losses = {
        name: value for name, value in (
            ("z_loss_multiplier", getattr(config, "z_loss_multiplier", 0.0)),
            ("load_balance_loss_weight", getattr(config, "load_balance_loss_weight", 0.0)),
            ("mtp_num_layers", getattr(config, "mtp_num_layers", 0)),
            ("indexer_loss_scaling_factor", getattr(config, "indexer_loss_scaling_factor", 0.0)),
        ) if value != 0
    }
    if unsupported_auxiliary_losses:
      settings = ", ".join(f"{name}={value!r}" for name, value in unsupported_auxiliary_losses.items())
      raise ValueError(
          "Block-diffusion SFT currently supports only weighted cross-entropy; "
          f"set unsupported auxiliary-loss settings to zero: {settings}"
      )


def setup_trainer_state(
    mt_config, goodput_recorder=None, *, model=None, mesh=None, train_iterator=None, eval_iterator=None,
    checkpoint_step=None, report_to=None, checkpoint_manager=None,
):
  validate_config(mt_config)
  tunix_config = get_tunix_config(mt_config, report_to=report_to)
  checkpoint_root = tunix_config.checkpoint_root_directory
  if checkpoint_step is not None or checkpoint_manager is not None:
    tunix_config.checkpoint_root_directory = None
  supplied_model = model is not None
  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    if model is None:
      if mesh is None:
        model, mesh = model_creation_utils.from_pretrained(mt_config)
      else:
        model = model_creation_utils.from_pretrained(mt_config, mesh=mesh)
    elif mesh is None:
      mesh = maxtext_utils.get_mesh_from_config(mt_config, None)
  with jax.set_mesh(mesh), logical_axis_rules(mt_config.logical_axis_rules):
    if mt_config.lora.enable_lora and not (supplied_model and lora_utils.is_lora_enabled(model)):
      model = lora_utils.apply_lora_to_model(model, mesh, mt_config)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule, model)
    if mt_config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold), optimizer)
    with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
      training_hooks = hooks.SFTTrainingHooks(mt_config, mesh, learning_rate_schedule, goodput_recorder)
      data_hooks = hooks.SFTDataHooks(
          mt_config, mesh, goodput_recorder, train_iterator=train_iterator, eval_iterator=eval_iterator
      )
      nnx.pop(model, nnx.Intermediate)
      if mt_config.lora.lora_restore_path:
        lora_utils.restore_lora_from_path(model, mt_config)
      trainer = _create_trainer(model, optimizer, tunix_config, mt_config)
      if checkpoint_step is not None or checkpoint_manager is not None:
        restore_trainer_step(trainer, checkpoint_root, checkpoint_step, checkpoint_manager)
      trainer.with_training_hooks(training_hooks)
      trainer.with_data_hooks(data_hooks)
      trainer = configure_training_objective(trainer, mt_config)
  return trainer, mesh


def train_model(mt_config, trainer, mesh, *, train_iterator=None, eval_iterator=None):
  with jax.set_mesh(mesh), logical_axis_rules(mt_config.logical_axis_rules):
    data_hooks = trainer.data_hooks
    if train_iterator is not None:
      data_hooks.train_data_iterator = train_iterator
      data_hooks.train_data_loader = DataLoader(
          data_hooks.config, data_hooks.mesh, train_iterator, goodput_recorder=data_hooks.goodput_recorder
      )
    if eval_iterator is not None:
      data_hooks.eval_data_iterator = eval_iterator
    is_block_diffusion = getattr(mt_config, "training_objective", "causal_lm") == "block_diffusion"
    records_internal_metrics = bool(getattr(mt_config, "record_internal_nn_metrics", False))
    enable_nnx_cache = mt_config.num_experts <= 1 and not (is_block_diffusion and records_internal_metrics)
    trainer.train(data_hooks.train_data_iterator, data_hooks.eval_data_iterator, cache_nnx_graph=enable_nnx_cache)
  return trainer


def train(
    mt_config, goodput_recorder=None, *, model=None, mesh=None, train_iterator=None, eval_iterator=None,
    checkpoint_step=None, reporter_factory=None, report_to=None, checkpoint_manager=None,
):
  trainer, mesh = setup_trainer_state(
      mt_config, goodput_recorder, model=model, mesh=mesh, train_iterator=train_iterator, eval_iterator=eval_iterator,
      checkpoint_step=checkpoint_step, report_to=report_to, checkpoint_manager=checkpoint_manager,
  )
  job_completed_gracefully = False
  reporter = None
  try:
    if reporter_factory is not None:
      reporter = reporter_factory(mt_config)
      trainer.training_hooks.metric_logger.reporter = reporter
    trainer = train_model(mt_config, trainer, mesh)
    job_completed_gracefully = True
  finally:
    try:
      if not job_completed_gracefully:
        trainer.training_hooks.metric_logger.flush_metrics_and_cleanup()
    finally:
      if reporter is not None:
        reporter.close()
    if job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
  return trainer, mesh
