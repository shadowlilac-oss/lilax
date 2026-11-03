import jax
import optax
from flax import nnx
from flax.core.spmd import logical_axis_rules
from orbax import checkpoint as ocp
from tunix.sft import metrics_logger
from tunix.sft.dpo.dpo_trainer import DPOTrainer, DPOTrainingConfig

from lilax._backend.common.goodput import GoodputEvent, RECORD_JOB_END_TIME, maybe_record_goodput, record_goodput
from lilax._backend.common.data_loader import DataLoader
from lilax._backend.input_pipeline import tokenizer
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.optimizers import optimizers
from lilax._backend.trainers.post_train.dpo import hooks
from lilax._backend.trainers.post_train.checkpoint_selection import restore_trainer_step
from lilax._backend.trainers.post_train.profiling import TrainingCapture
from lilax._backend.utils import device_mesh
from lilax._backend.utils import maxtext_utils
from lilax._backend.utils.model_creation import pretrained


def get_tunix_config(mt_config, report_to=None) -> DPOTrainingConfig:
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=mt_config.checkpoint_period,
      enable_async_checkpointing=mt_config.async_checkpointing,
  )
  metrics_logging_options = (metrics_logger.MetricsLoggerOptions(log_dir=mt_config.tensorboard_dir)
                             if report_to is None else None)

  return DPOTrainingConfig(
      eval_every_n_steps=mt_config.eval_interval,
      max_steps=mt_config.steps,
      gradient_accumulation_steps=mt_config.gradient_accumulation_steps,
      checkpoint_root_directory=mt_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
      metrics_logging_options=metrics_logging_options,
      profiler_options=None,
      algorithm=mt_config.dpo.algo,
      lambda_orpo=mt_config.dpo.orpo_lambda,
      beta=mt_config.dpo.dpo_beta,
      label_smoothing=mt_config.dpo.dpo_label_smoothing,
      max_prompt_length=mt_config.dpo.max_prompt_length,
      max_response_length=mt_config.max_target_length - mt_config.dpo.max_prompt_length,
  )


def setup_trainer_state(
    mt_config, goodput_recorder=None, test_only_training_hooks_class=None, *,
    model=None, mesh=None, train_iterator=None, eval_iterator=None, checkpoint_step=None, report_to=None,
    checkpoint_manager=None,
):
  validate_config(mt_config)
  tunix_config = get_tunix_config(mt_config, report_to=report_to)
  checkpoint_root = tunix_config.checkpoint_root_directory
  if checkpoint_step is not None or checkpoint_manager is not None:
    tunix_config.checkpoint_root_directory = None

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    tok = tokenizer.build_tokenizer(
        mt_config.tokenizer_path, mt_config.tokenizer_type, False, False, mt_config.hf_access_token
    )
    if model is None:
      if mesh is None:
        model, mesh = pretrained.from_pretrained(
            mt_config, wrap_with_tunix_adapter=True, tokenizer_pad_id=tok.pad_id
        )
      else:
        model = pretrained.from_pretrained(
            mt_config, mesh=mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=tok.pad_id
        )
    else:
      if mesh is None:
        mesh = device_mesh.get_mesh_from_config(mt_config, None)
      with mesh:
        model = TunixMaxTextAdapter(
            base_model=model, use_no_op_mappings="maxtext_config" in mt_config.vllm_additional_config,
            pad_id=tok.pad_id,
        )
        model.config = None
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule, model)
    if mt_config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold), optimizer)

  ref_model = nnx.clone(model) if mt_config.dpo.algo == "dpo" else None

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    training_hooks_class = test_only_training_hooks_class or hooks.DPOTrainingHooks
    training_hooks = training_hooks_class(mt_config, mesh, learning_rate_schedule, goodput_recorder)
    data_hooks = hooks.DPODataHooks(
        mt_config, mesh, goodput_recorder, train_iterator=train_iterator, eval_iterator=eval_iterator
    )
    with logical_axis_rules(mt_config.logical_axis_rules):
      trainer = DPOTrainer(
          model=model, ref_model=ref_model, optimizer=optimizer, training_config=tunix_config, tokenizer=None
      )
      if checkpoint_step is not None or checkpoint_manager is not None:
        restore_trainer_step(trainer, checkpoint_root, checkpoint_step, checkpoint_manager)
      trainer.with_training_hooks(training_hooks)
      trainer.with_data_hooks(data_hooks)

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
    data_hooks.capture = TrainingCapture(mt_config, trainer)
    try:
      trainer.train(data_hooks.train_data_iterator, data_hooks.eval_data_iterator)
    finally:
      data_hooks.finish_capture(trainer)
  return trainer


def train(
    mt_config, goodput_recorder=None, test_only_training_hooks_class=None, *,
    model=None, mesh=None, train_iterator=None, eval_iterator=None, checkpoint_step=None, reporter_factory=None,
    report_to=None, checkpoint_manager=None,
):
  trainer, mesh = setup_trainer_state(
      mt_config, goodput_recorder, test_only_training_hooks_class, model=model, mesh=mesh,
      train_iterator=train_iterator, eval_iterator=eval_iterator, checkpoint_step=checkpoint_step, report_to=report_to,
      checkpoint_manager=checkpoint_manager,
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


def validate_config(config):
  if config.optimizer_memory_host_offload:
    raise ValueError(
        "optimizer_memory_host_offload=True is not supported on the post-training "
        "DPO path because the underlying Tunix DPOTrainer does not support "
        "host offloading of the optimizer state."
    )

  if config.num_vocab_tiling > 1:
    raise ValueError(
        f"Vocab Tiling is not supported with DPO. "
        f"num_vocab_tiling was configured to {config.num_vocab_tiling}, but it must be 1 when running train_dpo."
    )
