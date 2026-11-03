"""Native pretraining host training loop."""

import datetime

from flax import nnx

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common import metric_logger, profiler
from lilax._backend.common.goodput import RECORD_JOB_END_TIME, record_goodput
from lilax._backend.common.telemetry import mllog_utils
from lilax._backend.trainers.pre_train.loop.iteration import training_loop_iteration
from lilax._backend.trainers.pre_train.loop.precompile import (
    build_shaped_eval_batch, precompile_eval_programs, precompile_train_programs,
)
from lilax._backend.trainers.pre_train.loop.programs import build_step_programs
from lilax._backend.trainers.pre_train.loop.run_clock import start_run_clock
from lilax._backend.trainers.pre_train.loop.te_moe_overflow import track_te_moe_overflow
from lilax._backend.trainers.pre_train.loop.warmup import build_warmup_fn
from lilax._backend.trainers.pre_train.start_step import get_first_step
from lilax._backend.utils import elastic_utils, exceptions, max_logging, train_utils


def train_loop(config, recorder, state=None, elastic_retry_cache=None, *, setup=None, reporter_factory=None,
               training_graph=None):
  if setup is None:
    setup = train_utils.setup_train_loop(config, recorder, elastic_retry_cache=elastic_retry_cache)
  (
      init_rng, checkpoint_manager, state_mesh_shardings, model, mesh, learning_rate_schedule,
      data_iterator, data_loader, rampup_manager, eval_data_iterator, state,
  ) = setup
  train_utils.maybe_apply_dcn_throttling(config)
  start_step = get_first_step(model, state)
  train_utils.validate_completed_steps(start_step, config.steps)
  state, state_mesh_shardings, programs = build_step_programs(
      config, model, mesh, state, state_mesh_shardings, eval_data_iterator, elastic_retry_cache
  )
  state, data_sharding, shaped_batch, precompiled_train = precompile_train_programs(
      config, mesh, state, state_mesh_shardings, init_rng, programs
  )
  shaped_eval_batch = build_shaped_eval_batch(config, mesh, programs["p_eval_step"])
  precompiled_eval = precompile_eval_programs(config, mesh, state, init_rng, shaped_eval_batch, programs)
  metric_logger_instance = metric_logger.MetricLogger(
      config=config, learning_rate_schedule=learning_rate_schedule, start_step=start_step
  )
  def on_capture_complete(**capture):
    if metric_logger_instance.reporter is not None:
      metric_logger_instance.reporter.log_capture(**capture)

  prof = profiler.Profiler(config, offset_step=start_step, on_capture_complete=on_capture_complete)
  if config.enable_diloco:
    setup_params = state.params
  else:
    _, setup_params, _ = nnx.split(state.model, nnx.Param, ...)
  metric_logger_instance.write_setup_info_to_tensorboard(setup_params)
  elastic_utils.record_elastic_reinit_end()
  jax_device_state = {
      "state": state, "init_rng": init_rng, "mesh": mesh, "state_mesh_shardings": state_mesh_shardings,
      **programs, "model": model,
  }
  python_vars = {
      "step": start_step, "last_step_completion": datetime.datetime.now(), "data_loader": data_loader,
      "rampup_manager": rampup_manager, "recorder": recorder, "checkpoint_manager": checkpoint_manager,
      "data_iterator": data_iterator, "eval_data_iterator": eval_data_iterator,
      "metric_logger_instance": metric_logger_instance, "prof": prof,
  }
  immutable_data = {
      "config": config, "training_graph": training_graph, "logical_axis_rules_for_train": config.logical_axis_rules,
      "logical_axis_rules_for_eval": config.logical_axis_rules_for_eval,
      "shard_optimizer_over_data": config.shard_optimizer_over_data, "shard_mode": config.shard_mode,
      "steps": config.steps, "eval_interval": config.eval_interval, "eval_steps": config.eval_steps,
      "eval_start_step": config.eval_start_step, "save_checkpoint_on_completion": config.save_checkpoint_on_completion,
      "start_step": start_step, "dump_hlo": config.dump_hlo, "dump_step": config.dump_step,
      "dump_hlo_local_dir": config.dump_hlo_local_dir, "dump_hlo_gcs_dir": config.dump_hlo_gcs_dir,
      "dump_hlo_module_name": config.dump_hlo_module_name, "dump_hlo_delete_local_after": config.dump_hlo_delete_local_after,
      "dump_hlo_upload_all": config.dump_hlo_upload_all,
  }
  warmup_fn = build_warmup_fn(
      config, programs, precompiled_train, precompiled_eval, state, mesh, init_rng, start_step,
      data_loader, data_sharding, shaped_batch, shaped_eval_batch,
  )
  job_completed_gracefully = False
  te_moe_overflow_window = []
  try:
    if reporter_factory is not None:
      metric_logger_instance.reporter = reporter_factory(config)
    train_utils.prepare_before_run_start(
        config, state, learning_rate_schedule, start_step, mesh, data_loader, shaped_batch, shaped_eval_batch=shaped_eval_batch
    )
    python_vars["last_step_completion"] = datetime.datetime.now()
    start_run_clock(config, start_step, warmup_fn=warmup_fn, python_vars=python_vars)
    python_vars["eval_prefill_thread"] = None
    if config.eval_cache_prefill_in_background and eval_data_iterator is not None:
      python_vars["eval_prefill_thread"] = train_utils.start_eval_cache_prefill(eval_data_iterator)
    while python_vars["step"] < immutable_data["steps"]:
      step = python_vars["step"]
      metrics = training_loop_iteration(jax_device_state, python_vars, immutable_data, elastic_retry_cache=elastic_retry_cache)
      python_vars["step"] += 1
      if getattr(config, "te_moe_block", False):
        track_te_moe_overflow(config, step, metrics, te_moe_overflow_window, prof)
    state = jax_device_state["state"]
    if immutable_data["save_checkpoint_on_completion"]:
      checkpointing.maybe_save_checkpoint(
          checkpoint_manager, state, config, data_iterator, training_graph=training_graph
      )
    if checkpoint_manager is not None:
      checkpointing.wait_until_finished(checkpoint_manager)
    job_completed_gracefully = True
  except exceptions.StopTraining as error:
    prof.deactivate()
    max_logging.log(f"Training stopped: {str(error)}")
    job_completed_gracefully = True
  finally:
    metric_logger_instance.flush_metrics_and_cleanup()
    if job_completed_gracefully:
      record_goodput(recorder, RECORD_JOB_END_TIME)
      samples_count = (python_vars["step"] - immutable_data["start_step"]) * config.global_batch_size_to_train_on
      mllog_utils.run_stop(status="aborted", current_epoch_num=samples_count, step=python_vars["step"])
    train_utils.maybe_cleanup_dcn_throttling(config)
  return state
