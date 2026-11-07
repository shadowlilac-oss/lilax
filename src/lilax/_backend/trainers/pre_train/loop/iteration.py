"""One host iteration of the native pretraining loop: step, persistence, evaluation, and metric reporting."""

import datetime
from typing import Any

from flax.core.spmd import logical_axis_rules
import jax
import numpy as np

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.checkpoints import snapshotting
from lilax._backend.common.goodput import GoodputEvent, maybe_record_goodput
from lilax._backend.common.telemetry import mllog_utils
from lilax._backend.trainers.pre_train.loop.evaluation import run_evaluation
from lilax._backend.utils import elastic_utils as elastic
from lilax._backend.storage import gcs
from lilax._backend.observability import max_logging
from lilax._backend.observability.diagnostics import pytree


def _maybe_log_required_rbf(config, step, metrics, program):
  if not getattr(config, "log_required_ragged_buffer_factor", False) or "moe_required_rbf" not in metrics:
    return
  per_layer = np.asarray(jax.device_get(metrics["moe_required_rbf"]), dtype=np.float32).ravel()
  if per_layer.size == 0:
    return
  per_layer_str = ",".join(f"{value:.4f}" for value in per_layer)
  max_logging.log(f"REQUIRED_RBF step={step} max={per_layer.max():.4f} per_layer=[{per_layer_str}] program={program}")


def _run_train_programs(
    config, state, step, start_step, example_batch, step_rng_args,
    p_train_step, p_train_step_dropless, p_train_step_first_phase,
):
  in_first_phase = config.retry_dropless_first_steps > 0 and step - start_step < config.retry_dropless_first_steps
  if in_first_phase and p_train_step_dropless is not None and p_train_step_first_phase is None:
    max_logging.log(f"Step {step}: dropless first-phase program")
    state, metrics = p_train_step_dropless(state, example_batch, *step_rng_args)
    _maybe_log_required_rbf(config, step, metrics, "dropless")
    return state, metrics
  if in_first_phase and p_train_step_first_phase is not None:
    max_logging.log(
        f"Step {step}: first-phase program (ragged_buffer_factor={config.first_phase_ragged_buffer_factor})"
    )
    p_attempt, attempt_name = p_train_step_first_phase, "first_phase"
  else:
    p_attempt, attempt_name = p_train_step, "normal"
  state, metrics = p_attempt(state, example_batch, *step_rng_args)
  _maybe_log_required_rbf(config, step, metrics, attempt_name)
  if config.moe_dropless_fallback == "step" and p_train_step_dropless is not None and bool(metrics.get("has_moe_overflow")):
    max_logging.log(f"Step {step}: MoE ragged buffer overflow detected! Replaying step with dropless buffer...")
    state, metrics = p_train_step_dropless(state, example_batch, *step_rng_args)
    _maybe_log_required_rbf(config, step, metrics, "dropless")
  return state, metrics


def training_loop_iteration(
    jax_device_state: dict[str, Any], python_vars: dict[str, Any], immutable_data: dict[str, Any],
    elastic_retry_cache: elastic.RetryCache | None = None,
):
  state = jax_device_state["state"]
  init_rng = jax_device_state["init_rng"]
  mesh = jax_device_state["mesh"]
  p_train_step = jax_device_state["p_train_step"]
  p_train_step_dropless = jax_device_state.get("p_train_step_dropless", None)
  p_train_step_first_phase = jax_device_state.get("p_train_step_first_phase", None)
  p_eval_step = jax_device_state["p_eval_step"]
  p_eval_step_dropless = jax_device_state.get("p_eval_step_dropless", None)
  step = python_vars["step"]
  last_step_completion = python_vars["last_step_completion"]
  data_loader = python_vars["data_loader"]
  rampup_manager = python_vars["rampup_manager"]
  recorder = python_vars["recorder"]
  checkpoint_manager = python_vars["checkpoint_manager"]
  data_iterator = python_vars["data_iterator"]
  eval_data_iterator = python_vars["eval_data_iterator"]
  metric_logger_instance = python_vars["metric_logger_instance"]
  prof = python_vars["prof"]
  config = immutable_data["config"]
  snapshotter = None if elastic_retry_cache is None else elastic_retry_cache.snapshotter
  logical_axis_rules_for_train = immutable_data["logical_axis_rules_for_train"]
  logical_axis_rules_for_eval = immutable_data["logical_axis_rules_for_eval"]
  eval_interval = immutable_data["eval_interval"]
  eval_steps = immutable_data["eval_steps"]
  start_step = immutable_data["start_step"]
  eval_start_step = immutable_data["eval_start_step"]
  dump_hlo = immutable_data["dump_hlo"]
  dump_step = immutable_data["dump_step"]
  dump_hlo_local_dir = immutable_data["dump_hlo_local_dir"]
  dump_hlo_gcs_dir = immutable_data["dump_hlo_gcs_dir"]
  dump_hlo_module_name = immutable_data["dump_hlo_module_name"]
  dump_hlo_delete_local_after = immutable_data["dump_hlo_delete_local_after"]
  dump_hlo_upload_all = immutable_data["dump_hlo_upload_all"]
  prof.maybe_activate_profiler(step, state)
  with jax.profiler.StepTraceAnnotation("train", step_num=step):
    example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
    if config.enable_diloco:
      step_rng_args = (jax.jit(jax.random.fold_in)(init_rng, step),)
    else:
      step_rng_args = ()
    with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
      with jax.set_mesh(mesh), logical_axis_rules(logical_axis_rules_for_train):
        state, metrics = _run_train_programs(
            config, state, step, start_step, example_batch, step_rng_args,
            p_train_step, p_train_step_dropless, p_train_step_first_phase,
        )

  step_time_delta = datetime.datetime.now() - last_step_completion
  last_step_completion = datetime.datetime.now()
  completed_step = step + 1
  jax_device_state["state"] = state
  lifecycle = python_vars.get("lifecycle")
  if lifecycle is not None:
    metric_logger_instance.record_train_metrics(metrics, step, step_time_delta.total_seconds())
    lifecycle._train_state = state
    lifecycle._end_step(completed_step, metrics["scalar"]["learning/loss"])
    if lifecycle.control.should_log:
      lifecycle.log({"loss": float(metrics["scalar"]["learning/loss"]),
                     **{name: float(value) for name, value in metrics["scalar"].items()}})
    ran_eval = lifecycle.control.should_evaluate
    if ran_eval:
      lifecycle.evaluate()
    receipt = checkpointing.maybe_save_checkpoint(
        checkpoint_manager, state, config, data_iterator, step, training_graph=immutable_data["training_graph"],
        **lifecycle._save_action(completed_step), return_result=True,
        force=lifecycle.control.should_save,
    )
    lifecycle._resolve_save(receipt, checkpoint_manager)
  else:
    checkpointing.maybe_save_checkpoint(
        checkpoint_manager, state, config, data_iterator, step, training_graph=immutable_data["training_graph"],
    )
  snapshotting.maybe_save_snapshot(snapshotter, config, state, step, checkpoint_manager, data_iterator)
  if dump_hlo and step == (dump_step if dump_step >= 0 else start_step):
    jax.block_until_ready(state)
    gcs.upload_dump(
        dump_hlo_local_dir, dump_hlo_gcs_dir, module_name=dump_hlo_module_name,
        delete_local_after=dump_hlo_delete_local_after, all_host_upload=dump_hlo_upload_all,
    )
  if lifecycle is None:
    ran_eval = (
        eval_interval > 0 and step >= start_step and step >= eval_start_step
        and (step - eval_start_step) % eval_interval == 0
    )
  if ran_eval and lifecycle is None:
    run_evaluation(
        config, mesh, state, step, completed_step, start_step, eval_steps, step_rng_args, logical_axis_rules_for_eval,
        p_eval_step, p_eval_step_dropless, eval_data_iterator, metric_logger_instance, python_vars,
    )
  prof.maybe_deactivate_profiler(step, state)
  if step == start_step:
    pytree.print_mem_stats("After params initialized")
  metric_logger_instance.buffer_and_write_metrics(metrics, step, step_time_delta, record_train=lifecycle is None)
  if not ran_eval:
    mllog_utils.step_end(config, completed_step)
  if config.elastic_enabled and completed_step < immutable_data["steps"]:
    snapshotting.maybe_back_up_and_scale_up(snapshotter, config, state, step, checkpoint_manager, data_iterator)
  jax_device_state["state"] = state
  python_vars["last_step_completion"] = last_step_completion
  return metrics
