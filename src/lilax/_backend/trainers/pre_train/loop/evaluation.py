"""In-loop evaluation pass run after a native pretraining step."""

import datetime
import time

from flax.core.spmd import logical_axis_rules
import jax

from lilax._backend.common.telemetry import mllog_utils
from lilax._backend.trainers.diloco import diloco_sharding
from lilax._backend.observability import max_logging
from lilax._backend.distributed import sharding


def run_evaluation(
    config, mesh, state, step, completed_step, start_step, eval_steps, step_rng_args, logical_axis_rules_for_eval,
    p_eval_step, p_eval_step_dropless, eval_data_iterator, metric_logger_instance, python_vars,
):
  assert eval_data_iterator
  prefill_thread = python_vars.get("eval_prefill_thread")
  if prefill_thread is not None and prefill_thread.is_alive():
    wait_t = time.perf_counter()
    prefill_thread.join(timeout=600)
    max_logging.log(f"eval cache prefill: waited {time.perf_counter() - wait_t:.3f} s before eval")
  if hasattr(eval_data_iterator, "reset"):
    eval_data_iterator.reset()
  metric_logger_instance.reset_eval_metrics()
  mllog_utils.eval_start(config, completed_step, start_step=start_step)
  max_logging.log(f"Starting eval after train step {step}")
  eval_step_count = 0
  last_eval_step_completion = datetime.datetime.now()
  metric_logger_instance.mark_eval_loop_start()
  for eval_batch in eval_data_iterator:
    if config.enable_diloco:
      eval_batch = diloco_sharding.reshape_first_axis_with_diloco(config.num_diloco_replicas, eval_batch)
    eval_batch = jax.device_put(
        eval_batch, sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
    )
    with jax.set_mesh(mesh), logical_axis_rules(logical_axis_rules_for_eval):
      eval_metrics = p_eval_step(state, eval_batch, *step_rng_args)
      if config.moe_dropless_fallback == "step" and p_eval_step_dropless is not None and bool(eval_metrics.get("has_moe_overflow")):
        max_logging.log(
            f"Eval step {eval_step_count}: MoE ragged buffer overflow detected! "
            "Replaying eval step with dropless buffer..."
        )
        eval_metrics = p_eval_step_dropless(state, eval_batch, *step_rng_args)
    eval_step_time_delta = datetime.datetime.now() - last_eval_step_completion
    last_eval_step_completion = datetime.datetime.now()
    metric_logger_instance.buffer_and_write_metrics(
        eval_metrics, eval_step_count, step_time_delta=eval_step_time_delta, is_training=False
    )
    eval_step_count += 1
    if 0 < eval_steps <= eval_step_count:
      break
