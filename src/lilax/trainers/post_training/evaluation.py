"""Evaluation for post-training."""

from flax.core.spmd import logical_axis_rules
import jax
from tunix.sft import metrics_logger

from lilax._backend.utils import maxtext_utils


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
