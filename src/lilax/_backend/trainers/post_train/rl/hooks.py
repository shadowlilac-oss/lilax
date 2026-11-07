from typing import Any, Callable, Optional

from tunix.sft import hooks as tunix_hooks

from lilax._backend.common.goodput import GoodputEvent, record_goodput
from lilax._backend.observability import max_logging


class RLTrainingHooks(tunix_hooks.TrainingHooks):
  def __init__(
      self, rl_cluster: Any, trainer_config: Any, test_dataset: Any, eval_interval: int,
      reward_fns: Optional[list[Callable[..., Any]]] = None, goodput_recorder: Any = None, *, evaluator: Callable,
  ):
    self._rl_cluster = rl_cluster
    self._trainer_config = trainer_config
    self._test_dataset = test_dataset
    self._eval_interval = eval_interval
    self._reward_fns = reward_fns
    self._goodput_recorder = goodput_recorder
    self._evaluator = evaluator
    self._last_step_evaluated = -1

  def on_train_start(self, train_ctx):
    del train_ctx

  def on_train_end(self, train_ctx):
    del train_ctx

  def on_train_step_start(self, train_ctx):
    if getattr(self._trainer_config, "enable_goodput_recording", False):
      record_goodput(self._goodput_recorder, f"record_{GoodputEvent.STEP.value}_start_time", train_ctx.train_steps)

  def on_eval_step_start(self, train_ctx):
    del train_ctx

  def on_eval_step_end(self, train_ctx, *args, **kwargs):
    del train_ctx, args, kwargs

  def on_train_step_end(self, trainer, step, loss):
    del trainer, loss
    if self._eval_interval <= 0 or getattr(self._trainer_config, "num_test_batches", 0) <= 0:
      return
    try:
      outer_step = int(self._rl_cluster.global_steps)
    except Exception:
      outer_step = int(step) if step is not None else -1
    if outer_step <= 0 or outer_step == self._last_step_evaluated:
      return
    if outer_step % self._eval_interval != 0:
      return
    self._last_step_evaluated = outer_step
    try:
      config = self._trainer_config
      (corr, total, accuracy, partial_accuracy, format_accuracy, mean_reward), _ = self._evaluator(
          config, self._test_dataset, rl_cluster=self._rl_cluster, num_passes=config.num_eval_passes,
          corr_lst=config.eval_corr_lst, make_lst=config.eval_make_lst, reward_fns=self._reward_fns,
      )
      max_logging.warning(
          f"Intermediate Eval (step={outer_step}): {corr=}, {total=},"
          f" {accuracy=}%, {partial_accuracy=}%, {format_accuracy=}%, {mean_reward=:.4f}"
      )
    except Exception as error:
      max_logging.warning(f"[intermediate-eval] step={outer_step} failed: {error!r}")
