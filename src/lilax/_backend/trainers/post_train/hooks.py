from collections import defaultdict
import abc
import time
from typing import override

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.sft import peft_trainer
from tunix.sft.hooks import DataHooks, TrainingHooks

from lilax._backend.input_pipeline.input_pipeline_interface import create_data_iterator
from lilax._backend.common.data_loader import DataLoader
from lilax._backend.common.goodput import GoodputEvent, record_goodput
from lilax._backend.common.metric_logger import MetricLogger, MetadataKey
from lilax._backend.utils import exceptions, gcs_utils, max_logging, max_utils, sharding


class GoodputTrainingHooks(TrainingHooks):
  def __init__(self, config, goodput_recorder):
    self.config = config
    self.goodput_recorder = goodput_recorder

  @override
  def on_train_start(self, train_ctx: peft_trainer.PeftTrainer):
    del train_ctx

  @override
  def on_train_end(self, train_ctx: peft_trainer.PeftTrainer):
    del train_ctx

  @override
  def on_train_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    if self.config.enable_goodput_recording:
      record_goodput(self.goodput_recorder, f"record_{GoodputEvent.STEP.value}_start_time", train_ctx.train_steps)

  @override
  def on_train_step_end(self, train_ctx: peft_trainer.PeftTrainer, train_step: int, train_loss: float, step_time: float = 0.0):
    del train_ctx, train_step, train_loss, step_time

  @override
  def on_eval_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    del train_ctx

  @override
  def on_eval_step_end(self, train_ctx: peft_trainer.PeftTrainer, eval_loss: float):
    del train_ctx, eval_loss


class BaseTrainingHooks(GoodputTrainingHooks, abc.ABC):
  def __init__(self, config, mesh, learning_rate_schedule, goodput_recorder):
    super().__init__(config, goodput_recorder)
    self.mesh = mesh
    self.metric_logger = MetricLogger(self.config, learning_rate_schedule)
    self.metadata = {}
    self.train_metadata = defaultdict(float)
    self.eval_metadata = defaultdict(float)
    self.step_start_time = 0.0

  @override
  def on_train_start(self, train_ctx: peft_trainer.PeftTrainer):
    state = nnx.state(train_ctx.model)
    params = state.filter(nnx.Param)
    if not self.config.using_pipeline_parallelism:
      sharding.assert_params_sufficiently_sharded(params, self.mesh, self.config.sharding_tolerance)
    self.metric_logger.write_setup_info_to_tensorboard(params)
    if MetadataKey.PER_DEVICE_TFLOPS in self.metric_logger.metadata:
      train_ctx._flops_measured = True
    if self.config.dump_hlo:
      jax.block_until_ready(state)
      gcs_utils.upload_dump(
          self.config.dump_hlo_local_dir, self.config.dump_hlo_gcs_dir,
          module_name=self.config.dump_hlo_module_name, delete_local_after=self.config.dump_hlo_delete_local_after,
          all_host_upload=self.config.dump_hlo_upload_all,
      )
    self.metadata["first_train_step"] = train_ctx.train_steps
    self.step_start_time = time.perf_counter()

  @override
  def on_train_end(self, train_ctx: peft_trainer.PeftTrainer):
    assert "first_train_step" in self.metadata, (
        "BaseTrainingHooks.on_train_start() must be called before BaseTrainingHooks.on_train_end()"
    )
    try:
      train_ctx.data_hooks.finish_capture(train_ctx)
    finally:
      if self.metric_logger:
        self.metric_logger.flush_metrics_and_cleanup()

  @override
  def on_train_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    super().on_train_step_start(train_ctx)
    self.train_metadata[train_ctx.train_steps] = {"total_weights": self.get_total_weights(train_ctx.data_hooks.train_batch)}

  @override
  def on_train_step_end(self, train_ctx: peft_trainer.PeftTrainer, train_step: int, train_loss: float, step_time: float = 0.0):
    assert train_step - 1 in self.train_metadata, (
        "BaseTrainingHooks.on_train_step_start() must be called before BaseTrainingHooks.on_train_step_end()"
    )
    if self.metadata["first_train_step"] == train_step - 1:
      max_utils.print_mem_stats("After params initialized")
    current_time = time.perf_counter()
    step_time = current_time - self.step_start_time
    self.step_start_time = current_time
    metrics = {"scalar": {
        "learning/loss": train_loss,
        "learning/total_weights": self.train_metadata[train_step - 1]["total_weights"],
    }}
    for mt_key, tunix_key in [
        ("learning/perplexity", "perplexity"), ("learning/lm_loss", "sft_loss"),
        ("learning/dpo_loss", "or_loss"), ("learning/grad_norm", "grad_norm"),
        ("learning/reward_accuracy", "rewards/accuracy"), ("learning/reward_margin", "rewards/margin"),
    ]:
      try:
        val = train_ctx.metrics_logger.get_metric(train_ctx.metrics_prefix, tunix_key, "train")
        metrics["scalar"][mt_key] = float(val)
      except:
        pass
    self.metric_logger.record_train_metrics(metrics, train_step, step_time)
    self.metric_logger.write_metrics(metrics, train_step)
    del self.train_metadata[train_step - 1]

  @override
  def on_eval_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    self.eval_metadata["eval_step_count"] += 1.0
    self.eval_metadata["total_weights"] += self.get_total_weights(train_ctx.data_hooks.eval_batch)

  @override
  def on_eval_step_end(self, train_ctx: peft_trainer.PeftTrainer, eval_loss: float):
    assert self.eval_metadata["eval_step_count"] != 0, (
        "BaseTrainingHooks.on_eval_step_start() must be called before BaseTrainingHooks.on_eval_step_end()"
    )
    avg_loss = eval_loss if self.eval_loss_is_preaveraged() else eval_loss / self.eval_metadata["eval_step_count"]
    metrics = {"scalar": {
        "eval/total_loss": eval_loss, "eval/avg_loss": avg_loss, "eval/total_weights": self.eval_metadata["total_weights"],
    }}
    for mt_key, tunix_key in [
        ("eval/avg_perplexity", "perplexity"), ("eval/avg_sft_loss", "sft_loss"),
        ("eval/avg_or_loss", "or_loss"), ("eval/avg_dpo_reward_accuracy", "rewards/accuracy"),
        ("eval/avg_dpo_reward_margin", "rewards/margin"),
    ]:
      try:
        val = train_ctx.metrics_logger.get_metric(train_ctx.metrics_prefix, tunix_key, "eval")
        metrics["scalar"][mt_key] = float(val)
      except:
        pass
    if "eval/avg_perplexity" not in metrics["scalar"]:
      metrics["scalar"]["eval/avg_perplexity"] = float(jnp.exp(avg_loss))
    self.metric_logger.write_metrics(metrics, train_ctx.train_steps, metric_type="eval")
    self.eval_metadata.clear()
    if avg_loss <= self.config.target_eval_loss:
      raise exceptions.StopTraining(f"Target loss {self.config.target_eval_loss=} is achieved.")

  def eval_loss_is_preaveraged(self) -> bool:
    return False

  @abc.abstractmethod
  def get_total_weights(self, batch) -> jax.Array:
    pass


class BaseDataHooks(DataHooks):
  def __init__(self, config, mesh, goodput_recorder, *, train_iterator=None, eval_iterator=None):
    self.config = config
    self.mesh = mesh
    self.goodput_recorder = goodput_recorder
    if train_iterator is None or (eval_iterator is None and config.eval_interval > 0):
      default_train, default_eval = create_data_iterator(config, mesh)
      if train_iterator is None:
        train_iterator = default_train
      if eval_iterator is None:
        eval_iterator = default_eval
    self.train_data_iterator, self.eval_data_iterator = train_iterator, eval_iterator
    self.train_data_loader = DataLoader(config, mesh, self.train_data_iterator, goodput_recorder=goodput_recorder)
    self.train_batch = None
    self.eval_batch = None
    self.capture = None

  @override
  def load_next_train_batch(self, train_ctx: peft_trainer.PeftTrainer):
    if self.capture is not None:
      self.capture.on_fetch(train_ctx.iter_steps, train_ctx.training_hooks.metric_logger.reporter)
    try:
      self.train_batch = self.train_data_loader.load_next_batch()
    except exceptions.StopTraining as e:
      if isinstance(e.__cause__ or e.__context__, StopIteration):
        max_logging.log(f"Training data exhausted: {str(e)}")
        self.train_batch = None
      else:
        raise
    return self.train_batch

  def finish_capture(self, train_ctx):
    if self.capture is not None:
      self.capture.finish(train_ctx.iter_steps, train_ctx.training_hooks.metric_logger.reporter)

  @override
  def load_next_eval_batch(self, train_ctx: peft_trainer.PeftTrainer):
    try:
      if self.config.eval_steps > 0 and train_ctx.training_hooks.eval_metadata["eval_step_count"] >= self.config.eval_steps:
        self.eval_batch = None
      else:
        self.eval_batch = next(self.eval_data_iterator)
    except StopIteration as e:
      max_logging.log(f"Evaluation data exhausted: {str(e)}")
      self.eval_batch = None
    return self.eval_batch
