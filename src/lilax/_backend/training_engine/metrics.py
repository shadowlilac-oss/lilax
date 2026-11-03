"""Metric buffering and logging for the training engine."""

from __future__ import annotations

from collections.abc import Callable
import enum
import os
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
from lilax._backend.configs import pyconfig
from lilax._backend.training_engine import abstract_engine
from lilax._backend.utils import max_logging
from lilax._backend.utils.diagnostics import config_summary
from lilax._backend.utils.diagnostics import pytree
from lilax._backend.utils.tensorboard_logging import summary_writing
import numpy as np

_METRICS_TO_LOG = ["learning_rate", "loss", "perplexity", "total_weights", "gradient_norm", "step_skipped", "step_time", "tflops"]
_DEFAULT_MAX_BUFFERED_STEPS = 128
_ALLOWED_SCALAR_ARRAY_TYPES = (jax.Array, np.ndarray, float, int, bool, np.number, np.bool_)


def _validate_metric(name: str, metric: Any) -> None:
  if isinstance(metric, abstract_engine.WeightedMetric):
    sum_val = metric.unreduced_sum
    denom_val = metric.denominator
    if not isinstance(sum_val, _ALLOWED_SCALAR_ARRAY_TYPES):
      raise TypeError(
          f"WeightedMetric '{name}' unreduced_sum must be a JAX/NumPy array or numeric scalar, "
          f"got {type(sum_val).__name__}"
      )
    if not isinstance(denom_val, _ALLOWED_SCALAR_ARRAY_TYPES):
      raise TypeError(
          f"WeightedMetric '{name}' denominator must be a JAX/NumPy array or numeric scalar, "
          f"got {type(denom_val).__name__}"
      )
    if hasattr(sum_val, "dtype") and not (
        np.issubdtype(sum_val.dtype, np.number) or np.issubdtype(sum_val.dtype, np.bool_)
    ):
      raise TypeError(f"WeightedMetric '{name}' unreduced_sum has non-numeric dtype: {sum_val.dtype}")
    if hasattr(denom_val, "dtype") and not (
        np.issubdtype(denom_val.dtype, np.number) or np.issubdtype(denom_val.dtype, np.bool_)
    ):
      raise TypeError(f"WeightedMetric '{name}' denominator has non-numeric dtype: {denom_val.dtype}")
    if hasattr(sum_val, "shape") and hasattr(denom_val, "shape"):
      if sum_val.ndim > 0 and denom_val.ndim > 0 and sum_val.shape != denom_val.shape:
        try:
          np.broadcast_shapes(sum_val.shape, denom_val.shape)
        except ValueError as error:
          raise ValueError(
              f"WeightedMetric '{name}' unreduced_sum shape {sum_val.shape} and "
              f"denominator shape {denom_val.shape} are incompatible: {error}"
          ) from error
    return
  if not isinstance(metric, _ALLOWED_SCALAR_ARRAY_TYPES):
    raise TypeError(f"Metric '{name}' must be a JAX/NumPy array, numeric scalar, or WeightedMetric; got {type(metric).__name__}")
  if hasattr(metric, "dtype") and not (np.issubdtype(metric.dtype, np.number) or np.issubdtype(metric.dtype, np.bool_)):
    raise TypeError(f"Metric '{name}' has non-numeric dtype: {metric.dtype}")


def _concat_1d(items: list[Any]) -> jax.Array:
  if len(items) == 1:
    return jnp.atleast_1d(jnp.asarray(items[0]))
  if all(jnp.ndim(item) == 0 for item in items):
    return jnp.stack(items)
  return jnp.concatenate([jnp.ravel(jnp.asarray(item)) for item in items])


class Mode(str, enum.Enum):
  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    return self.value


class MetricsRecorder:
  """Buffers metrics by step and materializes them on device."""

  def __init__(self, max_buffered_steps: int = _DEFAULT_MAX_BUFFERED_STEPS, mode: Mode = Mode.TRAIN) -> None:
    self._metrics_buffer: list[abstract_engine.MetricsBuffer] = []
    self._pending_weighted: dict[int, dict[str, list[abstract_engine.WeightedMetric]]] = {}
    self._pending_scalar: dict[int, dict[str, list[Any]]] = {}
    self._mode = mode
    self._max_buffered_steps = max_buffered_steps
    self._dropped_buffer_count = 0

  def buffer_metrics(
      self, train_step: int, name: str, metric: abstract_engine.WeightedMetric | jax.Array | float | int | None = None,
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
  ) -> None:
    if metric is None:
      return
    _validate_metric(name, metric)
    if not self._metrics_buffer or self._metrics_buffer[-1].id != train_step:
      self._metrics_buffer.append(abstract_engine.MetricsBuffer(id=train_step, mode=self._mode))
      self._evict_old_buffers()
    self._record_metric(name, metric, aggregation_fn=aggregation_fn)

  def _evict_old_buffers(self) -> None:
    if self._max_buffered_steps <= 0 or len(self._metrics_buffer) <= self._max_buffered_steps:
      return
    num_dropped = len(self._metrics_buffer) - self._max_buffered_steps
    oldest_dropped_id = self._metrics_buffer[0].id
    for evicted in self._metrics_buffer[:num_dropped]:
      self._pending_weighted.pop(id(evicted), None)
      self._pending_scalar.pop(id(evicted), None)
    del self._metrics_buffer[:num_dropped]
    self._dropped_buffer_count += num_dropped
    logging.log_every_n(
        logging.WARNING,
        "Metrics history is full at %d step(s); evicting buffers from step %s onwards "
        "(%d dropped so far). Drain it with MetricsRecorder.get_metrics_history() or the "
        "engine's get_metrics() if you need every step.",
        self._max_buffered_steps, self._max_buffered_steps, oldest_dropped_id, self._dropped_buffer_count,
    )

  def _record_metric(
      self, name: str, metric: abstract_engine.WeightedMetric | jax.Array | float | int,
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
  ) -> None:
    buffer = self._metrics_buffer[-1]
    key = id(buffer)
    if aggregation_fn is not None:
      buffer.aggregation_fns[name] = aggregation_fn
    if isinstance(metric, abstract_engine.WeightedMetric):
      if name in self._pending_scalar.get(key, {}) or name in buffer.scalar_metrics:
        raise TypeError(f"Metric '{name}' was previously recorded as a scalar metric, cannot record as WeightedMetric.")
      self._pending_weighted.setdefault(key, {}).setdefault(name, []).append(metric)
    else:
      if name in self._pending_weighted.get(key, {}) or name in buffer.weighted_metrics:
        raise TypeError(f"Metric '{name}' was previously recorded as a WeightedMetric, cannot record as a scalar metric.")
      self._pending_scalar.setdefault(key, {}).setdefault(name, []).append(metric)

  def _flush_buffer(self, buffer: abstract_engine.MetricsBuffer) -> None:
    key = id(buffer)
    pending_w = self._pending_weighted.get(key)
    pending_s = self._pending_scalar.get(key)
    if not pending_w and not pending_s:
      return
    new_weighted: dict[str, abstract_engine.WeightedMetric] = {}
    for name, items in (pending_w or {}).items():
      existing = buffer.weighted_metrics.get(name)
      prefix = [existing] if existing is not None else []
      template = items[0]
      new_weighted[name] = abstract_engine.WeightedMetric(
          unreduced_sum=_concat_1d([metric.unreduced_sum for metric in prefix + items]),
          denominator=_concat_1d([metric.denominator for metric in prefix + items]),
          eps=template.eps, min_denom=template.min_denom,
      )
    new_scalar: dict[str, jax.Array] = {}
    for name, items in (pending_s or {}).items():
      prefix = [buffer.scalar_metrics[name]] if name in buffer.scalar_metrics else []
      new_scalar[name] = _concat_1d(prefix + items)
    buffer.weighted_metrics.update(new_weighted)
    buffer.scalar_metrics.update(new_scalar)
    self._pending_weighted.pop(key, None)
    self._pending_scalar.pop(key, None)

  def get_metrics_history(self, clear_cache: bool = True) -> list[abstract_engine.MetricsBuffer]:
    for buffer in self._metrics_buffer:
      self._flush_buffer(buffer)
    metrics_to_return = self._metrics_buffer
    if clear_cache:
      self._metrics_buffer = []
      self._pending_weighted.clear()
      self._pending_scalar.clear()
    return metrics_to_return

  def get_step_metrics(self, step: int) -> abstract_engine.MetricsBuffer | None:
    if not self._metrics_buffer:
      return None
    latest = self._metrics_buffer[-1]
    if latest.id == step:
      self._flush_buffer(latest)
      return latest
    return None

  def cleanup(self) -> None:
    self._metrics_buffer = []
    self._pending_weighted.clear()
    self._pending_scalar.clear()


class MetricsLogger:
  """Processes and logs completed metrics."""

  def __init__(self, config: pyconfig.HyperParameters) -> None:
    self._tb_writer = None
    if config.enable_tensorboard:
      self._tb_writer = summary_writing.initialize_summary_writer(config.tensorboard_dir, config.run_name, config.enable_tensorboard)
    self._config = config

  def write_setup_info_to_tensorboard(self, params: Any) -> None:
    num_model_parameters = pytree.calculate_num_params_from_pytree(params)
    max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
    if self._tb_writer is None:
      return
    summary_writing.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self._tb_writer)
    summary_writing.add_text_to_summary_writer("libtpu_init_args", os.getenv("LIBTPU_INIT_ARGS", ""), self._tb_writer)
    config_summary.add_config_to_summary_writer(self._config, self._tb_writer)

  def _log_metrics(self, step: int, metrics: dict[str, Any], mode: Mode) -> None:
    if mode == Mode.TRAIN:
      log_message = [f"Train step: {step}"]
      for name in ["loss", "perplexity"]:
        if name in metrics:
          log_message.append(f"{name}: {metrics[name]:.3f}")
      if len(log_message) > 1:
        logging.info(", ".join(log_message))
    elif mode == Mode.EVAL:
      loss = metrics.get("loss")
      if loss is not None:
        logging.info("Eval step %d evaluation loss: %f", step, loss)

  def write_metrics(self, metrics: abstract_engine.MetricsBuffer, mode: Mode = Mode.TRAIN) -> None:
    processed_metrics = self.process_metrics(metrics)
    self._log_metrics(step=metrics.id, metrics=processed_metrics, mode=mode)
    self._write_metrics_to_tensorboard(step=metrics.id, metrics=processed_metrics, mode=mode)

  def _write_metrics_to_tensorboard(self, step: int, metrics: dict[str, Any], mode: Mode = Mode.TRAIN) -> None:
    if self._tb_writer is None:
      return
    if jax.process_index() == 0:
      for metric_name, value in metrics.items():
        self._tb_writer.add_scalar(f"{mode}/{metric_name}", value, step)
    if step % self._config.log_period == 0:
      logging.info("Flushing TensorBoard writer at step %d to %s", step, self._config.tensorboard_dir)
      self._tb_writer.flush()

  def process_metrics(self, metrics: abstract_engine.MetricsBuffer) -> dict[str, Any]:
    processed: dict[str, Any] = {}
    for name, weighted_metric in metrics.weighted_metrics.items():
      if name not in _METRICS_TO_LOG:
        continue
      host_val = np.asarray(weighted_metric.compute())
      if name in metrics.aggregation_fns:
        host_val = metrics.aggregation_fns[name](host_val)
      elif host_val.size > 1:
        host_val = np.mean(host_val)
      if isinstance(host_val, (np.ndarray, jax.Array)):
        host_val = host_val.item() if host_val.size == 1 else float(np.mean(host_val))
      processed[name] = host_val
    for name, scalar_val in metrics.scalar_metrics.items():
      if name not in _METRICS_TO_LOG:
        continue
      host_val = np.asarray(scalar_val)
      if name in metrics.aggregation_fns:
        host_val = metrics.aggregation_fns[name](host_val)
      elif host_val.size > 1:
        host_val = np.mean(host_val)
      if isinstance(host_val, (np.ndarray, jax.Array)):
        host_val = host_val.item() if host_val.size == 1 else float(np.mean(host_val))
      processed[name] = host_val
    if "loss" in processed:
      processed["perplexity"] = float(np.exp(np.asarray(processed["loss"], dtype=np.float64)))
    return processed

  def cleanup(self) -> None:
    if self._tb_writer is not None:
      summary_writing.close_summary_writer(self._tb_writer)
