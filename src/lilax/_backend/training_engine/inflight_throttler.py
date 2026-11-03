"""Inflight computation throttler."""

import queue
from typing import Any

import jax
from lilax._backend.configs import pyconfig
from lilax._backend.training_engine import abstract_engine
from lilax._backend.training_engine import metrics as metrics_module


class InflightThrottler:
  """Rate limits the number of inflight computations on TPU."""

  def __init__(self, config: pyconfig.HyperParameters, metrics_logger: metrics_module.MetricsLogger) -> None:
    self._inflight_queue = queue.Queue[Any](maxsize=config.max_inflight_computations)
    self._metrics_logger = metrics_logger
    self._pending_metrics: abstract_engine.MetricsBuffer | None = None

  def add_computation(self, computation: Any, metrics: abstract_engine.MetricsBuffer | None) -> None:
    self._inflight_queue.put((jax.tree.leaves(computation), metrics))
    self._flush_pending_metrics()

  def _flush_pending_metrics(self) -> None:
    if self._pending_metrics is None:
      return
    metrics, self._pending_metrics = self._pending_metrics, None
    self._metrics_logger.write_metrics(metrics)

  def wait_for_next(self) -> None:
    if self._inflight_queue.full():
      computation, metrics = self._inflight_queue.get()
      jax.block_until_ready(computation)
      if metrics is not None:
        self._flush_pending_metrics()
        self._pending_metrics = metrics

  def wait_for_all(self) -> None:
    while not self._inflight_queue.empty():
      computation, metrics = self._inflight_queue.get()
      jax.block_until_ready(computation)
      if metrics is not None:
        self._flush_pending_metrics()
        self._pending_metrics = metrics
    self._flush_pending_metrics()

  def cleanup(self) -> None:
    self.wait_for_all()
