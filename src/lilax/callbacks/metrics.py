"""Scalar metric persistence through Trainer events."""

from types import SimpleNamespace

from transformers import TrainerCallback

from lilax._backend.common.cloud import is_decoupled
from lilax._backend.observability.metric_storage import ScalarMetricStorage
from lilax._backend.storage.gcs import write_config_raw_keys_for_gcs


class MetricsCallback(TrainerCallback):
  def __init__(self, *, metrics_file=None, gcs_metrics=False, save_config_to_gcs=False):
    self.metrics_file, self.gcs_metrics = metrics_file, gcs_metrics
    self.save_config_to_gcs = save_config_to_gcs
    self._config = None
    self._storage = None

  def on_init_end(self, args, state, control, **kwargs):
    self._bind(self._config)

  def _bind(self, config):
    self._config = config
    self._storage = ScalarMetricStorage(SimpleNamespace(
        metrics_file=self.metrics_file, gcs_metrics=self.gcs_metrics, metrics_dir=config.metrics_dir,
        run_name=config.run_name, log_period=config.log_period, steps=config.steps,
    ))

  def on_train_begin(self, args, state, control, **kwargs):
    if self.save_config_to_gcs:
      write_config_raw_keys_for_gcs(dict(self._config.get_keys()) | {"save_config_to_gcs": True})

  def on_log(self, args, state, control, logs=None, **kwargs):
    self._write(state, logs, "train")

  def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    self._write(state, metrics, "eval")

  def _write(self, state, values, metric_type):
    if not state.is_world_process_zero:
      return
    metrics = {"scalar": {name: value for name, value in (values or {}).items() if isinstance(value, (int, float))}}
    if self.metrics_file:
      self._storage.write_locally(metrics, state.global_step)
    if self.gcs_metrics and not is_decoupled():
      self._storage.write_for_gcs(metrics, state.global_step, metric_type)
