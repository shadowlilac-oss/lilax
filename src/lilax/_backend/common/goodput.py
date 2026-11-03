"""Utilities for monitoring and recording job goodput."""

import contextlib
from enum import Enum

import jax
from ml_goodput_measurement import goodput, goodput_elastic, monitoring, monitoring_elastic

from lilax._backend.utils import elastic_utils, max_logging


def get_goodput_job_name(config) -> str:
  return config.goodput_job_name or config.run_name


class GoodputEvent(Enum):
  JOB = "job"
  TPU_INIT = "tpu_init"
  TRAINING_PREPARATION = "training_preparation"
  DATA_LOADING = "data_loading"
  STEP = "step"


RECORD_JOB_START_TIME = f"record_{GoodputEvent.JOB.value}_start_time"
RECORD_JOB_END_TIME = f"record_{GoodputEvent.JOB.value}_end_time"


def _construct_goodput_monitor(config, common_kwargs):
  job_name = common_kwargs["job_name"]
  if config.elastic_enabled:
    try:
      if elastic_utils.should_use_elastic(config):
        monitor = monitoring_elastic.ElasticGoodputMonitor(include_slice_efficiency=True, **common_kwargs)
        max_logging.log(f"Goodput: using ElasticGoodputMonitor for job: {job_name}")
        return monitor
    except Exception as e:
      max_logging.log(f"Goodput: could not create elastic goodput monitor, falling back to base monitor: {e}")
  monitor = monitoring.GoodputMonitor(pathway_enabled=config.enable_pathways_goodput, **common_kwargs)
  max_logging.log(f"Goodput: using GoodputMonitor for job: {job_name}")
  return monitor


@contextlib.contextmanager
def maybe_monitor_goodput(config):
  if not config.monitor_goodput or jax.process_index() != 0:
    yield
    return
  goodput_monitor = None
  try:
    if config.report_performance_metric_for_gcp_monitoring:
      config.enable_gcp_step_deviation_metrics = False
    gcp_options = monitoring.GCPOptions(
        enable_gcp_goodput_metrics=config.enable_gcp_goodput_metrics,
        enable_gcp_step_deviation_metrics=config.enable_gcp_step_deviation_metrics,
    )
    job_name = get_goodput_job_name(config)
    common_kwargs = {
        "job_name": job_name,
        "logger_name": f"goodput_{job_name}",
        "tensorboard_dir": config.tensorboard_dir,
        "upload_interval": config.goodput_upload_interval_seconds,
        "monitoring_enabled": True,
        "include_badput_breakdown": True,
        "include_step_deviation": config.monitor_step_time_deviation,
        "step_deviation_interval_seconds": config.step_deviation_interval_seconds,
        "gcp_options": gcp_options,
    }
    goodput_monitor = _construct_goodput_monitor(config, common_kwargs)
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Cumulative Goodput upload to Tensorboard & GCM in the background!")
    with maybe_monitor_rolling_window_goodput(goodput_monitor, config):
      yield
  finally:
    if goodput_monitor:
      goodput_monitor.stop_goodput_uploader()
      max_logging.log("Flushed final metrics and safe exited from cumulative Goodput monitoring.")


@contextlib.contextmanager
def maybe_monitor_rolling_window_goodput(goodput_monitor, config):
  if not goodput_monitor or not config.enable_rolling_window_goodput:
    yield
    return
  started = False
  try:
    goodput_monitor.start_rolling_window_goodput_uploader(config.rolling_windows_seconds)
    started = True
    max_logging.log("Started Rolling Window Goodput upload to Tensorboard & GCM in the background!")
  except Exception as e:
    max_logging.log(f"Goodput: failed to start rolling window goodput uploader, continuing without it: {e}")
  try:
    yield
  finally:
    if started:
      try:
        goodput_monitor.stop_rolling_window_goodput_uploader()
        max_logging.log("Flushed final metrics and safe exited from rolling window Goodput monitoring.")
      except Exception as e:
        max_logging.log(f"Goodput: failed to stop rolling window goodput uploader cleanly: {e}")


@contextlib.contextmanager
def maybe_record_goodput(recorder, event_name, *args):
  record_goodput(recorder, f"record_{event_name.value}_start_time", *args)
  completed = False
  try:
    yield
    completed = True
  finally:
    if completed:
      record_goodput(recorder, f"record_{event_name.value}_end_time", *args)


def record_goodput(recorder, event_name, *args):
  if recorder:
    record_func = getattr(recorder, event_name, None)
    if record_func:
      record_func(*args)


def create_goodput_recorder(config):
  if not config.enable_goodput_recording:
    return None
  job_name = get_goodput_job_name(config)
  logger_name = f"goodput_{job_name}"
  if config.elastic_enabled:
    try:
      if elastic_utils.should_use_elastic(config):
        recorder = goodput_elastic.ElasticGoodputRecorder(job_name, logger_name, jax.process_index() == 0)
        elastic_utils.record_slice_state(recorder)
        max_logging.log(f"Goodput: created ElasticGoodputRecorder for job: {job_name}")
        return recorder
    except Exception as e:
      max_logging.log(f"Goodput: could not create elastic goodput recorder, falling back to base recorder: {e}")
  recorder = goodput.GoodputRecorder(job_name, logger_name, jax.process_index() == 0)
  max_logging.log(f"Goodput: created base GoodputRecorder for job: {job_name}")
  return recorder
