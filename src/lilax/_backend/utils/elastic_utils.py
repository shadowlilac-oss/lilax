"""Utility functions for elastic training."""

from collections import Counter
import dataclasses
import functools
from types import SimpleNamespace
from typing import Any, Callable

import jax
import pathwaysutils
from pathwaysutils.elastic import elastic
from pathwaysutils.elastic import manager

from lilax._backend.input_pipeline.multihost_dataloading import cleanup_all_iterators
from lilax._backend.storage import gcs as gcs_utils
from lilax._backend.observability import max_logging

elastic_manager: manager.Manager | None = None
pending_reinit_recorder = None
pending_elastic_event_type = None


def record_slice_state(recorder, active_slices_override: int | None = None) -> None:
  """Record live slice counts with the goodput recorder."""
  if (
      recorder is None
      or not hasattr(recorder, "record_elastic_slice_counts")
      or not pathwaysutils.is_pathways_backend_used()
      or elastic_manager is None
  ):
    return

  available_slices = len(elastic.get_active_slice_indices())
  active_slices = (
      active_slices_override if active_slices_override is not None else len(elastic_manager.active_slice_indices)
  )
  total_slices = len(elastic.get_slice_to_devices(jax.devices()))
  recorder.record_elastic_slice_counts(
      available_slices=available_slices,
      active_slices=active_slices,
      total_slices=total_slices,
  )


def record_elastic_event_start(recorder, scale_up: bool) -> None:
  """Record the start of a scale-up or slice-down event."""
  global pending_elastic_event_type
  if scale_up:
    event_type = "elastic_scale_up"
  else:
    event_type = "elastic_slice_down"
  pending_elastic_event_type = event_type
  if recorder and hasattr(recorder, "record_elastic_wait_start_time"):
    recorder.record_elastic_wait_start_time(event_type=event_type)
    record_slice_state(recorder, active_slices_override=0)


def record_elastic_wait_end_and_reinit_start(recorder) -> None:
  """Record the end of the elastic wait and start of reinitialization."""
  global pending_reinit_recorder, pending_elastic_event_type
  if pending_elastic_event_type is None:
    return
  event_type = pending_elastic_event_type
  pending_elastic_event_type = None
  if recorder and hasattr(recorder, "record_elastic_wait_end_time"):
    recorder.record_elastic_wait_end_time(event_type=event_type)
    recorder.record_elastic_reinit_start_time()
    record_slice_state(recorder)
  pending_reinit_recorder = recorder


def record_elastic_reinit_end() -> None:
  """Record the end of elastic reinitialization."""
  global pending_reinit_recorder
  if pending_reinit_recorder is not None and hasattr(pending_reinit_recorder, "record_elastic_reinit_end_time"):
    pending_reinit_recorder.record_elastic_reinit_end_time()
    record_slice_state(pending_reinit_recorder)
  pending_reinit_recorder = None


def elastic_enabled(config) -> bool:
  """Return whether elastic mode is enabled."""
  return pathwaysutils.is_pathways_backend_used() and config.elastic_enabled


def elastic_snapshot(config) -> bool:
  """Return whether elastic snapshot mode is enabled."""
  return elastic_enabled(config) and config.elastic_backup_kind == "snapshot"


def maybe_bubble_elastic_exception(config, e: Exception) -> None:
  """Re-raise JAX and scale-up errors when elasticity is enabled."""
  if elastic_enabled(config) and isinstance(e, (jax.errors.JaxRuntimeError, manager.ScaleUpSignalError)):
    raise e


def should_use_elastic(config) -> bool:
  """Return whether elastic training should be used."""
  return config is not None and elastic_enabled(config)


def clean_up_incomplete_checkpoints(checkpoint_dir: str):
  """Delete the latest incomplete GCS checkpoint after an elastic event."""
  if not checkpoint_dir.startswith("gs://"):
    max_logging.log(f"Elastic utils: Not checking {checkpoint_dir} for incomplete checkpoints: it is not on GCS.")
    return
  max_logging.log("Elastic utils: Checking for incomplete checkpoint after an elastic event...")
  checkpoint_dir = gcs_utils.add_trailing_slash(checkpoint_dir)
  checkpoints = gcs_utils.gcs_list_directories(checkpoint_dir)
  checkpoints = [cp for cp in checkpoints if cp.isdigit()]
  if not checkpoints:
    max_logging.log("Found no existing checkpoints. Continuing")
    return
  checkpoints.sort(key=int)
  latest_checkpoint_name = checkpoints[-1]
  latest_checkpoint_path = f"{checkpoint_dir}{latest_checkpoint_name}/"
  max_logging.log(f"Checking latest checkpoint: {latest_checkpoint_path}")
  success_markers = gcs_utils.gcs_glob_pattern(f"{latest_checkpoint_path}commit_success*")
  if not success_markers:
    max_logging.log(f"No commit_success file found. Deleting {latest_checkpoint_path}...")
    gcs_utils.gcs_delete_directory(latest_checkpoint_path)
  else:
    max_logging.log(f"Found commit_success file. Keeping {latest_checkpoint_path}.")


def ensure_elastic_manager_initialized(config):
  """Initialize the elastic manager when elastic Pathways is enabled."""
  global elastic_manager
  if should_use_elastic(config) and elastic_manager is None:
    elastic_manager = manager.Manager()


def is_pause_resume(config) -> bool:
  """Return whether every attempt waits for all slices in the jobset."""
  ensure_elastic_manager_initialized(config)
  assert elastic_manager is not None
  if config.elastic_min_slice_count == -1:
    return True
  return config.elastic_min_slice_count == elastic_manager.total_slice_count


def get_local_batch_size(config) -> int:
  """Return the local batch size based on the configuration."""
  return config.per_device_batch_size * get_devices_per_host(config)


def live_devices(config=None):
  """Return the list of live devices."""
  if should_use_elastic(config):
    ensure_elastic_manager_initialized(config)
    assert elastic_manager is not None
    return [
        d for d in jax.devices() if d is not None and getattr(d, "slice_index", 0) in elastic_manager.active_slice_indices
    ]
  return jax.devices()


def live_slice_indices(config) -> set[int]:
  """Return the set of live slice indices."""
  return {getattr(d, "slice_index", 0) for d in live_devices(config) if d is not None}


def get_devices_per_host(config):
  """Calculate the number of chips per physical worker VM."""
  devices = Counter(d.task_id for d in live_devices(config))
  max_logging.log(f"elastic_utils: Device counts per task: {devices}")
  if not devices:
    raise ValueError("elastic_utils: get_devices_per_host: No devices found.")
  devices_per_host = next(iter(devices.values()))
  if devices_per_host == 0:
    raise ValueError("elastic_utils: get_devices_per_host: Devices per host is 0.")
  max_logging.log(f"elastic_utils: Devices per host: {devices_per_host}")
  return devices_per_host


def chain_callbacks(*funcs):
  """Chain callbacks in their given order."""
  def wrapper():
    for func in funcs:
      func()

  return wrapper


class RetryCache:
  """Reuse setup objects across attempts on the same slices."""

  def __init__(self):
    self._enabled = False
    self._slices: frozenset[int] | None = None
    self._values: dict[str, Any] = {}
    self.snapshotter: Any = None

  @property
  def enabled(self) -> bool:
    return self._enabled

  def reset(self, enabled: bool) -> None:
    """Empty the cache and turn it on or off."""
    self._enabled = enabled
    self._slices = None
    self._values.clear()

  def clear_if_slices_changed(self, active_slices: frozenset[int]) -> None:
    """Empty cached entries if the active slices have changed."""
    if self._slices != active_slices:
      self._values.clear()
      self._slices = active_slices

  def contains(self, name: str) -> bool:
    """Return whether the enabled cache contains an entry."""
    return self._enabled and name in self._values

  def get(self, name: str) -> Any:
    """Return a cached entry, or None when missing or disabled."""
    if not self._enabled:
      return None
    return self._values.get(name)

  def put(self, name: str, value: Any) -> None:
    """Cache a value when the cache is enabled."""
    if self._enabled:
      self._values[name] = value


def get_or_build(retry_cache: RetryCache | None, name: str, build_fn: Callable[[], Any]) -> Any:
  """Return a cached object or build and cache it."""
  if retry_cache is None:
    return build_fn()
  if retry_cache.contains(name):
    return retry_cache.get(name)
  value = build_fn()
  retry_cache.put(name, value)
  return value


@dataclasses.dataclass
class _AttemptOutcome:
  """Track whether the last attempt ended for scale-up."""

  scale_up: bool = False


def elastic_retry(config, callback_fn=None, pre_callback_fn=None, retry_cache=None):
  """Decorate a function to retry after elastic events."""
  if not elastic_enabled(config):
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled:"
        f" {config.elastic_enabled}, pathways backend used:"
        f" {pathwaysutils.is_pathways_backend_used()}"
    )
    raise ValueError(msg)

  max_logging.log("Elastic Retry Enabled")
  ensure_elastic_manager_initialized(config)
  assert elastic_manager is not None

  def cleanup_iterators_and_checkpoints():
    try:
      cleanup_all_iterators()
    except Exception as e:
      max_logging.log(f"Failed to cleanup iterators during elastic scale up: {e}")
    if config.enable_checkpointing or config.enable_emergency_checkpoint:
      clean_up_incomplete_checkpoints(config.checkpoint_dir)

  outcome = _AttemptOutcome()

  def effective_callback():
    scale_up = outcome.scale_up
    outcome.scale_up = False
    cleanup_iterators_and_checkpoints()
    if callback_fn is not None:
      callback_fn(scale_up=scale_up)

  def effective_pre_callback():
    if retry_cache is not None:
      retry_cache.clear_if_slices_changed(frozenset(elastic_manager.active_slice_indices))
    if pre_callback_fn is not None:
      pre_callback_fn()

  if config.elastic_min_slice_count == -1:
    minimum_slice_count = None
  else:
    minimum_slice_count = config.elastic_min_slice_count
  if minimum_slice_count is not None and minimum_slice_count > elastic_manager.total_slice_count:
    raise ValueError(
        f"elastic_min_slice_count ({minimum_slice_count}) is larger than the number of slices in the jobset"
        f" ({elastic_manager.total_slice_count})."
    )
  if not is_pause_resume(config) and config.dataset_type == "grain" and not config.grain_use_elastic_iterator:
    raise ValueError(
        "Replica resize (elastic_min_slice_count smaller than the number of slices in the jobset) restores the data"
        " iterator onto a different number of hosts, which only grain's ElasticIterator supports. Set"
        " grain_use_elastic_iterator=True."
    )

  pathways_retry = elastic_manager.elastic_retry(
      max_retries=config.elastic_max_retries,
      timeout=config.elastic_timeout_seconds,
      minimum_slice_count=minimum_slice_count,
      pre_callback=effective_pre_callback,
      on_elastic_event_callback=effective_callback,
  )

  def decorator(func):
    @functools.wraps(func)
    def attempt(*args, **kwargs):
      try:
        return func(*args, **kwargs)
      except manager.ScaleUpSignalError:
        outcome.scale_up = True
        raise

    retried_func = pathways_retry(attempt)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      if retry_cache is not None:
        retry_cache.reset(enabled=True)
      try:
        return retried_func(*args, **kwargs)
      finally:
        if retry_cache is not None:
          retry_cache.reset(enabled=False)
        outcome.scale_up = False

    return wrapper

  return decorator


def is_scale_up_event(config) -> bool:
  """Return whether a scale-up event is detected."""
  if elastic_enabled(config):
    ensure_elastic_manager_initialized(config)
    assert elastic_manager is not None
    return bool(elastic_manager.available_inactive_slices)
  return False


is_slice_down_error = elastic.is_error_due_to_slice_down


def maybe_elastic_scale_up(config, checkpoint_manager):
  """Wait for checkpoint completion before interrupting for scale-up."""
  if not should_use_elastic(config):
    max_logging.log("maybe_elastic_scale_up: Elastic training is not enabled.")
    return
  if is_scale_up_event(config):
    max_logging.log("New slices are available. Waiting for any checkpoint save in progress before interrupting.")
    if checkpoint_manager is not None:
      if hasattr(checkpoint_manager, "wait"):
        checkpoint_manager.wait()
      else:
        checkpoint_manager.wait_until_finished()
    max_logging.log("Interrupting the attempt to scale up.")
    try:
      cleanup_all_iterators()
    except Exception as e:
      max_logging.log(f"Error in maybe_elastic_scale_up cleanup: {e}")
    raise manager.ScaleUpSignalError()


def single_controller_mtc_init_kwargs(raw_keys):
  """Return topology kwargs for single-controller MTC initialization."""
  kwargs = {
      "data_parallelism": raw_keys["mtc_data_parallelism"],
      "num_slices": raw_keys["num_slices"],
  }
  if not raw_keys.get("elastic_enabled", False):
    return kwargs

  config = SimpleNamespace(**raw_keys)
  if not should_use_elastic(config):
    return kwargs

  active_devices = tuple(live_devices(config))
  active_slice_indices = {getattr(device, "slice_index", 0) for device in active_devices if device is not None}
  if not active_devices or not active_slice_indices:
    raise ValueError("Elastic single-controller MTC initialization found no active devices.")

  kwargs["devices"] = active_devices
  kwargs["num_slices"] = len(active_slice_indices)
  if not kwargs["data_parallelism"]:
    kwargs["data_parallelism"] = kwargs["num_slices"]
  max_logging.log(
      "Using active elastic devices for single-controller MTC initialization: "
      f"active_num_slices={kwargs['num_slices']}, "
      f"active_device_count={len(active_devices)}, "
      f"configured_num_slices={raw_keys['num_slices']}."
  )
  return kwargs
