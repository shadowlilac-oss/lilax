"""SPMD multihost dataloading utilities."""

from collections.abc import Iterable
from functools import partial
from typing import Sequence
import gc
import json
import os
import signal
import time
import weakref

from etils import epath
import jax
from jax.experimental import colocated_python
from jax.experimental.colocated_python.obj_backend import SINGLETON_OBJECT_STORE
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax.tree_util as jtu
import numpy as np
import tensorflow as tf

from lilax._backend.observability import max_logging

_TF_RETRYABLE_ERRORS = (tf.errors.FailedPreconditionError,)
_GLOBAL_REMOTE_ITERATORS = weakref.WeakSet()
_GLOBAL_MULTIHOST_ITERATORS = weakref.WeakSet()


def _close_iterator(it):
  """Safely closes an iterator or dataloader, handling nested Grain/TF iterators."""
  if it is None:
    return
  if hasattr(it, "close") and callable(it.close):
    try:
      it.close()
    except Exception as e:
      max_logging.log(f"CLEANUP: error closing iterator: {e}")
  elif hasattr(it, "_iterator") and hasattr(it._iterator, "close") and callable(it._iterator.close):
    try:
      it._iterator.close()
    except Exception as e:
      max_logging.log(f"CLEANUP: error closing underlying _iterator: {e}")


def _terminate_worker(w):
  try:
    if hasattr(w, "terminate") and callable(w.terminate):
      w.terminate()
      if hasattr(w, "join") and callable(w.join):
        w.join(timeout=0.2)
  except Exception:
    pass


def _terminate_iterator_workers(obj):
  """Terminates multiprocessing workers directly associated with an iterator or dataloader."""
  if obj is None:
    return
  candidates = [obj]
  for attr in ("_iterator", "_parent"):
    try:
      val = getattr(obj, attr, None)
      if val is not None:
        candidates.append(val)
    except Exception:
      pass
  try:
    parents = getattr(obj, "_parents", None)
    if parents and isinstance(parents, (list, tuple)):
      candidates.extend(p for p in parents if p is not None)
  except Exception:
    pass
  for c in candidates:
    workers = getattr(c, "_workers", None) or getattr(c, "workers", None)
    if workers and isinstance(workers, (list, tuple)):
      for w in workers:
        _terminate_worker(w)
    pool = getattr(c, "_pool", None)
    if pool and hasattr(pool, "terminate") and callable(pool.terminate):
      try:
        pool.terminate()
      except Exception:
        pass


def _cleanup_orphaned_multiprocessing_processes():
  """Cleans up multiprocessing spawn and resource tracker processes orphaned under PID 1."""
  current_pid = os.getpid()
  if current_pid == 1:
    return
  try:
    if not os.path.exists("/proc"):
      return
    orphan_pids = []
    for entry in os.listdir("/proc"):
      if not entry.isdigit():
        continue
      pid = int(entry)
      if pid in (current_pid, 1):
        continue
      try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
          cmd = f.read().decode("utf-8", errors="ignore")
        if "multiprocessing.spawn" not in cmd and "multiprocessing.resource_tracker" not in cmd:
          continue
        ppid = None
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
          for line in f:
            if line.startswith("PPid:"):
              ppid = int(line.split()[1])
              break
        if ppid == 1:
          max_logging.log(f"CLEANUP: Terminating orphaned multiprocessing process {pid} (PPid=1)")
          try:
            os.kill(pid, signal.SIGTERM)
          except ProcessLookupError:
            continue
          except Exception as e:
            max_logging.log(f"CLEANUP: SIGTERM failed for {pid}: {e}")
          orphan_pids.append(pid)
      except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
      except Exception as e:
        max_logging.log(f"CLEANUP: Error checking PID {pid}: {e}")
    if not orphan_pids:
      return
    time.sleep(0.05)
    for pid in orphan_pids:
      try:
        os.kill(pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      except Exception as e:
        max_logging.log(f"CLEANUP: SIGKILL failed for {pid}: {e}")
  except Exception as e:
    max_logging.log(f"CLEANUP: Failed scanning /proc: {e}")


def cleanup_all_iterators():
  """Cleans up all registered global multihost and remote iterators."""
  max_logging.log(
      f"CLEANUP: Cleaning up {len(_GLOBAL_MULTIHOST_ITERATORS)} multihost iterators "
      f"and {len(_GLOBAL_REMOTE_ITERATORS)} remote iterators"
  )
  for it in list(_GLOBAL_MULTIHOST_ITERATORS):
    try:
      if hasattr(it, "close"):
        it.close()
    except Exception as e:
      max_logging.log(f"CLEANUP Error: {e}")
  for it in list(_GLOBAL_REMOTE_ITERATORS):
    try:
      if hasattr(it, "close"):
        it.close()
    except Exception as e:
      max_logging.log(f"CLEANUP Error: {e}")
  _cleanup_orphaned_multiprocessing_processes()


def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
  sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))
  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
  return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
  """Put local sharded array into local devices."""
  global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
  try:
    local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
  except ValueError as array_split_error:
    raise ValueError(
        f"Unable to put to devices shape {array.shape} with "
        f"local device count {len(global_mesh.local_devices)} "
        f"at {jtu.keystr(path)}"
    ) from array_split_error
  local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)


class MultiHostDataLoadIterator:
  """Fold sharded batch loading into an iterator."""

  def __init__(
      self,
      dataloader: Iterable,
      global_mesh: Mesh,
      generate_padding_batch: bool = False,
      expansion_loading_factor_for_grain: int = -1,
  ):
    self._closed = False
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    if hasattr(self.dataloader, "as_numpy_iterator"):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")
    self.out_of_data = False
    self.last_local_data = None
    self._reset_and_unread = False
    self.generate_padding_batch = generate_padding_batch
    self.expansion_loading_factor_for_grain = expansion_loading_factor_for_grain
    _GLOBAL_MULTIHOST_ITERATORS.add(self)

  def reset(self):
    if hasattr(self.dataloader, "as_numpy_iterator"):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")
    self.out_of_data = False
    self.last_local_data = None
    self._reset_and_unread = True
    self._closed = False

  def close(self):
    if self._closed:
      return
    self._closed = True
    if hasattr(self, "local_iterator"):
      _close_iterator(self.local_iterator)
      _terminate_iterator_workers(self.local_iterator)
    if hasattr(self, "dataloader"):
      _close_iterator(self.dataloader)
      _terminate_iterator_workers(self.dataloader)
    _cleanup_orphaned_multiprocessing_processes()

  def __del__(self):
    self.close()

  def __iter__(self):
    if not self._reset_and_unread:
      self.reset()
    return self

  def __next__(self):
    self._reset_and_unread = False
    return self._get_next_batch_sharded()

  def _get_next_batch_sharded(self) -> jax.Array:
    """Splits the host loaded data equally over all devices."""
    if self.out_of_data and self.generate_padding_batch:
      local_data = self._make_padding_batch()
    else:
      SLEEP_TIME = 10
      MAX_DATA_LOAD_ATTEMPTS = 30
      local_data = None
      for _ in range(MAX_DATA_LOAD_ATTEMPTS):
        try:
          local_data = next(self.local_iterator)
          if self.expansion_loading_factor_for_grain > 1:
            local_data_list = [local_data]
            for _ in range(1, int(self.expansion_loading_factor_for_grain)):
              next_batch = next(self.local_iterator)
              local_data_list.append(next_batch)
            local_data = jtu.tree_map(lambda *xs: np.concatenate(xs, axis=0), *local_data_list)
          break
        except _TF_RETRYABLE_ERRORS as e:
          max_logging.log(f"Failed to get next data batch due to {e}, retrying")
          time.sleep(SLEEP_TIME)
        except StopIteration as e:
          if self.generate_padding_batch:
            max_logging.log(
                f"MultiHostDataLoadIterator: host {jax.process_index()} failed to load data with {type(e)} error: ({e}). "
                "It may have reached the end of the data. Generating a padding batch as generate_padding_batch=True."
            )
            self.out_of_data = True
            local_data = self._make_padding_batch()
            break
          else:
            raise e
      else:
        raise TimeoutError(f"Failed to load data after {MAX_DATA_LOAD_ATTEMPTS} retry attempts.")
      self.last_local_data = local_data
    input_gdas = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=self.global_mesh), local_data)
    return input_gdas

  def _make_padding_batch(self):
    if self.last_local_data is None:
      raise ValueError("last_local_data is None, cannot make padding batch.")
    return jtu.tree_map(lambda x: jnp.full_like(x, 0), self.last_local_data)


def _colocated_cpu_devices(devices: Sequence[jax.Device]) -> Sequence[jax.Device]:
  """Returns CPU devices colocated with the given devices."""
  return colocated_python.colocated_cpu_devices(devices)


def _colocated_cpu_mesh(mesh: Mesh) -> Mesh:
  """Returns a CPU mesh that has colocated CPU devices."""
  return colocated_python.colocated_cpu_devices(mesh)


_MAX_DIRECTORY_BYTES = 4096


def _state_file(directory_array) -> epath.Path:
  """Returns this colocated host's iterator state file."""
  directory = np.asarray(directory_array.addressable_data(0)).tobytes().rstrip(b"\0").decode()
  return epath.Path(directory) / f"process_{jax.process_index()}-of-{jax.process_count()}.json"


class RemoteIterator:
  """Iterator for a colocated Python class."""

  def __init__(self, get_ds_fn, preprocessing_fn, global_shape, checkpoint_path, elastic=False):
    try:
      with SINGLETON_OBJECT_STORE._lock:
        uids_to_del = []
        for uid, state in list(SINGLETON_OBJECT_STORE._storage.items()):
          if state.obj is not None and state.obj is not self:
            if isinstance(state.obj, RemoteIterator) or state.obj.__class__.__name__ == "RemoteIterator":
              if hasattr(state.obj, "close") and callable(state.obj.close):
                try:
                  state.obj.close()
                except Exception:
                  pass
              uids_to_del.append(uid)
        for uid in uids_to_del:
          SINGLETON_OBJECT_STORE._storage.pop(uid, None)
    except Exception as e:
      max_logging.log(f"Failed to cleanup SINGLETON_OBJECT_STORE in RemoteIterator: {e}")
    try:
      cleanup_all_iterators()
    except Exception as e:
      max_logging.log(f"Failed to cleanup old iterators in RemoteIterator: {e}")
    _cleanup_orphaned_multiprocessing_processes()
    try:
      gc.collect()
    except Exception:
      pass
    self.get_ds_fn = get_ds_fn
    self.preprocessing_fn = preprocessing_fn
    self.global_shape = global_shape
    self.checkpoint_path = checkpoint_path
    self.elastic = elastic
    _GLOBAL_REMOTE_ITERATORS.add(self)
    max_logging.log("RemoteIterator initiated")

  def build(self, dummy_array):
    """Builds the iterator over this host's shard of the mesh."""
    processes = sorted({device.process_index for device in dummy_array.sharding.device_set})
    local_processes = {device.process_index for device in dummy_array.sharding.addressable_devices}
    if len(local_processes) != 1:
      raise ValueError(
          "RemoteIterator.build expects this colocated process to address the devices of exactly one host of the"
          f" mesh, whose processes are {processes}; it addresses devices of processes {sorted(local_processes)}."
      )
    (process,) = local_processes
    self.shard_index, self.shard_count = processes.index(process), len(processes)
    max_logging.log(f"RemoteIterator reads data shard {self.shard_index} of {self.shard_count}")
    self.reset()
    return dummy_array

  def reset(self):
    """Rebuilds the iterator for this host's data shard, from the start of the data."""
    ds = self.get_ds_fn(dataloading_host_index=self.shard_index, dataloading_host_count=self.shard_count)
    if self.elastic:
      self.dataloader = self.preprocessing_fn(dataset=ds, shard_index=self.shard_index, shard_count=self.shard_count)
    else:
      self.dataloader = self.preprocessing_fn(dataset=ds)
    if hasattr(self.dataloader, "as_numpy_iterator"):
      self.iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be Iterable.")

  def close(self, dummy_array=None):
    if hasattr(self, "iterator"):
      _close_iterator(self.iterator)
      _terminate_iterator_workers(self.iterator)
    if hasattr(self, "dataloader"):
      _close_iterator(self.dataloader)
      _terminate_iterator_workers(self.dataloader)
    _cleanup_orphaned_multiprocessing_processes()

  def __del__(self):
    self.close()

  def get_next(self, dummy_array):
    """Gets the next batch of data and forms a global array."""
    local_data = next(self.iterator)

    def form_global_array_colocated_python(path, array, devices, global_shape, sharding):
      try:
        device_arrays = np.split(array, len(devices), axis=0)
      except ValueError as array_split_error:
        raise ValueError(
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(devices)} "
            f"at {jtu.keystr(path)}"
        ) from array_split_error
      device_arrays = jax.device_put(device_arrays, devices)
      return jax.make_array_from_single_device_arrays(shape=global_shape, sharding=sharding, arrays=device_arrays)

    return jtu.tree_map_with_path(
        partial(
            form_global_array_colocated_python,
            devices=list(dummy_array.sharding.addressable_devices),
            global_shape=self.global_shape,
            sharding=dummy_array.sharding,
        ),
        local_data,
    )

  def save_state(self, directory_array):
    """Writes this host's iterator state."""
    state = json.dumps(self.iterator.get_state(), indent=4)
    _state_file(directory_array).write_text(state)
    return directory_array

  def restore_state(self, directory_array):
    """Restores this host's iterator state."""
    self.iterator.set_state(json.loads(_state_file(directory_array).read_text()))
    return directory_array

  def get_state(self, dummy_array):
    words = np.array([self.iterator.get_state()["global_next_index"]], dtype=np.int64).view(np.int32)
    devices = list(dummy_array.sharding.addressable_devices)
    shards = jax.device_put([words[None, :] for _ in devices], devices)
    return jax.make_array_from_single_device_arrays((*dummy_array.shape, 2), dummy_array.sharding, shards)

  def set_state(self, index_array):
    index = int(np.asarray(index_array.addressable_data(0)).view(np.int64).item())
    self.iterator.set_state({"global_next_index": index})
    return index_array


class RemoteIteratorWrapper:
  """Wrapper for RemoteIterator that handles device placement."""

  def __init__(self, get_ds_fn, preprocessing_fn, global_mesh, global_shape, checkpoint_path="", elastic=False):
    self.elastic = elastic
    self.cpu_devices = _colocated_cpu_devices(tuple(global_mesh.devices.flat))
    self.cpu_mesh = _colocated_cpu_mesh(global_mesh)
    self.tpu_sharding = jax.sharding.NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))
    self.cpu_sharding = jax.sharding.NamedSharding(self.cpu_mesh, PartitionSpec(self.cpu_mesh.axis_names))
    self.dummy_array = jnp.zeros((len(self.cpu_devices)))
    self.dummy_array = jax.device_put(self.dummy_array, self.cpu_sharding)
    remote_iterator_cls = colocated_python.colocated_python_class(RemoteIterator)
    self.local_iterator = remote_iterator_cls(
        get_ds_fn,
        preprocessing_fn,
        global_shape,
        checkpoint_path,
        elastic=elastic,
    )
    jax.block_until_ready(self.local_iterator.build(self.dummy_array))
    _GLOBAL_REMOTE_ITERATORS.add(self)
    max_logging.log("RemoteIteratorWrapper initiated")

  def close(self):
    if hasattr(self, "local_iterator") and hasattr(self.local_iterator, "close") and callable(self.local_iterator.close):
      try:
        self.local_iterator.close(self.dummy_array)
      except Exception as e:
        max_logging.log(f"RemoteIteratorWrapper.close() ignored exception: {e}")

  def __del__(self):
    self.close()

  def __iter__(self):
    return self

  def reset(self):
    self.local_iterator.reset()

  def __next__(self):
    out = self.local_iterator.get_next(self.dummy_array)
    return jax.device_put(out, self.tpu_sharding)

  def _encode_directory(self, directory) -> jax.Array:
    path = os.fspath(directory).encode().ljust(_MAX_DIRECTORY_BYTES, b"\0")
    return jax.device_put(np.frombuffer(path, dtype=np.uint8), NamedSharding(self.cpu_mesh, PartitionSpec()))

  def save_state(self, directory):
    """Has each colocated host write its iterator state into the checkpoint directory."""
    directory_array = self._encode_directory(directory)
    jax.block_until_ready(self.local_iterator.save_state(directory_array))

  def restore_state(self, directory):
    """Has each colocated host restore its iterator state from the checkpoint directory."""
    directory_array = self._encode_directory(directory)
    jax.block_until_ready(self.local_iterator.restore_state(directory_array))

  def get_state(self) -> dict[str, int]:
    if not self.elastic:
      raise ValueError("RemoteIteratorWrapper.get_state requires elastic=True.")
    index_array = self.local_iterator.get_state(self.dummy_array)
    return {"global_next_index": int(np.asarray(index_array[0]).view(np.int64).item())}

  def set_state(self, state: dict[str, int]) -> None:
    if not self.elastic:
      raise ValueError("RemoteIteratorWrapper.set_state requires elastic=True.")
    words = np.array([state["global_next_index"]], dtype=np.int64).view(np.int32)
    index_array = jax.device_put(words, NamedSharding(self.cpu_mesh, PartitionSpec()))
    jax.block_until_ready(self.local_iterator.set_state(index_array))
