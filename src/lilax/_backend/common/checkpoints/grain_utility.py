"""Grain utility functions for checkpointing."""

import asyncio
import json
from typing import Any

import grain
from grain import experimental as grain_experimental
from grain import python
import jax
from lilax._backend.input_pipeline import multihost_dataloading
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import v1 as ocp

PyGrainCheckpointHandler = python.PyGrainCheckpointHandler
Composite = ocp_v0.args.Composite
RemoteIteratorWrapper = multihost_dataloading.RemoteIteratorWrapper
ElasticIterator = grain_experimental.ElasticIterator
_PROCESS_0_FILENAME = "process_0.json"


def _process_filename(process_index: int, process_count: int) -> str:
  return f"process_{process_index}-of-{process_count}.json"


def _state_to_text(iterator) -> str:
  if isinstance(iterator, grain.DatasetIterator):
    return json.dumps(iterator.get_state(), indent=4)
  return iterator.get_state().decode()


def _set_state_from_text(iterator, text: str) -> None:
  if isinstance(iterator, grain.DatasetIterator):
    iterator.set_state(json.loads(text))
  else:
    iterator.set_state(text.encode())


async def no_op():
  return None


class GrainCheckpointable(ocp.StatefulCheckpointable):
  """Orbax v1 checkpointable for Grain iterators and multihost state."""

  def __init__(self, item, *, restore_process_index=None, restore_process_count=None):
    self._item = item
    self._restore_process_index = restore_process_index
    self._restore_process_count = restore_process_count

  async def save(self, directory):
    item = self._item

    if isinstance(item, RemoteIteratorWrapper) and not item.elastic:
      item.save_state(await directory.await_creation())
      return no_op()

    if isinstance(item, (ElasticIterator, RemoteIteratorWrapper)):
      state = item.get_state()

      async def _write_elastic():
        path = await directory.await_creation()
        if jax.process_index() == 0:
          await asyncio.to_thread((path / _PROCESS_0_FILENAME).write_text, json.dumps(state, indent=4))

      return _write_elastic()

    if isinstance(item, list):
      snapshots = [(_state_to_text(it), idx, count) for it, idx, count in item]

      async def _write_list():
        path = await directory.await_creation()
        for text, idx, count in snapshots:
          await asyncio.to_thread((path / _process_filename(idx, count)).write_text, text)

      return _write_list()

    if hasattr(item, "save"):
      return await item.save(directory)

    state_text = _state_to_text(item)

    async def _write_single_fallback():
      path = await directory.await_creation()
      filename = path / _process_filename(jax.process_index(), jax.process_count())
      await asyncio.to_thread(filename.write_text, state_text)

    return _write_single_fallback()

  async def load(self, directory):
    item = self._item

    if isinstance(item, RemoteIteratorWrapper) and not item.elastic:
      item.restore_state(directory)
      return no_op()

    if isinstance(item, (ElasticIterator, RemoteIteratorWrapper)):

      async def _read_elastic():
        filename = directory / _PROCESS_0_FILENAME
        if not await asyncio.to_thread(filename.exists):
          raise ValueError(f"File {filename} does not exist.")
        item.set_state(json.loads(await asyncio.to_thread(filename.read_text)))

      return _read_elastic()

    if isinstance(item, list):
      specs = list(zip(item, self._restore_process_index))

      async def _read_list():
        for iterator, idx in specs:
          filename = directory / _process_filename(idx, self._restore_process_count)
          if not await asyncio.to_thread(filename.exists):
            raise ValueError(f"File {filename} does not exist.")
          _set_state_from_text(iterator, await asyncio.to_thread(filename.read_text))

      return _read_list()

    if self._restore_process_count is not None:
      index = self._restore_process_index if self._restore_process_index is not None else jax.process_index()

      async def _read_single():
        filename = directory / _process_filename(index, self._restore_process_count)
        if not await asyncio.to_thread(filename.exists):
          raise ValueError(f"File {filename} does not exist.")
        _set_state_from_text(item, await asyncio.to_thread(filename.read_text))

      return _read_single()

    if hasattr(item, "load"):
      return await item.load(directory)

    async def _read_single_fallback():
      filename = directory / _process_filename(jax.process_index(), jax.process_count())
      if not await asyncio.to_thread(filename.exists):
        raise ValueError(f"File {filename} does not exist.")
      _set_state_from_text(item, await asyncio.to_thread(filename.read_text))

    return _read_single_fallback()


def for_save(data_iterator: Any, expansion_factor_real_data: int) -> GrainCheckpointable:
  """Builds the v1 Grain checkpointable for saving the iterator."""
  if isinstance(data_iterator, RemoteIteratorWrapper):
    return GrainCheckpointable(data_iterator)

  if (
      not isinstance(data_iterator, list)
      and hasattr(data_iterator, "local_iterator")
      and isinstance(data_iterator.local_iterator, ElasticIterator)
  ):
    return GrainCheckpointable(data_iterator.local_iterator)

  iterators = data_iterator if isinstance(data_iterator, list) else [data_iterator]
  process_count_total = jax.process_count() * len(iterators)
  if expansion_factor_real_data > 1:
    process_count_total = process_count_total // expansion_factor_real_data

  if len(iterators) == 1 and process_count_total == jax.process_count():
    return GrainCheckpointable(iterators[0].local_iterator if hasattr(iterators[0], "local_iterator") else iterators[0])

  specs = [
      (
          di.local_iterator if hasattr(di, "local_iterator") else di,
          jax.process_index() + i * jax.process_count(),
          process_count_total,
      )
      for i, di in enumerate(iterators)
  ]
  return GrainCheckpointable(specs)


def for_restore(
    checkpoint_manager: Any, step: int, data_iterator: Any, expansion_factor_real_data: int
) -> GrainCheckpointable:
  """Builds the v1 Grain checkpointable for restoring the iterator."""
  if isinstance(data_iterator, RemoteIteratorWrapper):
    return GrainCheckpointable(data_iterator)

  if (
      not isinstance(data_iterator, list)
      and hasattr(data_iterator, "local_iterator")
      and isinstance(data_iterator.local_iterator, ElasticIterator)
  ):
    return GrainCheckpointable(data_iterator.local_iterator)

  directory = checkpoint_manager.directory / str(step) / "iter"
  process_count_jax = jax.process_count()
  process_count_stored = len(list(directory.glob("process_*-of-*.json")))

  if process_count_stored > process_count_jax:
    assert isinstance(data_iterator, list), (
        f"{process_count_stored} processes found in Grain checkpoint directory {directory}, but only "
        f"{process_count_jax} jax processes in this run, please set expansion_factor_real_data accordingly."
    )
    scaling_factor = len(data_iterator)
    expected = process_count_stored / process_count_jax
    assert scaling_factor == expected, (
        f"Found {process_count_stored} processes in checkpoint and {process_count_jax} JAX processes, "
        f"implying a scaling factor of {expected}, but the data_iterator list has {scaling_factor} items."
    )
    local_iterators = [x.local_iterator if hasattr(x, "local_iterator") else x for x in data_iterator]
    restore_process_index = [jax.process_index() + i * process_count_jax for i in range(scaling_factor)]
    return GrainCheckpointable(
        local_iterators, restore_process_index=restore_process_index, restore_process_count=process_count_stored
    )

  if process_count_stored == process_count_jax:
    assert not isinstance(data_iterator, list), (
        f"{process_count_stored} processes found in Grain checkpoint directory {directory}, matching the number of "
        "jax processes, please do not set expansion_factor_real_data."
    )
    return GrainCheckpointable(
        data_iterator.local_iterator if hasattr(data_iterator, "local_iterator") else data_iterator
    )

  if expansion_factor_real_data > 1 and process_count_stored == process_count_jax // expansion_factor_real_data:
    assert not isinstance(
        data_iterator, list
    ), "when expansion_factor_real_data > 1, the data iterator should not be a list."
    return GrainCheckpointable(
        data_iterator.local_iterator if hasattr(data_iterator, "local_iterator") else data_iterator,
        restore_process_index=jax.process_index(),
        restore_process_count=process_count_stored,
    )

  raise ValueError(
      f"Error restoring Grain checkpoint in {directory}: "
      f"The number of stored checkpoint files ({process_count_stored}) "
      f"is incompatible with the number of JAX processes ({process_count_jax}). "
      "If you are resuming training with a different number of chips, see instructions in "
      "https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/"
      "data_input_grain.md#using-grain"
  )
