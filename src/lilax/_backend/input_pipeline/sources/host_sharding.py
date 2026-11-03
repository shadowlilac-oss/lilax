"""Per-host file and record sharding for multi-host data loading."""

import grain.python as grain


def compute_file_sharding(file_count, host_index, host_count):
  """Compute per-host file slicing and optional row-shard parameters.

  When file_count >= host_count, each host reads a disjoint subset of files via the
  standard `[host_index::host_count]` slice. When file_count < host_count, every file
  is replicated across `ceil(host_count/file_count)` hosts (or `floor` for files past
  the remainder), and those hosts shard records by index within the file. This bounds
  concurrent readers per file at `ceil(host_count/file_count)`

  Returns:
    file_slice (slice): file slice this host should keep — also the (start, step)
      pair `(host_index, host_count)` to feed `tf.distribute.InputContext` when applicable.
    files_per_host (int): files this host's slice contains per epoch.
    row_shard (tuple|None): `(row_shard_index, row_shard_count)` when host_count > file_count
      and the file's group has >1 reader; otherwise None.
  """
  if file_count >= host_count:
    return (
        slice(host_index, None, host_count),
        max(file_count // host_count, 1),
        None,
    )
  file_idx = host_index % file_count
  row_shard_idx = host_index // file_count
  row_shard_count = (host_count // file_count) + (1 if file_idx < (host_count % file_count) else 0)
  row_shard = (row_shard_idx, row_shard_count) if row_shard_count > 1 else None
  return slice(file_idx, None, file_count), 1, row_shard


class _IndexShardDatasetIterator(grain.DatasetIterator):
  """Iterator that yields every nth element of its parent (round-robin by index)."""

  def __init__(self, parent: grain.DatasetIterator, host_index: int, host_count: int):
    super().__init__(parent)
    self._host_index = host_index
    self._host_count = host_count
    self._next_index = 0

  def __next__(self):
    while True:
      value = next(self._parent)
      current = self._next_index
      self._next_index += 1
      if current % self._host_count == self._host_index:
        return value

  def get_state(self):
    return {
        "next_index": self._next_index,
        "parent": self._parent.get_state(),
    }

  def set_state(self, state):
    self._next_index = state["next_index"]
    self._parent.set_state(state["parent"])


class IndexShardIterDataset(grain.IterDataset):
  """Shards an IterDataset across hosts by element index (host i keeps records where idx % N == i).

  Use when the upstream `IterDataset` order is deterministic and identical on every host;
  this guarantees disjoint, balanced slices without per-file sharding.
  """

  def __init__(self, parent: grain.IterDataset, host_index: int, host_count: int):
    super().__init__(parent)
    self._host_index = host_index
    self._host_count = host_count

  def __iter__(self) -> _IndexShardDatasetIterator:
    return _IndexShardDatasetIterator(
        self._parent.__iter__(),
        host_index=self._host_index,
        host_count=self._host_count,
    )
