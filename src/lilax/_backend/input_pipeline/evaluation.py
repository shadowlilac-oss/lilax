"""Exhaust finite evaluation data together across JAX processes."""

import json

import jax
from jax.experimental import multihost_utils
import numpy as np

from .multihost_dataloading import _close_iterator


class FiniteEvaluationLoader:
  """Supply zero-weight local rows until every process has exhausted its records."""

  def __init__(self, dataloader):
    self.dataloader = dataloader

  def __iter__(self):
    iterator = iter(self.dataloader)
    try:
      while True:
        batch = next(iterator, None)
        live = np.asarray(multihost_utils.process_allgather(np.array(batch is not None))).reshape(-1)
        if not live.any():
          return
        source = jax.process_index() == int(np.flatnonzero(live)[0])
        metadata = (json.dumps({key: [list(value.shape), str(value.dtype)] for key, value in batch.items()}).encode()
                    if source else b"")
        length = int(multihost_utils.broadcast_one_to_all(np.array(len(metadata), dtype=np.int32), is_source=source))
        buffer = np.frombuffer(metadata, dtype=np.uint8) if source else np.zeros(length, dtype=np.uint8)
        shapes = json.loads(multihost_utils.broadcast_one_to_all(buffer, is_source=source).tobytes())
        valid = batch is None or {key: [list(value.shape), str(value.dtype)] for key, value in batch.items()} == shapes
        if not np.asarray(multihost_utils.process_allgather(np.array(valid))).all():
          raise ValueError("Evaluation processes must produce the same batch shapes and dtypes before sharding.")
        if batch is None:
          batch = {key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in shapes.items()}
        yield batch
    finally:
      _close_iterator(iterator)

  def close(self):
    _close_iterator(self.dataloader)
