"""Data-axis placement of loss inputs on the engine mesh."""

from __future__ import annotations

from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
from lilax._backend.utils import sharding
import numpy as np

_REPLICATED_BATCH_DIM_WARNING = (
    "Loss input with batch dim %d does not divide mesh axis %r (size %d), so that "
    "dimension is replicated instead of sharded: every device along the axis holds and "
    "computes the whole micro-batch, %dx the work a sharded one would do there. Results "
    "stay correct. If it was not deliberate -- a sequence-packed micro-batch is always "
    "size 1 and has no alternative -- make the micro-batch a multiple of the axis size."
)


class BatchSharder:
  """Shards loss inputs over the mesh's batch axes, replicating dimensions that do not divide."""

  def __init__(self, config: Any, mesh: jax.sharding.Mesh | None) -> None:
    self._config = config
    self._mesh = mesh
    self._replicated_batch_warned = False

  def _batch_axis_size(self, axis_names: Any) -> int:
    if axis_names is None:
      return 1
    names = axis_names if isinstance(axis_names, tuple) else (axis_names,)
    size = 1
    for name in names:
      size *= self._mesh.shape[name]
    return size

  def _input_data_spec(self) -> tuple[Any, ...]:
    return tuple(sharding.get_input_data_sharding(self._config, self._mesh).spec)

  def _leaf_data_sharding(self, leaf: Any, data_spec: tuple[Any, ...]) -> jax.sharding.NamedSharding | None:
    if leaf is None:
      return None
    rank = jnp.ndim(leaf)
    spec = list(data_spec[:rank])
    if spec and spec[0] is not None:
      axis_size = self._batch_axis_size(spec[0])
      if leaf.shape[0] % axis_size:
        if not self._replicated_batch_warned:
          self._replicated_batch_warned = True
          logging.warning(_REPLICATED_BATCH_DIM_WARNING, leaf.shape[0], spec[0], axis_size, axis_size)
        spec[0] = None
    return jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec(*spec))

  def data_shardings(self, dynamic_batch: Any) -> Any:
    data_spec = self._input_data_spec()
    return jax.tree.map(lambda leaf: self._leaf_data_sharding(leaf, data_spec), dynamic_batch)

  def place_read_only_inputs(self, inputs: Any) -> Any:
    if self._mesh is None:
      return inputs
    data_spec = self._input_data_spec()

    def place(leaf: Any) -> Any:
      if not isinstance(leaf, (jax.Array, np.ndarray)):
        return leaf
      if leaf.ndim == 0 or 0 in leaf.shape:
        return leaf
      if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        return leaf
      target = self._leaf_data_sharding(leaf, data_spec)
      if target is None or getattr(leaf, "sharding", None) == target:
        return leaf
      return jax.device_put(leaf, target)

    return jax.tree.map(place, inputs)
