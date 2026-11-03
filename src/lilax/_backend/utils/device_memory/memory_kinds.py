"""Memory kind and memory space selection for device placement."""

import jax
from packaging.version import Version


def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind=memory_kind), t)


def device_space():
  """Version guard for jax.memory.Space.Device."""
  if Version(jax.__version__) >= Version("0.7.1"):
    return jax.memory.Space.Device
  else:
    return jax._src.sharding_impls.TransferToMemoryKind("device")
