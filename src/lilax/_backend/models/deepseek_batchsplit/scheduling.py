"""XLA scheduling groups for the DeepSeek BF16 batch-split schedule."""

import contextlib

import jax


def scheduling_group(group_id) -> contextlib.AbstractContextManager[None]:
  return jax.experimental.xla_metadata.set_xla_metadata(_scheduling_group_id=group_id)
