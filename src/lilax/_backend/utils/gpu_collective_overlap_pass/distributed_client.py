
"""Resolution of JAX's distributed coordination-service client and process identity."""

from __future__ import annotations

import os
import socket

from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger

# ---------------------------------------------------------------------------
# Cross-rank sync (JAX's coordination-service key-value store)
# ---------------------------------------------------------------------------
# The graph traversal here is cpu-heavy, try to avoid straining the system
# when it's running 1 process per GPU by only running the compilation on rank0
_dist_client = None
_dist_checked = False


def _get_distributed_client():
  """Lazily resolve jax's distributed coordination-service client."""
  global _dist_client, _dist_checked
  if not _dist_checked:
    _dist_checked = True
    try:
      from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel

      _dist_client = _jax_distributed.global_state.client
      _process_id = _jax_distributed.global_state.process_id
      _process_count = _jax_distributed.global_state.num_processes
      _logger.info(
          "gpu_collective_overlap_pass: JAX distributed client: process_id=%d process_count=%d host=%s pid=%d.%s",
          _process_id,
          _process_count,
          socket.gethostname(),
          os.getpid(),
          ""
          if (_dist_client is not None and _process_count > 1)
          else (
              " WARNING: no multi-process client; every rank will compute its own "
              "(possibly divergent) schedule. Was jax.distributed.initialize() called?"
          ),
      )
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: jax distributed client unavailable (%s); "
          "each rank will use its own schedule, which risks divergence.",
          exc,
      )
      _dist_client = None
  return _dist_client


def _jax_process_id() -> int:
  from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel

  return _jax_distributed.global_state.process_id


def _jax_process_count() -> int:
  from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel

  return _jax_distributed.global_state.num_processes
