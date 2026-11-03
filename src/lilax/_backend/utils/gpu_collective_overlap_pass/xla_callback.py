
"""Top-level POST_SCHEDULER callback: compute on rank 0 and share the result with every rank."""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import time
from typing import Optional

from lilax._backend.utils.gpu_collective_overlap_pass.distributed_client import (
    _get_distributed_client, _jax_process_count, _jax_process_id,
)
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.async_ops import _module_has_interesting_async_ops
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.overlap_computation import _compute_collective_overlap

# Count of invocations that got past the early-out; names the barriers and the
# KV key (see _gpu_collective_overlap_pass for why not a content hash).
_barrier_invocation_count = 0

_SHARE_TIMEOUT_MS = int(os.environ.get("COLLECTIVE_OVERLAP_SHARE_TIMEOUT_MS", str(20 * 60 * 1000)))

# Prefix bytes distinguishing "no transformation" from module bytes in the KV store.
_SHARE_NONE = b"\x00"
_SHARE_SOME = b"\x01"


def _gpu_collective_overlap_pass(serialized_hlo: bytes) -> Optional[bytes]:
  """Run the pass on rank 0 and share its output bytes with every rank.

  Every rank must compile a byte-identical executable, so only rank 0
  computes and everyone returns the same bytes, through the coordination
  service key-value store.
  """
  global _barrier_invocation_count

  # First statement, as cheap as possible, so the log shows the callback was
  # entered even if a hang happens later in XLA.
  _t_enter = time.monotonic()
  _logger.info(
      "gpu_collective_overlap_pass: ENTER host=%s pid=%d input_bytes=%d",
      socket.gethostname(),
      os.getpid(),
      len(serialized_hlo),
  )

  # Early-out before touching the distributed client: modules with no async op
  # need no synchronization at all.
  from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel

  module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
  if not _module_has_interesting_async_ops(module):
    _logger.info(
        "gpu_collective_overlap_pass: EXIT host=%s pid=%d module=%s result=no-op (no async ops; %.1f ms).",
        socket.gethostname(),
        os.getpid(),
        module.name,
        (time.monotonic() - _t_enter) * 1000,
    )
    return None

  client = _get_distributed_client()
  _in_rank = _jax_process_id() if client is not None else 0
  dump_modules = os.environ.get("COLLECTIVE_OVERLAP_DUMP_MODULE", "1") == "1"

  if dump_modules:
    # Every rank's raw input, so ranks' modules can be diffed.
    _module_text = module.to_string()
    _in_host = socket.gethostname()
    _in_pid = os.getpid()
    _content_digest = hashlib.sha256(_module_text.encode()).hexdigest()
    _logger.info(
        "gpu_collective_overlap_pass: PRE-PASS INPUT HASH rank=%d host=%s pid=%d module=%s sha256=%s bytes=%d text_bytes=%d",
        _in_rank,
        _in_host,
        _in_pid,
        module.name,
        _content_digest,
        len(serialized_hlo),
        len(_module_text),
    )
    sys.stderr.write(
        f"[gpu_collective_overlap_pass] === BEGIN PRE-PASS INPUT MODULE "
        f"(rank={_in_rank} host={_in_host} pid={_in_pid} "
        f"module={module.name} sha256={_content_digest}) ===\n"
        f"{_module_text}\n"
        f"[gpu_collective_overlap_pass] === END PRE-PASS INPUT MODULE "
        f"(rank={_in_rank} module={module.name}) ===\n"
    )

  is_root = client is None or _in_rank == 0
  multi_process = client is not None and _jax_process_count() > 1

  _entry_name = _exit_name = None
  if multi_process:
    _barrier_invocation_count += 1
    _entry_name = f"gpu_collective_overlap_pass_entry_{_barrier_invocation_count}"
    _exit_name = f"gpu_collective_overlap_pass_exit_{_barrier_invocation_count}"
    try:
      client.wait_at_barrier(_entry_name, _SHARE_TIMEOUT_MS)
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: entry barrier %s failed (%s); proceeding without it.",
          _entry_name,
          exc,
      )

  result_bytes: Optional[bytes] = None
  if is_root:
    _t_compute = time.monotonic()
    result_bytes = _compute_collective_overlap(serialized_hlo)
    _logger.info(
        "gpu_collective_overlap_pass: _compute_collective_overlap done (%s, %.1f ms).",
        "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
        (time.monotonic() - _t_compute) * 1000,
    )

  if multi_process:
    _key = f"gpu_collective_overlap_pass_kv_{_barrier_invocation_count}"
    try:
      if is_root:
        _payload = _SHARE_NONE if result_bytes is None else _SHARE_SOME + result_bytes
        client.key_value_set_bytes(_key, _payload)
      else:
        _payload = client.blocking_key_value_get_bytes(_key, _SHARE_TIMEOUT_MS)
        result_bytes = None if _payload == _SHARE_NONE else _payload[len(_SHARE_SOME) :]
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: sharing the scheduled module via the JAX distributed client "
          "failed (%s); using this rank's own result, which risks divergent schedules.",
          exc,
      )
      if not is_root:
        result_bytes = _compute_collective_overlap(serialized_hlo)

    try:
      client.wait_at_barrier(_exit_name, _SHARE_TIMEOUT_MS)
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: exit barrier %s failed (%s); proceeding without it.",
          _exit_name,
          exc,
      )

  # What this rank returns.
  if result_bytes is not None and dump_modules:
    _rank = _jax_process_id()
    _host = socket.gethostname()
    _pid = os.getpid()
    _digest = hashlib.sha256(result_bytes).hexdigest()
    _final_module = _hlo.HloModule.from_serialized_hlo_module_proto(result_bytes)
    _module_name = _final_module.name
    _logger.info(
        "gpu_collective_overlap_pass: POST-SHARE HASH rank=%d host=%s pid=%d module=%s sha256=%s bytes=%d",
        _rank,
        _host,
        _pid,
        _module_name,
        _digest,
        len(result_bytes),
    )
    sys.stderr.write(
        f"[gpu_collective_overlap_pass] === BEGIN POST-SHARE MODULE "
        f"(rank={_rank} host={_host} pid={_pid} module={_module_name} "
        f"sha256={_digest}) ===\n"
        f"{_final_module.to_string()}\n"
        f"[gpu_collective_overlap_pass] === END POST-SHARE MODULE "
        f"(rank={_rank} module={_module_name}) ===\n"
    )

  _logger.info(
      "gpu_collective_overlap_pass: EXIT host=%s pid=%d result=%s (total %.1f ms).",
      socket.gethostname(),
      os.getpid(),
      "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
      (time.monotonic() - _t_enter) * 1000,
  )
  return result_bytes
