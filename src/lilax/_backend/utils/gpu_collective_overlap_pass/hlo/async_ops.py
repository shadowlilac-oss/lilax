
"""Identification of the async collectives and te_ep calls the pass schedules."""

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.host_offload import (
    _host_offload_copy_names, _register_host_offload_copies,
)
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.profile_keys import _resolve_profile_key
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _opcode_str

# Dedicated Start opcodes. Reduce-scatter (and te_ep calls) use the generic
# async-start wrapper instead, which is async by construction.
_ASYNC_START_OPCODES = frozenset(
    {
        "all-gather-start",
        "all-reduce-start",
        "collective-permute-start",
    }
)

# Async collectives the reorder treats as relocatable; all-reduce is handled like all-gather/reduce-scatter.
_ASYNC_COLLECTIVE_PREFIXES = ("all-gather", "reduce-scatter", "all-reduce")


# This tells the main loop of the pass of what the interesting ops are
# for optimization. Add additional logic here to match any future ops
# to be scheduled.
def _is_async_start(inst: object) -> bool:
  """True if `inst` is the Start half of an async collective.

  Deliberately permissive: the generic "async-start" opcode covers NCCL
  collectives (all-gather/reduce-scatter/all-reduce), te_ep call-starts and
  trivial passthrough wrappers alike, and there is no cheap reliable way to
  tell them apart by opcode/attributes. A trivial wrapper never shows a
  deficit worth acting on. Host-offload copies are the exception: they are
  recognized (see _host_offload_copy_names) and excluded.
  """
  if inst.name in _host_offload_copy_names:
    return False
  opcode = _opcode_str(inst)
  return opcode in _ASYNC_START_OPCODES or opcode == "async-start"


def _module_has_interesting_async_ops(module) -> bool:
  """True if any non-fusion computation has an async-start (cheap early-out)."""
  _register_host_offload_copies(module)
  for comp in module.make_nonfusion_computations():
    for inst in comp.instructions():
      if _is_async_start(inst):
        return True
  return False


def _is_te_ep_call_start(inst, comp_by_name: dict) -> bool:
  """True for a te_ep dispatch/combine call-start (the multi-ms MoE windows)."""
  if not _is_async_start(inst):
    return False
  return _resolve_profile_key(inst, comp_by_name).startswith(("te_ep_combine", "te_ep_dispatch"))
