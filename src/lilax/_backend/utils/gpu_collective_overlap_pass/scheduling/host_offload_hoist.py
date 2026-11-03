
"""Hoisting host-offload copy starts to their earliest legal position."""

from __future__ import annotations

import os

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.host_offload import (
    _host_offload_copy_kind, _host_offload_copy_names,
)
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _opcode_str
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.legality import _earliest_legal_pos
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.sequence import _commit_sequence, _reinsert

_HOIST_HOST_OFFLOAD_COPIES = os.environ.get("COLLECTIVE_OVERLAP_HOIST_OFFLOAD_COPIES", "1") != "0"


def _hoist_host_offload_copy_starts(module, schedule, module_name: str) -> bool:
  """Move every host-offload copy start to its earliest legal position.

  Runs last, after all other relocation, so nothing leaves a copy later than
  its data allows. A D2H copy updates its host buffer in place, so a start
  whose host-buffer operand has other users is left alone (those readers must
  stay ordered before the in-place write). H2D copies only read it.
  """
  if not _HOIST_HOST_OFFLOAD_COPIES or not _host_offload_copy_names:
    return False
  comp_by_name = {c.name: c for c in module.computations()}
  changed = False
  for comp in module.make_nonfusion_computations():
    seq = schedule.sequence(comp)
    if seq is None:
      continue
    seq = list(seq)
    starts = [i for i in seq if i.name in _host_offload_copy_names and _opcode_str(i) == "async-start"]
    if not starts:
      continue
    positions = {inst: i for i, inst in enumerate(seq)}
    name_to_pos = {inst.name: i for inst, i in positions.items()}
    for start in starts:
      operands = list(start.operands())
      # Only the D2H copy writes its host buffer in place; the H2D slice
      # just reads it.
      if _host_offload_copy_kind(start, comp_by_name) == "d2h" and operands and len(list(operands[0].users())) > 1:
        _logger.info(
            "gpu_collective_overlap_pass [%s]: host-offload copy %s: host buffer %s has other users; not hoisting.",
            module_name,
            start.name,
            operands[0].name,
        )
        continue
      floor, movable, _ = _earliest_legal_pos(start, positions, name_to_pos, comp_by_name)
      old_pos = positions[start]
      if floor >= old_pos:
        continue
      new_seq = _reinsert(seq, movable, floor, start)
      # The start lands after its relocated operand chain, so a "move" can
      # leave it at the same index; commit only real moves.
      if new_seq.index(start) >= old_pos:
        continue
      seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
      changed = True
      _logger.info(
          "gpu_collective_overlap_pass [%s]: hoisted host-offload copy %s in %s from pos %d to %d "
          "with %d relocated operand(s).",
          module_name,
          start.name,
          comp.name,
          old_pos,
          positions[start],
          len(movable),
      )
  return changed
