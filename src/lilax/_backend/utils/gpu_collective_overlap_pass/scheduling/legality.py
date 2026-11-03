
"""Earliest legal schedule position of an instruction under data/control dependencies."""

from __future__ import annotations

from typing import Optional

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.movability import _is_trivially_movable_inst
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _control_predecessor_names

_MAX_RELOCATE_CHAIN = 64


def _earliest_legal_pos(
    async_start,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    cp_pins: Optional[list] = None,
) -> tuple[int, list, object]:
  """Earliest position async_start can legally move to.

  Walks async_start's operands transitively through trivially-movable
  instructions (no distance bound; trivial ops are free to move). The walk
  stops at any non-trivial instruction, which pins the floor to just after
  it. Control-predecessors outside the moving set pin it the same way, and
  are appended to `cp_pins` as (position, name) if a list is given.

  Returns (floor, movable, blocker): movable is the part of the trivial chain
  at or after the floor, in schedule order, excluding async_start (chain members
  already before the floor stay put; moving them would drag async_start's final
  position later). blocker is the non-trivial data producer pinning the
  floor, or None if the floor is 0 or pinned by a control-predecessor.
  """
  candidates: set = set()
  seen: set = set()
  floor = 0
  blocker = None
  queue = list(async_start.operands())
  while queue:
    inst = queue.pop()
    if inst in seen:
      continue
    seen.add(inst)
    if inst not in positions:
      continue
    if len(candidates) < _MAX_RELOCATE_CHAIN and _is_trivially_movable_inst(inst, comp_by_name):
      candidates.add(inst)
      queue.extend(inst.operands())
    else:
      if positions[inst] + 1 > floor:
        floor = positions[inst] + 1
        blocker = inst

  moving_names = {inst.name for inst in candidates} | {async_start.name}
  for inst in list(candidates) + [async_start]:
    for name in _control_predecessor_names(inst):
      if name in moving_names:
        continue
      cp_pos = name_to_pos.get(name)
      if cp_pos is not None and cp_pins is not None:
        cp_pins.append((cp_pos, name))
      if cp_pos is not None and cp_pos + 1 > floor:
        floor = cp_pos + 1
        blocker = None

  # The floor only grows as the walk proceeds, so early candidates may
  # already sit before the final one.
  movable = [inst for inst in candidates if positions[inst] >= floor]
  return floor, sorted(movable, key=lambda i: positions[i]), blocker
