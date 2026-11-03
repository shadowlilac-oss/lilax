
"""Pulling a hidden async collective's done ahead of the next te_ep call-start."""

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.costing.instruction_cost import _resolve_inst_cost
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.async_ops import _is_te_ep_call_start
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _control_predecessor_names
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.sequence import _commit_sequence, _reinsert


def _relocate_async_done_before_te_ep(
    async_start,
    async_done,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    collective_latency: float,
    module_name: str = "",
):
  """For an already-hidden async collective, pull `done` forward to just before
  the first te_ep call-start at or after the point where its latency is
  closed, if any.

  `done` gates its consumers, so a later done than needed delays them for no
  benefit; an earlier one frees them to fill te_ep windows (the largest,
  hardest-to-fill ones). Always legal: done's only operand is start, and its
  consumers already sit after its old position. Control-predecessors are
  respected.

  Returns (changed, seq, positions, name_to_pos).
  """
  start_pos = positions[async_start]
  done_pos = positions[async_done]
  if done_pos <= start_pos + 1:
    return False, seq, positions, name_to_pos

  done_floor = start_pos + 1
  for name in _control_predecessor_names(async_done):
    cp_pos = name_to_pos.get(name)
    if cp_pos is not None and cp_pos + 1 > done_floor:
      done_floor = cp_pos + 1

  cum = 0.0
  latency_closure_pos = None
  for i in range(start_pos + 1, done_pos):
    cum += _resolve_inst_cost(seq[i], comp_by_name)
    if cum >= collective_latency:
      latency_closure_pos = i + 1
      break
  if latency_closure_pos is None:
    return False, seq, positions, name_to_pos

  search_start = max(latency_closure_pos, done_floor)
  target = None
  for i in range(search_start, done_pos):
    inst = seq[i]
    if inst is async_done or inst is async_start:
      continue
    if _is_te_ep_call_start(inst, comp_by_name):
      target = i
      break
  if target is None:
    return False, seq, positions, name_to_pos

  new_seq, new_positions, new_name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, [], target, async_done))
  _logger.info(
      "gpu_collective_overlap_pass [%s]: moved %s done from pos %d to %d, before te_ep call-start %s "
      "(latency %.1f us closed by pos %d).",
      module_name,
      async_done.name,
      done_pos,
      target,
      seq[target].name,
      collective_latency,
      latency_closure_pos,
  )
  return True, new_seq, new_positions, new_name_to_pos
