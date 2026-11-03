
"""Innermost-first ordering of the scheduled computations and while-body detection."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

import re

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _opcode_str

_WHILE_CALLS_RE = re.compile(r"(?:condition|body)=%([A-Za-z0-9_.]+)")


def _innermost_first_computations(module, schedule) -> tuple[list, set]:
  """Return (computations, while_body_comps).

  computations are the scheduled non-fusion computations in innermost-first
  DFS order (a while body before the computation holding its while), the
  entry computation last. while_body_comps is every computation called as a
  while body, i.e. all but the true entry.
  """
  all_comps = [c for c in module.make_nonfusion_computations() if schedule.sequence(c) is not None]
  # Restricted to this accessor call so lookups stay identity-consistent with
  # all_comps (HloComputation has no custom __eq__/__hash__).
  local_comp_by_name = {c.name: c for c in all_comps}

  children: dict = {c: [] for c in all_comps}
  all_callees: set = set()
  for comp in all_comps:
    for inst in schedule.sequence(comp):
      if _opcode_str(inst) != "while":
        continue
      try:
        text = inst.to_string()
      except Exception:
        continue
      for name in _WHILE_CALLS_RE.findall(text):
        called = local_comp_by_name.get(name)
        if called is not None and schedule.sequence(called) is not None:
          children[comp].append(called)
          all_callees.add(called)

  roots = [c for c in all_comps if c not in all_callees]
  if not roots:
    return all_comps, all_callees

  def _collect(comp, visited: set, result: list) -> None:
    if comp in visited:
      return
    visited.add(comp)
    for child in children.get(comp, []):
      _collect(child, visited, result)
    result.append(comp)

  visited: set = set()
  result: list = []
  for root in roots:
    _collect(root, visited, result)
  for comp in all_comps:
    if comp not in visited:
      result.append(comp)
  return result, all_callees
