
"""Profile-coverage and final exposed-time reports for a scheduled module."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

import re

from lilax._backend.utils.gpu_collective_overlap_pass.costing.exposure import _prefix_costs_excluding
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.async_ops import _is_async_start
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.profile_keys import _resolve_profile_key
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _opcode_str
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _profile_costs

_WHILE_BODY_RE = re.compile(r"\bbody=%([A-Za-z0-9_.]+)")


def _log_profile_coverage_gaps(module, schedule, module_name: str) -> None:
  """Log fusion/custom-call instructions in while bodies with no PGLE cost
  entry; every cost lookup silently treats a missing one as free."""
  all_comps = [c for c in module.make_nonfusion_computations() if schedule.sequence(c) is not None]
  comp_by_name = {c.name: c for c in all_comps}
  while_bodies: set = set()
  for comp in all_comps:
    for inst in schedule.sequence(comp):
      if _opcode_str(inst) != "while":
        continue
      try:
        text = inst.to_string()
      except Exception:
        continue
      for name in _WHILE_BODY_RE.findall(text):
        body = comp_by_name.get(name)
        if body is not None:
          while_bodies.add(body)

  if not while_bodies:
    return

  missing: list[str] = []
  total = 0
  for comp in while_bodies:
    for inst in schedule.sequence(comp):
      opc = _opcode_str(inst)
      if opc not in ("fusion", "custom-call"):
        continue
      total += 1
      if inst.name not in _profile_costs:
        missing.append(f"{inst.name}({opc})")

  _logger.info(
      "gpu_collective_overlap_pass [%s]: profile coverage in %d while body(ies): %d/%d "
      "fusion/custom-call instructions have a PGLE cost (%d missing).",
      module_name,
      len(while_bodies),
      total - len(missing),
      total,
      len(missing),
  )
  if missing:
    _logger.info(
        "gpu_collective_overlap_pass [%s]: missing-profile instructions (first 150 of %d): %s",
        module_name,
        len(missing),
        ", ".join(missing[:150]),
    )


def _log_final_exposed_summary(module, schedule, module_name: str, label: str) -> None:
  """Log total and per-collective exposed time computed directly from the live
  schedule: ground truth, since intermediate totals go stale after a later
  schedule.update()/verify()."""
  comp_by_name = {c.name: c for c in module.computations()}
  per_comp = []
  per_collective = []  # (comp_name, collective_name, deficit_us)
  for comp in module.make_nonfusion_computations():
    seq = schedule.sequence(comp)
    if seq is None:
      continue
    seq = list(seq)
    positions = {inst: i for i, inst in enumerate(seq)}
    start_of_done: dict = {}
    for inst in seq:
      if _is_async_start(inst):
        users = list(inst.users())
        if len(users) == 1:
          start_of_done[users[0]] = inst
    if not start_of_done:
      continue

    prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    comp_total = 0.0
    for async_done, async_start in start_of_done.items():
      if async_start not in positions or async_done not in positions:
        continue
      latency = _profile_costs.get(_resolve_profile_key(async_start, comp_by_name))
      if latency is None or latency <= 0:
        continue
      s, d = positions[async_start], positions[async_done]
      overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
      deficit = max(0.0, latency - overlap)
      if deficit > 0:
        per_collective.append((comp.name, async_start.name, deficit))
        comp_total += deficit
    if comp_total > 0:
      per_comp.append((comp.name, comp_total))
  per_comp.sort(key=lambda x: -x[1])
  per_collective.sort(key=lambda x: -x[2])
  _logger.info(
      "gpu_collective_overlap_pass [%s]: FINAL exposed summary [%s]: grand_total=%.1f us across %d computation(s): %s",
      module_name,
      label,
      sum(t for _, t in per_comp),
      len(per_comp),
      ", ".join(f"{name}={total:.1f}us" for name, total in per_comp),
  )
  _logger.info(
      "gpu_collective_overlap_pass [%s]: FINAL per-collective exposed [%s]: %s",
      module_name,
      label,
      ", ".join(f"{comp_name}/{name}={deficit:.1f}us" for comp_name, name, deficit in per_collective),
  )
