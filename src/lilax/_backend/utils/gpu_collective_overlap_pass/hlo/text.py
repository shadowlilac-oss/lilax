
"""HLO text helpers: opcode names, called computations, roots and control-predecessors."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# HLO text helpers (the HloInstruction binding exposes little beyond name/opcode/
# operands/users/to_string, so the rest is parsed from to_string())
# ---------------------------------------------------------------------------
_CALLS_RE = re.compile(r"calls=%([A-Za-z0-9_.]+)")
_TO_APPLY_RE = re.compile(r"to_apply=%([A-Za-z0-9_.]+)")
_ROOT_RE = re.compile(r"\bROOT %([A-Za-z0-9_.\-]+) = ")
_CONTROL_PRED_RE = re.compile(r"control-predecessors=\{([^}]*)\}")


def _opcode_str(inst) -> str:
  """Hyphenated opcode string (e.g. "get-tuple-element") for an HloInstruction."""
  return re.sub(r"(?<!^)(?=[A-Z])", "-", inst.opcode.name[1:]).lower()


# TODO tixxx, remove this once control_predecessors is exposed through python binding.
def _control_predecessor_names(inst) -> list[str]:
  try:
    text = inst.to_string()
  except Exception:
    return []
  m = _CONTROL_PRED_RE.search(text)
  if not m:
    return []
  return [n.strip().lstrip("%") for n in m.group(1).split(",") if n.strip()]


# TODO tixxx, remove this once root_instruciton is exposed through python binding.
def _computation_root(comp):
  """comp's ROOT instruction"""
  try:
    return comp.root_instruction()
  except Exception:
    pass
  try:
    m = _ROOT_RE.search(comp.to_string())
    if m:
      for inst in comp.instructions():
        if inst.name == m.group(1):
          return inst
  except Exception:
    pass
  try:
    for inst in comp.instructions():
      if not list(inst.users()):
        return inst
  except Exception:
    pass
  return None
