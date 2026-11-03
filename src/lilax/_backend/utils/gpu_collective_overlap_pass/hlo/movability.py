
"""Classification of instructions that can be relocated without moving real compute."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _CALLS_RE, _opcode_str

# "parameter"/"constant" are leaves with no real dependency, so always movable.
_TRIVIAL_SINGLE_OPCODES = frozenset(
    {
        "bitcast",
        "convert",
        "transpose",
        "reshape",
        "broadcast",
        "get-tuple-element",
        "tuple",
        "copy",
        "slice",
        "parameter",
        "constant",
    }
)

_TRIVIAL_FUSED_OPCODES = frozenset(
    {
        "parameter",
        "constant",
        "iota",
        "convert",
        "bitcast",
        "reshape",
        "transpose",
        "broadcast",
        "copy",
        "concatenate",
        "dynamic-slice",
        "slice",
        "get-tuple-element",
        "tuple",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "abs",
        "maximum",
        "minimum",
        "and",
        "or",
        "not",
        "xor",
        "sqrt",
        "rsqrt",
        "exp",
        "log",
        "sign",
        "clamp",
        "floor",
        "ceil",
        "round-nearest-afz",
        "round-nearest-even",
        "compare",
        "select",
    }
)


def _is_trivial_fusion_body(inst, comp_by_name: dict) -> bool:
  """True if every op in inst's fused computation is trivial."""
  try:
    text = inst.to_string()
  except Exception:
    return False
  m = _CALLS_RE.search(text)
  if not m:
    return False
  comp = comp_by_name.get(m.group(1))
  if comp is None:
    return False
  try:
    for fi in comp.instructions():
      if _opcode_str(fi) not in _TRIVIAL_FUSED_OPCODES:
        return False
    return True
  except Exception:
    return False


def _is_trivially_movable_inst(inst, comp_by_name: dict) -> bool:
  """True if inst can be relocated without moving real compute."""
  opc = _opcode_str(inst)
  if opc in _TRIVIAL_SINGLE_OPCODES:
    return True
  if opc == "fusion":
    return _is_trivial_fusion_body(inst, comp_by_name)
  return False
