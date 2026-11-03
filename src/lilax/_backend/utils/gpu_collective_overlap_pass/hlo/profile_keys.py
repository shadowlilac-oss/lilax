
"""Resolution of the instruction name under which PGLE profiles an async op."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import (
    _CALLS_RE, _TO_APPLY_RE, _computation_root, _opcode_str,
)

_MAX_PROFILE_KEY_UNWRAP_HOPS = 8


def _unwrap_to_leaf_name(inst, comp_by_name: dict) -> str:
  """Follow nested call/async wrappers down to the instruction PGLE profiles.

  Stops at the first non-call/async op (it may simply have no profile entry,
  as for a trivial passthrough).
  """
  for _ in range(_MAX_PROFILE_KEY_UNWRAP_HOPS):
    opc = _opcode_str(inst)
    if opc not in ("call", "call-start", "async-start"):
      return inst.name
    try:
      text = inst.to_string()
    except Exception:
      return inst.name
    m = _TO_APPLY_RE.search(text) or _CALLS_RE.search(text)
    if not m:
      return inst.name
    comp = comp_by_name.get(m.group(1))
    if comp is None:
      return inst.name
    root = _computation_root(comp)
    if root is None:
      return inst.name
    inst = root
  return inst.name


def _resolve_profile_key(async_start, comp_by_name: dict) -> str:
  """Name under which async_start's latency is keyed in _profile_costs.

  Standard collectives expose their wrapped root via async_wrapped_root().
  te_ep call-starts use to_apply=%name instead (async_wrapped_root() fails or
  returns an intermediate "call.N"), so fall back to parsing it.
  """
  root = None
  try:
    root = async_start.async_wrapped_root()
  except Exception:
    root = None
  if root is None:
    try:
      text = async_start.to_string()
    except Exception:
      text = ""
    m = _TO_APPLY_RE.search(text)
    if m:
      comp = comp_by_name.get(m.group(1))
      if comp is not None:
        root = _computation_root(comp)
  if root is None:
    return async_start.name
  return _unwrap_to_leaf_name(root, comp_by_name)
