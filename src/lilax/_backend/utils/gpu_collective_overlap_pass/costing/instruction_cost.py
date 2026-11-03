
"""Profiled per-instruction cost resolution."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.host_offload import _host_offload_copy_names
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _CALLS_RE, _opcode_str
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _gemm_overrides, _profile_costs

# Minimum profiled cost (us) for an instruction to count as heavy compute.
_HEAVY_COMPUTE_MIN_US = 5.0


def _resolve_inst_cost(inst, comp_by_name: dict) -> float:
  """Profiled cost (us) of inst.

  A kind=kCustom fusion (e.g. dynamic-slice-fusion around te_gemm_v2_ffi plus
  a dynamic-update-slice) has no entry of its own, so sum the entries of
  every instruction in its nested computation.
  """
  if inst.name in _host_offload_copy_names:
    return 0.0  # DMA, not SM compute: hides nothing
  direct = _profile_costs.get(inst.name)
  if direct:
    return direct
  if _opcode_str(inst) != "fusion":
    return 0.0
  try:
    text = inst.to_string()
  except Exception:
    return 0.0
  if "kind=kCustom" not in text:
    return 0.0
  m = _CALLS_RE.search(text)
  if not m:
    return 0.0
  comp = comp_by_name.get(m.group(1))
  if comp is None:
    return 0.0
  try:
    return sum(_profile_costs.get(sub.name, 0.0) for sub in comp.instructions())
  except Exception:
    return 0.0


def _sim_inst_cost(inst, comp_by_name: dict) -> float:
  """_resolve_inst_cost, but with the per-invocation GEMM costs (_gemm_overrides)."""
  if not _gemm_overrides:
    return _resolve_inst_cost(inst, comp_by_name)
  direct = _gemm_overrides.get(inst.name)
  if direct:
    return direct
  if _opcode_str(inst) == "fusion":
    try:
      text = inst.to_string()
      if "kind=kCustom" in text:
        m = _CALLS_RE.search(text)
        comp = comp_by_name.get(m.group(1)) if m else None
        if comp is not None:
          total = sum(_gemm_overrides.get(sub.name) or _profile_costs.get(sub.name, 0.0) for sub in comp.instructions())
          if total:
            return total
    except Exception:
      pass
  return _resolve_inst_cost(inst, comp_by_name)
