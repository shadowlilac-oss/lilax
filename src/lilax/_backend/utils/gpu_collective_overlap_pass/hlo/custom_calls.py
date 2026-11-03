
"""Classification of heavy GEMM/cuDNN/cuBLAS custom-call kernels."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

import re

from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _CALLS_RE, _opcode_str

_TE_GEMM_TARGET_RE = re.compile(r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*)"')
_HEAVY_ANCHOR_TARGET_RE = re.compile(
    r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*|[^"]*cudnn[^"]*|[^"]*cublas[^"]*)"'
)


def _wraps_custom_call(inst, target_re, comp_by_name: dict) -> bool:
  """True if inst is a custom-call matching target_re, or a kind=kCustom
  fusion whose nested computation holds one (such a fusion never gets its own
  PGLE entry; the kernel is tagged with the inner custom-call's name)."""
  opc = _opcode_str(inst)
  try:
    text = inst.to_string()
  except Exception:
    return False
  if opc == "custom-call":
    return bool(target_re.search(text))
  if opc == "fusion" and "kind=kCustom" in text:
    m = _CALLS_RE.search(text)
    if not m:
      return False
    comp = comp_by_name.get(m.group(1))
    if comp is None:
      return False
    try:
      return any(_opcode_str(sub) == "custom-call" and target_re.search(sub.to_string()) for sub in comp.instructions())
    except Exception:
      return False
  return False


def _is_te_gemm_custom_call(inst, comp_by_name: dict) -> bool:
  """te_grouped_gemm/te_gemm kernel (the MoE GEMMs that dominate compute);
  prioritized when filling async collective windows."""
  return _wraps_custom_call(inst, _TE_GEMM_TARGET_RE, comp_by_name)


def _is_heavy_anchor_custom_call(inst, comp_by_name: dict) -> bool:
  """te_gemm/cudnn/cublas kernel: where _chase_async_start_blocker_toward_heavy_compute
  stops. Broader than _is_te_gemm_custom_call, kept separate so this chase
  path doesn't change the validated fill priority."""
  return _wraps_custom_call(inst, _HEAVY_ANCHOR_TARGET_RE, comp_by_name)
