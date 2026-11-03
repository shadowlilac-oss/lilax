
"""PGLE FDO profile parsing and merging into the pooled per-instruction costs."""

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _profile_costs, _te_ep_overrides
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.protos import _ensure_protos


def _load_costs_from_fdo(fdo_bytes: bytes) -> dict[str, float]:
  _ensure_protos()
  from tsl.profiler.protobuf import profiled_instructions_pb2 as _pi  # type: ignore  # pylint: disable=import-outside-toplevel

  profiled = _pi.ProfiledInstructionsProto()
  profiled.ParseFromString(fdo_bytes)
  return {c.name: c.cost_us for c in profiled.costs}


def _update_profile(fdo_bytes: bytes) -> None:
  """Merge an FDO profile chunk into the pooled per-instruction costs."""
  if not fdo_bytes:
    return
  try:
    costs = _load_costs_from_fdo(fdo_bytes)
    if not costs:
      return
    _profile_costs.update(costs)
    # PGLE's own te_ep entries are underestimates; keep the live corrections.
    _profile_costs.update(_te_ep_overrides)
    n_coll = sum(1 for k in costs if "all-gather" in k or "reduce-scatter" in k)
    _logger.info(
        "gpu_collective_overlap_pass: merged PGLE profile chunk: %d entries (%d collectives); pool now %d.",
        len(costs),
        n_coll,
        len(_profile_costs),
    )
  except Exception as exc:  # pylint: disable=broad-except
    _logger.warning("gpu_collective_overlap_pass: failed to parse FDO profile: %s", exc)
