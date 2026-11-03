
"""Rank-local schedule rewrite of one serialized HLO module using the PGLE profile."""

from __future__ import annotations

from typing import Optional

from lilax._backend.utils.gpu_collective_overlap_pass.diagnostics import (
    _log_final_exposed_summary, _log_profile_coverage_gaps,
)
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.host_offload import _register_host_offload_copies
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _profile_costs
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.fixed_point import _reorder_to_fixed_point


def _compute_collective_overlap(serialized_hlo: bytes) -> Optional[bytes]:
  """Reorder the schedule using the PGLE profile. Runs on rank 0 only; see
  _gpu_collective_overlap_pass."""
  if not _profile_costs:
    return None

  from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel

  module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
  schedule = module.schedule()
  if schedule is None:
    return None

  _register_host_offload_copies(module)
  module_name = module.name

  _log_profile_coverage_gaps(module, schedule, module_name)

  if not _reorder_to_fixed_point(module, schedule, module_name):
    return None
  _log_final_exposed_summary(module, schedule, module_name, "final")
  return module.as_serialized_hlo_module_proto()
