
"""Reordering to a fixed point across schedule.update()/verify(), then host-offload hoisting."""

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling import fill_strategy
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.collective_reorder import _reorder_collectives
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.host_offload_hoist import (
    _hoist_host_offload_copy_starts,
)

# Cap on reorder re-runs against a freshly update()'d module. Normal runs
# converge in 1-2 iterations.
_MAX_REORDER_ITERS = 4


def _reorder_to_fixed_point(module, schedule, module_name: str) -> bool:
  """Run _reorder_collectives to a fixed point; returns True if anything changed.

  schedule.update()/verify() (needed so XLA can canonicalize the schedule after
  raw seq mutations) can itself relocate an instruction whose dependency one
  of our earlier moves invalidated, without _reorder_collectives's bookkeeping
  noticing, so a window scored hidden can end up exposed. Re-running on the
  post-update schedule re-fills it.
  """
  fill_strategy._fill_strategy_cache = {}  # pylint: disable=protected-access
  changed = False
  for _ in range(_MAX_REORDER_ITERS):
    if not _reorder_collectives(module, schedule, module_name):
      break
    changed = True
    schedule.update()
    schedule.verify()
    module.set_schedule(schedule)
  else:
    # Cap reached: the last update() was never rechecked, so run one more pass.
    _logger.warning(
        "gpu_collective_overlap_pass [%s]: reorder fixed point not reached after %d iterations; "
        "running one final pass.",
        module_name,
        _MAX_REORDER_ITERS,
    )
    if _reorder_collectives(module, schedule, module_name):
      changed = True
      schedule.update()
      schedule.verify()
      module.set_schedule(schedule)
  if _hoist_host_offload_copy_starts(module, schedule, module_name):
    changed = True
    schedule.update()
    schedule.verify()
    module.set_schedule(schedule)
  return changed
