
"""Per-computation selection and caching of the best heavy-compute fill strategy."""

from __future__ import annotations

from typing import Optional

from lilax._backend.utils.gpu_collective_overlap_pass.costing.exposure import _total_exposed_us
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.heavy_compute_fill import (
    _fill_exposed_collectives_with_heavy_compute,
)
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.stuck_candidates import _GIVE_UP_MODES

# Which _GIVE_UP_MODES strategy won per computation, so the ~3x-cost comparison
# runs once per _reorder_to_fixed_point call rather than on every re-entry
# Cleared at the start of each _reorder_to_fixed_point call.
_fill_strategy_cache: dict[str, str] = {}


def _fill_exposed_collectives_best_of(
    seq: list,
    schedule,
    comp,
    start_of_done: dict,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    module_name: str,
    is_while_body: bool = False,
) -> tuple[bool, list, dict, dict]:
  """Run the fill with each _GIVE_UP_MODES strategy from the same baseline and
  keep the one with the lowest _total_exposed_us.

  No strategy dominates: "streak" wins on small while bodies,
  "floor_aware" on the large entry computation. Re-entries for
  a computation replay the cached winner with a single trial.
  """
  cached_mode = _fill_strategy_cache.get(comp.name)
  if cached_mode is not None:
    pre_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
    changed, seq, positions, name_to_pos = _fill_exposed_collectives_with_heavy_compute(
        list(seq),
        schedule,
        comp,
        start_of_done,
        dict(positions),
        dict(name_to_pos),
        comp_by_name,
        module_name,
        give_up_mode=cached_mode,
        is_while_body=is_while_body,
    )
    _logger.info(
        "gpu_collective_overlap_pass [%s]: comp %s: cached fill strategy '%s': total exposed "
        "%.1f -> %.1f us (changed=%s).",
        module_name,
        comp.name,
        cached_mode,
        pre_total,
        _total_exposed_us(start_of_done, positions, seq, comp_by_name),
        changed,
    )
    return changed, seq, positions, name_to_pos

  baseline_seq = list(seq)
  baseline_positions = dict(positions)
  baseline_name_to_pos = dict(name_to_pos)

  best: Optional[tuple] = None  # (total_exposed, changed, seq, positions, name_to_pos, mode)
  for mode in _GIVE_UP_MODES:
    schedule.set_sequence(comp, list(baseline_seq))
    trial_changed, trial_seq, trial_positions, trial_name_to_pos = _fill_exposed_collectives_with_heavy_compute(
        list(baseline_seq),
        schedule,
        comp,
        start_of_done,
        dict(baseline_positions),
        dict(baseline_name_to_pos),
        comp_by_name,
        module_name,
        give_up_mode=mode,
        is_while_body=is_while_body,
    )
    total = _total_exposed_us(start_of_done, trial_positions, trial_seq, comp_by_name)
    _logger.info(
        "gpu_collective_overlap_pass [%s]: fill strategy '%s' for comp %s: total exposed %.1f us (changed=%s).",
        module_name,
        mode,
        comp.name,
        total,
        trial_changed,
    )
    if best is None or total < best[0] - 1e-6:  # pylint: disable=unsubscriptable-object
      best = (total, trial_changed, trial_seq, trial_positions, trial_name_to_pos, mode)

  total, changed, seq, positions, name_to_pos, mode = best
  schedule.set_sequence(comp, seq)
  _fill_strategy_cache[comp.name] = mode
  _logger.info(
      "gpu_collective_overlap_pass [%s]: comp %s: selected fill strategy '%s' (total exposed %.1f us).",
      module_name,
      comp.name,
      mode,
      total,
  )
  return changed, seq, positions, name_to_pos
