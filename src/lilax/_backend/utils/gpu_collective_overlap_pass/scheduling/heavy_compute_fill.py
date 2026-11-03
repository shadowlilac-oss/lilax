
"""Filling still-exposed collective windows with heavy compute."""

# The fill loops are deeply nested by nature.
# pylint: disable=too-many-nested-blocks

from __future__ import annotations

import os

from lilax._backend.utils.gpu_collective_overlap_pass.costing.exposure import (
    _compute_position_margins, _prefix_costs_excluding,
)
from lilax._backend.utils.gpu_collective_overlap_pass.costing.instruction_cost import (
    _HEAVY_COMPUTE_MIN_US, _resolve_inst_cost,
)
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.async_ops import _ASYNC_COLLECTIVE_PREFIXES
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.custom_calls import _is_te_gemm_custom_call
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.movability import _is_trivially_movable_inst
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.profile_keys import _resolve_profile_key
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _profile_costs
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.blocker_relocation import (
    _MAX_PRODUCER_RELOCATE_HOPS, _chase_exposed_collective_blocker_earlier, _chase_heavy_compute_blocker_chain,
)
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.legality import _earliest_legal_pos
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.sequence import _commit_sequence, _reinsert
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.stuck_candidates import _StuckCandidateTracker

# Bound on hops when chasing a fill candidate's blocker chain via
# _try_relocate_blocker_earlier (net-benefit-gated per hop, though cross-hop
# error can still accumulate). COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS=0 disables.
_MAX_HEAVY_COMPUTE_CHASE_HOPS = int(os.environ.get("COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS", "8"))

# Outer fixed-point passes in _fill_exposed_collectives_with_heavy_compute: a
# fill or chase can expose a window that was hidden when the worklist was built.
_MAX_FILL_FIXED_POINT_ITERS = 3


def _fill_exposed_collectives_with_heavy_compute(
    seq: list,
    schedule,
    comp,
    start_of_done: dict,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    module_name: str,
    give_up_mode: str = "streak",
    is_while_body: bool = False,
) -> tuple[bool, list, dict, dict]:
  """Pull heavy compute backward into still-exposed collective windows.

  Outer fixed point (up to _MAX_FILL_FIXED_POINT_ITERS passes): each pass
  rebuilds the worklist from EVERY collective's fresh deficit, earliest start
  first, because filling or chasing one window can steal coverage from
  another that was already hidden. A window
  whose deficit is unchanged since the last pass was already scanned to
  exhaustion and is skipped. Stops when a pass changes nothing.

  For each window, scan forward from its `done` for a heavy instruction
  (cost >= _HEAVY_COMPUTE_MIN_US, not trivial) whose real dependencies
  (_earliest_legal_pos) already sit at or before `done`, and move it into the
  window. This never breaks the candidate's consumers (they already sit after
  its producer). Candidates are restricted to those whose cost fits within
  the slack of the collectives covering their position
  (_compute_position_margins), so a direct placement can never push another
  collective below its latency. A candidate whose floor lies past `done` goes
  to the chase fallbacks instead (see _chase_blocked).

  The deficit is recomputed from the real prefix sums on every iteration. An
  incremental "deficit -= cost" double-credited instructions that a chase
  displaced and then re-placed.
  """
  changed = False
  pass_changed = False
  chase_hops = 0
  # Full-range margins for the chase fallbacks (a blocker's position can be
  # anywhere, unlike the scan's done_pos+1.. queries). Computed lazily once per
  # while-iteration and discarded after a successful chase mutates positions.
  chase_margins = None
  stuck_tracker = None
  given_up_candidates: set = set()
  prev_deficit_by_name: dict[str, float] = {}

  def _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones) -> bool:
    """`inst`'s floor lies past the window's done: record the stall and try
    the two chase fallbacks. Returns True if a chase moved something (the
    caller must restart its scan; positions have shifted).

    Fallback 1 relocates the start of another still-exposed collective whose
    done pins the candidate. Fallback 2 chases an arbitrary movable heavy
    blocker chain (_chase_heavy_compute_blocker_chain).
    """
    nonlocal seq, positions, name_to_pos, changed, pass_changed, chase_hops, chase_margins
    streak, give_up_now = stuck_tracker.note_and_maybe_give_up(inst.name, floor)
    if give_up_now:
      given_up_candidates.add(inst.name)
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: window %s: giving up on %s after %d unresolved "
          "chase restarts (mode=%s, floor stuck at %d).",
          module_name,
          window["start"].name,
          inst.name,
          streak,
          give_up_mode,
          floor,
      )
    if blocker is not None and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
      moved, seq, positions, name_to_pos = _chase_exposed_collective_blocker_earlier(
          blocker,
          start_of_done,
          exposed_dones,
          start_pos,
          done_pos,
          seq,
          schedule,
          comp,
          positions,
          name_to_pos,
          comp_by_name,
      )
      if moved:
        chase_hops += 1
        changed = pass_changed = True
        _logger.info(
            "gpu_collective_overlap_pass [%s]: chased exposed-collective blocker %s earlier "
            "to unblock %s for window %s (deficit %.1f us).",
            module_name,
            start_of_done[blocker].name,
            inst.name,
            window["start"].name,
            window["deficit"],
        )
        return True
    if _MAX_HEAVY_COMPUTE_CHASE_HOPS > 0 and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
      if chase_margins is None:
        chase_margins = _compute_position_margins(
            start_of_done,
            positions,
            seq,
            comp_by_name,
        )
      moved, seq, positions, name_to_pos, _, _, _ = _chase_heavy_compute_blocker_chain(
          inst,
          comp,
          schedule,
          seq,
          positions,
          name_to_pos,
          comp_by_name,
          window["start"],
          done_pos,
          window["latency"],
          start_of_done,
          _MAX_HEAVY_COMPUTE_CHASE_HOPS,
          module_name,
          chase_margins,
          is_while_body=is_while_body,
      )
      if moved:
        chase_hops += 1
        changed = pass_changed = True
        return True
    return False

  for _fp_iter in range(_MAX_FILL_FIXED_POINT_ITERS):
    pass_changed = False

    windows: list[dict] = []
    cur_deficit_by_name: dict[str, float] = {}
    worklist_prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    for async_done, async_start in start_of_done.items():
      if async_start not in positions or async_done not in positions:
        continue
      profile_key = _resolve_profile_key(async_start, comp_by_name)
      latency = _profile_costs.get(profile_key)
      if latency is None or latency <= 0:
        continue
      start_pos = positions[async_start]
      done_pos = positions[async_done]
      overlap = worklist_prefix[done_pos] - worklist_prefix[start_pos + 1]
      deficit = latency - overlap
      cur_deficit_by_name[async_start.name] = deficit
      if deficit > 0:
        prev = prev_deficit_by_name.get(async_start.name)
        windows.append(
            {
                "start": async_start,
                "done": async_done,
                "deficit": deficit,
                "is_async": profile_key.startswith(_ASYNC_COLLECTIVE_PREFIXES),
                "latency": latency,
                "skip": _fp_iter > 0 and prev is not None and abs(prev - deficit) < 1e-6,
            }
        )

    prev_deficit_by_name = cur_deficit_by_name

    if not windows:
      break

    windows.sort(key=lambda w: positions[w["start"]])
    excluded = set(start_of_done.keys()) | set(start_of_done.values())
    # Only genuinely under-hidden collectives are chase targets, so
    # adequately overlapped ones aren't disturbed.
    exposed_dones = {w["done"] for w in windows}

    # Windows can overlap in time; an instruction placed in a shared
    # zone counts toward both, so such placements are preferred. Best-effort
    # snapshot per pass.
    sibling_ranges: dict[str, list[tuple[int, int]]] = {}
    for w in windows:
      s, d = positions[w["start"]], positions[w["done"]]
      for other in windows:
        if other is w:
          continue
        os_, od = positions[other["start"]], positions[other["done"]]
        if os_ < d and s < od:
          sibling_ranges.setdefault(w["start"].name, []).append((os_, od))

    for window in windows:
      if window["skip"]:
        continue
      window_siblings = sibling_ranges.get(window["start"].name, [])
      chase_hops = 0
      # One window is drained fully before the next.
      stuck_tracker = _StuckCandidateTracker(give_up_mode)
      given_up_candidates = set()
      while window["deficit"] > 0:
        done_pos = positions[window["done"]]
        start_pos = positions[window["start"]]
        iter_prefix = _prefix_costs_excluding(seq, (), comp_by_name)
        window["deficit"] = max(0.0, window["latency"] - (iter_prefix[done_pos] - iter_prefix[start_pos + 1]))
        if window["deficit"] <= 0:
          break
        # The scan only queries positions >= done_pos + 1.
        position_margins = _compute_position_margins(
            start_of_done,
            positions,
            seq,
            comp_by_name,
            prefix=iter_prefix,
            min_pos=done_pos + 1,
        )
        chase_margins = None
        candidate = None
        cand_floor = None
        cand_chain: list = []
        restart = False

        # async windows prefer te_gemm candidates: of all legally
        # reachable ones take the earliest floor (it depends on the
        # dependency chain, not scan order), so the GEMM overlaps more
        # of the transfer. Falls through to the unrestricted scan if
        # none is reachable.
        if window["is_async"]:
          best_floor = None
          for i in range(done_pos + 1, len(seq)):
            inst = seq[i]
            if inst in excluded or inst.name in given_up_candidates:
              continue
            if _is_trivially_movable_inst(inst, comp_by_name):
              continue
            if not _is_te_gemm_custom_call(inst, comp_by_name):
              continue
            cost = _resolve_inst_cost(inst, comp_by_name)
            if cost < _HEAVY_COMPUTE_MIN_US:
              continue
            if cost > position_margins.get(i, float("inf")):
              continue
            floor, chain, blocker = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
            if floor > done_pos:
              if _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones):
                restart = True
                break
              continue
            if best_floor is None or floor < best_floor:
              best_floor = floor
              candidate = inst
              cand_floor = floor
              cand_chain = chain
          if restart:
            continue

        if candidate is None:
          # Collect every legal candidate rather than taking the
          # first. Best fit is picked below.
          legal_candidates: list[tuple] = []
          for i in range(done_pos + 1, len(seq)):
            inst = seq[i]
            if inst in excluded or inst.name in given_up_candidates:
              continue
            if _is_trivially_movable_inst(inst, comp_by_name):
              continue
            cost = _resolve_inst_cost(inst, comp_by_name)
            if cost < _HEAVY_COMPUTE_MIN_US:
              continue
            if cost > position_margins.get(i, float("inf")):
              continue
            floor, chain, blocker = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
            if floor > done_pos:
              if _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones):
                restart = True
                break
              continue
            legal_candidates.append((inst, floor, chain, cost))
          if restart:
            continue

          if legal_candidates:

            # Only called within this iteration, so the loop-variable capture is safe.
            # pylint: disable=cell-var-from-loop
            def _lands_in_sibling(floor: int) -> bool:
              return any(s <= floor < d for s, d in window_siblings)

            sufficient = [c for c in legal_candidates if c[3] >= window["deficit"]]
            if sufficient:
              # Smallest candidate that closes the deficit,
              # preferring one landing in a sibling window.
              candidate, cand_floor, cand_chain, _ = min(
                  sufficient,
                  key=lambda c: (not _lands_in_sibling(c[1]), c[3]),
              )
            else:
              candidate, cand_floor, cand_chain, _ = max(
                  legal_candidates,
                  key=lambda c: (_lands_in_sibling(c[1]), c[3]),
              )

        if candidate is None:
          break

        prev_deficit = window["deficit"]
        target = max(cand_floor, start_pos + 1)
        seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, cand_chain, target, candidate))
        changed = pass_changed = True
        _logger.info(
            "gpu_collective_overlap_pass [%s]: relocated heavy compute %s (cost %.1f us) into "
            "window of %s (pre-relocation deficit %.1f us).",
            module_name,
            candidate.name,
            _resolve_inst_cost(candidate, comp_by_name),
            window["start"].name,
            prev_deficit,
        )
    if not pass_changed:
      break
  return changed, seq, positions, name_to_pos
