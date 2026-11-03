
"""One reorder pass over every computation: move under-hidden async starts earlier."""

# The reorder loops are deeply nested by nature.
# pylint: disable=too-many-nested-blocks

from __future__ import annotations

import os

from lilax._backend.utils.gpu_collective_overlap_pass.costing.instruction_cost import _resolve_inst_cost
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.async_ops import _ASYNC_COLLECTIVE_PREFIXES, _is_async_start
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.profile_keys import _resolve_profile_key
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _profile_costs
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.async_start_hoist import _hoist_async_starts_to_floor
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.blocker_relocation import (
    _MAX_PRODUCER_RELOCATE_HOPS, _chase_async_start_blocker_toward_heavy_compute,
)
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.comm_stream import _ASYNC_STREAM_MODEL
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.computation_order import _innermost_first_computations
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.done_relocation import (
    _relocate_async_done_before_te_ep,
)
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.fill_strategy import _fill_exposed_collectives_best_of
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.legality import _earliest_legal_pos
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.sequence import _commit_sequence, _reinsert

# COLLECTIVE_OVERLAP_WHILE_BODY_ONLY=1 skips the entry computation, which is by
# far the largest sequence and dominates compile time.
# Trades its overlap away for faster compiles.
_WHILE_BODY_ONLY = os.environ.get("COLLECTIVE_OVERLAP_WHILE_BODY_ONLY") == "1"

# COLLECTIVE_OVERLAP_HOIST_ASYNC_STARTS=0 disables the final per-while-body step
# that moves async starts (and their trivial operand chain) toward their
# earliest legal position.
_HOIST_ASYNC_STARTS = os.environ.get("COLLECTIVE_OVERLAP_HOIST_ASYNC_STARTS", "1") != "0"


def _reorder_collectives(module, schedule, module_name: str) -> bool:
  """Move async collective starts earlier where latency is under-hidden.

  Returns True if the schedule changed.
  """
  changed = False
  # Includes fusion sub-computations, unlike make_nonfusion_computations().
  comp_by_name = {c.name: c for c in module.computations()}

  ordered_comps, while_body_comps = _innermost_first_computations(module, schedule)
  for comp in ordered_comps:
    is_while_body = comp in while_body_comps
    if _WHILE_BODY_ONLY and not is_while_body:
      continue
    seq = list(schedule.sequence(comp))

    start_of_done: dict = {}
    for inst in seq:
      if _is_async_start(inst):
        users = list(inst.users())
        if len(users) == 1:
          start_of_done[users[0]] = inst
        else:
          _logger.debug(
              "gpu_collective_overlap_pass [%s]: async-start %s has %d users (expected its done only); skipping.",
              module_name,
              inst.name,
              len(users),
          )

    if not start_of_done:
      continue

    positions = {inst: i for i, inst in enumerate(seq)}
    name_to_pos = {inst.name: i for inst, i in positions.items()}

    for async_done, async_start in start_of_done.items():
      profile_key = _resolve_profile_key(async_start, comp_by_name)
      collective_latency = _profile_costs.get(profile_key)
      if collective_latency is None or collective_latency <= 0:
        _logger.debug(
            "gpu_collective_overlap_pass [%s]: no profile entry for %s (resolved key: %s).",
            module_name,
            async_start.name,
            profile_key,
        )
        continue

      async_start_pos = positions[async_start]
      async_done_pos = positions[async_done]
      is_async = profile_key.startswith(_ASYNC_COLLECTIVE_PREFIXES)
      stream_owns_done = _ASYNC_STREAM_MODEL and is_while_body

      current_overlap = sum(_resolve_inst_cost(seq[i], comp_by_name) for i in range(async_start_pos + 1, async_done_pos))
      if current_overlap >= collective_latency:
        if is_async and not stream_owns_done:
          done_moved, seq, positions, name_to_pos = _relocate_async_done_before_te_ep(
              async_start,
              async_done,
              comp,
              schedule,
              seq,
              positions,
              name_to_pos,
              comp_by_name,
              collective_latency,
              module_name,
          )
          changed = changed or done_moved
        continue

      deficit = collective_latency - current_overlap
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: %s deficit=%.1f us (latency=%.1f us, overlap=%.1f us).",
          module_name,
          async_start.name,
          deficit,
          collective_latency,
          current_overlap,
      )

      # Always move all the way to the floor: earlier only adds headroom
      # and puts the collective ahead of heavy compute it doesn't depend on.
      floor, to_move, _ = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
      orig_async_start_pos = async_start_pos

      if is_async:
        # The floor stops at the first non-trivial producer; try moving
        # that producer out of the way too.
        (
            chased_blocker,
            seq,
            positions,
            name_to_pos,
            floor,
            to_move,
            _,
        ) = _chase_async_start_blocker_toward_heavy_compute(
            async_start,
            comp,
            schedule,
            seq,
            positions,
            name_to_pos,
            comp_by_name,
            _MAX_PRODUCER_RELOCATE_HOPS,
            module_name,
        )
        if chased_blocker:
          changed = True
          async_start_pos = positions[async_start]
          async_done_pos = positions[async_done]

      # The chain lands right before async_start, so its post-move position
      # is floor + len(to_move); comparing the bare floor would treat an
      # already-packed start as movable.
      if floor + len(to_move) >= async_start_pos:
        continue

      seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, to_move, floor, async_start))
      changed = True

      new_overlap = sum(
          _resolve_inst_cost(seq[i], comp_by_name) for i in range(positions[async_start] + 1, positions[async_done])
      )
      _logger.info(
          "gpu_collective_overlap_pass: moving %s from pos %d to %d with %d relocated operand(s) "
          "(overlap %.1f -> %.1f us, latency=%.1f us).",
          async_start.name,
          orig_async_start_pos,
          positions[async_start],
          len(to_move),
          current_overlap,
          new_overlap,
          collective_latency,
      )

      if new_overlap >= collective_latency and is_async and not stream_owns_done:
        done_moved, seq, positions, name_to_pos = _relocate_async_done_before_te_ep(
            async_start,
            async_done,
            comp,
            schedule,
            seq,
            positions,
            name_to_pos,
            comp_by_name,
            collective_latency,
            module_name,
        )
        changed = changed or done_moved

    heavy_changed, seq, positions, name_to_pos = _fill_exposed_collectives_best_of(
        seq,
        schedule,
        comp,
        start_of_done,
        positions,
        name_to_pos,
        comp_by_name,
        module_name,
        is_while_body=is_while_body,
    )
    changed = changed or heavy_changed

    if is_while_body and _HOIST_ASYNC_STARTS:
      hoist_changed, seq, positions, name_to_pos = _hoist_async_starts_to_floor(
          seq,
          schedule,
          comp,
          start_of_done,
          positions,
          name_to_pos,
          comp_by_name,
          module_name,
      )
      changed = changed or hoist_changed

  return changed
