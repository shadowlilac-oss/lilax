
"""Relocating and chasing the blockers that pin a collective's or candidate's floor."""

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.costing.exposure import (
    _compute_position_margins, _prefix_costs_excluding, _total_exposed_us,
)
from lilax._backend.utils.gpu_collective_overlap_pass.costing.instruction_cost import (
    _HEAVY_COMPUTE_MIN_US, _resolve_inst_cost,
)
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.custom_calls import _is_heavy_anchor_custom_call
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.profile_keys import _resolve_profile_key
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _profile_costs
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.legality import _earliest_legal_pos
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.sequence import _commit_sequence, _reinsert

# Bound on "blocker's own blocker" hops per chase. The chain is meant to be
# walked to the computation's inputs; each hop self-terminates when a blocker has
# no legal room, so this only guards against a pathological chain.
_MAX_PRODUCER_RELOCATE_HOPS = 64


def _find_best_blocker_position(
    blocker,
    chain: list,
    floor: int,
    blocker_pos: int,
    async_start,
    async_done_pos: int,
    async_latency: float,
    start_of_done: dict,
    positions: dict,
    seq: list,
    comp_by_name: dict,
) -> tuple[int, float]:
  """Position in [floor, blocker_pos] that minimizes TOTAL exposed time over
  every collective in the computation when `blocker` (and its trivial chain)
  moves there.

  Moving blocker to P changes async_start's own overlap (its new floor is P +
  len(chain) + 1) and, for every other window, removes blocker's cost if it
  currently sits inside and wouldn't at P, or adds it in the reverse case.
  Chain members are trivial (~0 cost) and ignored. This is an approximation
  (pre-move positions decide window containment) but cheap, and it catches
  the dominant effect of stealing overlap from already-hidden windows.

  Candidates are blocker's *landing* positions, so "leave it" is unambiguously
  new_pos == blocker_pos (evaluating in insertion-start units made the no-op
  baseline depend on len(chain) and allowed oscillation between two blockers).
  Ties prefer the position closest to blocker_pos.

  Returns (best insertion-start position, best total exposed us). A result of
  blocker_pos - len(chain) means don't move.
  """
  prefix = _prefix_costs_excluding(seq, set(chain) | {blocker}, comp_by_name)
  block_cost = _resolve_inst_cost(blocker, comp_by_name)

  other_windows = []  # (start_pos, done_pos, latency, overlap excluding blocker)
  for od, os_ in start_of_done.items():
    if os_ is async_start or os_ not in positions or od not in positions:
      continue
    latency2 = _profile_costs.get(_resolve_profile_key(os_, comp_by_name))
    if latency2 is None or latency2 <= 0:
      continue
    s2, d2 = positions[os_], positions[od]
    base_overlap = prefix[d2] - prefix[s2 + 1] if d2 > s2 + 1 else 0.0
    other_windows.append((s2, d2, latency2, base_overlap))

  lo = floor + len(chain)
  new_pos_candidates = {lo, blocker_pos}
  for s2, d2, _, _ in other_windows:
    if lo <= s2 < blocker_pos:
      new_pos_candidates.add(s2 + 1)
    if lo < d2 <= blocker_pos:
      new_pos_candidates.add(d2)
  new_pos_candidates = sorted(p for p in new_pos_candidates if lo <= p <= blocker_pos)

  def total_exposed(blocker_new_pos: int) -> float:
    async_new_start = blocker_new_pos + 1
    async_overlap = prefix[async_done_pos] - prefix[async_new_start] if async_done_pos > async_new_start else 0.0
    total = max(0.0, async_latency - async_overlap)
    for s2, d2, latency2, base_overlap in other_windows:
      contains_new = s2 < blocker_new_pos < d2
      overlap2 = base_overlap + (block_cost if contains_new else 0.0)
      total += max(0.0, latency2 - overlap2)
    return total

  best_new_pos, best_exposed = blocker_pos, None
  for new_pos in new_pos_candidates:
    e = total_exposed(new_pos)
    if best_exposed is None or e < best_exposed - 1e-6 or (abs(e - best_exposed) <= 1e-6 and new_pos > best_new_pos):
      best_exposed = e
      best_new_pos = new_pos
  return best_new_pos - len(chain), best_exposed


def _try_relocate_blocker_earlier(
    blocker,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    async_start,
    async_done_pos: int,
    async_latency: float,
    start_of_done: dict,
    module_name: str = "",
    position_margins: dict | None = None,
):
  """Relocate `blocker` (a non-trivial instruction pinning some collective's
  floor) and its trivial chain earlier, if it is heavy and one of two gates
  passes:

  1. Strict: the best position in [floor, current] strictly lowers TOTAL
     exposed time (see _find_best_blocker_position).
  2. Margin-safe fallback (only when `position_margins` is given, i.e. from
     the heavy-compute chase): move straight to the floor if blocker's cost
     fits within the slack of every collective covering its position. Path 1
     is myopic across a chain: moving a small op a few positions looks like a
     wash even when it is a necessary link to a candidate several hops away.
     Path 2 can never push a fine collective below its latency.

  Moving a producer earlier never breaks its consumers; the risk is stealing
  overlap another collective relied on, which both gates guard against.

  Returns the new (seq, positions, name_to_pos) if a move happened, else None.
  """
  if blocker not in positions:
    return None
  cost = _resolve_inst_cost(blocker, comp_by_name)
  if cost < _HEAVY_COMPUTE_MIN_US:
    return None
  floor, chain, _ = _earliest_legal_pos(blocker, positions, name_to_pos, comp_by_name)
  blocker_pos = positions[blocker]
  # The chain lands right before blocker, so its final position is
  # floor + len(chain). Comparing against the bare floor would report room to
  # move for a blocker already packed after its chain and loop forever.
  final_pos = floor + len(chain)
  if final_pos >= blocker_pos:
    return None

  best_pos, best_exposed = _find_best_blocker_position(
      blocker,
      chain,
      floor,
      blocker_pos,
      async_start,
      async_done_pos,
      async_latency,
      start_of_done,
      positions,
      seq,
      comp_by_name,
  )
  current_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
  improves_total = best_pos < blocker_pos - len(chain) and best_exposed < current_total - 1e-6

  target = None
  if improves_total:
    target = best_pos
    _logger.debug(
        "gpu_collective_overlap_pass [%s]: TRC %s: pos %d -> %d (chain %d), total exposed %.1f -> %.1f us (path 1).",
        module_name,
        blocker.name,
        blocker_pos,
        best_pos + len(chain),
        len(chain),
        current_total,
        best_exposed,
    )
  elif position_margins is not None:
    margin = position_margins.get(blocker_pos, float("inf"))
    if cost <= margin:
      target = floor
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: TRC %s: pos %d -> %d (chain %d), margin-safe "
          "(cost %.1f <= margin %.1f us) (path 2).",
          module_name,
          blocker.name,
          blocker_pos,
          floor + len(chain),
          len(chain),
          cost,
          margin,
      )

  if target is None:
    return None
  return _commit_sequence(schedule, comp, _reinsert(seq, chain, target, blocker))


def _chase_async_start_blocker_toward_heavy_compute(
    async_start,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    max_hops: int,
    module_name: str = "",
):
  """async-only: when async_start has no earlier legal position because its direct
  non-trivial producer (`blocker`) sits right next to it, relocate that
  blocker earlier too, repeating until the blocker is a heavy anchor
  (_is_heavy_anchor_custom_call) or has no room.

  Ungated (no net-exposure or margin check): an earlier gated version existed
  to protect other windows, but async operand producers (quantize/dbias ops)
  are specific to that collective. If validation shows another collective
  regressing, the gate needs to come back.

  Returns (changed, seq, positions, name_to_pos, floor, to_move, blocker),
  with floor/to_move/blocker being async_start's current _earliest_legal_pos.
  """
  changed = False
  floor, to_move, blocker = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
  hops = 0
  while blocker is not None and not _is_heavy_anchor_custom_call(blocker, comp_by_name) and hops < max_hops:
    b_floor, b_chain, _ = _earliest_legal_pos(blocker, positions, name_to_pos, comp_by_name)
    blocker_pos = positions[blocker]
    if b_floor + len(b_chain) >= blocker_pos:
      break
    seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, b_chain, b_floor, blocker))
    changed = True
    hops += 1
    floor, to_move, blocker = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
  return changed, seq, positions, name_to_pos, floor, to_move, blocker


def _chase_exposed_collective_blocker_earlier(
    blocker_done,
    start_of_done: dict,
    exposed_dones: set,
    window_start_pos: int,
    window_done_pos: int,
    seq: list,
    schedule,
    comp,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
):
  """If `blocker_done` is the done of ANOTHER still-exposed collective, try to
  move that collective's start earlier, so it sits between the caller's window
  and the heavy-compute candidate it is blocking.

  Narrower than the general recursive blocker chase it fires only for a
  real dependency on another exposed collective's done. The move is capped at
  `window_done_pos` and bounded by the chased start's own legal floor.

  Returns (changed, seq, positions, name_to_pos); unchanged if no move.
  """
  if blocker_done not in exposed_dones:
    return False, seq, positions, name_to_pos
  chase_start = start_of_done.get(blocker_done)
  if chase_start is None or chase_start not in positions:
    return False, seq, positions, name_to_pos
  chase_start_pos = positions[chase_start]
  if chase_start_pos <= window_start_pos:
    return False, seq, positions, name_to_pos
  chase_floor, chase_chain, _ = _earliest_legal_pos(chase_start, positions, name_to_pos, comp_by_name)
  target = min(max(chase_floor, window_start_pos + 1), window_done_pos)
  if target + len(chase_chain) >= chase_start_pos:
    return False, seq, positions, name_to_pos

  new_seq, new_positions, new_name_to_pos = _commit_sequence(
      schedule, comp, _reinsert(seq, chase_chain, target, chase_start)
  )
  return True, new_seq, new_positions, new_name_to_pos


def _find_relocatable_ancestor(
    inst,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    max_depth: int,
    skip_cheap_ancestors: bool = False,
):
  """Walk `inst`'s blocker chain to the first ancestor with structural room to
  move (its floor + chain lands before its current position).

  Purely structural unless skip_cheap_ancestors is set; the result must still
  pass _try_relocate_blocker_earlier's gating before it is moved.

  skip_cheap_ancestors also requires cost >= _HEAVY_COMPUTE_MIN_US, so the
  walk passes over near-zero-cost connector ops (dynamic-slice/select
  fusions, a collective's own call-start/done) that the cost gate would
  refuse anyway.

  Returns the ancestor (possibly `inst`), or None if the chain bottoms out,
  cycles or exhausts max_depth.
  """
  seen = set()
  cur = inst
  for _ in range(max_depth + 1):
    if cur is None or cur in seen or cur not in positions:
      return None
    seen.add(cur)
    floor, chain, sub_blocker = _earliest_legal_pos(cur, positions, name_to_pos, comp_by_name)
    has_room = floor + len(chain) < positions[cur]
    if has_room and (not skip_cheap_ancestors or _resolve_inst_cost(cur, comp_by_name) >= _HEAVY_COMPUTE_MIN_US):
      return cur
    cur = sub_blocker
  return None


def _chase_heavy_compute_blocker_chain(
    candidate,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    async_start,
    async_done_pos: int,
    async_latency: float,
    start_of_done: dict,
    max_hops: int,
    module_name: str = "",
    position_margins: dict | None = None,
    is_while_body: bool = False,
):
  """Relocate whatever blocks `candidate` from reaching a legal position in
  window [async_start, async_done_pos), up to `max_hops` times.

  The blocker can be any movable heavy op, e.g. dot_product_attention_fwd
  blocked by a GEMM blocked by another GEMM.
  Each hop descends via _find_relocatable_ancestor to an op that can move, and
  hands it to _try_relocate_blocker_earlier, the sole committer. Reaching an
  indirect blocker typically takes two hops: the deep ancestor first, then the
  now-movable direct blocker on the next iteration.

  `position_margins` is forwarded to enable the margin-safe path, and
  recomputed after every committed hop so several hops can't cumulatively
  draw more than a window's slack.

  Stops on a refused hop, no relocatable ancestor, max_hops, or once the
  candidate's floor is resolved.

  Returns (changed, seq, positions, name_to_pos, floor, chain, blocker),
  with floor/chain/blocker being candidate's current _earliest_legal_pos.
  """
  changed = False
  floor, chain, blocker = _earliest_legal_pos(candidate, positions, name_to_pos, comp_by_name)
  hops = 0
  while floor > async_done_pos and blocker is not None and hops < max_hops:
    target = _find_relocatable_ancestor(
        blocker,
        positions,
        name_to_pos,
        comp_by_name,
        max_hops - hops,
        skip_cheap_ancestors=is_while_body,
    )
    if target is None:
      break
    result = _try_relocate_blocker_earlier(
        target,
        comp,
        schedule,
        seq,
        positions,
        name_to_pos,
        comp_by_name,
        async_start,
        async_done_pos,
        async_latency,
        start_of_done,
        module_name,
        position_margins,
    )
    hops += 1
    if result is None:
      break
    seq, positions, name_to_pos = result
    changed = True
    if position_margins is not None:
      position_margins = _compute_position_margins(start_of_done, positions, seq, comp_by_name)
    floor, chain, blocker = _earliest_legal_pos(candidate, positions, name_to_pos, comp_by_name)
  _logger.debug(
      "gpu_collective_overlap_pass [%s]: CHC %s: %s after %d/%d hop(s), floor=%d, done_pos=%d.",
      module_name,
      candidate.name,
      "resolved" if floor <= async_done_pos else "gave up",
      hops,
      max_hops,
      floor,
      async_done_pos,
  )
  return changed, seq, positions, name_to_pos, floor, chain, blocker
