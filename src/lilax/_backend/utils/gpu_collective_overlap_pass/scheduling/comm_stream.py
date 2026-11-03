
"""Comm-stream model: simulated stalls and stall-minimizing async start/done placement."""

from __future__ import annotations

import bisect
import os

from lilax._backend.utils.gpu_collective_overlap_pass.costing.instruction_cost import _HEAVY_COMPUTE_MIN_US

# COLLECTIVE_OVERLAP_ASYNC_STREAM_MODEL=0 disables the comm-stream model that
# places async starts/dones in the hoist step (_best_async_placement). While on,
# it also owns async done placement in while bodies and
# _relocate_async_done_before_te_ep is skipped there.
_ASYNC_STREAM_MODEL = os.environ.get("COLLECTIVE_OVERLAP_ASYNC_STREAM_MODEL", "1") != "0"
# Simulated totals within this many us of the best are treated as equal when
# placing an async start, so the earliest start (largest window) wins. Without
# it a ~30 us modeled gain moved collective starts next to their dones.
_ASYNC_STREAM_TOL_US = float(os.environ.get("COLLECTIVE_OVERLAP_ASYNC_STREAM_TOL_US", "250"))
# SM gating: a launched heavy GEMM holds the SMs, so a collective kernel can't
# begin until it ends. COLLECTIVE_OVERLAP_SM_GATE=0 disables; GEMMs shorter than
# COLLECTIVE_OVERLAP_SM_GATE_MIN_US are ignored.
_SM_GATE = os.environ.get("COLLECTIVE_OVERLAP_SM_GATE", "1") != "0"
_SM_GATE_MIN_US = float(os.environ.get("COLLECTIVE_OVERLAP_SM_GATE_MIN_US", "1000"))


def _simulate_comm_stream(order, cost: dict, lat: dict, done_start: dict, gate=None):
  """Simulate compute (in `order`) against one FIFO communication stream.

  Compute ops run back to back at their cost. Every async start is queued on
  the comm stream and begins at the later of its issue time and the end of the
  previous queued op; an async done stalls compute until its start has
  finished. Returns (total stall us, {done: stall us}).

  NCCL collectives and custom comm calls share one stream, so a
  collective issued behind a custom comm call waits for it and vice versa; the
  static window cost used elsewhere sees neither effect.

  `gate` is an optional set of SM-saturating GEMMs: a collective can't begin
  while one is running. The GEMM timeline depends on the stalls, which depend
  on the collective timeline, so it is iterated from the previous pass's GEMM
  intervals (a few passes converge).
  """
  intervals: list = []
  total = 0.0
  stalls: dict = {}
  for _ in range(3 if gate else 1):
    t = 0.0
    comm = 0.0
    end: dict = {}
    total = 0.0
    stalls = {}
    found: list = []
    for inst in order:
      start = done_start.get(inst)
      if start is not None:
        e = end.get(start)
        if e is not None and e > t:
          stalls[inst] = e - t
          total += e - t
          t = e
      elif inst in lat:
        begin = t if t > comm else comm
        for gs, ge in intervals:
          if gs < begin < ge:
            begin = ge
        comm = begin + lat[inst]
        end[inst] = comm
      else:
        c = cost.get(inst, 0.0)
        if gate and inst in gate:
          found.append((t, t + c))
        t += c
    if not gate or found == intervals:
      break
    intervals = found
  return total, stalls


def _best_async_placement(
    async_start,
    async_done,
    floor,
    to_move,
    seq,
    positions,
    cost,
    lat,
    done_start,
    ctrl_succs,
    protect=None,
    gate=None,
    diag=None,
):
  """Choose where to put an async start (and its done) to minimize simulated stalls.

  Candidates are insertion points for the start between its earliest legal
  position and the first consumer of its done. The done may stay put or move
  to just before its first consumer, so a start placed after its current done
  shifts the done along with it. Topological order is kept: the start after
  its operands and control-predecessors, the done after the start and before
  its consumers and control-successors. A start moving earlier brings its
  trivial operand chain (`to_move`); one moving later leaves it.

  Candidates within _ASYNC_STREAM_TOL_US of the best simulated total count as
  equal; among them the earliest start wins, then the latest done (largest
  window; a later done never adds a stall in the sim).

  `protect` is an optional set of dones (the async-class ones): a candidate is
  dropped if it raises their combined stall by more than the tolerance, so
  trading one async collective's exposure for another's or a te_ep stall never
  counts as a gain.

  Returns ((total, done_moved, base_idx), moved_chain, start_idx, done_idx, base)
  or None; indices are into `base` (seq without the moved instructions).
  """
  start_pos = positions[async_start]
  done_pos = positions[async_done]
  consumers = [positions[u] for u in async_done.users() if u in positions]
  consumers += [positions[x] for x in ctrl_succs.get(async_done.name, []) if x in positions]
  if not consumers:
    return None
  user_pos = min(consumers)
  p_max = min(
      [user_pos] + [positions[x] for x in ctrl_succs.get(async_start.name, []) if x in positions and x is not async_done]
  )
  cands = []
  protected_now = 0.0
  if protect:
    _, stalls_now = _simulate_comm_stream(seq, cost, lat, done_start, gate)
    protected_now = sum(stalls_now.get(d, 0.0) for d in protect)
  chain = sorted(to_move, key=lambda x: positions[x])
  chain_pos = [positions[x] for x in chain]
  prepared: dict = {}

  def prepare(k):
    # Variant k moves chain[k:] with the start; chain[:k] already sit before
    # the insertion point and stay, so nothing moves later than it is now
    # (other users of a chain member may sit in between).
    if k not in prepared:
      rem = {async_start, async_done} | set(chain[k:])
      removed_before = [0] * (len(seq) + 1)
      count = 0
      for i, inst in enumerate(seq):
        removed_before[i] = count
        if inst in rem:
          count += 1
      removed_before[len(seq)] = count
      prepared[k] = ([inst for inst in seq if inst not in rem], removed_before)
    return prepared[k]

  for early in (True, False):
    lo, hi = (floor, start_pos) if early else (start_pos + 1, p_max)
    hi = min(hi, p_max)
    for i in range(lo, hi + 1):
      # Interior points only matter next to heavy compute or an async op.
      if i < len(seq) and i not in (lo, hi, start_pos):
        inst = seq[i]
        if not (cost.get(inst, 0.0) >= _HEAVY_COMPUTE_MIN_US or inst in lat or inst in done_start):
          continue
      k = bisect.bisect_left(chain_pos, i) if early else len(chain)
      base, removed_before = prepare(k)
      moved = chain[k:] if early else []
      q_latest = user_pos - removed_before[user_pos]
      q_cur = done_pos - removed_before[done_pos]
      pb = i - removed_before[i] if i < len(seq) else len(base)
      for q in sorted({q_latest, q_cur}):
        if q < pb:
          continue
        order = base[:pb] + moved + [async_start] + base[pb:q] + [async_done] + base[q:]
        total, stalls = _simulate_comm_stream(order, cost, lat, done_start, gate)
        if protect and sum(stalls.get(d, 0.0) for d in protect) > protected_now + _ASYNC_STREAM_TOL_US:
          if diag is not None:
            diag.append((total, i, True))
          continue
        if diag is not None:
          diag.append((total, i, False))
        cands.append((total, q != q_cur, pb, k, q, moved, base))
  if not cands:
    return None
  limit = min(c[0] for c in cands) + _ASYNC_STREAM_TOL_US
  total, _moved, pb, _, q, moved, base = min((c for c in cands if c[0] <= limit), key=lambda c: (c[2], -c[4]))
  return ((round(total, 1), _moved, pb), moved, pb, q, base)
