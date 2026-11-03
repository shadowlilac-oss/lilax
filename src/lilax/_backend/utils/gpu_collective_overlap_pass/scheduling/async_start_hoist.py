
"""Hoisting while-body async starts to their floor (or the comm-stream model's placement)."""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional.
# pylint: disable=broad-exception-caught

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.costing.instruction_cost import _sim_inst_cost
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.async_ops import _ASYNC_COLLECTIVE_PREFIXES, _is_async_start
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.custom_calls import _is_te_gemm_custom_call
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.profile_keys import _resolve_profile_key
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.text import _control_predecessor_names, _opcode_str
from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import (
    _TE_EP_PREFIX, _collective_overrides, _profile_costs,
)
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.blocker_relocation import _MAX_PRODUCER_RELOCATE_HOPS
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.comm_stream import (
    _ASYNC_STREAM_MODEL, _ASYNC_STREAM_TOL_US, _SM_GATE, _SM_GATE_MIN_US, _best_async_placement, _simulate_comm_stream,
)
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.legality import _earliest_legal_pos
from lilax._backend.utils.gpu_collective_overlap_pass.scheduling.sequence import _commit_sequence, _reinsert


def _may_hoist_control_pred(pred, positions, comp_by_name) -> bool:
  """Whether control-predecessor `pred` may move earlier.

  Anything may, except the done of an async or te_ep collective. The recursive
  hoist reached such a done through a consumer right after it and pulled it
  next to its start.
  """
  opc = _opcode_str(pred)
  if not (opc == "async-done" or opc.endswith("-done")):
    return True
  ops = list(pred.operands())
  start = ops[0] if ops else None
  if start is None or start not in positions:
    return False
  try:
    key = _resolve_profile_key(start, comp_by_name)
  except Exception:
    key = ""
  return not key.startswith(_ASYNC_COLLECTIVE_PREFIXES + (_TE_EP_PREFIX,))


# Recursion bound when hoisting a control-predecessor whose own latest operand
# must move first.
_MAX_PRED_HOIST_DEPTH = 8


def _hoist_async_starts_to_floor(
    seq,
    schedule,
    comp,
    start_of_done,
    positions,
    name_to_pos,
    comp_by_name,
    module_name,
):
  """Move each async start in a while body to its earliest legal position.

  Runs after the fill step. The per-collective loop in _reorder_collectives skips a
  collective once its static window looks hidden, and later fill/chase moves
  can land compute ahead of it with no dependency on it.

  With the stream model on, placement is chosen by _best_async_placement
  (which may also move the done); otherwise the start goes to its floor and
  the done stays put, so the window only grows.
  """
  changed = False

  def _move_earlier(inst, depth) -> bool:
    """Move inst to just after its latest operand/control-predecessor,
    hoisting that one first (recursively, up to `depth`) if it sits right
    before inst."""
    nonlocal seq, positions, name_to_pos, changed
    pos = positions[inst]
    floor, binder = 0, None
    for op in inst.operands():
      p = positions.get(op)
      if p is not None and p + 1 > floor:
        floor, binder = p + 1, op
    for name in _control_predecessor_names(inst):
      p = name_to_pos.get(name)
      if p is not None and p + 1 > floor:
        floor, binder = p + 1, seq[p]
    if floor < pos:
      if not _may_hoist_control_pred(inst, positions, comp_by_name):
        return False
      new_seq = [i for i in seq if i is not inst]
      if len(new_seq) != len(seq) - 1:
        _logger.warning(
            "gpu_collective_overlap_pass [%s]: HOISTDIAG %s not found exactly once in sequence; skipping move.",
            module_name,
            inst.name,
        )
        return False
      new_seq.insert(floor, inst)
      seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
      changed = True
      return True
    if binder is None or depth <= 0:
      return False
    if not _move_earlier(binder, depth - 1):
      return False
    return _move_earlier(inst, depth - 1)

  sim_cost: dict = {}
  sim_lat: dict = {}
  async_dones: set = set()
  sim_gate: set = set()
  n_lat_overridden = 0
  ctrl_succs: dict = {}
  if _ASYNC_STREAM_MODEL:
    for inst in seq:
      if _is_async_start(inst) or _opcode_str(inst).endswith("-done"):
        sim_cost[inst] = 0.0
      else:
        sim_cost[inst] = _sim_inst_cost(inst, comp_by_name)
        if _SM_GATE and sim_cost[inst] >= _SM_GATE_MIN_US and _is_te_gemm_custom_call(inst, comp_by_name):
          sim_gate.add(inst)
      for name in _control_predecessor_names(inst):
        ctrl_succs.setdefault(name, []).append(inst)
    for d_inst, s_inst in start_of_done.items():
      try:
        key_s = _resolve_profile_key(s_inst, comp_by_name)
        sim_lat[s_inst] = _collective_overrides.get(key_s) or _profile_costs.get(key_s, 0.0) or 0.0
        if key_s in _collective_overrides:
          n_lat_overridden += 1
        if key_s.startswith(_ASYNC_COLLECTIVE_PREFIXES):
          async_dones.add(d_inst)
      except Exception:
        sim_lat[s_inst] = 0.0
    _logger.info(
        "gpu_collective_overlap_pass [%s]: stream model: %s simulated total exposed %.1f us before async placement "
        "(%d collective latencies from per-invocation clustering).",
        module_name,
        comp.name,
        _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done, sim_gate)[0],
        n_lat_overridden,
    )
  for async_done, async_start in sorted(start_of_done.items(), key=lambda kv: positions[kv[1]]):
    profile_key = _resolve_profile_key(async_start, comp_by_name)
    if not profile_key.startswith(_ASYNC_COLLECTIVE_PREFIXES):
      continue
    orig_pos = positions[async_start]
    for _ in range(_MAX_PRODUCER_RELOCATE_HOPS):
      cp_pins: list = []
      floor, to_move, _ = _earliest_legal_pos(
          async_start,
          positions,
          name_to_pos,
          comp_by_name,
          cp_pins,
      )
      if not cp_pins:
        break
      pin_pos, pin_name = max(cp_pins)
      if pin_pos + 1 < floor or pin_pos + 1 < 1:
        break  # a data dependency pins the floor, not a control-predecessor
      if not _move_earlier(seq[pin_pos], _MAX_PRED_HOIST_DEPTH):
        _logger.debug(
            "gpu_collective_overlap_pass [%s]: %s floor %d pinned by control-predecessor %s "
            "(pos %d) which could not be hoisted.",
            module_name,
            async_start.name,
            floor,
            pin_name,
            pin_pos,
        )
        break
    floor, to_move, _ = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
    start_pos = positions[async_start]
    if _ASYNC_STREAM_MODEL and sim_lat.get(async_start, 0.0) > 0:
      before = _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done, sim_gate)[0]
      cand_diag: list = []
      best = _best_async_placement(
          async_start,
          async_done,
          floor,
          to_move,
          seq,
          positions,
          sim_cost,
          sim_lat,
          start_of_done,
          ctrl_succs,
          protect=async_dones,
          gate=sim_gate,
          diag=cand_diag,
      )
      if cand_diag:
        top = sorted(cand_diag)[:4]
        later = sorted((c for c in cand_diag if c[1] > start_pos))[:2]

        def fmt(cs):
          return ", ".join(f"seq idx {c[1]}: {c[0]:.1f} us{' [dropped]' if c[2] else ''}" for c in cs) or "none"

        _logger.debug(
            "gpu_collective_overlap_pass [%s]: STREAMCANDS %s: %d candidates (%d dropped by async-stall guard); "
            "current %.1f us; lowest %s; lowest later than current %s.",
            module_name,
            async_start.name,
            len(cand_diag),
            sum(1 for c in cand_diag if c[2]),
            before,
            fmt(top),
            fmt(later),
        )
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: STREAMDIAG %s: start pos %d, done pos %d, floor %d, to_move %d, "
          "latency %.1f us, simulated total %.1f us; best %s.",
          module_name,
          async_start.name,
          start_pos,
          positions[async_done],
          floor,
          len(to_move),
          sim_lat.get(async_start, 0.0),
          before,
          "none"
          if best is None
          else f"total {best[0][0]:.1f} us at base idx {best[2]}, done idx {best[3]}, " f"{len(best[1])} chain ops moved",
      )
      if best is not None:
        key, moved, pb, q, base = best
        if key[0] <= before + _ASYNC_STREAM_TOL_US:
          done_pos_old = positions[async_done]
          new_seq = base[:pb] + list(moved) + [async_start] + base[pb:q] + [async_done] + base[q:]
          if len(new_seq) == len(seq) and set(new_seq) == set(seq) and new_seq != seq:
            seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
            changed = True
            _logger.info(
                "gpu_collective_overlap_pass [%s]: stream model placed %s: start pos %d -> %d, done pos %d -> %d "
                "(simulated total exposed %.1f -> %.1f us).",
                module_name,
                async_start.name,
                orig_pos,
                positions[async_start],
                done_pos_old,
                positions[async_done],
                before,
                key[0],
            )
        continue
    if floor + len(to_move) >= start_pos:
      continue
    seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, to_move, floor, async_start))
    changed = True
    _logger.info(
        "gpu_collective_overlap_pass [%s]: hoisted %s from pos %d to %d with %d relocated operand(s).",
        module_name,
        async_start.name,
        orig_pos,
        positions[async_start],
        len(to_move),
    )
  return changed, seq, positions, name_to_pos
