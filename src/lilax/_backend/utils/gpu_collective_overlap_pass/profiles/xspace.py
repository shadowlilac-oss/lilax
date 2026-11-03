
"""Per-invocation cost extraction from a raw PGLE XSpace."""

from __future__ import annotations

from typing import Optional

from lilax._backend.utils.gpu_collective_overlap_pass.logger import _logger
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _TE_EP_PREFIX
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.protos import _ensure_protos

# TODO tixxx remove the above invocation cost once PGLE can return the
# correct aggregated cost for multi-kernel ops.
# ---------------------------------------------------------------------------
# Per-invocation cost corrections from the live PGLE XSpace
# ---------------------------------------------------------------------------
# PGLE's xplane->FDO conversion averages cost_us over every GPU kernel launch
# tagged with an HLO op's name. That is fine for single-kernel ops but wrong for
# ops that run as many kernels per execution.
# _patch_pgle_profiler intercepts the raw XSpace and
# clusters events into invocations (a new one starts after a gap of
# _INVOCATION_GAP_US) to recover the per-invocation cost.
_INVOCATION_GAP_US = 100.0


def _average_invocation_costs(by_op: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
  """{hlo_op: [(ts_us, dur_us), ...]} -> {hlo_op: mean per-invocation span}."""
  costs: dict[str, float] = {}
  for hlo_op, events in by_op.items():
    if not events:
      continue
    events.sort()
    invocation_spans: list[float] = []
    group_start, group_end = events[0][0], events[0][0] + events[0][1]
    for ts, dur in events[1:]:
      if ts - group_end > _INVOCATION_GAP_US:
        invocation_spans.append(group_end - group_start)
        group_start = ts
      group_end = max(group_end, ts + dur)
    invocation_spans.append(group_end - group_start)
    costs[hlo_op] = sum(invocation_spans) / len(invocation_spans)
  return costs


def _load_costs_from_xspace_bytes(xspace_bytes: bytes, prefixes=None) -> dict[str, float]:
  """Per-invocation costs for HLO ops starting with `prefixes`, from a raw XSpace.

  Walks GPU device planes, resolves each event's `hlo_op` stat (event or event
  metadata; str_value or interned ref_value) and clusters by invocation.
  """
  _ensure_protos()
  from tsl.profiler.protobuf import xplane_pb2 as _xp  # type: ignore  # pylint: disable=import-outside-toplevel

  if prefixes is None:
    prefixes = (_TE_EP_PREFIX,)
  try:
    xspace = _xp.XSpace()
    xspace.ParseFromString(xspace_bytes)
  except Exception as exc:  # pylint: disable=broad-except
    _logger.warning("gpu_collective_overlap_pass: failed to parse live XSpace profile: %s", exc)
    return {}

  by_op: dict[str, list[tuple[float, float]]] = {}
  for plane in xspace.planes:
    if not plane.name.startswith("/device:GPU:"):
      continue
    stat_meta_name = {m.id: m.name for m in plane.stat_metadata.values()}
    event_meta_stats = {m.id: list(m.stats) for m in plane.event_metadata.values()}

    # Called only within this iteration, so capturing the loop variable is safe.
    # pylint: disable=cell-var-from-loop
    def _resolve_hlo_op(stats) -> Optional[str]:
      for stat in stats:
        if stat_meta_name.get(stat.metadata_id) != "hlo_op":
          continue
        which = stat.WhichOneof("value")
        if which == "str_value":
          return stat.str_value
        if which == "ref_value":
          return stat_meta_name.get(stat.ref_value)
      return None

    for line in plane.lines:
      for event in line.events:
        if event.WhichOneof("data") != "offset_ps":
          continue
        dur_us = event.duration_ps / 1e6
        if dur_us <= 0:
          continue
        hlo_op = _resolve_hlo_op(event_meta_stats.get(event.metadata_id, ())) or _resolve_hlo_op(event.stats)
        if not hlo_op or not hlo_op.startswith(prefixes):
          continue
        ts_us = (line.timestamp_ns * 1000 + event.offset_ps) / 1e6
        by_op.setdefault(hlo_op, []).append((ts_us, dur_us))

  return _average_invocation_costs(by_op)
