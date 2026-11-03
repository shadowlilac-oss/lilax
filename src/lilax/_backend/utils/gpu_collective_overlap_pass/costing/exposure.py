
"""Static collective-window exposure metrics over a schedule sequence."""

from __future__ import annotations

from lilax._backend.utils.gpu_collective_overlap_pass.costing.instruction_cost import _resolve_inst_cost
from lilax._backend.utils.gpu_collective_overlap_pass.hlo.profile_keys import _resolve_profile_key
from lilax._backend.utils.gpu_collective_overlap_pass.profiles.costs import _profile_costs


def _prefix_costs_excluding(seq: list, exclude: set, comp_by_name: dict) -> list:
  """Prefix sum of costs along `seq`, with `exclude`d instructions counted as 0."""
  prefix = [0.0] * (len(seq) + 1)
  for i, inst in enumerate(seq):
    c = 0.0 if inst in exclude else _resolve_inst_cost(inst, comp_by_name)
    prefix[i + 1] = prefix[i] + c
  return prefix


def _total_exposed_us(
    start_of_done: dict,
    positions: dict,
    seq: list,
    comp_by_name: dict,
) -> float:
  """Sum of positive deficits over every collective with a known latency, on
  the schedule as it stands: the metric that decides whether a move is a net win."""
  prefix = _prefix_costs_excluding(seq, (), comp_by_name)
  total = 0.0
  for async_done, async_start in start_of_done.items():
    if async_start not in positions or async_done not in positions:
      continue
    latency = _profile_costs.get(_resolve_profile_key(async_start, comp_by_name))
    if latency is None or latency <= 0:
      continue
    s, d = positions[async_start], positions[async_done]
    overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
    total += max(0.0, latency - overlap)
  return total


def _compute_position_margins(
    start_of_done: dict,
    positions: dict,
    seq: list,
    comp_by_name: dict,
    prefix: list | None = None,
    min_pos: int = 0,
) -> dict:
  """For every position >= min_pos, the minimum slack (overlap minus latency)
  among the collectives whose [start+1, done) window covers it: how much cost
  can be pulled out of that position without dropping any covering collective
  below its latency. Uncovered positions have no entry (callers treat that as
  infinite margin).

  Over-covered time is as safe to take from as idle time, as long as no more
  than the slack is taken.

  min_pos trims the work for a caller that only queries positions >= min_pos
  (the candidate scan, done_pos + 1). A caller that queries earlier
  positions, like the margin-safe fallback at a blocker's current position,
  must keep the default 0.
  """
  if prefix is None:
    prefix = _prefix_costs_excluding(seq, (), comp_by_name)
  margins: dict[int, float] = {}
  for async_done, async_start in start_of_done.items():
    if async_start not in positions or async_done not in positions:
      continue
    latency = _profile_costs.get(_resolve_profile_key(async_start, comp_by_name))
    if latency is None or latency <= 0:
      continue
    s, d = positions[async_start], positions[async_done]
    if d <= s + 1 or d <= min_pos:
      continue
    slack = prefix[d] - prefix[s + 1] - latency
    for p in range(max(s + 1, min_pos), d):
      if p not in margins or slack < margins[p]:
        margins[p] = slack
  return margins
