
"""Schedule sequence rewriting and committing."""

from __future__ import annotations


def _reinsert(seq: list, moving: list, at: int, last):
  """seq without `moving`/`last`, with `moving` then `last` inserted at `at`."""
  drop = set(moving) | {last}
  new_seq = [inst for inst in seq if inst not in drop]
  new_seq[at:at] = list(moving) + [last]
  return new_seq


def _commit_sequence(schedule, comp, new_seq: list):
  """Set comp's sequence; returns (seq, positions, name_to_pos)."""
  schedule.set_sequence(comp, new_seq)
  positions = {inst: i for i, inst in enumerate(new_seq)}
  name_to_pos = {inst.name: i for inst, i in positions.items()}
  return new_seq, positions, name_to_pos
