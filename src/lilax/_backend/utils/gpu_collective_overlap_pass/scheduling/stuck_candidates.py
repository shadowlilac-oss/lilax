
"""Give-up strategies for fill candidates that stay blocked across scan restarts."""

from __future__ import annotations

# Give up on a blocked candidate once it has been the first blocked thing on
# this many consecutive scan restarts, so it can't starve reachable ones.
_STUCK_CANDIDATE_GIVE_UP_STREAK = 4

_GIVE_UP_MODES = ("none", "streak", "floor_aware")


class _StuckCandidateTracker:
  """Decides when a repeatedly blocked candidate is skipped so others get a
  turn. Strategies (compared per computation by
  _fill_exposed_collectives_best_of):

  - "none": never give up.
  - "streak": counts consecutive restarts blocked on the same candidate; a
    different name resets it. Cheap and safest in small while bodies.
  - "floor_aware": per-candidate, counting a repeat only if its floor failed
    to improve. Better on the large entry computation, where
    multi-hop chains converge slowly.
  """

  def __init__(self, give_up_mode: str):
    self.give_up_mode = give_up_mode
    self._last_name: str | None = None
    self._last_streak = 0
    self._streak_by_name: dict[str, int] = {}
    self._last_floor_by_name: dict[str, int] = {}

  def note_and_maybe_give_up(self, name: str, floor: int) -> tuple[int, bool]:
    """Record that `name` was found blocked at `floor`; returns (streak, give_up)."""
    if self.give_up_mode == "none":
      return 0, False
    if self.give_up_mode == "streak":
      if name == self._last_name:
        self._last_streak += 1
      else:
        self._last_name = name
        self._last_streak = 1
      streak = self._last_streak
    elif self.give_up_mode == "floor_aware":
      prev_floor = self._last_floor_by_name.get(name)
      if prev_floor is not None and floor >= prev_floor:
        self._streak_by_name[name] = self._streak_by_name.get(name, 0) + 1
      else:
        self._streak_by_name[name] = 1
      self._last_floor_by_name[name] = floor
      streak = self._streak_by_name[name]
    else:
      raise ValueError(f"unknown give_up_mode: {self.give_up_mode!r}")
    return streak, streak >= _STUCK_CANDIDATE_GIVE_UP_STREAK
