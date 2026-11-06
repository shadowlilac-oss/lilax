"""Environment-specific mapping of terminal results to Tunix statuses."""

from collections.abc import Callable
from typing import Any

from tunix.rl.agentic.agents import agent_types


TerminalStatusMapper = Callable[[Any, float, dict[str, Any]], agent_types.TrajectoryStatus]


def terminal_status(
    mapper: TerminalStatusMapper, observation: Any, reward: float, info: dict[str, Any]
) -> agent_types.TrajectoryStatus:
  """Map an environment's authoritative terminal information to a Tunix status."""
  status = mapper(observation, reward, info)
  if not isinstance(status, agent_types.TrajectoryStatus) or status == agent_types.TrajectoryStatus.RUNNING:
    raise ValueError("Terminal status mapping must return a non-running Tunix trajectory status.")
  return status
