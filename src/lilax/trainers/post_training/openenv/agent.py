"""Conversation agent for an OpenEnv-specific Tunix environment adapter."""

import copy
from collections.abc import Callable
from typing import Any

from tunix.rl.agentic.agents import agent_types, base_agent

from .environment import InitialObservation
from .status import TerminalStatusMapper, terminal_status


class OpenEnvAgent(base_agent.ConversationAgentBase):
  """Record turns while delegating environment-specific formats to the caller."""

  def __init__(
      self,
      system_prompt: str,
      action_from_response: Callable[[str], Any],
      observation_to_text: Callable[[Any], str],
      status_mapper: TerminalStatusMapper,
  ):
    super().__init__(system_prompt=system_prompt)
    self.action_from_response = action_from_response
    self.observation_to_text = observation_to_text
    self.status_mapper = status_mapper

  def _observation_to_messages(self, observation: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
    self.chat_completions.append({"role": "user", "content": self.observation_to_text(observation)})

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    action = agent_types.Action(action=self.action_from_response(response))
    self.chat_completions.append({"role": "assistant", "content": response})
    self.trajectory.steps.append(agent_types.Step(
        chat_completions=copy.deepcopy(self.chat_completions), action=action, model_response=response,
    ))
    return action

  def update_from_env(
      self, observation: Any, reward: float, done: bool, info: dict[str, Any] | None = None, **kwargs
  ) -> None:
    if isinstance(observation, InitialObservation):
      self.trajectory.task = copy.deepcopy(observation.task)
      observation = observation.observation
    if done:
      self.trajectory.status = terminal_status(self.status_mapper, observation, reward, info or {})
    super().update_from_env(observation, reward, done, info, **kwargs)
