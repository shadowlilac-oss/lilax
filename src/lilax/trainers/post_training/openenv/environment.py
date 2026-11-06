"""Adapt one OpenEnv client session to a Tunix episode."""

import threading
from dataclasses import dataclass
from typing import Any

from tunix.rl.agentic.environments import base_environment


@dataclass
class InitialObservation:
  task: dict[str, Any]
  observation: Any


class OpenEnvEnvironment(base_environment.BaseEnv):
  """Use a caller-created synchronous OpenEnv client for one episode.

  The factory receives the task, group ID and pair index and returns a fresh
  synchronous client. Reset arguments and step rewards are mapped by callers.
  Close is serialized with client operations; a timed-out operation can still
  run, and closing the client does not acknowledge server-side cancellation.
  """

  def __init__(self, task, *, group_id, pair_index, client_factory, reset_kwargs, reward_from_result, max_steps):
    super().__init__()
    self.task = task
    self.extra_kwargs = {"group_id": group_id, "pair_index": pair_index}
    self.client_factory = client_factory
    self.reset_kwargs = reset_kwargs
    self.reward_from_result = reward_from_result
    self.max_steps = max_steps
    self.client = None
    self.closed = False
    self.lock = threading.Lock()

  def reset(self):
    with self.lock:
      if self.closed:
        raise RuntimeError("Cannot reset a closed OpenEnv episode.")
      self.client = self.client_factory(self.task, **self.extra_kwargs)
      result = self.client.reset(**self.reset_kwargs(self.task, **self.extra_kwargs))
      if result.done:
        raise ValueError("OpenEnv reset returned a terminal episode; Tunix requires an active initial observation.")
      return InitialObservation(self.task, result.observation), result.metadata or {}

  def step(self, action):
    with self.lock:
      result = self.client.step(action)
      return result.observation, self.reward_from_result(result), result.done, result.metadata or {}

  def close(self):
    with self.lock:
      self.closed = True
      if self.client is not None:
        client, self.client = self.client, None
        client.close()
