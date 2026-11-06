"""Trainer-owned tool rollouts for environment factories."""

import copy
import inspect
import json
import threading
from dataclasses import dataclass

from transformers.utils import get_json_schema
from tunix.rl.agentic.agents import agent_types, base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.parser.tool_parser.qwen_parser import QwenToolParser
from tunix.rl.agentic import utils as agentic_utils

from lilax.trainers.post_training.openenv.registry import ENVIRONMENT_TOKEN


@dataclass
class FactoryReset:
  task: dict
  prompt: str
  schemas: list


class FactoryAgent(base_agent.ConversationAgentBase):
  def __init__(self, system_prompt):
    super().__init__(system_prompt=system_prompt)

  def update_from_model(self, response, **kwargs):
    response = response.removesuffix("<|im_end|>")
    action = agent_types.Action(action=response)
    self.chat_completions.append({"role": "assistant", "content": response})
    self.trajectory.steps.append(agent_types.Step(
        chat_completions=copy.deepcopy(self.chat_completions), action=action, model_response=response,
    ))
    return action

  def update_from_env(self, observation, reward, done, info=None, **kwargs):
    if isinstance(observation, FactoryReset):
      self.trajectory.task = copy.deepcopy(observation.task)
      self.chat_completions[0]["tools"] = json.dumps(observation.schemas)
      self.chat_completions.append({"role": "user", "content": observation.prompt})
      return
    super().update_from_env(observation, reward, done, info, **kwargs)

  def _observation_to_messages(self, observation, reward, done, info):
    for output in observation:
      self.chat_completions.append({"role": "tool", "content": output})


class FactoryEnvironment(base_environment.BaseEnv):
  def __init__(self, task, *, group_id, pair_index, environment_factory, has_reward_funcs,
               environment_registry=None):
    super().__init__()
    self.task = task
    self.extra_kwargs = {"group_id": group_id, "pair_index": pair_index}
    self.environment_factory = environment_factory
    self.has_reward_funcs = has_reward_funcs
    self.max_steps = float("inf")
    self.environment = None
    self.lock = threading.Lock()
    self.closed = False
    self.tool_parser = QwenToolParser()
    self.environment_registry = environment_registry
    self.environment_token = None

  def reset(self):
    with self.lock:
      if self.closed:
        raise RuntimeError("Cannot reset a closed environment.")
      self.environment = self.environment_factory()
      environment = self.environment
      tools = {name: method for name, method in inspect.getmembers(environment, predicate=inspect.ismethod)
               if not name.startswith("_") and name not in {"reset", "get_reward", "close"}}
      self.tools = tools
      if any(inspect.iscoroutinefunction(method) for method in tools.values()):
        raise TypeError("Async environment tools are not supported.")
      schemas = [get_json_schema(method) for method in tools.values()]
      reset_prompt = environment.reset()
      if reset_prompt is not None and not isinstance(reset_prompt, str):
        raise TypeError("environment.reset() must return a string or None.")
      message = agentic_utils.convert_messages_to_string({"role": "user", "content": self.task["prompts"]})
      prompt = message["content"] + (reset_prompt or "")
      reward_fn = getattr(environment, "get_reward", None)
      if reward_fn is not None and (
          not callable(reward_fn) or inspect.iscoroutinefunction(reward_fn)
      ):
        raise TypeError("environment.get_reward must be a synchronous callable.")
      if reward_fn is None and not self.has_reward_funcs:
        raise ValueError("An environment get_reward() or reward_funcs is required.")
      if inspect.iscoroutinefunction(getattr(environment, "close", None)):
        raise TypeError("environment.close must be synchronous.")
      self.final_reward_fn = self._final_reward if reward_fn is not None else None
      if self.environment_registry is not None:
        if ENVIRONMENT_TOKEN in self.task or "environments" in self.task:
          raise ValueError("Dataset fields 'environments' and '_lilax_environment_token' are reserved.")
        self.environment_token = self.environment_registry.add(environment)
        self.task[ENVIRONMENT_TOKEN] = [self.environment_token]
      return FactoryReset(self.task, prompt, schemas), {}

  def _final_reward(self):
    with self.lock:
      return self.environment.get_reward()

  def step(self, response):
    with self.lock:
      if response.count("<tool_call>") != response.count("</tool_call>"):
        raise ValueError("Incomplete Qwen tool call in model response.")
      calls = self.tool_parser.parse(response)
      if len(calls) != response.count("<tool_call>"):
        raise ValueError("Invalid Qwen tool call in model response.")
      if not calls:
        return [], 0.0, True, {}
      outputs = []
      for call in calls:
        result = self.tools[call.name](**call.arguments)
        if inspect.isawaitable(result):
          raise TypeError("Async environment tools are not supported.")
        outputs.append(str(result))
      return outputs, 0.0, False, {}

  def close(self):
    with self.lock:
      self.closed = True
      close = getattr(self.environment, "close", None)
      if close is not None:
        result = close()
        if inspect.isawaitable(result):
          raise TypeError("environment.close must be synchronous.")
      if self.environment_registry is not None and self.environment_token is not None:
        self.environment_registry.mark_closed(self.environment_token)

  def mark_close_timeout(self):
    if self.environment_registry is not None and self.environment_token is not None:
      self.environment_registry.mark_timeout(self.environment_token)
