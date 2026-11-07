"""Agent and environment bindings for existing agentic rollouts."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentBinding(BaseModel):
  model_config = ConfigDict(extra="forbid")
  agent_class: type
  environment_class: type
  agent_kwargs: dict[str, Any] | None = None
  environment_kwargs: dict[str, Any] | None = None


class AgenticRolloutConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  enabled: bool = False
  max_concurrency: int = 256
  off_policy_steps: int = 0
  system_prompt: str = ""
  agent: AgentBinding | None = None

  def _source_values(self):
    return {"use_agentic_rollout": self.enabled, "max_concurrency": self.max_concurrency,
            "off_policy_steps": self.off_policy_steps, "system_prompt": self.system_prompt}
