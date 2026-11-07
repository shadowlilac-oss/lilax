"""Action attribution for optional Trainer checkpoint save receipts."""

from dataclasses import dataclass, replace
from typing import Any, Literal

from orbax.checkpoint import v1 as ocp


@dataclass(frozen=True)
class SaveReceipt:
  action_id: str
  completed_step: int
  checkpoint_step: int
  outcome: Literal["pending", "no_save", "completed"]
  response: Any = None
  tier: str | None = None
  reason: str | None = None

  def wait(self, checkpoint_manager):
    if self.outcome != "pending":
      return self
    if self.reason is not None:
      return self
    try:
      saved = self.response if isinstance(self.response, bool) else self.response.result()
    except ocp.training.errors.StepAlreadyExistsError:
      return replace(self, outcome="no_save")
    if not saved:
      return replace(self, outcome="no_save")
    metadata = checkpoint_manager.checkpointables_metadata(self.checkpoint_step)
    action = {"action_id": self.action_id, "completed_step": self.completed_step}
    if (metadata is None or metadata.step != self.checkpoint_step or metadata.commit_timestamp_nsecs is None
        or (metadata.custom_metadata or {}).get("lilax_save_action") != action):
      raise RuntimeError(f"Committed checkpoint metadata does not identify save action {self.action_id!r}.")
    return replace(self, outcome="completed", tier="orbax_checkpoint_commit")


def save_receipt(action_id, completed_step, checkpoint_step, response=None, *, unsupported_tier=False):
  if response is None or response is False:
    return SaveReceipt(action_id, completed_step, checkpoint_step, "no_save", response=response)
  return SaveReceipt(action_id, completed_step, checkpoint_step, "pending", response=response,
                     reason="Emergency or replicated completion tier is not established." if unsupported_tier else None)
