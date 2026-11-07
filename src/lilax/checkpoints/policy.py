"""Trainer checkpoint policy, separate from deployment storage settings."""

from pydantic import BaseModel, ConfigDict


class TrainingCheckpointConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  enabled: bool = True
  async_checkpointing: bool = True
  max_num_checkpoints_to_keep: int | None = None
  save_checkpoint_on_start: bool = True
  save_checkpoint_on_completion: bool = True
  enable_continuous_checkpointing: bool = False
  enable_autocheckpoint: bool = False
  save_only_lora_params: bool | None = None
  restore_only_lora_params: bool | None = None
