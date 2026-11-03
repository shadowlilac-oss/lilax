"""Validation of native checkpoint configuration."""

from typing import Any


def validate_checkpoint_source_config(config: Any):
  if config.load_parameters_path and config.load_full_state_path:
    raise ValueError("At most one of `load_parameters_path` or `load_full_state_path` should be set.")
  if config.source_checkpoint_layout == "safetensors_dynamic" and config.enable_single_controller:
    raise ValueError(
        "`source_checkpoint_layout='safetensors_dynamic'` is not supported"
        " on the Pathways backend (`enable_single_controller=True`)."
    )


def validate_checkpoint_storage_config(config: Any):
  if config.load_full_state_path and not config.enable_checkpointing:
    raise ValueError("You must set enable_checkpointing=True to resume from load_full_state_path.")
  if config.enable_multi_tier_checkpointing:
    if not config.local_checkpoint_directory:
      raise ValueError("`local_checkpoint_directory` must be set for multi-tier checkpointing.")
    if config.local_checkpoint_period <= 0:
      raise ValueError("`local_checkpoint_period` must be > 0 for multi-tier checkpointing.")
    if (config.multi_tier_checkpointing_backup_interval_minutes is None) == (
        config.multi_tier_checkpointing_backup_interval_steps is None
    ):
      raise ValueError(
          "Exactly one of `multi_tier_checkpointing_backup_interval_minutes`"
          " or `multi_tier_checkpointing_backup_interval_steps` must be"
          " specified."
      )
    if (
        config.multi_tier_checkpointing_backup_interval_steps is not None
        and config.multi_tier_checkpointing_backup_interval_steps < config.local_checkpoint_period
    ):
      raise ValueError(
          "`multi_tier_checkpointing_backup_interval_steps`"
          f" ({config.multi_tier_checkpointing_backup_interval_steps}) must be"
          f" >= `local_checkpoint_period` ({config.local_checkpoint_period})."
      )
  if config.colocated_python_checkpointing and not config.enable_single_controller:
    raise ValueError("`colocated_python_checkpointing` is only supported with `enable_single_controller` set to True.")
  if config.enable_emergency_checkpoint:
    if not config.local_checkpoint_directory:
      raise ValueError("`local_checkpoint_directory` must be set for emergency checkpointing.")
    if config.local_checkpoint_period <= 0:
      raise ValueError("`local_checkpoint_period` must be > 0 for emergency checkpointing.")
