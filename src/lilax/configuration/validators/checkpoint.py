def validate_checkpoint_source_options(self):
  if self.load_parameters_path and self.load_full_state_path:
    raise ValueError("At most one of `load_parameters_path` or `load_full_state_path` should be set.")
  if self.source_checkpoint_layout == "safetensors_dynamic" and self.enable_single_controller:
    raise ValueError(
        "`source_checkpoint_layout='safetensors_dynamic'` is not supported"
        " on the Pathways backend (`enable_single_controller=True`)."
    )


def validate_checkpoint_storage_options(self):
  if self.load_full_state_path and not self.enable_checkpointing:
    raise ValueError("You must set enable_checkpointing=True to resume from load_full_state_path.")
  if self.enable_multi_tier_checkpointing:
    if not self.local_checkpoint_directory:
      raise ValueError("`local_checkpoint_directory` must be set for multi-tier checkpointing.")
    if self.local_checkpoint_period <= 0:
      raise ValueError("`local_checkpoint_period` must be > 0 for multi-tier checkpointing.")
    if (self.multi_tier_checkpointing_backup_interval_minutes is None) == (
        self.multi_tier_checkpointing_backup_interval_steps is None
    ):
      raise ValueError(
          "Exactly one of `multi_tier_checkpointing_backup_interval_minutes`"
          " or `multi_tier_checkpointing_backup_interval_steps` must be specified."
      )
    if (
        self.multi_tier_checkpointing_backup_interval_steps is not None
        and self.multi_tier_checkpointing_backup_interval_steps < self.local_checkpoint_period
    ):
      raise ValueError(
          "`multi_tier_checkpointing_backup_interval_steps`"
          f" ({self.multi_tier_checkpointing_backup_interval_steps}) must be"
          f" >= `local_checkpoint_period` ({self.local_checkpoint_period})."
      )
  if self.colocated_python_checkpointing and not self.enable_single_controller:
    raise ValueError("`colocated_python_checkpointing` is only supported with `enable_single_controller` set to True.")
  if self.enable_emergency_checkpoint:
    if not self.local_checkpoint_directory:
      raise ValueError("`local_checkpoint_directory` must be set for emergency checkpointing.")
    if self.local_checkpoint_period <= 0:
      raise ValueError("`local_checkpoint_period` must be > 0 for emergency checkpointing.")
