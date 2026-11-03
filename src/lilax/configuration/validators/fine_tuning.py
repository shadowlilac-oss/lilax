def validate_dpo_packing(self):
  if self.use_dpo and self.packing:
    raise ValueError("DPO does not support packing. Set `packing=False`.")


def validate_fine_tuning_modes(self):
  if self.use_sft and self.use_dpo:
    raise ValueError("Only one of `use_sft` or `use_dpo` can be True.")


def validate_dpo_prompt_length(self):
  if self.use_dpo:
    if self.dpo.max_prompt_length is None:
      self.dpo.max_prompt_length = self.max_target_length // 2
    if self.dpo.max_prompt_length >= self.max_target_length:
      raise ValueError(
          f"dpo.max_prompt_length ({self.dpo.max_prompt_length}) must be less than max_target_length"
          f" ({self.max_target_length})."
      )
