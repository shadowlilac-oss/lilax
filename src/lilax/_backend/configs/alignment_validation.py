"""Validation and derivation of native SFT and DPO/ORPO alignment settings."""


def validate_alignment_objectives(config):
  if config.use_dpo:
    if config.packing:
      raise ValueError("For DPO/ORPO, `packing` is not supported.")
    if config.dpo.max_prompt_length is None:
      config.dpo.max_prompt_length = config.max_target_length // 2
    if config.dpo.max_prompt_length >= config.max_target_length:
      raise ValueError(
          f"dpo.max_prompt_length ({config.dpo.max_prompt_length}) must be less than max_target_length"
          f" ({config.max_target_length})."
      )
  if config.use_sft and config.use_dpo:
    raise ValueError("Only one of `use_sft` or `use_dpo` can be True.")
