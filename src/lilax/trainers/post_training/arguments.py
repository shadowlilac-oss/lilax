"""Shared post-training arguments."""

from ..arguments.training_args import TrainingArguments


class _PostTrainingArguments(TrainingArguments):
  def __init__(self, *args, save_only_lora_params=None, restore_only_lora_params=None, **kwargs):
    super().__init__(*args, **kwargs)
    for name, value in (("save_only_lora_params", save_only_lora_params),
                        ("restore_only_lora_params", restore_only_lora_params)):
      if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be None or a boolean.")
    self.save_only_lora_params = save_only_lora_params
