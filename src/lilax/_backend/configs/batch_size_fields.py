"""Native global, micro, and ramp-up batch size fields derived from per-device batch sizes."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class GlobalBatchSizes(BaseModel):
  """Global and micro batch sizes for training, evaluation, and data loading."""

  global_batch_size_to_train_on: None | int = Field(
      None,
      description="The total batch size for training across all devices. Derived from `per_device_batch_size` and data"
      "parallelism.",
  )
  global_batch_size_to_eval_on: None | int = Field(
      None,
      description="The total batch size for evaluation across all devices. Derived from `eval_per_device_batch_size` and"
      " data parallelism.",
  )
  global_batch_size_to_load: None | int = Field(
      None,
      description="The global batch size for the training dataloader, potentially scaled by `expansion_factor_real_data`.",
  )
  global_batch_size_to_load_eval: None | int = Field(
      None,
      description="The global batch size for the evaluation dataloader, potentially scaled by `expansion_factor_real_data`.",
  )
  micro_batch_size_to_train_on: None | int = Field(
      None,
      description="The size of each micro-batch for training, used in pipeline parallelism. Derived from "
      "`global_batch_size_to_train_on`.",
  )
  micro_batch_size_to_eval_on: None | int = Field(
      None,
      description="The size of each micro-batch for evaluation, used in pipeline parallelism. Derived from "
      "`global_batch_size_to_eval_on`.",
  )


class BatchSizeRampupEnd(BaseModel):
  """Training step at which the global batch size ramp-up concludes."""

  rampup_end_step: None | int = Field(None, description="The step at which the batch size ramp-up phase concludes.")


class BatchSizeRampupIncrements(BaseModel):
  """Starting batch size, increment, and samples per increment of the dataloader batch size ramp-up."""

  global_batch_size_to_load_start: None | int = Field(None, description="Starting global batch size for rampup.")
  global_batch_size_to_load_increment: None | int = Field(
      None, description="Increment for global batch size during rampup."
  )
  rampup_samples_per_increment_to_load: None | float = Field(None, description="Samples per increment for rampup.")
