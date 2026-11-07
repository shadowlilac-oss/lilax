"""Hugging Face-style arguments for one trainer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from lilax._backend.configs.runtime import resolve_initialized_config
from lilax.checkpoints.policy import TrainingCheckpointConfig
from lilax.distributed import DiLoCoConfig, ParallelismConfig
from lilax.training import BatchRampupConfig, LRSchedulerConfig, ObjectiveConfig, OptimizerConfig, RematConfig

from ._input_args import MMAP_FIELDS, TFDS_C4_FIELDS
from ._precision_args import PRECISION_FIELDS
from ._optimizer_args import optimizer_overrides
from ._execution_args import execution_overrides


class TrainingArguments(BaseModel):
  """Optimization, duration and common execution controls."""

  model_config = ConfigDict(extra="forbid", validate_default=True)
  output_dir: str
  run_name: str = ""
  max_steps: int = 150_001
  num_train_epochs: int = 1
  per_device_train_batch_size: int | float = 12
  per_device_eval_batch_size: int | float = 0.0
  gradient_accumulation_steps: int = 1
  learning_rate: float = 3e-5
  optim: Literal["adamw", "adam_pax", "sgd", "muon"] = "adamw"
  adam_beta1: float = 0.9
  adam_beta2: float = 0.95
  adam_epsilon: float = 1e-8
  weight_decay: float = 0.1
  max_grad_norm: float = 1.0
  lr_scheduler_type: Literal["cosine", "wsd"] = "cosine"
  warmup_ratio: float = 0.1
  lr_scheduler_config: LRSchedulerConfig | None = None
  optimizer_config: OptimizerConfig | None = None
  logging_steps: int = 100
  eval_steps: int = -1
  save_steps: int = 10_000
  eval_on_start: bool = False
  checkpoint_config: TrainingCheckpointConfig | None = None
  report_to: str | list[str] = "none"
  logging_dir: str | None = None
  parallelism_config: ParallelismConfig | None = None
  jax_distributed_initialization_timeout: int = 300
  diloco_config: DiLoCoConfig | None = None
  gradient_checkpointing: bool = True
  remat_config: RematConfig | None = None
  objective_config: ObjectiveConfig | None = None
  batch_rampup_config: BatchRampupConfig | None = None

  @field_validator("report_to")
  @classmethod
  def validate_reporting(cls, value):
    names = (["tensorboard", "wandb", "trackio"] if value == "all" else [] if value == "none"
             else [value] if isinstance(value, str) else value)
    if any(name not in ("tensorboard", "wandb", "trackio") for name in names):
      raise ValueError("report_to must select none, all, tensorboard, wandb or trackio")
    return names

  def _overrides(self):
    values = {
        "base_output_directory": self.output_dir, "steps": self.max_steps, "num_epoch": self.num_train_epochs,
        "per_device_batch_size": self.per_device_train_batch_size,
        "eval_per_device_batch_size": self.per_device_eval_batch_size,
        "gradient_accumulation_steps": self.gradient_accumulation_steps, "learning_rate": self.learning_rate,
        "log_period": self.logging_steps, "eval_interval": self.eval_steps, "eval_steps": -1,
        "checkpoint_period": self.save_steps,
        "enable_tensorboard": False, "enable_wandb": False,
    }
    if self.run_name:
      values["run_name"] = self.run_name
    if self.logging_dir is not None:
      values["tensorboard_dir"] = self.logging_dir
    if self.checkpoint_config is not None:
      values.update({"enable_checkpointing" if key == "enabled" else key: value
                     for key, value in self.checkpoint_config.model_dump(
                         exclude={"save_only_lora_params", "restore_only_lora_params"},
                     ).items()})
    for domain in (self.objective_config, self.batch_rampup_config):
      if domain is not None:
        values.update(domain.model_dump())
    values.update(optimizer_overrides(self))
    values.update(execution_overrides(self))
    return {name: value for name, value in values.items() if value is not None}

  def _resolve(self, runtime, *, train_dataset_provided=False, eval_dataset_provided=False):
    return resolve_initialized_config(
        runtime.public_config, runtime.config, overrides=runtime._overrides | self._overrides(),
        train_dataset_provided=train_dataset_provided, eval_dataset_provided=eval_dataset_provided,
    )
