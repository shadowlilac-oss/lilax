"""Native optimizer, learning-rate schedule, AdamW, and Muon configuration fields."""

from enum import Enum

from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import NonNegativeFloat, PositiveInt


class OptimizerType(str, Enum):
  """Supported optimizer algorithms."""

  ADAMW = "adamw"
  ADAM_PAX = "adam_pax"
  SGD = "sgd"
  MUON = "muon"


class LearningRateScheduleType(str, Enum):
  """Supported learning rate schedule types."""

  COSINE = "cosine"
  WSD = "wsd"


class WsdDecayStyle(str, Enum):
  """Supported decay styles for WSD schedule."""

  LINEAR = "linear"
  COSINE = "cosine"


class Optimizer(BaseModel):
  """Configuration for the optimizer and learning rate schedule."""

  opt_type: OptimizerType = Field(OptimizerType.ADAMW, description="The type of optimizer to use.")
  skip_step_on_spikes: bool = Field(
      False,
      description="If True, skip the training step when loss or gradient spike is detected.",
  )
  skip_step_interval: PositiveInt = Field(
      128,
      description="The rolling interval to calculate the mean and standard deviation.",
  )
  skip_step_scaling_factor: float = Field(6.0, description="The scaling factor to determine if a spike occurred.")
  gradient_accumulation_steps: PositiveInt = Field(
      1, description="Number of steps to accumulate gradients before updating."
  )
  use_tunix_gradient_accumulation: bool = Field(
      False,
      description="Whether to use the Tunix implementation for gradient accumulation.",
  )
  gradient_clipping_threshold: NonNegativeFloat = Field(
      1.0, description="The threshold for gradient clipping. 0 disables clipping."
  )
  learning_rate: NonNegativeFloat = Field(3.0e-5, description="The peak learning rate.")
  lr_schedule_type: LearningRateScheduleType = Field(
      LearningRateScheduleType.COSINE,
      description="The type of learning rate schedule to use.",
  )
  learning_rate_final_fraction: float = Field(
      0.1,
      description="Final LR as a fraction of peak LR (applies to both cosine and WSD schedules).",
  )
  wsd_decay_steps_fraction: float = Field(
      0.1,
      ge=0.0,
      le=1.0,
      description="Fraction of total steps for decay phase in WSD schedule.",
  )
  wsd_decay_style: WsdDecayStyle = Field(
      WsdDecayStyle.LINEAR,
      description="The decay style for WSD schedule ('linear' or 'cosine').",
  )
  warmup_steps_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of total steps for LR warmup.")
  learning_rate_schedule_steps: int = Field(
      -1,
      ge=-1,
      description="Total steps for the LR schedule. -1 defaults to `steps`.",
  )
  trainable_parameters_mask: list[str] = Field(
      default_factory=list,
      description=(
          "List of parameter names/patterns to train. If non-empty, all other parameters will be frozen, "
          "example: ['.*indexer.*']. If empty (default), all parameters are trained."
      ),
  )


class AdamW(BaseModel):
  """Configuration specific to the AdamW optimizer."""

  adam_b1: float = Field(
      0.9,
      description="Exponential decay rate for the first moment of past gradients (beta1).",
  )
  adam_b2: float = Field(
      0.95,
      description="Exponential decay rate for the second moment of past gradients (beta2).",
  )
  adam_eps: float = Field(
      1.0e-8,
      description="A small constant for numerical stability (epsilon), applied outside of the square root.",
  )
  adam_eps_root: float = Field(
      0.0,
      description="A small constant for numerical stability (epsilon), applied inside of the square root.",
  )
  adam_weight_decay: float = Field(0.1, description="Weight decay regularization.")
  adamw_mask: list[str] = Field(
      default_factory=list,
      description=(
          "List of parameter names/patterns to exclude from weight decay in AdamW," " like ['bias', '.*norm', '.*ln.*']"
      ),
  )
  mu_dtype: str = Field(
      "",
      description="Data type for 'mu' (first moment) in AdamW. Inherits from weight_dtype if empty.",
  )


class Muon(BaseModel):
  """Configuration specific to the Muon optimizer."""

  muon_type: str = Field(
      "optax_muon",
      description=("Type of Muon optimizer: 'optax_muon' (or 'optax') vs 'maxtext_muon' (or 'maxtext')."),
  )
  muon_beta: float = Field(0.95, description="Decay rate for the exponentially weighted average of grads.")
  muon_weight_decay: float = Field(
      0.0,
      description=("Strength of the weight decay regularization. This is multiplied with" " the learning rate."),
  )
  muon_consistent_rms: float | None = Field(
      None,
      description="If None, apply width scaling to updates. If float, apply consistent rms scaling (recommend 0.2).",
  )
  muon_include_routers: bool = Field(
      True,
      description=(
          "Whether to apply Muon updates to MoE router matrices. If False," " routers are optimized with AdamW."
      ),
  )
  muon_use_all_to_all: bool = Field(
      False,
      description=(
          "Whether to use all-to-all communication during Newton-Schulz" " iterations in the maxtext_muon optimizer."
      ),
  )
