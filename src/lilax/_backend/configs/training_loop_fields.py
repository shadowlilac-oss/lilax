"""Native training objective, loop, evaluation cadence, and reproducibility configuration fields."""

from typing import Literal

from pydantic.fields import Field
from pydantic.main import BaseModel


class TrainingLoop(BaseModel):
  """Configuration for the main training loop, evaluation, and reproducibility."""

  training_objective: Literal["causal_lm", "block_diffusion"] = Field(
      "causal_lm",
      description="The token-prediction objective used to prepare targets and compute loss.",
  )
  block_diffusion_mask_id: int = Field(
      -1,
      description="The tokenizer mask-token id required by the block-diffusion training objective.",
  )
  block_diffusion_min_noise: float = Field(
      1.0e-3,
      gt=0.0,
      le=1.0,
      description="The minimum corruption probability sampled independently for each block.",
  )
  block_diffusion_logit_alignment: Literal["same_position", "shifted"] = Field(
      "same_position",
      description="How model logits align to clean target-token positions.",
  )
  block_diffusion_canvas_policy: Literal["all_masked", "seed_and_mask"] = Field(
      "all_masked",
      description="Whether every block is fully maskable or begins with a clean anchor token.",
  )
  steps: int = Field(
      150_001,
      ge=-1,
      description="Total number of training steps. -1 defaults to learning_rate_schedule_steps.",
  )
  log_period: int = Field(100, description="Frequency (in steps) to log metrics and flush Tensorboard.")
  eval_start_step: int = Field(
      0,
      ge=0,
      description="Start evaluation when training step is >= eval_start_step.",
  )
  eval_interval: int = Field(
      -1,
      description="Run evaluation every N training steps. -1 disables interval-based evaluation.",
  )
  eval_steps: int = Field(
      -1,
      description="Number of steps to run for each evaluation. -1 runs on entire eval split.",
  )
  target_eval_loss: float = Field(
      0.0,
      description="If set, training will stop early when this evaluation loss is reached.",
  )
  abort_on_nan_loss: bool = Field(True, description="Check for NaN values and abort training.")
  abort_on_inf_loss: bool = Field(True, description="Check for Inf values and abort training.")
  enable_dropout: bool = Field(True, description="Enables dropout in the model.")
  dropout_rate: float = Field(0.0, ge=0.0, le=1.0, description="The dropout rate.")
  enable_data_shuffling: bool = Field(True, description="Enables shuffling of the training data.")
  data_shuffle_seed: int = Field(0, description="Seed for data shuffling.")
  init_weights_seed: int = Field(0, description="Seed for model weight initialization.")
  max_inflight_computations: int = Field(2, description="Maximum number of inflight computations on device.")
  block_state_before_run_start: bool = Field(
      False,
      description=(
          "Before init_stop/run_start: jax.block_until_ready(state), evaluate the learning-rate schedule once "
          "(compiles its small eager ops) and run a cross-host barrier. Does not touch the dataset."
      ),
  )
  warm_input_reshard_before_run_start: bool = Field(
      False,
      description=(
          "Before init_stop/run_start: device_put an all-zero synthetic batch through the same host->global->input "
          "sharding path as the data loader, so the reshard program is compiled outside the timer. No dataset access."
      ),
  )
  warm_eval_input_reshard_before_run_start: bool = Field(
      False,
      description=(
          "Before init_stop/run_start: device_put an all-zero synthetic eval batch through the eval loop's "
          "host->global->eval-input sharding path (the same jax.device_put the eval loop performs), so the eval "
          "reshard program is compiled outside the timer. No dataset access."
      ),
  )
  eval_cache_prefill_in_background: bool = Field(
      False,
      description=(
          "Right after run_start (inside the timed region): a daemon thread iterates the eval tf.data pipeline once "
          "so its in-memory cache() is complete before the first eval; the eval file reads then overlap training "
          "steps instead of being paid serially inside the first eval. tf.data (c4_mlperf) eval loaders only."
      ),
  )
  train_shard_in_read: bool = Field(
      False,
      description=(
          "c4_mlperf train split: shard by file in tfds (host i opens files i, i+n, ...) instead of reading the full "
          "stream and keeping every n-th example. Needs at least as many files as data-loading hosts."
      ),
  )
  train_interleave_cycle_length: int = Field(
      -1,
      description=(
          "c4_mlperf train split: tfds interleave_cycle_length, i.e. number of files read concurrently per host "
          "(-1 = tfds default of 16). With train_shard_in_read, keep this below the files per host so that "
          "data_shuffle_seed changes which files are read first."
      ),
  )
