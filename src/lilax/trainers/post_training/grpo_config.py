"""GRPO algorithm and evaluation preferences with data-derived duration."""

from typing import Literal

from lilax.rollout import MathRewardsConfig
from lilax.training.grpo import GRPOEvaluationConfig
from ..arguments.training_args import TrainingArguments


class GRPOConfig(TrainingArguments):
  max_steps: None = None
  gradient_accumulation_steps: None = None
  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.08
  epsilon: float = 0.2
  epsilon_high: float | None = None
  loss_type: Literal["grpo", "gspo-token"] = "grpo"
  loss_aggregation: Literal["token-mean", "sequence-mean", "sequence-mean-token-mean"] = "sequence-mean-token-mean"
  use_rollout_logps: bool = True
  reward_num_workers: int = 0
  reward_worker_timeout_seconds: float = 180.0
  rewards: MathRewardsConfig | None = None
  batch_size: int = 1
  num_batches: int = 4
  train_micro_batch_size: int = -1
  rollout_micro_batch_size: int = -1
  max_seq_token_per_tpu: int = 0
  evaluation_config: GRPOEvaluationConfig | None = None

  def _overrides(self):
    values = super()._overrides()
    values.update({name: getattr(self, name) for name in (
        "num_generations", "num_iterations", "epsilon_high", "use_rollout_logps", "reward_num_workers",
        "reward_worker_timeout_seconds", "batch_size", "num_batches", "train_micro_batch_size",
        "rollout_micro_batch_size", "max_seq_token_per_tpu",
    )})
    values.update(grpo_beta=self.beta, grpo_epsilon=self.epsilon, loss_algo=self.loss_type,
                  loss_agg_mode=self.loss_aggregation)
    if self.rewards is not None:
      values.update(self.rewards._source_values())
    if self.evaluation_config is not None:
      values.update(self.evaluation_config._source_values())
    return values
