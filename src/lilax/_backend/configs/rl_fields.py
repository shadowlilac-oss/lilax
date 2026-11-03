"""Native RL cluster, algorithm, dataset, evaluation, reward, and special-token configuration fields."""

from typing import Any, Literal, Optional

from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import PositiveInt


class RLCluster(BaseModel):
  """Cluster configurations specific to RL training."""

  trainer_devices_fraction: float = Field(0.5, gt=0.0, le=1.0, description="Fraction of devices to use for the trainer.")
  sampler_devices_fraction: float = Field(0.5, gt=0.0, le=1.0, description="Fraction of devices to use for the sampler.")
  chips_per_vm: int = Field(4, description="Number of accelerator chips per VM.")
  use_pathways: bool = Field(True, description="Whether to use Pathways for multihost orchestration.")
  num_trainer_slices: int = Field(-1, description="Number of slices for the trainer.")
  num_samplers_slices: int = Field(-1, description="Number of slices for the samplers.")
  rollout_data_parallelism: int = Field(
      -1,
      description="Total model replicas for rollout. It should only be specified when you would like to use more "
      "than one model replica in rollout.",
  )
  rollout_tensor_parallelism: int = Field(
      -1,
      description="Tensor parallelism per replica for rollout. If not specified, it will be auto-determined.",
  )
  rollout_expert_parallelism: int = Field(1, description="Expert parallelism per replica for rollout")
  inference_replicas: int = Field(1, description="Legacy experimental GRPO: number of inference (sampler) replicas.")
  inference_devices_per_replica: int = Field(
      4, description="Legacy experimental GRPO: devices per inference replica (single-controller device split)."
  )
  inference_rollouts: int = Field(
      1, description="Legacy experimental GRPO: refresh rollouts every N steps (step % inference_rollouts)."
  )
  use_pathways_reshard: bool = Field(
      True, description="Legacy experimental GRPO: use Pathways resharding to move policy params to the sampler."
  )
  gc_collect_after_weight_sync: bool = Field(
      True,
      description=(
          "Run a full host gc.collect() after every trainer->sampler weight sync "
          "(tunix ClusterConfig.gc_collect_after_weight_sync). Each sync leaves another copy of the weights on "
          "HBM until Python's garbage collector releases it; with this disabled the copies accumulate and the "
          "run can OOM over time. The collection costs about one second per step on a colocated setup, so "
          "disable it only when there is enough HBM headroom."
      ),
  )


class RL(BaseModel):
  """Configuration for RL algorithms like Group Relative Policy Optimization (GRPO) among others."""

  profiler_start_invocation: int = Field(
      2,
      ge=0,
      description=(
          "Zero-based invocation index at which tracing starts for each RL"
          " role: `update_actor` for the actor and `generate` for the rollout"
          " engine (tunix `RLProfileConfig.start_invocation`). This counts role"
          " function calls, not global training steps: `update_actor` runs once"
          " per micro-batch (mini-batch x gradient accumulation x"
          " num_iterations per step) and eval actor passes share the actor"
          " counter. Default 2 skips JIT compilation and warm-up."
      ),
  )
  profiler_num_invocations: int = Field(
      1,
      ge=1,
      description=(
          "Number of consecutive invocations traced per role once"
          " `profiler_start_invocation` is reached (tunix"
          " `RLProfileConfig.num_invocations`). One session is written per"
          " invocation as `<role>_invocation_<n>`."
      ),
  )
  profiler_timeout_secs: float = Field(
      60.0,
      gt=0,
      description=(
          "Seconds to wait for tunix's background trace stop and upload thread"
          " before starting the next trace or closing the cluster (tunix"
          " `RLProfileConfig.timeout_secs`)."
      ),
  )
  profiler_max_num_hosts: PositiveInt = Field(
      1,
      description=(
          "Pathways only: maximum number of worker hosts to trace with the"
          " xplane profiler across all RL roles (trainer and sampler). Defaults"
          " to 1 (worker 0 only). Values > 1 trace worker hosts across all"
          " slices on the Pathways backend."
      ),
  )

  num_generations: int = Field(2, description="Number of responses to generate per prompt (G in GRPO paper).")
  num_iterations: int = Field(1, ge=1, description="Number of iterations per batch (μ in GRPO paper).")
  grpo_beta: float = Field(0.08, description="Coefficient for the KL divergence penalty (β).")
  grpo_epsilon: float = Field(0.2, description="Epsilon value for clipping in the GRPO loss.")
  loss_algo: Literal["grpo", "gspo-token"] = Field("grpo", description="Loss algorithm, i.e., 'grpo' or 'gspo-token'.")
  loss_agg_mode: Literal["token-mean", "sequence-mean", "sequence-mean-token-mean"] = Field(
      "sequence-mean-token-mean",
      description="Specifies the method for aggregating loss across the batch.",
  )
  use_agentic_rollout: bool = Field(
      False,
      description="If True, uses the asynchronous AgenticGRPOLearner for online vLLM rollouts.",
  )
  max_concurrency: int = Field(
      256,
      description="Maximum number of concurrent rollout requests (agentic rollout only).",
  )
  off_policy_steps: int = Field(
      0,
      description="Number of off-policy steps tolerated before requiring a policy update (agentic only).",
  )
  system_prompt: str = Field(
      "",
      description="System prompt injected into the agent at rollout time (agentic only).",
  )
  epsilon_high: Optional[float] = Field(
      None,
      description="Upper-bound clipping epsilon for GRPO loss. Defaults to epsilon when None (agentic only).",
  )
  reshard_chunk_size: Optional[int] = Field(
      None,
      description=(
          "Number of model keys to chunk for resharding tensors between trainer and rollout devices."
          "If None, no chunking is applied, which may lead to OOM errors if tensors are too large."
      ),
  )
  use_rollout_logps: bool = Field(
      True,
      description=(
          "Use rollout engine's logprobs as old_per_token_logps. "
          "False selects the step-0 re-forward path (trainer recomputes them)"
      ),
  )


class RLDataset(BaseModel):
  """Dataset settings for RL training."""

  dataset_name: str = Field("openai/gsm8k", description="Name of the training dataset.")
  eval_dataset_name: str = Field("openai/gsm8k", description="Name of the evaluation dataset.")
  train_split: str = Field("train", description="Dataset split for training.")
  eval_split: str = Field("test", description="Dataset split for evaluation.")
  hf_subset: None | str = Field(None, description="Subset name of the Hugging Face dataset.")
  hf_train_files: None | str = Field(None, description="Files for the HF training split.")
  hf_eval_files: None | str = Field(None, description="Files for the HF evaluation split.")
  data_shuffle_seed: int = Field(42, description="Seed for shuffling the dataset.")
  batch_size: int = Field(1, description="Global batch size for the dataset loader in RL.")
  num_batches: int = Field(4, ge=1, description="Number of batches for RL training.")
  num_test_batches: int = Field(5, description="Number of batches for RL evaluation.")
  eval_batch_size: int = Field(-1, description="Batch size for RL evaluation.")
  test_batch_start_index: int = Field(0, description="Start index for the test dataset")
  train_fraction: float = Field(1.0, gt=0.0, le=1.0, description="Fraction of the dataset to be used for training.")
  train_micro_batch_size: int = Field(-1, description="Micro batch size for training.")
  rollout_micro_batch_size: int = Field(-1, description="Micro batch size for rollout.")
  max_seq_token_per_tpu: int = Field(
      0,
      ge=0,
      description=(
          "Token budget per packed training row (Tunix `max_seq_token_per_tpu`). When > 0, rollout sequences are packed "
          "into rows of this many tokens for the actor/reference passes instead of one padded row per sequence; a maximal "
          "sequence (max_prefill_predict_length + generation length) must fit in one row. 0 disables packing."
      ),
  )
  dataset_processor_path: str = Field(
      "",
      description=(
          "Optional path to a user-provided Python file with a `process_data` function. "
          "When set, replaces the built-in dataset processor for custom datasets."
      ),
  )


class RLEvaluation(BaseModel):
  """Settings for RL evaluation."""

  eval_sampling_strategy: str = Field("greedy", description="Sampling strategy for evaluation.")
  generation_configs: dict[str, Any] = Field(
      default_factory=dict,
      description="Configurations for different generation strategies.",
  )
  num_eval_passes: int = Field(1, description="Number of generation passes during evaluation.")
  eval_corr_lst: bool = Field(
      False,
      description="If True, only include correct responses in the list during evaluation.",
  )
  eval_make_lst: bool = Field(
      False,
      description="If True, return a list of (question, answer, responses) during evaluation.",
  )
  eval_mode: Literal["pass", "maj", "pass_at_1"] = Field(
      "pass",
      description="Evaluation mode to use ('pass' for pass@K, 'maj' for maj@K, 'pass_at_1' for pass@1 estimation).",
  )


class RLReward(BaseModel):
  """Configuration for the reward/penalty model in RL."""

  reward_exact_answer: float = Field(5.0, description="Reward for an exact answer match.")
  reward_exact_format_match: float = Field(3.0, description="Reward for an exact format match.")
  reward_white_space_format_match: float = Field(1.5, description="Reward for a format match ignoring whitespace.")
  reward_partial_format_match: float = Field(0.5, description="Reward for a partial format match.")
  reward_ratio_guess_to_answer_high: float = Field(
      0.5, description="Reward when guess/answer ratio is within a high threshold."
  )
  reward_ratio_guess_to_answer_low: float = Field(
      0.25, description="Reward when guess/answer ratio is within a low threshold."
  )
  penalty_incorrect_format: float = Field(-0.5, description="Penalty for an incorrect format.")
  penalty_incorrect_answer: float = Field(-1.0, description="Penalty for an incorrect answer.")
  math_verify_timeout: int = Field(300, description="Global timeout (seconds) for math_verify calls per batch.")
  math_verify_num_procs: int | None = Field(
      None,
      description=("Max worker processes for the math_verify pool. None ⇒ " "min(batch_size, cpu_count())."),
  )
  reward_num_workers: int = Field(
      0,
      description=(
          "Worker processes tunix uses to evaluate the reward functions over the batch "
          "(GrpoConfig.reward_num_workers). 0 = serial (default), -1 = one worker per CPU."
      ),
  )
  reward_worker_timeout_seconds: float = Field(
      180.0,
      description="Seconds to wait for one reward-function chunk in a worker before falling back to the parent process.",
  )
  reward_functions_path: str = Field(
      "",
      description=(
          "Optional path to a user Python file containing custom reward functions. "
          "Used with `reward_functions` to fully replace the built-in reward stack."
      ),
  )
  reward_functions: str = Field(
      "",
      description=(
          "Comma-separated names of reward functions to import from `reward_functions_path`. "
          "Each function signature: (prompts, completions, tmvp_config, **kwargs) -> list[float]."
      ),
  )


class RLSpecialTokens(BaseModel):
  """Special tokens used for formatting prompts and responses in RL."""

  reasoning_start_token: str = Field("<reasoning>", description="Token to mark the beginning of a reasoning section.")
  reasoning_end_token: str = Field("</reasoning>", description="Token to mark the end of a reasoning section.")
  reasoning_start_token_in_prompt: bool = Field(
      False,
      description="Whether the chat template prefilled the reasoning start token, so it is absent from the completion.",
  )
  solution_start_token: str = Field("<answer>", description="Token to mark the beginning of a solution section.")
  solution_end_token: str = Field("</answer>", description="Token to mark the end of a solution section.")
