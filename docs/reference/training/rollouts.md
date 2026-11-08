# Rollouts and rewards

Import rollout configurations from `lilax.rollout`. See [GRPO guide](../../guides/training/grpo.md).

## RolloutConfig

Import from `lilax.rollout`. Pydantic model with keyword fields.

Selects rollout generation and execution through keyword fields. `None`, including an omitted `rollout_config`, selects vLLM.

`reshard_chunk_size` counts model keys per chunk when tensors move between trainer and rollout devices. `None` disables chunking, which can exhaust memory for large tensors. This option applies to the vLLM workflow.

`generation_config` is a `GenerationConfig`. `max_prompt_length` selects prompt capacity. Generation length derives from this prompt capacity plus `max_new_tokens` when supplied.

<details>
<summary>Parameters</summary>

- `generation_config` (`GenerationConfig | None`, optional, defaults to `None`). Generation settings.
- `max_prompt_length` (`int | None`, optional, defaults to `None`). Prompt token capacity.
- `engine` (`Literal["maxtext", "vllm"] | None`, optional, defaults to `None`). Rollout workflow.
- `reshard_chunk_size` (`int | None`, optional, defaults to `None`). Model keys per transfer chunk. `None` disables chunking.
- `cluster` (`RolloutClusterConfig | None`, optional, defaults to `None`). Device allocation.
- `vllm` (`VLLMRolloutConfig | None`, optional, defaults to `None`). vLLM options.
- `weight_sync` (`WeightSyncConfig | None`, optional, defaults to `None`). Weight transfer options.

</details>

## RolloutClusterConfig

Import from `lilax.rollout`. Pydantic model with keyword fields.

Device fractions must be greater than zero and at most one. Device allocation must leave devices for both roles.

<details>
<summary>Parameters</summary>

- `trainer_devices_fraction` (`float`, optional, defaults to `0.5`). Trainer device fraction.
- `sampler_devices_fraction` (`float`, optional, defaults to `0.5`). Sampler device fraction.
- `chips_per_vm` (`int`, optional, defaults to `4`). Accelerator chips per VM.
- `use_pathways` (`bool`, optional, defaults to `True`). Pathways multihost orchestration.
- `num_trainer_slices` (`int`, optional, defaults to `-1`). Trainer slices.
- `num_samplers_slices` (`int`, optional, defaults to `-1`). Sampler slices.
- `rollout_data_parallelism` (`int`, optional, defaults to `-1`). Rollout replicas. Set when requesting more than one replica.
- `rollout_tensor_parallelism` (`int`, optional, defaults to `-1`). Tensor parallelism per replica. Unspecified values are determined automatically.
- `rollout_expert_parallelism` (`int`, optional, defaults to `1`). Expert parallelism per replica.
- `inference_replicas` (`int`, optional, defaults to `1`). Fixed-reward sampler replicas.
- `inference_devices_per_replica` (`int`, optional, defaults to `4`). Devices per fixed-reward replica.
- `inference_rollouts` (`int`, optional, defaults to `1`). Fixed-reward rollout refresh cadence.
- `use_pathways_reshard` (`bool`, optional, defaults to `True`). Pathways transfer for fixed-reward parameters.
- `gc_collect_after_weight_sync` (`bool`, optional, defaults to `True`). Host garbage collection after synchronization to release obsolete device copies.

</details>

## VLLMRolloutConfig and WeightSyncConfig

### VLLMRolloutConfig

Import from `lilax.rollout`. Pydantic model with keyword fields.

Configures vLLM through keyword fields.

<details>
<summary>Parameters</summary>

- `kv_cache_buffer` (`int`, optional, defaults to `256`). KV cache buffer.
- `hbm_utilization_vllm` (`float`, optional, defaults to `0.72`). Target HBM fraction.
- `swap_space_vllm_gb` (`int`, optional, defaults to `2`). Swap space in gigabytes.
- `enable_dp_attention` (`bool`, optional, defaults to `False`). `attn_dp` mesh axis selection.
- `enable_expert_parallel` (`bool`, optional, defaults to `False`). Expert parallelism.
- `async_scheduling` (`bool`, optional, defaults to `False`). Asynchronous scheduling.
- `max_num_batched_tokens` (`int | None`, optional, defaults to `None`). Batched token limit.
- `max_num_seqs` (`int | None`, optional, defaults to `None`). Sequence limit.
- `vllm_block_size` (`int | None`, optional, defaults to `None`). Cache page size. `None` lets the backend choose from engine shape.
- `stop_strings` (`list[str] | None`, optional, defaults to `None`). Generation stop strings.
- `vllm_hf_config_path` (`str`, optional, defaults to `""`). HF configuration path for the model.
- `additional_config` (`dict[str, Any]`, optional, defaults to `{}`). Additional vLLM options.
- `hf_overrides` (`dict[str, Any]`, optional, defaults to `{}`). HF model configuration overrides.

</details>

### WeightSyncConfig

Import from `lilax.rollout`. Pydantic model with keyword fields.

`WeightSyncConfig` controls cache release, standalone conversion, converter transport, rollout weight dtype, tensor-parallel splits, and conversion diagnostics. These controls apply to the vLLM workflow, not the fixed-reward engine.

Transport and cache-release support also depend on the installed Tunix version.

<details>
<summary>Parameters</summary>

- `free_kv_cache_during_weight_sync` (`bool`, optional, defaults to `True`). Sampler KV pool release for a second weight copy. `False` retains the pool and resets prefix entries, requiring enough free HBM.
- `use_standalone_converter` (`bool`, optional, defaults to `False`). Standalone Torchax vLLM converter.
- `use_weight_converter` (`bool`, optional, defaults to `True`). Explicit trainer weight conversion before transfer.
- `use_raiden_ffi` (`bool | None`, optional, defaults to `None`). Raiden FFI transport selection.
- `rollout_backend` (`Literal["maxtext", "vllm_torchax"]`, optional, defaults to `"maxtext"`). Trainer-side converter backend.
- `rollout_weight_dtype` (`str`, optional, defaults to `""`). Target dtype. Empty uses `weight_dtype`.
- `weight_sync_debug` (`bool`, optional, defaults to `False`). Placement logging and serialized conversion steps to identify failing parameters.
- `log_weight_sync_time` (`bool`, optional, defaults to `False`). Completed wall-time logging with a barrier. Initial sync includes compilation.
- `kv_tp_size` (`int`, optional, defaults to `1`). Cache and attention tensor parallelism.
- `moe_mlp_tp_size` (`int`, optional, defaults to `1`). Expert MLP tensor parallelism.
- `vllm_load_format` (`str`, optional, defaults to `"dummy"`). Converter validation load format, `auto` or `dummy`.
- `debug_converter` (`bool`, optional, defaults to `False`). Key coverage checks, weight statistics, and GCS upload.
- `gcs_debug_path` (`str`, optional, defaults to `""`). Upload destination for layer-zero and global NumPy tensors during debug conversion.

</details>

## Rewards

These are implementation defaults, not recommended reward weights for other tasks. `GRPOTrainer.reward_funcs` supplies custom reward functions on compatible rollout workflows.

Custom rewards are called with keyword `prompts` and `completions`, each a list of text strings, plus selected additional keywords. Return a numeric list of the same length. Returning `None` or the wrong list length fails. Multiple reward functions are summed.

### MathRewardsConfig

Import from `lilax.rollout`. Pydantic model with keyword fields.

Keyword fields configure the built-in math rewards.

<details>
<summary>Parameters</summary>

- `exact_answer` (`float`, optional, defaults to `5.0`). Exact answer reward.
- `exact_format_match` (`float`, optional, defaults to `3.0`). Exact format reward.
- `white_space_format_match` (`float`, optional, defaults to `1.5`). Whitespace-insensitive format reward.
- `partial_format_match` (`float`, optional, defaults to `0.5`). Partial format reward.
- `ratio_guess_to_answer_high` (`float`, optional, defaults to `0.5`). Reward within the high guess-to-answer ratio threshold.
- `ratio_guess_to_answer_low` (`float`, optional, defaults to `0.25`). Reward within the low ratio threshold.
- `incorrect_format` (`float`, optional, defaults to `-0.5`). Incorrect format penalty.
- `incorrect_answer` (`float`, optional, defaults to `-1.0`). Incorrect answer penalty.
- `math_verify_timeout` (`int`, optional, defaults to `300`). Global math verification timeout in seconds per batch.
- `math_verify_num_procs` (`int | None`, optional, defaults to `None`). Verification processes. `None` uses the smaller of batch size and CPU count.
- `special_tokens` (`RewardSpecialTokens | None`, optional, defaults to `None`). Parsing delimiters.

</details>

### RewardSpecialTokens

Import from `lilax.rollout`. Pydantic model with keyword fields.

Keyword fields select response delimiters.

<details>
<summary>Parameters</summary>

- `reasoning_start_token` (`str`, optional, defaults to `"<reasoning>"`). Reasoning opening marker.
- `reasoning_end_token` (`str`, optional, defaults to `"</reasoning>"`). Reasoning closing marker.
- `reasoning_start_token_in_prompt` (`bool`, optional, defaults to `False`). Opening reasoning marker is prefilled in the prompt and absent from completion.
- `solution_start_token` (`str`, optional, defaults to `"<answer>"`). Solution opening marker.
- `solution_end_token` (`str`, optional, defaults to `"</answer>"`). Solution closing marker.

</details>

## Agentic environments

### AgenticRolloutConfig

Import from `lilax.rollout`. Pydantic model with keyword fields.

Keyword fields select agentic vLLM execution.

<details>
<summary>Parameters</summary>

- `enabled` (`bool`, optional, defaults to `False`). Agentic rollout selection.
- `max_concurrency` (`int`, optional, defaults to `256`). Concurrent request limit.
- `off_policy_steps` (`int`, optional, defaults to `0`). Allowed policy-step lag.
- `system_prompt` (`str`, optional, defaults to `""`). Prompt injected into the agent.
- `agent` (`AgentBinding | None`, optional, defaults to `None`). Agent and environment binding.

</details>

### AgentBinding

Import from `lilax.rollout`. Pydantic model with keyword fields.

Keyword fields bind classes and constructor options.

<details>
<summary>Parameters</summary>

- `agent_class` (`type`). Agent class.
- `environment_class` (`type`). Environment class.
- `agent_kwargs` (`dict[str, Any] | None`, optional, defaults to `None`). Agent constructor options.
- `environment_kwargs` (`dict[str, Any] | None`, optional, defaults to `None`). Environment constructor options.

</details>

### OpenEnv exports

`lilax.openenv` exports synchronous `OpenEnvAgent`, `OpenEnvEnvironment`, `InitialObservation`, and `TerminalStatus`.

### InitialObservation

Import from `lilax.openenv`. Dataclass preserving the originating task alongside the reset observation.

```python
InitialObservation(task: dict[str, Any], observation: Any)
```

`OpenEnvEnvironment.reset()` returns this record. `OpenEnvAgent.update_from_env()` copies its task into the trajectory before formatting the observation.

<details>
<summary>Parameters</summary>

- `task` (`dict[str, Any]`). Originating episode task.
- `observation` (`Any`). Initial client observation.

</details>

### TerminalStatus

Import from `lilax.openenv`. Alias of Tunix's [TrajectoryStatus enum](https://github.com/google/tunix/blob/385551df4643ee3ac3205e813a93c2ff217dffa5/tunix/rl/agentic/agents/agent_types.py#L90). A terminal status mapper must return a member other than `RUNNING`.

| Member | Meaning |
| --- | --- |
| `SUCCEEDED` | Successful episode. |
| `RUNNING` | Active episode. Invalid as a terminal mapper result. |
| `MAX_STEPS_REACHED` | Episode step limit reached. |
| `MAX_CONTEXT_LIMIT_REACHED` | Response token budget exhausted. |
| `TIMEOUT` | Episode timeout. |
| `ENV_TIMEOUT` | Environment operation timeout. |
| `FAILED` | Unhandled model, environment, or reward failure. |
| `CANCELLED` | Episode cancellation. |

### OpenEnvEnvironment

Import from `lilax.openenv`. Synchronous environment adapter class.

```python
OpenEnvEnvironment(task, *, group_id, pair_index, client_factory, reset_kwargs, reward_from_result, max_steps)
```

Creates a fresh synchronous client on reset. A terminal reset raises `ValueError`.
Resetting a closed environment raises `RuntimeError`.

<details>
<summary>Parameters</summary>

- `task` (`dict[str, Any]`). Episode task.
- `group_id` (`Any`). Opaque group identifier forwarded unchanged to the client and reset factories.
- `pair_index` (`Any`). Opaque pair index forwarded unchanged to the client and reset factories.
- `client_factory` (`Callable`). Factory receiving task, group ID, and pair index and returning a fresh synchronous client.
- `reset_kwargs` (`Callable`). Function receiving the same values and returning client reset arguments.
- `reward_from_result` (`Callable`). Mapper from client results to episode rewards.
- `max_steps` (`int`). Episode step limit stored by the environment.

</details>

#### reset

```python
environment.reset()
```

Returns `InitialObservation` and metadata. Creates and resets the client under the environment lock.

#### step

```python
environment.step(action)
```

Returns observation, mapped reward, done, and metadata.

<details>
<summary>Parameters</summary>

- `action` (`Any`). Action forwarded unchanged to the synchronous client. Its accepted value is client-specific.

</details>

#### close

```python
environment.close()
```

Closes the client and marks the environment closed.

Returns `None`.

### OpenEnvAgent

Import from `lilax.openenv`. Conversation agent class.

```python
OpenEnvAgent(
	system_prompt: str, action_from_response: Callable[[str], Any],
	observation_to_text: Callable[[Any], str], status_mapper: TerminalStatusMapper,
)
```

Converts model responses to actions and observations to messages using caller-supplied formatting and status mapping.

<details>
<summary>Parameters</summary>

- `system_prompt` (`str`). Conversation system message.
- `action_from_response` (`Callable[[str], Any]`). Converter from response text to environment actions.
- `observation_to_text` (`Callable[[Any], str]`). Converter from observations to conversation text.
- `status_mapper` (`TerminalStatusMapper`). Terminal observation and reward mapper to Tunix status.

</details>

#### update_from_model

```python
agent.update_from_model(response: str, **kwargs) -> Action
```

Appends the assistant response and a trajectory step.

Returns the converted Tunix action.

<details>
<summary>Parameters</summary>

- `response` (`str`). Model response.
- `**kwargs` (`Any`). Accepted but unused by this method.

</details>

#### update_from_env

```python
agent.update_from_env(observation: Any, reward: float, done: bool, info: dict[str, Any] | None = None, **kwargs) -> None
```

Records the initial task when present, maps terminal status on completion, and updates conversation state.

Returns `None`.

<details>
<summary>Parameters</summary>

- `observation` (`Any`). Environment observation or `InitialObservation`.
- `reward` (`float`). Step reward.
- `done` (`bool`). Episode completion.
- `info` (`dict[str, Any] | None`, optional, defaults to `None`). Step metadata.
- `**kwargs` (`Any`). Forwarded to the base agent's environment update.

</details>

## GRPO evaluation

### GRPOEvaluationConfig

Import from `lilax.training`. Pydantic model whose keyword fields configure evaluation.

Each supplied generation configuration translates only `temperature`, `top_k`, and `top_p`. `do_sample=False` sets the translated temperature to zero. Negative `top_p` becomes `1.0`. Evaluation ignores `max_new_tokens` and `eos_token_id` on these objects. Its generation budget is the resolved `max_target_length - max_prefill_predict_length`. A rollout configuration with `max_prompt_length` and `generation_config.max_new_tokens` sets those lengths during trainer construction. See the [matching-name example](../../guides/training/grpo.md#evaluate-and-generate).

<details>
<summary>Parameters</summary>

- `eval_batch_size` (`int`, optional, defaults to `-1`). Evaluation batch size.
- `sampling_strategy` (`str`, optional, defaults to `"greedy"`). Key in resolved generation settings. A missing key fails.
- `generation_configs` (`dict[str, GenerationConfig] | None`, optional, defaults to `None`). Named settings. `None` retains runtime-source settings.
- `num_eval_passes` (`int`, optional, defaults to `1`). Generation passes.
- `response_selection` (`Literal["correct", "incorrect"]`, optional, defaults to `"incorrect"`). Retained response selection.
- `return_responses` (`bool`, optional, defaults to `False`). Selected question, answer, and response records in metrics.
- `mode` (`Literal["pass", "maj", "pass_at_1"]`, optional, defaults to `"pass"`). Pass@K, majority@K, or pass@1 evaluation.

</details>

## Engine restrictions

The `"maxtext"` engine uses a fixed length reward and requires stochastic generation with positive temperature, one iteration, and per-device batch sizes of at least one. It rejects evaluation input, custom rewards, agentic environments, vLLM weight-sync settings, LoRA, and QAT. Its supported cluster fields are `inference_replicas`, `inference_devices_per_replica`, `inference_rollouts`, and `use_pathways_reshard`.

Explicitly setting any of these `GRPOConfig` fields raises `ValueError` with this engine, even when the value equals its default:

- `batch_size`, `train_micro_batch_size`, `rollout_micro_batch_size`, and `max_seq_token_per_tpu`.
- `epsilon_high`, `loss_type`, `loss_aggregation`, and `use_rollout_logps`.
- `reward_num_workers`, `reward_worker_timeout_seconds`, `rewards`, and `evaluation_config`.

For example, omit `loss_type` rather than pass `loss_type="grpo"`. Evaluation also rejects `eval_on_start=True` or positive `eval_steps`. Non-`None` `RolloutConfig.vllm`, `weight_sync`, or `reshard_chunk_size` is rejected. Explicit cluster fields outside the supported list are rejected even at their defaults.
