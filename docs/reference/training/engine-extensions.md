# Training engine extensions

Import from `lilax.trainers.engine`. See [engine guide](../../guides/training/engine-extensions.md).

## AbstractTrainingEngine

Import from `lilax.trainers.engine`. Abstract base class.

```python
AbstractTrainingEngine(training_config: TrainingConfig) -> None
```

Abstract protocol. `train_step` exposes the current step.
The signatures below describe this protocol. Concrete extensions follow under `MaxTextTrainingEngine`.

<details>
<summary>Parameters</summary>

- `training_config` (`TrainingConfig`). Protocol-level training settings.

</details>

### with_loss_fn

```python
engine.with_loss_fn(customized_fn: Callable[..., Any], has_aux: bool = False) -> AbstractTrainingEngine
```

Sets the loss function.

Returns the engine.

<details>
<summary>Parameters</summary>

- `customized_fn` (`Callable[..., Any]`). Loss callable.
- `has_aux` (`bool`, optional, defaults to `False`). Loss returns auxiliary values.

</details>

### with_gen_model_input_fn

```python
engine.with_gen_model_input_fn(gen_model_input_fn: Callable[[Any], dict[str, Any]]) -> AbstractTrainingEngine
```

Sets the payload-to-input adapter.

Returns the engine.

<details>
<summary>Parameters</summary>

- `gen_model_input_fn` (`Callable[[Any], dict[str, Any]]`). Payload-to-model keyword input adapter.

</details>

### compile

```python
engine.compile(dummy_data: TrainerPayload | None = None) -> None
```

Compiles representative data.

Returns `None`.

<details>
<summary>Parameters</summary>

- `dummy_data` (`TrainerPayload | None`, optional, defaults to `None`). Representative payload.

</details>

### fwd_bwd

```python
engine.fwd_bwd(payload: TrainerPayload) -> None
```

Runs forward and backward passes.

Returns `None`.

<details>
<summary>Parameters</summary>

- `payload` (`TrainerPayload`). Training batch.

</details>

### update

```python
engine.update() -> int
```

Applies accumulated gradients.

Returns the step.

### eval_step

```python
engine.eval_step(payload: TrainerPayload) -> None
```

Runs evaluation.

Returns `None`.

<details>
<summary>Parameters</summary>

- `payload` (`TrainerPayload`). Evaluation batch.

</details>

### fwd_only

```python
engine.fwd_only(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any
```

Runs a read-only function on the model.

Returns the function's result.

<details>
<summary>Parameters</summary>

- `fn` (`Callable[..., Any]`). Function receiving the model followed by supplied arguments.
- `*args` (`Any`). Positional function arguments.
- `**kwargs` (`Any`). Keyword function arguments.

</details>

### save_checkpoint

```python
engine.save_checkpoint(metadata: Any = None, *, step: int | None = None, force: bool = False) -> None
```

Saves model and optimizer state. The concrete engine rejects mid-step saves with `ValueError` and skips saves when checkpointing is disabled.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metadata` (`Any`, optional, defaults to `None`). Additional checkpoint metadata.
- `step` (`int | None`, optional, defaults to `None`). Saved step. Concrete implementation uses metadata's step or current step when absent.
- `force` (`bool`, optional, defaults to `False`). Save-scheduling bypass.

</details>

### restore_checkpoint

```python
engine.restore_checkpoint(*, step: int | None = None) -> Any
```

Restores model and optimizer state.

Returns checkpoint metadata, or `None` without a restored step in the concrete engine.

<details>
<summary>Parameters</summary>

- `step` (`int | None`, optional, defaults to `None`). Selected checkpoint step.

</details>

### get_metrics

```python
engine.get_metrics(clear_cache: bool = True) -> MetricsBuffer
```

Returns accumulated metrics. The concrete implementation returns only the newest step buffer and drops older buffers.
Call once per update to retain each step.

<details>
<summary>Parameters</summary>

- `clear_cache` (`bool`, optional, defaults to `True`). Retrieved metric history clearing.

</details>

### prepare_weight_sync

```python
engine.prepare_weight_sync(*, sync_request: Any = None) -> Any
```

Stages weights for transfer.

Returns implementation-specific staged weights.

<details>
<summary>Parameters</summary>

- `sync_request` (`Any`, optional, defaults to `None`). Synchronization request. Unused by the concrete engine.

</details>

### release_weight_sync

```python
engine.release_weight_sync(*, sync_request: Any = None) -> Any
```

Releases staged buffers.

Returns `True` in the concrete engine.

<details>
<summary>Parameters</summary>

- `sync_request` (`Any`, optional, defaults to `None`). Synchronization request. Unused by the concrete engine.

</details>

### close

```python
engine.close() -> None
```

Cleans up resources.

Returns `None`.

## TrainingConfig and metric records

### TrainingConfig

Import from `lilax.trainers.engine`. Dataclass with training hyperparameters and operational settings.

`TrainerPayload`, `RLTrainerPayload`, `LossOutput`, and `WeightedMetric` retain their Tunix dependency identities.

<details>
<summary>Parameters</summary>

- `eval_every_n_steps` (`int`, optional, defaults to `0`). Evaluation cadence.
- `max_steps` (`int | None`, optional, defaults to `None`). Training duration.
- `gradient_accumulation_steps` (`int | None`, optional, defaults to `None`). Accumulation count.
- `checkpoint_root_directory` (`str | None`, optional, defaults to `None`). Checkpoint root.
- `metrics_prefix` (`str`, optional, defaults to `""`). Metric prefix.
- `max_inflight_computations` (`int`, optional, defaults to `2`). Inflight computation limit.

</details>

### MetricsBuffer

Import from `lilax.trainers.engine`. Flax struct record for unreduced device metrics. Dictionary defaults are fresh per record.

<details>
<summary>Parameters</summary>

- `id` (`Any`). Buffer identity.
- `weighted_metrics` (`dict[str, WeightedMetric]`, optional, defaults to `{}`). Weighted values.
- `scalar_metrics` (`dict[str, jax.Array]`, optional, defaults to `{}`). Scalar values.
- `aggregation_fns` (`dict[str, Callable[[jax.Array], Any]]`, optional, defaults to `{}`). Per-metric reductions. Not a pytree node.
- `mode` (`str`, optional, defaults to `"train"`). Metric mode. Not a pytree node.

</details>

### TrainerPayload

Import from `lilax.trainers.engine`. Frozen, keyword-only Flax struct dataclass with no fields or constructor arguments. This is Tunix's base payload identity, not a dictionary schema. Concrete payloads supply the engine inputs.

See the [Tunix payload fields](https://github.com/google/tunix/blob/385551df4643ee3ac3205e813a93c2ff217dffa5/tunix/dist_rl/common/datatypes.py#L458).

### RLTrainerPayload

Import from `lilax.trainers.engine`. Frozen, keyword-only Flax struct dataclass extending `TrainerPayload`. Types below use `ArrayLike` from `jax.typing`. The five token, mask, and advantage fields are required.

Batched prompts and completions use widths `P` and `C`. Segment metadata can cover the full row or completions. Router replay requires full-row segment widths. `routed_experts` can carry raw `int16`, shifted `uint8`, or packed shifted `uint10` data. The dependency's routing decoders interpret those encodings. Lilax's `router_replay_gen_model_input_fn` forwards the array without decoding, so that path requires decoded expert IDs. Static fields `num_segments`, `sampler_agreement_applied`, and `metadata` are not pytree nodes. Metadata defaults to a fresh dictionary.

See the [engine construction example](../../guides/training/engine-extensions.md) for a payload built from caller-supplied arrays.

<details>
<summary>Parameters</summary>

- `prompt_ids` (`ArrayLike`). Prompt token IDs. Unbatched rows may be one-dimensional before batch assembly.
- `prompt_mask` (`ArrayLike`). Prompt validity mask, batched as `(B, P)`.
- `completion_ids` (`ArrayLike`). Completion token IDs, batched as `(B, C)`.
- `completion_mask` (`ArrayLike`). Completion or action validity mask.
- `advantages` (`ArrayLike`). Advantages with shape `(B,)` or `(B, C)`.
- `segment_ids` (`ArrayLike | None`, optional, defaults to `None`). Packing segments.
- `segment_positions` (`ArrayLike | None`, optional, defaults to `None`). Positions within packing segments.
- `ref_per_token_logps` (`ArrayLike | None`, optional, defaults to `None`). Reference-policy token log probabilities.
- `old_per_token_logps` (`ArrayLike | None`, optional, defaults to `None`). Behavior-policy token log probabilities.
- `sampler_is_weights` (`ArrayLike | None`, optional, defaults to `None`). Sampler importance weights.
- `routed_experts` (`ArrayLike | None`, optional, defaults to `None`). Encoded rollout expert selections.
- `returns` (`ArrayLike | None`, optional, defaults to `None`). Value-baseline returns.
- `old_values` (`ArrayLike | None`, optional, defaults to `None`). Previous critic estimates.
- `num_segments` (`int | None`, optional, defaults to `None`). Static packed-segment upper bound.
- `sampler_agreement_applied` (`bool`, optional, defaults to `False`). Static sampler-agreement marker.
- `metadata` (`dict[str, Any]`, optional, defaults to `{}`). Extra static payload metadata.

</details>

### WeightedMetric

Import from `lilax.trainers.engine`. Flax struct dataclass retaining the [pinned Tunix metric identity](https://github.com/google/tunix/blob/385551df4643ee3ac3205e813a93c2ff217dffa5/tunix/sft/utils.py#L173). Stores scalar numerator and denominator arrays for weighted reduction. `eps` and `min_denom` are static fields.

`metric.compute_scale()` returns the reciprocal denominator after adding `eps` and then applying `min_denom`, when supplied. A resulting zero denominator returns zero scale. `metric.compute()` returns `unreduced_sum * metric.compute_scale()`.

<details>
<summary>Parameters</summary>

- `unreduced_sum` (`jax.Array`). Scalar metric numerator.
- `denominator` (`jax.Array`). Scalar weight or valid-item count.
- `eps` (`float | None`, optional, defaults to `None`). Term added to the denominator before bounding.
- `min_denom` (`float | None`, optional, defaults to `None`). Minimum denominator after the epsilon adjustment.

</details>

### LossOutput

Import from `lilax.trainers.engine`. Flax struct dataclass retaining the [pinned Tunix loss-result identity](https://github.com/google/tunix/blob/385551df4643ee3ac3205e813a93c2ff217dffa5/tunix/sft/utils.py#L272). Both fields are required. A scalar alone is not this loss contract.

For scalar JAX arrays `loss_sum` and `token_count` from your objective, construct the result with:

```python
from lilax.trainers.engine import LossOutput, WeightedMetric

result = LossOutput(
	primary_loss=WeightedMetric(unreduced_sum=loss_sum, denominator=token_count),
	aux_metrics={},
)
```

<details>
<summary>Parameters</summary>

- `primary_loss` (`WeightedMetric`). Unreduced loss to optimize.
- `aux_metrics` (`Mapping[str, WeightedMetric | jax.Array]`). Auxiliary metrics. An empty mapping is accepted and must be supplied explicitly.

</details>

## MaxTextTrainingEngine

Import from `lilax.trainers.engine`. Concrete training engine class.

```python
MaxTextTrainingEngine(
	training_config, mesh=None, wrap_with_tunix_adapter=False, tokenizer_pad_id=None, *, model=None,
) -> None
```

Requires resolved configuration (`runtime.config`) produced by `runtime = model.runtime.initialize()`. The model's `config`, runtime-source configuration (`model.runtime.public_config`), and protocol `TrainingConfig` have different identities and are rejected. See [configuration objects](../execution/runtime.md#configuration-objects).

Adapter wrapping requires a mesh and tokenizer pad ID. LoRA is unsupported. Model name is required. The constructor can create model and optimizer state and checkpoint resources.

`compile()` without data defers compilation until a payload arrives. `get_metrics()` returns the newest step buffer; call once per update to retain each step. Public state properties are `model`, `optimizer`, `state`, `train_step`, `total_micro_steps`, `micro_step_count`, and `has_accumulated_grads`.

<details>
<summary>Parameters</summary>

- `training_config` (`object`). Resolved configuration with model name from the initialized model runtime.
- `mesh` (`jax.sharding.Mesh | None`, optional, defaults to `None`). Execution mesh. Required for adapter wrapping.
- `wrap_with_tunix_adapter` (`bool`, optional, defaults to `False`). Tunix input adapter wrapping.
- `tokenizer_pad_id` (`int | None`, optional, defaults to `None`). Padding ID required for wrapping.
- `model` (`nnx.Module | None`, optional, defaults to `None`). Existing model. Otherwise constructs a model.

</details>

### compile

```python
engine.compile(
	dummy_data: TrainerPayload | None = None, compiler_options: dict[str, Any] | None = None,
) -> None
```

Requests compilation. Without data, defers to the first payload.

Returns `None`.

<details>
<summary>Parameters</summary>

- `dummy_data` (`TrainerPayload | None`, optional, defaults to `None`). Representative payload.
- `compiler_options` (`dict[str, Any] | None`, optional, defaults to `None`). XLA compiler options.

</details>

### compile_kernels

```python
engine.compile_kernels(
	dummy_data: TrainerPayload, compiler_options: dict[str, Any] | None = None,
) -> dict[str, jax.stages.Compiled]
```

Returns compiled training kernels.

<details>
<summary>Parameters</summary>

- `dummy_data` (`TrainerPayload`). Representative payload.
- `compiler_options` (`dict[str, Any] | None`, optional, defaults to `None`). XLA options.

</details>

### eval_context

```python
engine.eval_context()
```

Returns the evaluation context manager.

### record_metrics

```python
engine.record_metrics(
	name: str, metric: WeightedMetric | jax.Array | float | int | dict[str, Any],
	aggregation_fn: Callable[[jax.Array], Any] | None = None, mode: Mode = Mode.TRAIN,
) -> None
```

Buffers metrics for training or evaluation.

Returns `None`.

<details>
<summary>Parameters</summary>

- `name` (`str`). Metric name.
- `metric` (`WeightedMetric | jax.Array | float | int | dict[str, Any]`). Metric or nested metric dictionary.
- `aggregation_fn` (`Callable[[jax.Array], Any] | None`, optional, defaults to `None`). Metric reduction.
- `mode` (`Mode`, optional, defaults to `Mode.TRAIN`). Training or evaluation mode from the engine metrics module.

</details>

### set_target_state

```python
engine.set_target_state(target_state: Any) -> None
```

Selects the weight-sync target state.

Returns `None`.

<details>
<summary>Parameters</summary>

- `target_state` (`Any`). Transfer destination state.

</details>

### prepare_weight_sync

```python
engine.prepare_weight_sync(*, sync_request: Any = None, staging_transport: str = "raiden") -> Any
```

Returns staged trainable parameters. Unknown transport names raise `ValueError`.

<details>
<summary>Parameters</summary>

- `sync_request` (`Any`, optional, defaults to `None`). Unused synchronization request.
- `staging_transport` (`str`, optional, defaults to `"raiden"`). Only supported transport.

</details>

## Router replay

### router_replay_gen_model_input_fn

```python
router_replay_gen_model_input_fn(payload: RLTrainerPayload) -> dict[str, Any]
```

Joins prompt and completion IDs, masks, positions, and supplied forced routing. Packed segment widths must match
the full model row width or `ValueError` is raised.

Returns model keyword inputs.

<details>
<summary>Parameters</summary>

- `payload` (`RLTrainerPayload`). Rollout training payload.

</details>

### make_router_replay_loss_fn

```python
make_router_replay_loss_fn(
	config, dropout_rng=None,
) -> Callable[..., Any]
```

Returns the replay loss function. Enabled dropout without an RNG raises `ValueError`.

`config` accepts resolved configuration (`runtime.config`) from `runtime = model.runtime.initialize()`. See [configuration objects](../execution/runtime.md#configuration-objects).

<details>
<summary>Parameters</summary>

- `config` (`object`). Compatible resolved configuration from the initialized model runtime.
- `dropout_rng` (`jax.Array | None`, optional, defaults to `None`). Dropout random key. Required when dropout is enabled.

</details>
