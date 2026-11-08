# Trainer constructors and methods

Import trainer classes and task configurations from `lilax`.

## Constructors

Trainers require model objects, not repository strings. `args` is keyword-only.

These constructors initialize the model runtime and training mesh when needed, before `train()`. GRPO also allocates trainer and sampler devices. Distillation initializes the teacher runtime on the student's mesh. Weight construction or restoration remains deferred until training or another operation needs weights. Set device placement, mesh, and sharding before construction. Incompatible changes after initialization raise `ValueError`. See [runtime controls](../execution/runtime.md).

## Trainer

Import from `lilax`. Causal trainer class.

```python
Trainer(
	model=None, *, args, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None,
	callbacks=None, optimizers=(None, None), model_init=None,
)
```

Creates a causal training workflow. Missing both `model` and `model_init` raises `ValueError`.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel | None`, optional, defaults to `None`). Model to train.
- `args` (`TrainingArguments`). Training configuration.
- `data_collator` (`Callable | None`, optional, defaults to `None`). Custom batching. See [collator contract](#data_collator).
- `train_dataset` (`Dataset | IterableDataset | SyntheticDataset | TeacherLogitsDataset | None`, optional, defaults to `None`). Training input.
- `eval_dataset` (`Dataset | IterableDataset | TeacherLogitsDataset | dict[str, Dataset | IterableDataset | TeacherLogitsDataset] | None`, optional, defaults to `None`). Finite evaluation input.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin | None`, optional, defaults to `None`). Input processing and persistence.
- `callbacks` (`Sequence[TrainerCallback | type[TrainerCallback]] | None`, optional, defaults to `None`). Callback instances or classes.
- `optimizers` (`tuple[optax.GradientTransformation | None, Callable | None]`, optional, defaults to `(None, None)`). Optax transformation and step-to-rate callable. See [optimizer contract](#optimizers).
- `model_init` (`Callable[[], MaxTextModel] | None`, optional, defaults to `None`). Model factory when `model` is absent.

</details>

## SFTTrainer

Import from `lilax`. Supervised fine-tuning trainer class.

```python
SFTTrainer(
	model, *, args: SFTConfig, formatting_func=None, data_collator=None, train_dataset=None,
	eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None),
)
```

Creates supervised fine-tuning. Multimodal input rejects `formatting_func`. SFT rejects mmap data settings.
See [SFT guide](../../guides/training/sft.md).

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Model to fine-tune.
- `args` (`SFTConfig`). Training configuration.
- `formatting_func` (`Callable[[dict], str | list[str]] | None`, optional, defaults to `None`). Record formatter returning a string or list of strings.
- `data_collator` (`Callable | None`, optional, defaults to `None`). [Custom batching](#data_collator).
- `train_dataset` (`Dataset | IterableDataset | SyntheticDataset | TeacherLogitsDataset | None`, optional, defaults to `None`). Training input.
- `eval_dataset` (`Dataset | IterableDataset | TeacherLogitsDataset | dict[str, Dataset | IterableDataset | TeacherLogitsDataset] | None`, optional, defaults to `None`). Finite evaluation input.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin | None`, optional, defaults to `None`). Input processing.
- `callbacks` (`Sequence[TrainerCallback | type[TrainerCallback]] | None`, optional, defaults to `None`). Callback instances or classes.
- `optimizers` (`tuple[optax.GradientTransformation | None, Callable | None]`, optional, defaults to `(None, None)`). [Optax transformation and schedule](#optimizers).

</details>

## SFTConfig

Import from `lilax`. Pydantic model extending [TrainingArguments](training-arguments.md) with the keyword fields below.

SFT rejects mmap data settings. See [SFT guide](../../guides/training/sft.md).

<details>
<summary>Parameters</summary>

- `learning_rate` (`float`, optional, defaults to `2e-5`). Peak learning rate, replacing the common default.
- `max_length` (`int | None`, optional, defaults to `None`). Sequence length override.
- `packing` (`bool`, optional, defaults to `True`). Training example packing.
- `completion_only_loss` (`bool`, optional, defaults to `True`). Loss restricted to completion tokens.
- `dataset_text_field` (`str`, optional, defaults to `"text"`). Dataset text column.
- `xprof_dir` (`str | None`, optional, defaults to `None`). XProf output directory selection.

</details>

## DPOTrainer

Import from `lilax`. Preference trainer class.

```python
DPOTrainer(
	model, *, args: DPOConfig, data_collator=None, train_dataset=None, eval_dataset=None,
	processing_class=None, callbacks=None, optimizers=(None, None),
)
```

Creates DPO or ORPO preference training. See [preference guide](../../guides/training/dpo.md).

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Policy model.
- `args` (`DPOConfig`). Preference configuration.
- `data_collator` (`Callable | None`, optional, defaults to `None`). [Preference batching](#data_collator).
- `train_dataset` (`Dataset | IterableDataset | None`, optional, defaults to `None`). Preference input.
- `eval_dataset` (`Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None`, optional, defaults to `None`). Finite evaluation input.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin | None`, optional, defaults to `None`). Input processing.
- `callbacks` (`Sequence[TrainerCallback | type[TrainerCallback]] | None`, optional, defaults to `None`). Callback instances or classes.
- `optimizers` (`tuple[optax.GradientTransformation | None, Callable | None]`, optional, defaults to `(None, None)`). [Optax transformation and schedule](#optimizers).

</details>

## DPOConfig

Import from `lilax`. Pydantic model extending [TrainingArguments](training-arguments.md). One preference trainer handles DPO and ORPO.
See [preference guide](../../guides/training/dpo.md).

The model configuration's `max_position_embeddings` maps to runtime `max_target_length`. Set it on the configuration passed to `AutoModelForCausalLM.from_pretrained` or `from_config`, before loading or materializing the model. `max_prompt_length=None` resolves to `max_target_length // 2`. A value greater than or equal to the resolved `max_target_length` raises `ValueError`. DPO disables packing unless the model runtime explicitly set it. An explicit `packing=True` raises `ValueError`.

<details>
<summary>Parameters</summary>

- `algorithm` (`Literal["dpo", "orpo"]`, optional, defaults to `"dpo"`). Preference objective.
- `beta` (`float`, optional, defaults to `0.1`). DPO beta.
- `label_smoothing` (`float`, optional, defaults to `0.0`). DPO label smoothing.
- `orpo_lambda` (`float`, optional, defaults to `0.1`). ORPO loss coefficient.
- `max_prompt_length` (`int | None`, optional, defaults to `None`). Prompt capacity. `None` uses half the target length.

</details>

## DistillationTrainer

Import from `lilax`. Teacher distillation trainer class.

```python
DistillationTrainer(
	model, teacher_model, *, args: DistillationConfig, data_collator=None, train_dataset=None,
	eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None),
)
```

Creates teacher distillation. Teacher and student require the same mesh and compatible batch settings.
`teacher_model=None` requires `TeacherLogitsDataset`, otherwise construction raises `ValueError`.
See [distillation guide](../../guides/training/distillation.md).

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Student model.
- `teacher_model` (`MaxTextModel | None`). Teacher, or `None` for offline logits.
- `args` (`DistillationConfig`). Distillation configuration.
- `data_collator` (`Callable | None`, optional, defaults to `None`). [Custom HF-input batching](#data_collator).
- `train_dataset` (`Dataset | IterableDataset | TeacherLogitsDataset | None`, optional, defaults to `None`). Training input.
- `eval_dataset` (`Dataset | IterableDataset | TeacherLogitsDataset | dict[str, Dataset | IterableDataset | TeacherLogitsDataset] | None`, optional, defaults to `None`). Finite evaluation input.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin | None`, optional, defaults to `None`). Input processing.
- `callbacks` (`Sequence[TrainerCallback | type[TrainerCallback]] | None`, optional, defaults to `None`). Callback instances or classes.
- `optimizers` (`tuple[optax.GradientTransformation | None, Callable | None]`, optional, defaults to `(None, None)`). [Optimizer and distillation schedule contract](#optimizers).

</details>

## DistillationConfig

Import from `lilax`. Pydantic model extending [TrainingArguments](training-arguments.md) with distillation keyword fields.

A nonconstant schedule requires its matching end value. `alpha_end` must be in `[0, 1]`, `temperature_end` must be positive, and `beta_end` must be nonnegative. Positive initial `beta` requires `scan_layers=True`. Starting at `beta=0` and ending at positive `beta_end` raises `ValueError` because feature extraction starts disabled. These invalid combinations raise `ValueError`.

Constant schedules retain the initial value even when an end value is supplied. Linear and cosine schedules interpolate using progress clipped to `[0, 1]` over the configured training steps. The feature loss is added to the logit loss, rather than sharing a normalized weight budget with `alpha`.

<details>
<summary>Parameters</summary>

- `alpha` (`float`, optional, defaults to `0.5`). Soft-logit loss mixture coefficient.
- `temperature` (`float`, optional, defaults to `1.0`). Distillation temperature.
- `beta` (`float`, optional, defaults to `0.0`). Feature-loss coefficient, added to logit loss.
- `feature_loss_type` (`Literal["cosine", "l2"]`, optional, defaults to `"cosine"`). Feature comparison.
- `layer_indices` (`list[int] | None`, optional, defaults to `None`). Features selected for loss.
- `alpha_end` (`float | None`, optional, defaults to `None`). Scheduled final alpha in `[0, 1]`.
- `alpha_schedule` (`Literal["constant", "linear", "cosine"]`, optional, defaults to `"constant"`). Alpha interpolation.
- `temperature_end` (`float | None`, optional, defaults to `None`). Positive final temperature.
- `temperature_schedule` (`Literal["constant", "linear", "cosine"]`, optional, defaults to `"constant"`). Temperature interpolation.
- `beta_end` (`float | None`, optional, defaults to `None`). Nonnegative final feature-loss weight.
- `beta_schedule` (`Literal["constant", "linear", "cosine"]`, optional, defaults to `"constant"`). Feature-loss interpolation.
- `learn_to_init` (`LearnToInitConfig | None`, optional, defaults to `None`). [Weight-copy, parameter-filter, and structural settings](#learn-to-init-maps-and-parameter-selection).

</details>

## GRPOTrainer

Import from `lilax`. Group-relative policy optimization trainer class.

```python
GRPOTrainer(
	model, reward_funcs=None, *, args: GRPOConfig, train_dataset=None, eval_dataset=None,
	processing_class=None, callbacks=None, optimizers=(None, None), ref_model=None, environment_factory=None,
	rollout_config=None, agentic_config=None,
)
```

Creates GRPO training with engine-specific [rollout restrictions](rollouts.md#engine-restrictions).
Environment factories require vLLM and `AgenticRolloutConfig(enabled=True)`. Environment evaluation is unsupported.
See [GRPO guide](../../guides/training/grpo.md).

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Policy model.
- `reward_funcs` (`Callable | Sequence[Callable] | None`, optional, defaults to `None`). [Custom reward protocol](rollouts.md#rewards).
- `args` (`GRPOConfig`). Algorithm and training settings.
- `train_dataset` (`Dataset | IterableDataset | None`, optional, defaults to `None`). Training records.
- `eval_dataset` (`Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None`, optional, defaults to `None`). Finite evaluation records.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin | None`, optional, defaults to `None`). Prompt processing.
- `callbacks` (`Sequence[TrainerCallback | type[TrainerCallback]] | None`, optional, defaults to `None`). Callback instances or classes.
- `optimizers` (`tuple[optax.GradientTransformation | None, Callable | None]`, optional, defaults to `(None, None)`). Optax transformation and step-to-rate callable.
- `ref_model` (`MaxTextModel | None`, optional, defaults to `None`). KL reference policy. See [selection](../../guides/training/grpo.md#supply-a-reference-policy).
- `environment_factory` (`Callable | None`, optional, defaults to `None`). No-argument factory for a synchronous tool environment. See [contracts](../../guides/training/environments.md#train-with-a-tool-environment).
- `rollout_config` (`RolloutConfig | None`, optional, defaults to `None`). Engine and generation settings.
- `agentic_config` (`AgenticRolloutConfig | None`, optional, defaults to `None`). Agentic execution settings.

</details>

## GRPOConfig

Import from `lilax`. Pydantic model extending [TrainingArguments](training-arguments.md). Duration is data-derived. The [loss and engine restrictions](#grpoconfig-and-grpotrainer) apply.

<details>
<summary>Parameters</summary>

- `max_steps` (`None`, optional, defaults to `None`). Only `None` is accepted.
- `gradient_accumulation_steps` (`None`, optional, defaults to `None`). Only `None` is accepted.
- `num_generations` (`int`, optional, defaults to `2`). Completions per prompt.
- `num_iterations` (`int`, optional, defaults to `1`). Policy updates per rollout.
- `beta` (`float`, optional, defaults to `0.08`). Reference-policy KL weight.
- `epsilon` (`float`, optional, defaults to `0.2`). Clipping bound.
- `epsilon_high` (`float | None`, optional, defaults to `None`). Separate upper bound for agentic training. `None` uses epsilon.
- `loss_type` (`Literal["grpo", "gspo-token"]`, optional, defaults to `"grpo"`). Policy ratio loss.
- `loss_aggregation` (`Literal["token-mean", "sequence-mean", "sequence-mean-token-mean"]`, optional, defaults to `"sequence-mean-token-mean"`). Loss reduction. See [training-time restrictions](#grpoconfig-and-grpotrainer).
- `use_rollout_logps` (`bool`, optional, defaults to `True`). Agentic old-policy probabilities from rollout instead of trainer recomputation.
- `reward_num_workers` (`int`, optional, defaults to `0`). Reward worker count.
- `reward_worker_timeout_seconds` (`float`, optional, defaults to `180.0`). Reward worker timeout.
- `rewards` (`MathRewardsConfig | None`, optional, defaults to `None`). Built-in math reward configuration.
- `batch_size` (`int`, optional, defaults to `1`). Prompt batch size.
- `num_batches` (`int`, optional, defaults to `4`). Training data batch budget.
- `train_micro_batch_size` (`int`, optional, defaults to `-1`). Training microbatch size. `-1` selects automatic sizing.
- `rollout_micro_batch_size` (`int`, optional, defaults to `-1`). Rollout microbatch size. `-1` selects automatic sizing.
- `max_seq_token_per_tpu` (`int`, optional, defaults to `0`). Packed row token capacity. Zero disables packing.
- `evaluation_config` (`GRPOEvaluationConfig | None`, optional, defaults to `None`). [Evaluation controls](rollouts.md#grpo-evaluation).

</details>

## Lifecycle methods

See [checkpoint selection](checkpoints.md) and [callbacks](callbacks.md).

### train

```python
trainer.train(resume_from_checkpoint=None, *, checkpoint_step=None)
```

Runs training.

Returns Transformers `TrainOutput` with `global_step`, `training_loss`, and
`metrics={"train_loss": training_loss}`. Loss averages completed optimizer-step losses in this call.

<details>
<summary>Parameters</summary>

- `resume_from_checkpoint` (`str | bool | None`, optional, defaults to `None`). Checkpoint root, numeric step path, or `True` for the latest run checkpoint. See [resume selection](checkpoints.md#resume-selection).
- `checkpoint_step` (`int | None`, optional, defaults to `None`). Nonnegative step. Boolean, negative, or combined resume requests raise `ValueError`.

</details>

### evaluate

```python
trainer.evaluate(eval_dataset=None, ignore_keys=None, metric_key_prefix="eval")
```

Returns a [metrics dictionary](#evaluation-metrics). Missing evaluation data raises `ValueError`.
Named datasets receive name-specific prefixes.

<details>
<summary>Parameters</summary>

- `eval_dataset` (`Dataset | IterableDataset | TeacherLogitsDataset | dict[str, Dataset | IterableDataset | TeacherLogitsDataset] | None`, optional, defaults to `None`). Finite input overriding stored evaluation data. Teacher-logit input requires a compatible distillation path.
- `ignore_keys` (`list[str] | None`, optional, defaults to `None`). Accepted evaluation compatibility argument.
- `metric_key_prefix` (`str`, optional, defaults to `"eval"`). Metric prefix.

</details>

### save_model

```python
trainer.save_model(output_dir=None)
```

Saves weights and the processing class. Ordinary model weights go under `_lilax_model` in the destination.
Repeated saves replace the declared Lilax-owned payload and update named configuration and processing files,
preserving run checkpoints, metrics, TensorBoard logs, and unrelated artifacts.
See [ownership collisions](checkpoints.md#model-saving) and [adapter persistence](adapters.md#peftmodel-persistence).

Returns `None`.

<details>
<summary>Parameters</summary>

- `output_dir` (`str | os.PathLike | None`, optional, defaults to `None`). Destination. `None` uses `args.output_dir`.

</details>

### save_state

```python
trainer.save_state()
```

Saves resumable training state. Before training-state initialization it raises `RuntimeError`.
`Trainer` and SFT with multimodal input, DiLoCo, or optimizer host offload require setup and training state.
Other SFT, DPO, and distillation require an initialized trainer. GRPO requires its initialized RL cluster or fixed-reward setup. Constructor runtime initialization alone
is insufficient. These guards do not require `train()` to have returned. Evaluation can initialize some paths.
A missing checkpoint manager, disabled checkpointing, or missing GRPO checkpoint root raises `ValueError`
after the state guard.

Returns `None`.

### add_callback

```python
trainer.add_callback(callback)
```

Adds a callback instance or class.

Returns `None`.

<details>
<summary>Parameters</summary>

- `callback` (`TrainerCallback | type[TrainerCallback]`). Callback to add.

</details>

### pop_callback

```python
trainer.pop_callback(callback)
```

Removes the matching callback instance. Removing a TensorBoard, W&B, or Trackio callback immediately closes its active writer or run before returning the object.

Returns the first matching callback instance, or `None` when no callback matches.

<details>
<summary>Parameters</summary>

- `callback` (`TrainerCallback | type[TrainerCallback]`). Callback to remove.

</details>

### remove_callback

```python
trainer.remove_callback(callback)
```

Removes a callback instance or class. Removing a TensorBoard, W&B, or Trackio callback immediately closes its active writer or run.

Returns `None`.

<details>
<summary>Parameters</summary>

- `callback` (`TrainerCallback | type[TrainerCallback]`). Callback to remove.

</details>

### log

```python
trainer.log(logs, start_time=None)
```

Appends metrics to `state.log_history` and dispatches `on_log`. With a start time, adds elapsed seconds as `train_runtime`.

Returns `None`.

<details>
<summary>Parameters</summary>

- `logs` (`Mapping[str, Any]`). Metrics to record.
- `start_time` (`float | None`, optional, defaults to `None`). Start time for elapsed runtime.

</details>

## data_collator

`data_collator=None` uses Lilax's built-in batching, padding or packing, and objective-specific target preparation. It does not instantiate a Transformers collator. A custom collator applies to HF-dataset input, including evaluation, after preprocessing and batching. It does not process `SyntheticDataset` or `TeacherLogitsDataset` input.

For causal training, the callable receives a list of processed example dictionaries with `input_ids`, `position_ids`, `attention_mask`, `segment_ids`, and `labels`. Ignored labels are `-100`. Other prepared fields, including media fields, remain in the dictionaries. The result is a dictionary of arrays convertible to NumPy, with batched `input_ids` and optional `attention_mask`, `segment_ids`, `labels`, and `position_ids`. Omitted fields default to an all-ones attention mask, segment IDs equal to that mask, labels equal to `input_ids`, and positions `0` through sequence length minus one for each row.

Lilax converts that result into `inputs`, `inputs_position`, `inputs_segmentation`, `targets`, `targets_position`, and `targets_segmentation`. It masks `-100`, attention-mask zeros, and padding labels from loss. Text causal input also masks BOS labels. Targets and their segmentation shift left by one token, so the collator must return unshifted labels. Other returned fields pass through. A result containing `inputs` bypasses this conversion and must already contain the complete training batch layout.

For DPO or ORPO, each input dictionary instead contains unpadded token lists `prompt_ids`, `chosen_ids`, and `rejected_ids`. The result can contain the batched native fields `prompt_ids`, `chosen_ids`, `rejected_ids`, `prompt_mask`, `chosen_mask`, and `rejected_mask`. A result containing `prompt_ids` passes through without further formatting. Alternatively, it must contain `input_ids`, `attention_mask`, and `completion_mask` with all chosen rows first, followed by their rejected rows in matching order. Zero completion-mask values select the prompt from the chosen row. Nonzero values select each completion. Lilax then left-pads prompts and right-pads completions to the configured lengths. This is the supported TRL-style layout, not a guarantee that every upstream preference collator is compatible.

Block diffusion uses the causal example and result fields without a target shift. Its configured block-diffusion corruption runs after collation. See [block-diffusion SFT](../../guides/training/block-diffusion.md).

## optimizers

The pair is `(optax.GradientTransformation, schedule)`, with either element optionally `None`. A schedule maps the optimizer step to a learning rate. For `Trainer`, SFT, and DPO, each missing element defaults independently. A missing schedule uses the configured schedule. A missing transformation uses the configured optimizer with that schedule. A supplied transformation must already incorporate its learning rate or schedule. Supplying the second element does not attach it to the first. The separate schedule is also available for learning-rate reporting and callback state where the training path binds it.

For SFT without multimodal input, DiLoCo, or optimizer host offload, positive `args.max_grad_norm` prepends `optax.clip_by_global_norm` even to a supplied transformation. Distillation constructs its default schedule only when building its default optimizer. See [optimization](optimization.md) for its schedule and [custom optimizer example](../../guides/training/optimization.md#supply-an-optax-optimizer).

## Logged metrics

`trainer.state.log_history` stores each `trainer.log(...)` dictionary with a `step` field added. That step counts completed optimizer updates. It is distinct from the returned `TrainOutput.metrics["train_loss"]`.

With positive `logging_steps`, regular training logs occur at completed steps divisible by `logging_steps`. Callbacks can also request logging. A log request still pending at the end of training can produce a cumulative `train_loss` entry. Regular step entries use the following keys.

| Training workflow | Regular log keys |
| --- | --- |
| `Trainer`, and SFT with multimodal data, DiLoCo, or optimizer host offload | `loss`, `learning/loss`, `learning/lm_loss`, `learning/perplexity`, `learning/z_loss`, `learning/moe_lb_loss`, `learning/indexer_loss`, `learning/mtp_loss`, `learning/total_weights`, `learning/current_learning_rate`, and `perf/step_time_seconds`. Norm and performance keys depend on the selected settings. |
| Other SFT, DPO, distillation, and vLLM GRPO | `loss`, `perplexity`, `learning_rate` when available, and the task's additional reduced metrics. Additional names pass through unchanged. |
| GRPO with `engine="maxtext"` | `learning/loss`, `learning/avg_reward`, `learning/avg_reward_std`, `learning/avg_advantage`, `learning/avg_kl`, `learning/completion_length`, `learning/moe_lb_loss`, and `learning/total_weights`. Norm keys are included when optimizer host offload is disabled. |

The norm keys are `learning/grad_norm`, `learning/raw_grad_norm`, and `learning/param_norm`. `Trainer` can also emit `optim/step_skipped` when spike skipping is enabled. Performance keys include `perf/per_device_tflops` and `perf/per_device_tflops_per_sec` after batch rampup. Optional model instrumentation can add metrics.

Distillation adds `distill/total_loss`, `distill/hard_loss`, `distill/soft_loss`, `distill/teacher_loss`, `distill/kl_div_T1`, `distill/kl_div_at_T`, `distill/student_perplexity`, `distill/teacher_perplexity`, `distill/out_proj_feature_loss`, `distill/moe_lb_loss`, `distill/teacher_moe_lb_loss`, `distill/temperature`, `distill/alpha`, and `distill/beta_feature`. These retain their `distill/` names in history and callback reporting.

`report_to` selects callbacks for `"tensorboard"`, `"wandb"`, or `"trackio"`. `"all"` selects all three and `"none"` selects none. Reporting runs on world process zero.

| Tracker | Payload and cadence |
| --- | --- |
| TensorBoard | Numeric `on_log` values as scalar tags under their unchanged keys, at `state.global_step`. The writer flushes on each log event. Evaluation sends numeric evaluation keys at evaluation events. |
| Weights & Biases | The `on_log` dictionary under unchanged keys, with `step=state.global_step`. Evaluation sends its metrics dictionary at evaluation events. |
| Trackio | The `on_log` dictionary under unchanged keys, with `step=state.global_step`. Evaluation sends its metrics dictionary at evaluation events. |

The automatically added history `step` is not added to the tracker payload. Trackers receive the step separately. `evaluate()` dispatches `on_evaluate`, so its metrics reach these trackers without automatically appending an evaluation entry to `log_history`. Explicit `trainer.log(...)` calls append the supplied keys and dispatch them immediately. With `start_time`, `log` also adds elapsed seconds as `train_runtime`.

## Evaluation metrics

`Trainer.evaluate()` and SFT with multimodal input, DiLoCo, or optimizer host offload return the following numeric keys. Losses other than `eval_loss` are averaged across evaluation batches.

| Key | Meaning |
| --- | --- |
| `eval_loss` | Total cross-entropy divided by total evaluation token weight |
| `eval_perplexity` | Exponential of `eval_loss` |
| `eval_total_weights` | Sum of evaluation token weights |
| `eval_moe_lb_loss` | Expert load-balancing loss |
| `eval_indexer_loss` | Sparse-attention indexer auxiliary loss |
| `eval_mtp_loss` | Multi-token prediction auxiliary loss, currently zero in evaluation |
| `eval_mtp_acceptance_rate_percent` | Multi-token prediction acceptance rate as a percentage |
| `eval_z_loss` | Logit z-loss |

Inactive auxiliary losses can be zero. Other SFT, DPO, and distillation return `eval_loss` and `eval_perplexity`.

For example, `trainer.evaluate(metric_key_prefix="validation")` returns `validation_loss` instead of `eval_loss`. With `eval_dataset={"heldout": dataset}`, the default prefix produces `eval_heldout_loss`. Values depend on the actual evaluation run.

`GRPOTrainer.evaluate()` with vLLM returns these keys:

| Key | Meaning |
| --- | --- |
| `eval_correct` | Sum of correctness scores under the selected evaluation mode |
| `eval_total` | Number of evaluated records |
| `eval_accuracy` | `100 * eval_correct / eval_total`, or zero for no records |
| `eval_partial_accuracy` | Percentage from partial-correctness scores |
| `eval_format_accuracy` | Percentage from response-format scores |
| `eval_mean_reward` | Sum of reward-function scores averaged over successfully scored responses, or zero without such responses |
| `eval_responses` | Optional list of `(question, answers, responses)` tuples when `GRPOEvaluationConfig(return_responses=True)` is selected |

GRPO correctness depends on the evaluation mode. `pass_at_1` averages per-response scores, while `maj` scores the majority extracted answer. `pass` returns the first response satisfying correctness, partial correctness, and format together, or the last response's scores. `eval_correct` can be fractional. See [rollout evaluation controls](rollouts.md).

## Learn-to-init maps and parameter selection

`weights_copy_map` maps teacher source patterns to student destination replacements. `student_weights_share_map` maps student source patterns to student destination replacements. Paths are slash-separated NNX graph paths, such as `token_embedder/embedding`, rather than Hugging Face state-dictionary names. Both maps use Python `re.fullmatch` on source paths. Replacements use `match.expand`, so `r"\1"` and `r"\g<name>"` refer to captured groups. Identical source and replacement strings preserve the matched path directly.

Sharing runs first and replaces the destination with the source student parameter object. Updates therefore affect both aliases. Copying then assigns teacher values into existing student parameters without aliasing the teacher parameter object. Each source pattern must match at least one path, otherwise `ValueError` is raised. Destinations must exist and parameter shapes must match exactly. Maps must select parameter leaves, not modules. Missing destinations or incompatible shapes fail during preparation.

`student_params_to_update` uses regex **search**, unlike the maps' full match. With a nonempty list, only matching student parameters remain trainable. An empty list or `None` adds no parameter filter. These maps and the filter are consumed even with `enabled=False`. See the [copy and filter example](../../guides/training/distillation.md#copy-teacher-weights-and-select-student-parameters).

For models loaded through `AutoModelForCausalLM`, `enabled=True` raises `AssertionError` during finalization after the distillation training loop. It does not provide a usable structural attention workflow. Use `enabled=False` for weight copying, parameter sharing, and trainable-parameter filtering.

`attention_module_name`, `layer_indices`, and `use_general_linear_map` configure structural attention changes. They do not make `enabled=True` usable with these models. `LearnToInitConfig.layer_indices` is separate from `DistillationConfig.layer_indices`, which selects features for the loss.

### LearnToInitConfig

Import from `lilax.training`. Pydantic model whose keyword fields follow the map and structural restrictions above.

<details>
<summary>Parameters</summary>

- `enabled` (`bool`, optional, defaults to `False`). Structural attention request. With public `AutoModelForCausalLM` models, `True` raises `AssertionError` during finalization after training.
- `use_general_linear_map` (`bool`, optional, defaults to `False`). Full `W` map instead of factor matrices `A` and `B`.
- `weights_copy_map` (`dict[str, str] | None`, optional, defaults to `None`). Teacher-to-student path replacements using regex full matches.
- `student_weights_share_map` (`dict[str, str] | None`, optional, defaults to `None`). Student-to-student aliases using regex full matches.
- `attention_module_name` (`str | None`, optional, defaults to `None`). Attention attribute selected for structural modification.
- `layer_indices` (`list[int] | None`, optional, defaults to `None`). Zero-based attention layers when scanning is disabled. Distinct from feature-loss layer selection.
- `student_params_to_update` (`list[str] | None`, optional, defaults to `None`). Regex searches selecting trainable student paths. `None` or `[]` adds no filter.

</details>

## Teacher inputs and offline output

Teacher and student require the same mesh and compatible batch settings. `teacher_model=None` requires `TeacherLogitsDataset`, otherwise construction raises `ValueError`.

### generate_teacher_logits

```python
trainer.generate_teacher_logits(
	*, dataset=None, top_k=128,
	optional_keys=("inputs_position", "inputs_segmentation", "targets_segmentation", "targets"),
	gcs_upload_path=None, local_tmp_dir="/tmp", steps_per_file=50, upload_workers=4, serialization_workers=16,
)
```

Writes offline teacher records. Requires a teacher model.
See [distillation guide](../../guides/training/distillation.md).

Returns `None`.

<details>
<summary>Parameters</summary>

- `dataset` (`Dataset | IterableDataset | None`, optional, defaults to `None`). Teacher input. Absent input uses the training dataset.
- `top_k` (`int`, optional, defaults to `128`). Teacher logits retained per position.
- `optional_keys` (`Sequence[str]`, optional, defaults to `("inputs_position", "inputs_segmentation", "targets_segmentation", "targets")`). Batch metadata to retain.
- `gcs_upload_path` (`str | None`, optional, defaults to `None`). GCS output destination.
- `local_tmp_dir` (`str`, optional, defaults to `"/tmp"`). Local output staging directory.
- `steps_per_file` (`int`, optional, defaults to `50`). Records grouped by training steps per file.
- `upload_workers` (`int`, optional, defaults to `4`). Upload workers.
- `serialization_workers` (`int`, optional, defaults to `16`). Serialization workers.

</details>

## GRPOConfig and GRPOTrainer

`max_steps` and `gradient_accumulation_steps` accept only `None`. Duration is data-derived.

Loss type accepts `"grpo"` or `"gspo-token"`. Aggregation accepts `"token-mean"`, `"sequence-mean"`, or `"sequence-mean-token-mean"`.

For vLLM, `"grpo"` uses a per-token current-policy to old-policy probability ratio in the clipped policy loss. `"gspo-token"` uses the exponentiated mean completion log-ratio as its sequence-level ratio value, with per-token gradients. `beta` weights the reference-policy KL penalty. `epsilon` controls clipping, while `epsilon_high` supplies a separate upper clipping bound only in the agentic workflow. `None` uses `epsilon` there.

`"token-mean"` divides summed completion-token losses by the total completion-token count. `"sequence-mean-token-mean"` averages each completion's token mean across nonempty completions. Packed execution uses individual sequences rather than treating an entire packed row as one sequence. `GRPOConfig` accepts `"sequence-mean"`, but with the Tunix version selected by both supplied containers, vLLM training raises `ValueError` when computing the loss, not when constructing `GRPOConfig`.

`use_rollout_logps=True` selects rollout-engine log probabilities as the old-policy token log probabilities. `False` selects trainer recomputation at step zero. Lilax forwards this selection only to the agentic learner. The non-agentic learner uses its dependency's default.

`train_micro_batch_size=-1` and `rollout_micro_batch_size=-1` select automatic sizing. Without packing, a positive training value controls microbatches for gradient accumulation and must divide `batch_size`. A positive rollout value controls rollout batching. With packing, the learner ignores the training microbatch size when assembling prompt batches and derives updates from `batch_size * num_generations` sequences. `GRPOConfig` accepts an explicit microbatch value of `0`, but with the Tunix version selected by both supplied containers, `train()` raises `ValueError` during RL configuration initialization, including with packing. Use `-1` for automatic sizing.

`max_seq_token_per_tpu=0` disables packing. A positive value packs rollout sequences into rows of that token capacity for actor and reference passes. Each maximal sequence must fit in a row, so the budget must be at least the prompt capacity plus generation length. This is a token capacity, not a hardware throughput rate. These controls apply to vLLM. The [fixed-reward engine rejects explicitly supplied fields](rollouts.md#engine-restrictions), including values equal to defaults.

`ref_model=None` accepts a reference model object for the KL penalty. See [reference policy selection](../../guides/training/grpo.md#supply-a-reference-policy).

`environment_factory=None` accepts a no-argument callable returning a synchronous tool environment. It requires vLLM and `AgenticRolloutConfig(enabled=True)`. `reset()` returns a prompt suffix or `None`, and `get_reward()` supplies an episode reward when no `reward_funcs` is given. With custom rewards, the `environments` keyword supplies one closed environment per completion and `reward_num_workers` must be `0` or `1`. Evaluation with an environment factory is unsupported. See [environment contracts and example](../../guides/training/environments.md#train-with-a-tool-environment).

### GRPOTrainer.generate

```python
trainer.generate(prompts, num_passes=1)
```

Requires completed training setup. Engine-specific evaluation and adapter restrictions are listed in [rollouts](rollouts.md).
See [GRPO guide](../../guides/training/grpo.md).

Returns a list per prompt, each containing responses from successive generation passes.

Missing training setup raises `RuntimeError`.

<details>
<summary>Parameters</summary>

- `prompts` (`Sequence[str]`). Prompts to generate from.
- `num_passes` (`int`, optional, defaults to `1`). Generation passes.

</details>
