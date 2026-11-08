# Evaluation

Import from `lilax.evaluation`. See [evaluation guide](../../guides/inference/evaluation.md).

## evaluate

```python
evaluate(
	model, *, config: DatasetEvalConfig | HarnessEvalConfig | SimpleEvalsConfig,
	output: EvalOutputConfig, processing_class=None, eval_dataset=None, compute_metrics=None, token=None,
)
```

Dispatches to the configured evaluator. `model` is a loaded model or an already entered `ServingEndpoint`. `output` is an `EvalOutputConfig`.

For a model, the evaluator starts and stops a `ServingEndpoint` with `host="localhost"`, port `8000`, `tensor_parallel_size=4`, and `max_model_len=model.config.max_position_embeddings`. SimpleEvals and evalchemy enable chat. An explicitly entered endpoint supplies custom server settings and remains caller-owned.

For materialized weights, endpoint startup saves a full checkpoint copy under `~/tmp/lilax/serving/`. The automatically owned endpoint cleans its temporary directory on exit or startup failure. Evaluation writes results and may download task data or tokenizer files.

The result `payload` contains `benchmark`, `model_name`, `timestamp_utc`, `scores`, `generation_stats`, and `config`. The file is named `<benchmark>_<safe_model>_<UTC timestamp>.json`, where `/` and `:` in the model name become `_`. Harness and SimpleEvals join task names with `+` for the benchmark name. SimpleEvals with `log_debug_info=True` also writes a debug report.

Unsupported configuration types raise `TypeError`. Synthetic evaluation input raises `ValueError`. SimpleEvals and evalchemy require a chat-capable endpoint.

Returns `{"results": payload, "local_path": absolute_json_path}` on JAX process zero, with `debug_local_path` also present for SimpleEvals with `log_debug_info=True`. Other processes return `{}` after synchronization.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel | ServingEndpoint`). Loaded model or entered endpoint to evaluate.
- `config` (`DatasetEvalConfig | HarnessEvalConfig | SimpleEvalsConfig`). Evaluator and request controls.
- `output` (`EvalOutputConfig`). Result destinations.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin | None`, optional, defaults to `None`). Compatible processing class overriding the endpoint's processing class.
- `eval_dataset` (`Iterable | dict[str, Iterable] | None`, optional, defaults to `None`). Task-compatible data as described under [custom input and metrics](#custom-input-and-metrics).
- `compute_metrics` (`Callable[[list[str], list[str]], dict] | None`, optional, defaults to `None`). Receives positional response and reference lists and returns JSON-serializable metric values.
- `token` (`str | None`, optional, defaults to `None`). Authentication token for evaluator and endpoint loading.

</details>

## DatasetEvalConfig

Import from `lilax.evaluation`. Pydantic settings for a registered dataset evaluator. Unknown fields are rejected.

The built-in benchmark names are `"mlperf_openorca"` and `"openorca"`, case-insensitively. Unknown names raise `KeyError`. Both stream `Open-Orca/OpenOrca`, split `train`, unless `eval_dataset` supplies records. Records require `question` and a nonempty `response`. Missing or blank responses are skipped. Optional `system_prompt` defaults to `"You are a helpful assistant. Summarize the following conversation."` when absent or empty. The processing tokenizer applies its chat template to system and user messages, with a generation prompt.

Without `compute_metrics`, both names use the ROUGE scorer. It sentence-tokenizes stripped text, multiplies the returned ROUGE values by 100, rounds to four decimals, and adds `gen_num` for the prediction count. This scorer loads the upstream `rouge` metric.

<details>
<summary>Parameters</summary>

- `benchmark` (`str`). Registered benchmark name.
- `num_samples` (`int | None`, optional, defaults to `None`). Sample limit, or all records when unset.
- `max_tokens` (`int`, optional, defaults to `1024`). Completion limit.
- `temperature` (`float`, optional, defaults to `0.0`). Request sampling temperature.
- `concurrency` (`int`, optional, defaults to `64`). Concurrent requests.
- `request_timeout` (`int`, optional, defaults to `600`). Request timeout in seconds.
- `max_retries` (`int`, optional, defaults to `3`). Request retry limit.
- `skip_warmup` (`bool`, optional, defaults to `False`). Skip evaluator warmup.

</details>

## Custom input and metrics

`compute_metrics(responses, references)` receives two positional lists and returns a mapping of metric names to JSON-serializable values. This is Lilax's callable contract, rather than the Transformers Trainer `EvalPrediction` contract.

| Configuration | `eval_dataset` contract | Custom metrics |
| --- | --- | --- |
| `DatasetEvalConfig` | One finite iterable of the records above. | Receives generated text and the records' `response` strings. Replaces the built-in scorer. Metric names are unchanged. |
| `HarnessEvalConfig`, either backend | A task-compatible dataset, or a dictionary keyed by loaded task name. Replaces each task's test split, or validation split if it has no test documents. | Uses logged `filtered_resps[0]` and `target`. Both must be strings or `ValueError` is raised. Merges metrics as `<task>_<metric>`. |
| `SimpleEvalsConfig` | Task-specific records, or a dictionary keyed by task name. For example, `gsm8k` requires `question` and `answer`, with the target taken after `####` in `answer`. | Uses the last conversation response and `correct_answer` metadata. Predictions and single references must be strings or `ValueError` is raised. Merges metrics as `<task>_<metric>`. |

Harness scores map recognized accuracy and normalized accuracy fields to `<task>_accuracy` and `<task>_accuracy_norm`, multiplied by 100 and rounded to two decimals. Instruction-following strict and loose accuracy fields retain their task-prefixed names and receive the same percentage conversion. Unrecognized task metrics remain available in `generation_stats["<backend>_results"]`. SimpleEvals maps its main score to `<task>_accuracy` as a percentage and other numeric metrics to `<task>_<name>`, rounded to four decimals. Custom metric values receive no automatic scaling.

## HarnessEvalConfig

Import from `lilax.evaluation`. Pydantic settings for evalchemy or lm-eval harness evaluation. Unknown fields are rejected.

For either backend, a supplied `generation_config` becomes evaluator `gen_kwargs`. Lilax always passes `do_sample` and `top_k`, and passes `temperature=0.0` when `do_sample=False`, otherwise the configured temperature. Non-`None` `max_new_tokens` becomes `max_gen_toks`. Nonnegative `top_p` is included. Non-`None` `eos_token_id` becomes `stop_token_ids`, with a single ID converted to a list. This translation applies only to `HarnessEvalConfig`. `DatasetEvalConfig` and `SimpleEvalsConfig` use their own `max_tokens` and `temperature` fields and reject `generation_config` as an unknown field.

<details>
<summary>Parameters</summary>

- `backend` (`Literal["evalchemy", "lm_eval"]`, optional, defaults to `"evalchemy"`). Harness implementation.
- `tasks` (`tuple[str, ...] | list[str]`, optional, defaults to `("mmlu",)`). Task names.
- `num_fewshot` (`int`, optional, defaults to `0`). Few-shot example count.
- `num_samples` (`int | float | None`, optional, defaults to `None`). Sample count or fraction.
- `apply_chat_template` (`bool`, optional, defaults to `False`). Apply the tokenizer chat template.
- `fewshot_as_multiturn` (`bool`, optional, defaults to `False`). Format few-shot examples as turns.
- `generation_config` (`GenerationConfig | None`, optional, defaults to `None`). Generation options translated above.
- `skip_warmup` (`bool`, optional, defaults to `False`). Skip evaluator warmup.

</details>

## SimpleEvalsConfig

Import from `lilax.evaluation`. Pydantic settings for SimpleEvals. Unknown fields are rejected. Debug logging writes an additional report.

Tasks are `mmlu`, `gpqa`, `gsm8k`, `drop`, `mgsm`, `mgsm_en`, `aime2024`, and `aime2025`.

<details>
<summary>Parameters</summary>

- `tasks` (`tuple[Literal["mmlu", "gpqa", "gsm8k", "drop", "mgsm", "mgsm_en", "aime2024", "aime2025"], ...]`, optional, defaults to `("mmlu",)`). Tasks to evaluate.
- `num_samples` (`int | None`, optional, defaults to `None`). Sample limit.
- `n_repeats` (`int | None`, optional, defaults to `None`). Task repetition override.
- `max_tokens` (`int`, optional, defaults to `2048`). Completion limit.
- `temperature` (`float`, optional, defaults to `0.5`). Sampling temperature.
- `system_message` (`str | None`, optional, defaults to `"You are a helpful assistant."`). System message.
- `reasoning_effort` (`str | None`, optional, defaults to `None`). Reasoning effort sent to requests.
- `log_debug_info` (`bool`, optional, defaults to `False`). Capture and write request diagnostics.
- `continue_on_request_error` (`bool`, optional, defaults to `False`). Continue after request failures.
- `skip_warmup` (`bool`, optional, defaults to `False`). Skip evaluator warmup.

</details>

## EvalOutputConfig

Import from `lilax.evaluation`. Pydantic output settings. A local results path is created as needed and receives the JSON file. A `gs://` path uploads the JSON and leaves a local copy in a system temporary directory. `local_path` always names the absolute local copy. Unknown fields are rejected.

<details>
<summary>Parameters</summary>

- `results_path` (`str`). Local or GCS result destination.
- `gcs_results_path` (`str | None`, optional, defaults to `None`). Additional GCS upload destination.

</details>
