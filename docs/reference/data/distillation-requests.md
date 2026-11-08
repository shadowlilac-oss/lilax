# Distillation request preparation

Import from `lilax.data_pipeline.api`. See [teacher request workflow](../../guides/data/distillation-requests.md).

## InputRequest

Import from `lilax.data_pipeline.api`. Dataclass describing one teacher request. Each instance receives a fresh token list.

<details>
<summary>Parameters</summary>

- `prompt` (`str`, optional, defaults to `""`). Prompt text.
- `prompt_token_ids` (`list[int]`, optional, defaults to `[]`). Tokenized prompt. Each instance receives a fresh list.
- `actual_completion` (`str`, optional, defaults to `""`). Dataset completion text.
- `max_output_tokens` (`int`, optional, defaults to `0`). Requested completion token limit.

</details>

## load_distillation_dataset

```python
load_distillation_dataset(dataset_path, data_split, hf_access_token, dataset_type="huggingface")
```

Loads an HF dataset split. Other dataset types fail with an assertion.

Returns the HF dataset split.

<details>
<summary>Parameters</summary>

- `dataset_path` (`str`). Dataset source passed to HF loading.
- `data_split` (`str`). Split selection.
- `hf_access_token` (`str | bool | None`). Authentication passed to HF loading.
- `dataset_type` (`str`, optional, defaults to `"huggingface"`). Only supported dataset type.

</details>

## process_distillation_dataset

```python
process_distillation_dataset(dataset, data_columns)
```

Separates user and assistant messages and extracts their text. Unsupported columns and nonconversational features fail with assertions.

Returns a dataset with lists of text under `prompt` and `completion`.

<details>
<summary>Parameters</summary>

- `dataset` (`Dataset`). Conversational records.
- `data_columns` (`list[str]`). Either `messages`, or paired `prompt` and `completion` columns.

</details>

## filter_distillation_dataset

```python
filter_distillation_dataset(dataset, tokenizer, max_target_length, max_prefill_length, use_chat_template=False)
```

Uses the first prompt and completion text. Removes prompts above the prefill limit and caps output tokens by actual completion length and the target-minus-prefill budget.

Returns a list of `InputRequest`.

<details>
<summary>Parameters</summary>

- `dataset` (`Dataset`). Prepared prompt and completion text lists.
- `tokenizer` (`PreTrainedTokenizerBase`). Prompt and completion tokenizer.
- `max_target_length` (`int`). Combined length budget.
- `max_prefill_length` (`int`). Prompt length limit and reserved prefill budget.
- `use_chat_template` (`bool`, optional, defaults to `False`). Apply the chat template to one user message.

</details>

## Teacher service generation

```python
generate_distillation_data(
	dataset_path, data_split, data_columns, tokenizer_path, batch_size, num_batches, *,
	hf_access_token=None, dataset_type="huggingface", use_chat_template=False,
	max_prefill_length=256, max_target_length=2048, num_generations=1,
	server_address="localhost:9000", upload, hf_repo_id=None,
	gcs_bucket=None, gcs_data_path=None, remove_local_dataset_files=False,
)
```

Returns `None`. It calls a JetStream gRPC teacher, writes `distillation-data-<batch>.parquet`, and uploads each batch. Output records contain conversational `prompt`, generated `completion`, and `actual_completion`.

`upload` must be `upload-to-hf` with `hf_repo_id`, or `upload-to-gcs` with `gcs_bucket` and `gcs_data_path`. Missing or invalid destinations raise `ValueError`. Prefill length must be below target length or an assertion fails. `remove_local_dataset_files=True` removes each local file after upload. This operation uses `asyncio.run` and must be called outside an already running event loop.

<details>
<summary>Parameters</summary>

- `dataset_path` (`str`). HF dataset source.
- `data_split` (`str`). Dataset split.
- `data_columns` (`list[str]`). Conversational columns accepted by `process_distillation_dataset`.
- `tokenizer_path` (`str`). Teacher tokenizer source.
- `batch_size` (`int`). Dataset records per batch.
- `num_batches` (`int`). Number of batches to generate.
- `hf_access_token` (`str | bool | None`, optional, defaults to `None`). Dataset, tokenizer, and upload authentication.
- `dataset_type` (`str`, optional, defaults to `"huggingface"`). Dataset loader selection.
- `use_chat_template` (`bool`, optional, defaults to `False`). Format each prompt as one user message.
- `max_prefill_length` (`int`, optional, defaults to `256`). Prompt token limit.
- `max_target_length` (`int`, optional, defaults to `2048`). Combined token budget.
- `num_generations` (`int`, optional, defaults to `1`). Teacher responses per prompt.
- `server_address` (`str`, optional, defaults to `"localhost:9000"`). JetStream gRPC endpoint.
- `upload` (`str`). Required upload mode, as described above.
- `hf_repo_id` (`str | None`, optional, defaults to `None`). HF upload destination.
- `gcs_bucket` (`str | None`, optional, defaults to `None`). GCS upload bucket.
- `gcs_data_path` (`str | None`, optional, defaults to `None`). GCS upload prefix.
- `remove_local_dataset_files` (`bool`, optional, defaults to `False`). Delete each local batch file after upload.

</details>
