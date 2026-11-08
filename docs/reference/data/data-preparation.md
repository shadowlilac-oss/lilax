# Corpus preparation

See [preparation guide](../../guides/data/preparation.md).

## prepare_llava_video_178k

Import from `lilax.data_operations.llava_video`.

```python
prepare_llava_video_178k(
	output_dir, fold="0_30_s_academic_v0_1", download_videos=False, video_subset=None, local_temp_dir=None,
)
```

Downloads metadata and optionally videos. GCS output stages locally, uploads, and removes the staged fold.

Returns `None`.

<details>
<summary>Parameters</summary>

- `output_dir` (`str`). Local or GCS destination.
- `fold` (`str`, optional, defaults to `"0_30_s_academic_v0_1"`). Dataset fold.
- `download_videos` (`bool`, optional, defaults to `False`). Download and extract video archives.
- `video_subset` (`int | None`, optional, defaults to `None`). Download the first this many archives. `None` selects all eight.
- `local_temp_dir` (`str | None`, optional, defaults to `None`). GCS staging directory. The default uses `llava_video_178k_temp` under `TMPDIR`, or `/tmp` when unset.

</details>

## copy_c4

Import from `lilax.data_operations.c4`.

```python
copy_c4(destination, source="gs://allennlp-tensorflow-datasets/c4/en/3.0.1")
```

Copies source objects under the destination's `c4/en/3.0.1` path. External command failure propagates.

Returns `None`.

<details>
<summary>Parameters</summary>

- `destination` (`str`). Destination root for the GCS copy.
- `source` (`str`, optional, defaults to `"gs://allennlp-tensorflow-datasets/c4/en/3.0.1"`). Source directory copied through `gcloud storage cp`.

</details>

## generate_distillation_data

Import from `lilax.data_operations.distillation_vllm`.

```python
generate_distillation_data(
	*, hf_access_token, dataset_path="HuggingFaceH4/ultrachat_200k", data_split="train_sft",
	data_columns=None, use_chat_template=False, max_prefill_length=256, max_target_length=2048,
	num_generations=1, num_prompts=5120, output_file=None, teacher_model=None, tp_size=4,
	max_model_len=4096, max_new_tokens=512, max_num_batched_tokens=2048, max_num_seqs=256,
	gpu_memory_utilization=0.98,
)
```

Generates responses with vLLM and saves conversational records as Parquet.

`hf_access_token`, `max_prefill_length`, and `max_target_length` are accepted but unused. They do not enforce authentication or length limits. Supply the teacher and output path explicitly to avoid environment-derived paths.

Returns `None`.

<details>
<summary>Parameters</summary>

- `hf_access_token` (`Any`). Required but unused. No type-dependent operation is performed on it.
- `dataset_path` (`str`, optional, defaults to `"HuggingFaceH4/ultrachat_200k"`). Dataset repository.
- `data_split` (`str`, optional, defaults to `"train_sft"`). Dataset split.
- `data_columns` (`Sequence[str] | None`, optional, defaults to `None`). First element selects the prompt column. `None` uses `messages`.
- `use_chat_template` (`bool`, optional, defaults to `False`). Format prompts using the teacher tokenizer.
- `max_prefill_length` (`int`, optional, defaults to `256`). Accepted but unused.
- `max_target_length` (`int`, optional, defaults to `2048`). Accepted but unused.
- `num_generations` (`int`, optional, defaults to `1`). Responses per prompt.
- `num_prompts` (`int`, optional, defaults to `5120`). Maximum selected prompts.
- `output_file` (`str | None`, optional, defaults to `None`). Parquet destination. `None` uses `<BASE_DIRECTORY>/datasets/distillation_data.parquet`, with `.` as the environment fallback.
- `teacher_model` (`str | None`, optional, defaults to `None`). vLLM model source. `None` uses `<BASE_DIRECTORY>/qwen3-32b`, with `.` as the environment fallback.
- `tp_size` (`int`, optional, defaults to `4`). Tensor parallel size.
- `max_model_len` (`int`, optional, defaults to `4096`). vLLM context length.
- `max_new_tokens` (`int`, optional, defaults to `512`). Generation token limit.
- `max_num_batched_tokens` (`int`, optional, defaults to `2048`). vLLM batched token limit.
- `max_num_seqs` (`int`, optional, defaults to `256`). vLLM concurrent sequence limit.
- `gpu_memory_utilization` (`float`, optional, defaults to `0.98`). vLLM GPU memory fraction.

</details>
