# Datasets and processing

Import dataset descriptors, RL preparation, and tokenizer training from `lilax.data`.
See [dataset loading](../../guides/data/datasets.md) and [processing](../../guides/data/processing.md) for examples.

## Dataset objects

`lilax.data.Dataset`, `IterableDataset`, and `load_dataset` are Hugging Face objects. Their general methods follow the official [Datasets reference](https://huggingface.co/docs/datasets/package_reference/main_classes) and [loading reference](https://huggingface.co/docs/datasets/package_reference/loading_methods).

Lilax trainers consume these objects as `train_dataset` or `eval_dataset`. Evaluation accepts a finite dataset or a dictionary of named datasets.

### SyntheticDataset

Import from `lilax.data`. Frozen, keyword-only dataclass.

```python
SyntheticDataset(*, reuse_example_batch: int = 0, max_segments_per_seq: int = -1)
```

Frozen synthetic training descriptor. Evaluation rejects it with `ValueError`.

<details>
<summary>Parameters</summary>

- `reuse_example_batch` (`int`, optional, defaults to `0`). Synthetic batch reuse setting.
- `max_segments_per_seq` (`int`, optional, defaults to `-1`). Synthetic segment-count setting.

</details>

### TeacherLogitsDataset

Import from `lilax.data`. Frozen dataclass.

```python
TeacherLogitsDataset(files: str)
```

Selects offline ArrayRecord teacher-logit input for distillation with `teacher_model=None`.

<details>
<summary>Parameters</summary>

- `files` (`str`). Teacher-logit input files.

</details>

## RL preparation

### prepare_rl_dataset

Import from `lilax.data`.

```python
prepare_rl_dataset(dataset, *, processing_class, options=None)
```

Maps records. A supplied `options.processor` takes precedence. Without one, records containing all of `prompts`, `question`, and `answer` pass through unchanged. Other records use this conversion:

| Output | Input selection and conversion |
| --- | --- |
| `question` | First present column among `problem`, `prompt`, and `question`. A dictionary `extra_info` containing `raw_problem` overrides that choice. |
| `answer` | First present column among `answer`, `solution`, and `expected_answer`. A dictionary `reward_model` containing `ground_truth` overrides that choice. Values are decoded from bytes or converted to strings. |
| `prompts` | The selected question is formatted with the packaged `gsm8k_rl.json` math template, then the tokenizer's chat template with `tokenize=False` and `add_generation_prompt=True`. |

The dataset's `info.dataset_name` selects GSM8K extraction. `"gsm8k"` is normalized to `"openai/gsm8k"`. Only that name extracts the answer after `####`. The default conversion JSON-encodes the acceptable-answer list into `answer`. `question_type="MCQ"` also accepts the matching option letter or text. Other question types produce two copies of the selected answer. Records need both a question and an answer source.

The packaged template requests reasoning between `<reasoning>` and `</reasoning>` and a single numerical answer between `<answer>` and `</answer>`. It embeds that instruction and the question in turn markers inside one user message before the tokenizer applies its own chat template. Prepared records and custom processors bypass this formatting.

Returns the prepared dataset.

<details>
<summary>Parameters</summary>

- `dataset` (`Dataset | IterableDataset`). Input records with a question and answer source.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin`). Supplies the tokenizer for chat formatting.
- `options` (`RLDataOptions | None`, optional, defaults to `None`). Preparation settings. `None` creates default options.

</details>

### RLDataOptions

Import from `lilax.data`. Pydantic model whose keyword fields control preparation and trainer batching.
`train_fraction` must be greater than zero and at most one.

The vLLM RL pipeline keeps records whose `len(tokenizer.tokenize(record["prompts"]))` is at most the resolved prompt capacity. `RolloutConfig.max_prompt_length`, when supplied, sets that capacity. Longer records are silently dropped, including during evaluation. After filtering, training takes up to `int(num_batches * batch_size * train_fraction)` records, repeats them for the configured epochs, and drops the final partial batch. The count applies to retained records, not the original dataset size. Evaluation instead skips `test_batch_start_index` retained records and keeps its final partial batch. This filtering is part of trainer input preparation, not `prepare_rl_dataset()` itself.

<details>
<summary>Parameters</summary>

- `train_fraction` (`float`, optional, defaults to `1.0`). Fraction used in the retained training-record budget.
- `test_batch_start_index` (`int`, optional, defaults to `0`). Retained records skipped before evaluation.
- `processor` (`Callable[..., dict] | None`, optional, defaults to `None`). Custom record-to-dictionary conversion.

</details>

## AutoTokenizer.from_pretrained

Import from `lilax`.

```python
AutoTokenizer.from_pretrained(
	pretrained_model_name_or_path, *, revision=None, token=None, cache_dir=None, local_files_only=False,
	tokenize: bool | None = None, use_chat_template: bool | None = None,
	use_truncation: bool | None = None, **kwargs,
)
```

Returns the actual HF tokenizer. Lilax preprocessing defaults are `tokenize=True`, `use_chat_template=False`, and `use_truncation=True`. See the official [tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer) for encode, decode, templates, and persistence.

<details>
<summary>Parameters</summary>

- `pretrained_model_name_or_path` (`str | os.PathLike`). Repository or local tokenizer path.
- `revision` (`str | None`, optional, defaults to `None`). Snapshot revision.
- `token` (`str | bool | None`, optional, defaults to `None`). Snapshot access token.
- `cache_dir` (`str | os.PathLike | None`, optional, defaults to `None`). Snapshot cache directory.
- `local_files_only` (`bool`, optional, defaults to `False`). Use only local cached files.
- `tokenize` (`bool | None`, optional, defaults to `None`). Override saved preprocessing choice. New tokenizer default is `True`.
- `use_chat_template` (`bool | None`, optional, defaults to `None`). Override saved choice. New tokenizer default is `False`.
- `use_truncation` (`bool | None`, optional, defaults to `None`). Override saved choice. New tokenizer default is `True`.
- `**kwargs` (`Any`). Forwarded to HF tokenizer loading.

</details>

## AutoProcessor.from_pretrained

Import from `lilax`.

```python
AutoProcessor.from_pretrained(
	pretrained_model_name_or_path, *, revision=None, token=None, cache_dir=None, local_files_only=False,
	tokenize: bool | None = None, use_chat_template: bool | None = None, use_truncation: bool | None = None,
	image_column: str | list[str] | None = None, image_placeholder: str | None = None,
	video_placeholder: str | None = None, max_num_images_per_example: int | None = None,
	audio_placeholder: str | None = None, use_audio_in_video: bool | None = None,
	video_max_grid_t: int | None = None, video_max_grid_h: int | None = None,
	video_max_grid_w: int | None = None, **kwargs,
)
```

The result accepts `(images=None, text=None, videos=None, audio=None, **kwargs)`. Text-only calls use its tokenizer. Media calls prepare matching model inputs. `return_token_type_ids=True` raises `ValueError`. Model input names include `input_ids`, `attention_mask`, `position_ids`, `images`, `image_masks`, `videos`, `video_masks`, `video_grid_thw`, and `audio_values`.

Other processor methods follow the wrapped [HF processor](https://huggingface.co/docs/transformers/main_classes/processors).

<details>
<summary>Parameters</summary>

- `pretrained_model_name_or_path` (`str | os.PathLike`). Repository or local processor path.
- `revision` (`str | None`, optional, defaults to `None`). Snapshot revision.
- `token` (`str | bool | None`, optional, defaults to `None`). Snapshot access token.
- `cache_dir` (`str | os.PathLike | None`, optional, defaults to `None`). Snapshot cache directory.
- `local_files_only` (`bool`, optional, defaults to `False`). Use local cached files only.
- `tokenize` (`bool | None`, optional, defaults to `None`). Override saved choice. New processor default is `True`.
- `use_chat_template` (`bool | None`, optional, defaults to `None`). Override saved choice. New processor default is `False`.
- `use_truncation` (`bool | None`, optional, defaults to `None`). Override saved choice. New processor default is `True`.
- `image_column` (`str | list[str] | None`, optional, defaults to `None`). Image columns. Effective new-processor default is `"image"`.
- `image_placeholder` (`str | None`, optional, defaults to `None`). Image marker. Effective default is `"<|image|>"`.
- `video_placeholder` (`str | None`, optional, defaults to `None`). Video marker. Effective default is `"<|video|>"`.
- `max_num_images_per_example` (`int | None`, optional, defaults to `None`). Image limit. Effective default `-1` imposes no limit.
- `audio_placeholder` (`str | None`, optional, defaults to `None`). Audio marker. Effective default is `"<|audio|>"`.
- `use_audio_in_video` (`bool | None`, optional, defaults to `None`). Extract audio from video. Effective default is `False`.
- `video_max_grid_t` (`int | None`, optional, defaults to `None`). Temporal grid limit.
- `video_max_grid_h` (`int | None`, optional, defaults to `None`). Height grid limit.
- `video_max_grid_w` (`int | None`, optional, defaults to `None`). Width grid limit.
- `**kwargs` (`Any`). Forwarded to HF processor loading.

</details>

### Processor call

```python
processor(images=None, text=None, videos=None, audio=None, **kwargs)
```

Returns tokenized text or matching-family media model inputs. Text-only calls use the tokenizer.
`return_token_type_ids=True` raises `ValueError`.

<details>
<summary>Parameters</summary>

- `images` (`ImageInput | str | os.PathLike | list | None`, optional, defaults to `None`). Image objects or paths. Nested lists group images by text input.
- `text` (`str | list[str] | None`, optional, defaults to `None`). Text input. Media processing requires strings.
- `videos` (`str | os.PathLike | Sequence[str | os.PathLike] | None`, optional, defaults to `None`). One video file path per text input.
- `audio` (`str | os.PathLike | Sequence[str | os.PathLike] | None`, optional, defaults to `None`). One audio file path per text input.
- `**kwargs` (`Any`). Text or media processing options. Token-type IDs are unsupported.

</details>

### Processor save_pretrained

```python
processor.save_pretrained(save_directory, **kwargs)
```

Persists model-source preferences and processor settings through the wrapped HF processor.
Returns the wrapped processor's save result.

<details>
<summary>Parameters</summary>

- `save_directory` (`str | os.PathLike`). Processor destination.
- `**kwargs` (`Any`). Forwarded to the wrapped save method.

</details>

## train_tokenizer_from_files

```python
train_tokenizer_from_files(
	files: str, *, file_type: Literal["parquet", "arrayrecord", "tfrecord"] = "parquet",
	data_column: str = "text", vocab_size: int = 32768, max_corpus_chars: int = 10000000,
	assets_path: str = MAXTEXT_ASSETS_ROOT, vocab_model_name: str = "tokenizer",
)
```

Trains SentencePiece assets. `file_type` accepts `"parquet"`, `"arrayrecord"`, or `"tfrecord"`. The default assets directory is package-owned; supply your own writable `assets_path` for output.

Returns `None`.

<details>
<summary>Parameters</summary>

- `files` (`str`). Training file pattern.
- `file_type` (`Literal["parquet", "arrayrecord", "tfrecord"]`, optional, defaults to `"parquet"`). Input format.
- `data_column` (`str`, optional, defaults to `"text"`). Corpus text column.
- `vocab_size` (`int`, optional, defaults to `32768`). SentencePiece vocabulary size.
- `max_corpus_chars` (`int`, optional, defaults to `10000000`). Corpus character budget.
- `assets_path` (`str`, optional, defaults to `MAXTEXT_ASSETS_ROOT`). Package-owned assets directory.
- `vocab_model_name` (`str`, optional, defaults to `"tokenizer"`). Output tokenizer basename.

</details>
