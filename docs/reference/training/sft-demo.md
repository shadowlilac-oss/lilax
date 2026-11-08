# Public SFT demo

Import functions from `lilax.trainers.examples.sft_demo`. See the [SFT demo workflow](../../guides/training/sft.md#evaluate-before-and-after-sft-with-the-public-demo).

## get_test_dataset

```python
get_test_dataset(processing_class, *, token=None)
```

Loads and caches `openai/gsm8k`, `data_dir="main"`, split `"test"`. Shuffles with seed `42` and adds `prompt` and `target_answer` using the packaged math chat template. Answer extraction splits on the packaged separator. Records without it receive `target_answer=None`, which is not a valid string target for `score_response`.

Returns a Hugging Face `Dataset` retaining the original columns with the added formatted columns.

<details>
<summary>Parameters</summary>

- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin`). Processing object providing `apply_chat_template`.
- `token` (`str | bool | None`, optional, defaults to `None`). Authentication passed to `datasets.load_dataset`.

</details>

## evaluate_model

```python
evaluate_model(dataset, model, processing_class, debug=True)
```

Consumes batches with `prompt`, `question`, and `target_answer`. A tokenized prompt exceeding 256 tokens raises `ValueError`. Generation requests up to 768 new tokens with `do_sample=True`, `temperature=1e-4`, `top_k=1`, and `top_p=1.0`. A progress bar is displayed. Debug output prints questions, responses, targets, and scoring exceptions.

Returns a dictionary with `correct`, `partially_correct`, and `correct_format` percentages. Empty input divides by zero. The combined helper below supplies batches of four.

<details>
<summary>Parameters</summary>

- `dataset` (`Iterable[dict[str, list]]`). Nonempty batched records, such as `test_dataset.batch(4)`.
- `model` (`MaxTextModel | PeftModel`). Model providing public `generate`.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin`). Prompt encoder providing `batch_decode`.
- `debug` (`bool`, optional, defaults to `True`). Printed evaluation details and scoring exceptions.

</details>

## score_response

```python
score_response(target, prediction, debug=True)
```

Extracts a numeric prediction after `<answer>` and removes commas, spaces, and dollar signs from it and the target. Exact agreement compares converted floats. Partial agreement requires a prediction-to-target ratio in `[0.9, 1.1]`. Format agreement requires nonempty `<reasoning>...</reasoning>` and `<answer>...</answer>` sections and no trailing non-whitespace text. Numeric conversion and division share one exception handler, so a zero target prevents the format check after the division fails. Non-string targets fail before that handler.

Returns `(is_correct, is_partially_correct, has_correct_format)`, a tuple of booleans. These are the demo's fixed rules, not a custom reward function.

<details>
<summary>Parameters</summary>

- `target` (`str`). Numeric reference text.
- `prediction` (`str`). Generated response text.
- `debug` (`bool`, optional, defaults to `True`). Printed numeric conversion or division exceptions.

</details>

## safe_string_to_float

```python
safe_string_to_float(text)
```

Returns a string after removing commas, spaces, and dollar signs. Despite its name, this helper does not convert the result to a float. `score_response` performs the numeric conversion separately.

<details>
<summary>Parameters</summary>

- `text` (`str`). Numeric text to clean.

</details>

## finetune_and_evaluate

```python
finetune_and_evaluate(model, args, train_dataset, test_dataset, processing_class, debug=True)
```

Constructs `SFTTrainer`, evaluates the supplied model, trains it in place, and evaluates `trainer.model`. Does not attach adapters. Uses the caller's training and checkpoint settings.

Returns `(trainer, output, before, after)`. `output` is `TrainOutput` with `global_step`, `training_loss`, and `metrics={"train_loss": ...}`. The two evaluation dictionaries follow `evaluate_model` above.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel | PeftModel`). Base or already adapted model.
- `args` (`SFTConfig`). Caller-selected SFT settings.
- `train_dataset` (`Dataset | IterableDataset`). Training records accepted by `SFTTrainer`.
- `test_dataset` (`Dataset`). Nonempty formatted test records supporting `.batch(4)`.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin`). Shared training and evaluation processing object.
- `debug` (`bool`, optional, defaults to `True`). Printed evaluation details.

</details>
