# Direct task evaluation

Import from `lilax.evaluation.benchmarks`. See [task evaluation guide](../../guides/inference/task-evaluation.md).

These methods use the supplied model's parameters and processing preferences, materialize the model, and set JAX's default PRNG implementation to `"unsafe_rbg"`. Synthetic evaluation data is rejected. Media evaluation requires compatible model and processor inputs; supplying a text model does not add vision capability.

Both media evaluators upload CSV predictions when `base_output_directory` is configured and the process index is zero. `results_path` is the cloud destination or `None`. Temporary CSV removal applies after upload when `remove_tmp_results=True`.

## evaluate_mmlu

```python
evaluate_mmlu(model, *, processing_class, eval_dataset, prompt_template=mmlu_eval.DEFAULT_PROMPT_TEMPLATE)
```

Evaluates supplied multiple-choice records. The source default `mmlu_eval.DEFAULT_PROMPT_TEMPLATE` includes subject, question, choices, and an answer prefix. A custom template uses `subject`, `question`, and `choices` placeholders.

Returns `accuracy`, `subject_accuracy`, `subcategory_accuracy`, and `category_accuracy`. Empty input yields no overall accuracy. Invalid answer extraction counts as an incorrect prediction.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Loaded model to evaluate.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin`). Compatible tokenizer or processor supplying tokenization and processing preferences.
- `eval_dataset` (`Iterable[dict]`). Supplied multiple-choice records.
- `prompt_template` (`str`, optional, defaults to `mmlu_eval.DEFAULT_PROMPT_TEMPLATE`). Multiple-choice prompt template.

</details>

## evaluate_multimodal

```python
evaluate_multimodal(model, *, processing_class, eval_dataset, num_examples=-1,
	tmp_results_file="mm_eval_results.csv", remove_tmp_results=True, image_resize=-1)
```

Evaluates supplied multimodal examples and writes CSV predictions.

Returns a dictionary with `accuracy`, `results`, and `results_path`. Accuracy is `None` for no processed examples.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Loaded compatible multimodal model to evaluate.
- `processing_class` (`ProcessorMixin`). Compatible processor supplying model inputs and processing preferences.
- `eval_dataset` (`Iterable[dict]`). Supplied multimodal records.
- `num_examples` (`int`, optional, defaults to `-1`). Example limit. `-1` uses all available examples.
- `tmp_results_file` (`str`, optional, defaults to `"mm_eval_results.csv"`). Temporary CSV path.
- `remove_tmp_results` (`bool`, optional, defaults to `True`). Remove the temporary CSV after upload.
- `image_resize` (`int`, optional, defaults to `-1`). Image resize setting. `-1` preserves input size.

</details>

## evaluate_omni

```python
evaluate_omni(model, *, processing_class, eval_dataset, num_examples=-1,
	tmp_results_file="omni_eval_results.csv", remove_tmp_results=True, image_resize=-1)
```

Evaluates supplied omni examples, writes CSV predictions, and logs running and final accuracy.

Returns `None`.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Loaded compatible omni model to evaluate.
- `processing_class` (`ProcessorMixin`). Compatible processor supplying model inputs and processing preferences.
- `eval_dataset` (`Iterable[dict]`). Supplied omni records.
- `num_examples` (`int`, optional, defaults to `-1`). Example limit. `-1` uses all available examples.
- `tmp_results_file` (`str`, optional, defaults to `"omni_eval_results.csv"`). Temporary CSV path.
- `remove_tmp_results` (`bool`, optional, defaults to `True`). Remove the temporary CSV after upload.
- `image_resize` (`int`, optional, defaults to `-1`). Image resize setting. `-1` preserves input size.

</details>
