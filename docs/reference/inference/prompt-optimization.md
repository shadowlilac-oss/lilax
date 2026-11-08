# Math prompt optimization

Import from `lilax.evaluation.gepa`. See [workflow](../../guides/inference/prompt-optimization.md).

## load_math_dataset

```python
load_math_dataset()
```

Streams `AI-MO/NuminaMath-CoT`, train split. It keeps up to 600 AMC and AIME records with numeric boxed solutions and shuffles with seed 42.

Returns a tuple of training, validation, and test lists with up to 200, 300, and 100 records respectively. Records contain `input`, `additional_context.solution`, and `answer`.

## extract_math_answer

```python
extract_math_answer(prediction)
```

Extracts the first boxed answer while tracking nested braces.

Returns the answer string, or an empty string for an absent or unclosed box.

<details>
<summary>Parameters</summary>

- `prediction` (`str`). Generated solution text.

</details>

## MathAdapter

Import from `lilax.evaluation.gepa`. `MathAdapter` extends GEPA's `DefaultAdapter`. Constructor and client options belong to [GEPA](https://github.com/gepa-ai/gepa).

### evaluate

```python
adapter.evaluate(batch, candidate, capture_traces=False)
```

Scores use float equality between the extracted answer and the record's answer, or zero for failed numeric conversion. With traces enabled, trajectories include `extracted_answer`.

Returns a GEPA `EvaluationBatch` with outputs, numeric scores, and trajectories.

<details>
<summary>Parameters</summary>

- `batch` (`list[DefaultDataInst]`). Math records with `input`, `additional_context`, and `answer`, using GEPA's record type.
- `candidate` (`dict[str, str]`). Candidate passed to `DefaultAdapter.evaluate`.
- `capture_traces` (`bool`, optional, defaults to `False`). Evaluation trajectory capture.

</details>

## optimize_math_prompt

```python
optimize_math_prompt(model, reflection_lm, trainset, valset, seed_candidate)
```

Optimizes the math prompt through GEPA. It configures two LiteLLM workers, generation `max_tokens=8192`, and temperature zero. The fixed budget is 1000 metric calls, reflection batches of eight, no progress bar, perfect-score skipping, and merging enabled with at most five merge invocations and a validation overlap floor of five. The workflow issues model and reflection requests.

Returns the result of [GEPA optimization](https://github.com/gepa-ai/gepa/blob/v0.1.4/src/gepa/api.py). Record and completion-callable types are defined by the [GEPA adapter](https://github.com/gepa-ai/gepa/blob/v0.1.4/src/gepa/adapters/default_adapter/default_adapter.py).

<details>
<summary>Parameters</summary>

- `model` (`str | ChatCompletionCallable`). LiteLLM model name or GEPA chat-completion callable evaluated by `MathAdapter`.
- `reflection_lm` (`str | LanguageModel`). GEPA model used to propose prompt changes.
- `trainset` (`list[DefaultDataInst]`). Training records.
- `valset` (`list[DefaultDataInst]`). Validation records.
- `seed_candidate` (`dict[str, str]`). Initial candidate for optimization.

</details>
