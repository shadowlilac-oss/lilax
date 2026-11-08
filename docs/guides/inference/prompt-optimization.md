# Optimize a math prompt

The GEPA helper optimizes a system prompt against boxed numeric answers. It calls a task model and a reflection model; configure those clients before starting optimization.

Prepare the [public import dependencies](../../getting-started/environment.md#set-up-a-tpu-container) and add the [`gepa` extra](../../getting-started/environment.md#select-optional-dependencies). If you host either client with Lilax's vLLM engine, also complete the [TPU-vLLM setup](../../getting-started/environment.md#tpu-vllm-workflows).

## Prepare the dataset

```python
from lilax.evaluation.gepa import load_math_dataset

trainset, valset, testset = load_math_dataset()
```

The loader streams NuminaMath-CoT, keeps AMC and AIME examples with numeric boxed answers, and returns train, validation, and test lists.

## Optimize a supplied prompt

Configure `model_client` and `reflection_client` for your own services before this call. It sends model requests with a fixed budget of 1,000 metric calls.

```python
from lilax.evaluation.gepa import optimize_math_prompt

result = optimize_math_prompt(
	model_client, reflection_client, trainset, valset,
	{"system_prompt": "Solve the problem and put the numeric answer in \\boxed{...}."},
)
```

The task client is a LiteLLM model name or compatible chat completion callable accepted by GEPA. The reflection client is the value accepted by GEPA's optimizer. These clients are external services or already configured clients, not Lilax model wrappers.

The input lists contain records shaped like `{"input": "What is 2 + 2?", "additional_context": {"solution": "2 + 2 = 4"}, "answer": "4"}`. The return value is GEPA's optimization result. See [prompt optimization reference](../../reference/inference/prompt-optimization.md).
