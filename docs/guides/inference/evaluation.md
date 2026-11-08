# Evaluate a model

Benchmark evaluation uses a serving endpoint. Trainer evaluation measures held-out training loss and is a separate workflow.

## Run a task suite

Complete the [TPU-vLLM setup](../../getting-started/environment.md#tpu-vllm-workflows), including the `evaluation` extra. Evaluation starts a local service and writes JSON files under `results_path`. If the model is already materialized, endpoint startup first saves a full weight copy under `~/tmp/lilax/serving/`. Reserve disk space for that copy. The automatically owned endpoint removes its copy on exit, including startup failure.

The evaluator owns the service lifecycle when given a model. It uses `host="localhost"`, port `8000`, `tensor_parallel_size=4`, and `max_model_len=model.config.max_position_embeddings`. It enables chat for SimpleEvals and evalchemy. Use an explicit endpoint below to change placement or capacity.

```python
from lilax import AutoModelForCausalLM, AutoTokenizer
from lilax.evaluation import EvalOutputConfig, SimpleEvalsConfig, evaluate

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
results = evaluate(
	model, config=SimpleEvalsConfig(tasks=("gsm8k",), num_samples=10),
	output=EvalOutputConfig(results_path="./evaluation"), processing_class=tokenizer,
)
```

The sample count illustrates a subset, not an evaluation recommendation.

On process zero, `results["results"]["scores"]` contains the scores and `results["local_path"]` is the absolute JSON path. Other JAX processes return `{}`.

## Evaluate your own records

Supply a finite Hugging Face dataset with `question` and nonempty `response` strings. `system_prompt` is optional. Both `"openorca"` and `"mlperf_openorca"` select this record format. Without `eval_dataset`, the evaluator streams `Open-Orca/OpenOrca` from the Hub.

This example writes scores to `./dataset-results`. The custom metric measures exact text agreement on one illustrative record. It replaces the built-in dataset scorer.

```python
from datasets import Dataset
from lilax import AutoModelForCausalLM, AutoTokenizer
from lilax.evaluation import DatasetEvalConfig, EvalOutputConfig, evaluate

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([
	{"system_prompt": "Answer with one word.", "question": "What is the capital of France?", "response": "Paris"},
])

def compute_metrics(responses, references):
	return {"exact_match": sum(r.strip() == t.strip() for r, t in zip(responses, references)) / len(references)}

output = evaluate(
	model, config=DatasetEvalConfig(benchmark="openorca"),
	output=EvalOutputConfig(results_path="./dataset-results"), processing_class=tokenizer,
	eval_dataset=dataset, compute_metrics=compute_metrics,
)
if output:
	print(output["results"]["scores"]["exact_match"])
	print(output["local_path"])
```

See the upstream [Dataset.from_list reference](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.from_list) for constructing datasets. Lilax's metric callable takes two positional lists, `responses` and `references`.

## Choose a harness

```python
from lilax.evaluation import HarnessEvalConfig

results = evaluate(
	model, config=HarnessEvalConfig(backend="lm_eval", tasks=("mmlu",)),
	output=EvalOutputConfig(results_path="./harness-results"), processing_class=tokenizer,
)
```

`backend="evalchemy"` requires a chat-capable endpoint. Harness datasets must match the selected task's schema. Custom metrics require text predictions and text references. Harness and SimpleEvals add task prefixes to custom metric keys. See the [evaluation reference](../../reference/inference/evaluation.md#custom-input-and-metrics) for each backend's restrictions.

## Reuse a running endpoint

Entering the context starts a service and may write the full temporary weight copy described above. Exiting stops the service and removes that copy. Pass the model and tokenizer loaded above. Set `tensor_parallel_size` to the partition count required by your deployment.

```python
from lilax.serving import EvaluationServerConfig, ServingEndpoint

with ServingEndpoint(
	model, config=EvaluationServerConfig(max_model_len=model.config.max_position_embeddings, enable_chat_api=True),
	processing_class=tokenizer,
) as endpoint:
	results = evaluate(
		endpoint, config=SimpleEvalsConfig(tasks=("gsm8k",), num_samples=10),
		output=EvalOutputConfig(results_path="./endpoint-results"), processing_class=tokenizer,
	)
```

See [evaluation reference](../../reference/inference/evaluation.md) for task choices and output settings.
