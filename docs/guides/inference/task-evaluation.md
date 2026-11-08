# Evaluate task datasets directly

Use task-specific evaluators when you want to run the supplied model directly rather than start an HTTP evaluation service.

Prepare the [model import and JetStream dependencies](../../getting-started/environment.md#set-up-a-tpu-container) before using these evaluators.

`evaluate_mmlu`, `evaluate_multimodal`, and `evaluate_omni` change the process-wide JAX default PRNG implementation to `unsafe_rbg`. They do not restore it after evaluation. Use a separate process if other work needs a different default.

## Evaluate multiple-choice questions

```python
from lilax import AutoModelForCausalLM, AutoTokenizer
from lilax.data import load_dataset
from lilax.evaluation.benchmarks import evaluate_mmlu

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("lighteval/mmlu", "all", split="test[:10]")
metrics = evaluate_mmlu(model, processing_class=tokenizer, eval_dataset=dataset)
```

Records contain `subject`, `question`, `choices`, and a zero-based integer `answer`. The evaluator reports overall, subject, subcategory, and category accuracy.

## Evaluate chart questions

Continue with the `AutoModelForCausalLM` and `load_dataset` imports from the preceding section.

With a configured output directory, process zero writes and uploads predictions, then removes the temporary CSV by default.

The registry maps `Qwen/Qwen3-VL-8B-Instruct` to the Lilax model name `weaver-max`.

```python
from lilax import AutoProcessor
from lilax.evaluation.benchmarks import evaluate_multimodal

model_id = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
dataset = load_dataset("HuggingFaceM4/ChartQA", split="val")
metrics = evaluate_multimodal(model, processing_class=processor, eval_dataset=dataset, num_examples=10)
```

ChartQA records supply an RGB-convertible `image`, `query`, and answer list `label`. The evaluator reads the first label.

`evaluate_omni` exposes the corresponding direct multimodal evaluation path for compatible Omni models. It logs accuracy and returns `None`. It is distinct from the sample-logging `generation.omni.run_evaluation` workflow.

See [task evaluation reference](../../reference/inference/task-evaluation.md).
