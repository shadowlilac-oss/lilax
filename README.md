# Lilax

Train and generate with supported language models through a Hugging Face-style Python API on JAX.

[Documentation home](docs/README.md) | [Quickstart](docs/getting-started/quickstart.md) | [API reference](docs/reference/README.md)

Lilax provides model and tokenizer loading, text generation, and trainers for fine-tuning supported models.
Use familiar `from_pretrained`, `generate`, and `SFTTrainer` workflows on your JAX accelerator.

## Get started

Follow [environment setup](docs/getting-started/environment.md) for the GPU or TPU installation before running
an example. Choose hardware with enough memory for the model and, for training, gradients and optimizer state.

The examples below are independent scripts. Run them separately. Training settings illustrate API use,
not tuning recommendations.

## Load a model

Load Qwen3-8B and its tokenizer from the Hugging Face Hub. Weights are restored when an operation needs them.
First use converts Hub weights on CPU and caches an additional copy. See the
[loading guide](docs/guides/models/loading.md) for disk requirements and local checkpoints.

<details>
<summary>Load a pretrained model and tokenizer</summary>

```python
from lilax import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(model.config)
```

</details>

## Generate text

Tokenize a prompt, generate a continuation, and decode the returned tokens. The output includes the prompt.
See the [generation guide](docs/guides/inference/generation.md) for sampling and batched inputs.

<details>
<summary>Generate a text continuation</summary>

```python
from lilax import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
inputs = tokenizer("Explain gradient accumulation.", return_tensors="np", return_token_type_ids=False)
tokens = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

</details>

## Supervised fine-tuning

Load conversations from a Hub dataset, fine-tune the model, and save it locally. This example uses the
dataset's `messages` column and the model tokenizer's chat template. See the
[SFT guide](docs/guides/training/sft.md) for text and prompt-completion datasets, evaluation, and callbacks.

<details>
<summary>Fine-tune Qwen3-8B with SFTTrainer</summary>

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.data import load_dataset

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("trl-lib/Capybara", split="train").select_columns(["messages"])

args = SFTConfig(
	output_dir="./sft-run", max_steps=10, per_device_train_batch_size=1, max_length=256, packing=False,
)
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
print(result.global_step, result.metrics["train_loss"])
trainer.save_model("./sft-model")
```

</details>

Configure device placement before constructing the trainer. Training writes run artifacts under
`output_dir`. The final save writes model weights and configuration for local reloading.
See [training checkpoints](docs/guides/training/checkpoints.md) to resume optimizer state and progress.

## Explore the documentation

The [documentation home](docs/README.md) groups guides and API references by task.

- [Load and configure models](docs/guides/models/loading.md).
- [Prepare datasets](docs/guides/data/datasets.md).
- [Fine-tune with SFT](docs/guides/training/sft.md), [LoRA](docs/guides/training/lora.md), or [preferences](docs/guides/training/dpo.md).
- [Generate text](docs/guides/inference/generation.md) and [serve a model](docs/guides/inference/serving.md).
- [Train with FP8](docs/guides/models/fp8.md), [NVFP4](docs/guides/models/nvfp4.md), or [QAT](docs/guides/models/qat.md).
- [Look up the public API](docs/reference/README.md).
