<h1 align="center">Lilax</h1>
<h3 align="center">Language model training and generation on JAX</h3>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="docs/getting-started/environment.md"><img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-blue.svg"></a>
  <a href="docs/README.md"><img alt="Documentation" src="https://img.shields.io/badge/Docs-Guides_and_API-blue.svg"></a>
</p>

<p align="center">
  <a href="docs/README.md">Documentation</a> |
  <a href="#quickstart">Quickstart</a> |
  <a href="docs/reference/README.md">API reference</a> |
  <a href="docs/reference/models/models.md#supported-model-identities">Supported models</a>
</p>

## Overview

Lilax is a JAX library for training and running supported language models on GPUs and TPUs.
Load Hub checkpoints, fine-tune on your datasets, and save trained models through familiar
Hugging Face-style model, tokenizer, and Trainer APIs.

## Highlights

- Load [models and tokenizers](docs/guides/models/loading.md) with `from_pretrained`, then generate text with `generate`.
- Fine-tune with [`SFTTrainer`](docs/guides/training/sft.md) or train [LoRA and QLoRA adapters](docs/guides/training/lora.md).
- Choose [FP8](docs/guides/models/fp8.md), [NVFP4](docs/guides/models/nvfp4.md), or [quantization-aware training](docs/guides/models/qat.md).
- Configure [device placement and multi-node execution](docs/guides/execution/launch.md) for your training script.

## Installation

Lilax requires Python 3.12 or later. Use the supplied container for your accelerator.
See [environment setup](docs/getting-started/environment.md) for host requirements and optional dependencies.

<details>
<summary>NVIDIA GPU installation</summary>

On a host with Docker and the NVIDIA Container Toolkit configured:

```bash
git clone https://github.com/shadowlilac-oss/lilax.git
cd lilax
docker build -f Dockerfile.cuda -t lilax-cuda .
docker run --rm --gpus all -it lilax-cuda bash
```

</details>

<details>
<summary>TPU installation</summary>

On an x86_64 Linux TPU host with Docker and access to the TPU devices:

```bash
git clone https://github.com/shadowlilac-oss/lilax.git
cd lilax
docker build --platform linux/amd64 -f Dockerfile.tpu -t lilax-tpu .
docker run --rm --privileged --network host -it lilax-tpu bash
```

</details>

> [!NOTE]
> Allow accelerator memory for the model and training state. First use of Hub weights also creates a
> converted copy on disk. See [model loading](docs/guides/models/loading.md) for storage requirements and
> [runtime setup](docs/guides/execution/runtime.md) before configuring a training mesh.

## Quickstart

Expand an example to load a model, generate text, or fine-tune on a Hub dataset.
Each example is a standalone script. Run them separately in your accelerator environment.

### AutoModelForCausalLM

Load Qwen3-8B and its tokenizer from the Hugging Face Hub with
[`AutoModelForCausalLM`](docs/guides/models/loading.md).

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

### Text generation

Use [`generate`](docs/guides/inference/generation.md) to continue a prompt and decode the result.

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

The returned tokens include the prompt and its continuation.

</details>

### SFTTrainer

Fine-tune on conversations with [`SFTTrainer`](docs/guides/training/sft.md), then save the trained model.

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

The tokenizer formats the `messages` column with its chat template. This short run illustrates the API.
`save_model` writes the trained model to `./sft-model`. Training checkpoints and logs stay under `./sft-run`.

</details>

## Documentation

The [documentation home](docs/README.md) lists all guides and references by task.

| Start here | What you'll find |
| --- | --- |
| [Model loading](docs/guides/models/loading.md) | Hub and local checkpoints. |
| [Supervised fine-tuning](docs/guides/training/sft.md) | Datasets, training, and evaluation. |
| [LoRA](docs/guides/training/lora.md) | Adapter training and saving. |
| [Precision recipes](docs/guides/models/quantization.md) | FP8, NVFP4, and QAT. |
| [Generation](docs/guides/inference/generation.md) | Sampling and batched prompts. |
| [API reference](docs/reference/README.md) | Classes, arguments, and return values. |

## License

Lilax is available under the [Apache 2.0 license](LICENSE).
