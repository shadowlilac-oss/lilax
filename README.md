<h1 align="center">Lilax</h1>
<h3 align="center">Language model training and inference in JAX</h3>

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

# This repository is a work in progress

## Overview

Lilax brings the capabilities, integration and performance of Google's [JAX](https://github.com/jax-ml/jax) framework to
the already well known HuggingFace training. 
Train
[LoRA or QLoRA adapters](docs/guides/training/lora.md), teach models to operate agentic environments via [GRPO](docs/guides/training/grpo.md), or [train a model from scratch](docs/guides/training/pretraining.md)
with familiar Hugging Face-style model, dataset, and Trainer APIs.

Below is an example of how finetuning a model with [SFT](docs/guides/training/sft.md) on your own data is done in Lilax compared
to HuggingFace's TRL. 
### Lilax
```py
from datasets import load_dataset
from lilax import (
    AutoModelForCausalLM, AutoTokenizer,
    SFTConfig, SFTTrainer,
)

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
)

trainer = SFTTrainer(
    model,
    args=SFTConfig(
        output_dir="./sft", max_steps=10,
        per_device_train_batch_size=1,
        max_length=256, packing=False,
        completion_only_loss=False,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### HuggingFace
```py
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
)

trainer = SFTTrainer(
    model,
    args=SFTConfig(
        output_dir="./sft", max_steps=10,
        per_device_train_batch_size=1,
        max_length=256, packing=False,
        completion_only_loss=False,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```
## Why Lilax?

- **Integration with HuggingFace hub.** Models are automatically converted between the HuggingFace and JAX file formats. Models you train in Lilax can be uploaded to HuggingFace hub without .
- **Mixed precision training.** Utilize the data types you were promised by your GPU's specs sheet, but were impossible to use for training like [FP8](docs/guides/models/fp8.md) or [NVFP4](docs/guides/models/nvfp4.md) for an up to 4x higher compute ceiling on supported hardware.
- **QAT Training** Finetune LLMs to learn to be quantized. With [QAT](docs/guides/models/qat.md) training you can reduce the quantization error significantly 
- **Fit larger models in memory.** Use [Gradient Checkpointing](docs/reference/execution/execution-profile-compute.md#remat)
  and [CPU offload](docs/guides/training/optimization.md#offload-memory) to fit models on your GPUs that previously ran out of memory.
- **Find what slows training** Use the [XProf Web UI](docs/guides/training/callbacks.md#capture-profiles) to see how efficiently your model is training compared to what your hardware allows, and what is slowing it down.
- **Less communication between machines** Speed up your training on multi-gpu setups lacking NVLink or fast PCI Express lanes with Google's [DiLoCo](docs/reference/execution/runtime.md#dilococonfig) training algorithm.
- **Distil from larger models** Train smaller student models to imitate more powerful teachers, without having to keep both models loaded using [Offline distillation](docs/guides/training/distillation.md#train-from-saved-records).

Lilax is a [JAX](https://github.com/jax-ml/jax) framework for training and running language models on GPUs
and TPUs. [Fine-tune on your own data](docs/guides/training/sft.md), train
[LoRA or QLoRA adapters](docs/guides/training/lora.md), teach models through
[rewards](docs/guides/training/grpo.md), or [train from scratch](docs/guides/training/pretraining.md)
with familiar Hugging Face-style model, dataset, and Trainer APIs.

JAX's compiled training, memory controls, and tools for splitting work across devices are integrated
into those workflows. You can select FP8 or NVFP4 training, add quantization-aware training, and capture
performance profiles without writing a separate training loop.

Much of Lilax's code comes from [MaxText](https://github.com/AI-Hypercomputer/maxtext), including model and
distributed-training implementations. Lilax reworks the architecture and user workflows and adds its own
implementations and integrations.

Supported model families include Qwen, Llama, Gemma, DeepSeek, Mixtral, and GPT-OSS, with dense and
mixture-of-experts models. See the [model list](docs/reference/models/models.md#supported-model-identities)
for exact checkpoints and supported loading and export formats.

## Why Lilax?

- **FP8 and NVFP4 mixed-precision training.** [FP8](docs/guides/models/fp8.md) and
  [NVFP4](docs/guides/models/nvfp4.md) training math includes activations, not just compressed weights,
  to use your [hardware's low-precision compute](docs/guides/models/precision-hardware.md).
- **Quantization-aware training.** [QAT](docs/guides/models/qat.md) is part of the normal training workflow,
  helping models tolerate the rounding errors of lower-precision inference.
- **Fit larger models in memory.** [Selective recomputation](docs/reference/execution/execution-profile-compute.md#remat)
  and [CPU offload](docs/guides/training/optimization.md#offload-memory) reduce training memory use.
- **GPU and TPU scaling.** The same Python workflow supports
  [training across devices and machines](docs/guides/execution/launch.md).
- **Find what slows training.** Built-in [XProf captures](docs/guides/training/callbacks.md#capture-profiles)
  show GPU and TPU activity. Optional [cloud reports](docs/reference/execution/execution-profile-monitoring.md#goodput)
  show time spent on useful training versus overhead.
- **Less communication across machines.** [DiLoCo](docs/reference/execution/runtime.md#dilococonfig)
  syncs less often, with [streaming updates](docs/reference/execution/execution-profile-compute.md#diloco)
  spread across training steps.
- **Train from another model's saved predictions.** [Offline distillation](docs/guides/training/distillation.md#train-from-saved-records)
  avoids keeping a teacher model running alongside each student.
- **More training objectives.** [Multi-token prediction](docs/reference/models/model-loss.md#mtpconfig)
  predicts several future tokens. [Block diffusion](docs/guides/training/block-diffusion.md) learns to fill
  masked token blocks in compatible models.
- **Fine-tune with more than text.** Supported multimodal models learn from paired text and
  [images, video, or audio](docs/guides/data/processing.md).
- **Tool-use training.** [GRPO](docs/guides/training/grpo.md) supports rewards for interactions with
  [environments and tools](docs/guides/training/environments.md).

## Coming from Hugging Face

The model-loading, SFT, and adapter APIs follow the patterns in Transformers, TRL, and PEFT.
[Transformers v5 focuses on PyTorch](https://huggingface.co/blog/transformers-v5#code-reduction).
Lilax uses JAX for model execution and training on GPUs and TPUs.

For training with less frequent communication, [Accelerate's Local SGD guide](https://huggingface.co/docs/accelerate/usage_guides/local_sgd)
adds synchronization to a custom training loop. Lilax's DiLoCo configuration keeps that work inside the
Trainer and also offers streaming updates. These are different training algorithms, not interchangeable settings.

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
> [runtime setup](docs/guides/execution/runtime.md) before choosing how to split training across devices.

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
