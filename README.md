<h1 align="center">Lilax</h1>
<h3 align="center">A simpler, Hugging Face-style Python API for MaxText, built with JAX</h3>

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

Lilax makes [MaxText](https://github.com/AI-Hypercomputer/maxtext) easier to use through a Hugging Face-style
Python API. It builds on MaxText's model and training implementations, powered by
[JAX](https://github.com/jax-ml/jax) on GPUs and TPUs.

Load models with `from_pretrained`, prepare datasets, and train through familiar Trainer APIs while using
MaxText's distributed execution, low-precision computation, and model features.

## Highlights

- [Model loading](docs/guides/models/loading.md). Load supported Hub checkpoints or local weights with `AutoModelForCausalLM` and `AutoTokenizer`.
- [Pretraining](docs/guides/training/pretraining.md) and [supervised fine-tuning](docs/guides/training/sft.md). Train with text, prompt-completion pairs, conversations, and supported multimodal records.
- [LoRA and QLoRA](docs/guides/training/lora.md). Train adapters, quantize base weights, and save or reload adapter checkpoints.
- [Preference training](docs/guides/training/dpo.md) and [distillation](docs/guides/training/distillation.md). Use DPO, ORPO, or teacher-guided training.
- [Reinforcement learning](docs/guides/training/grpo.md). Train with GRPO, custom rewards, and [OpenEnv environments](docs/guides/training/environments.md).
- [Low-precision training](docs/guides/models/quantization.md). Select FP8 variants, NVFP4 with or without RHT, int8 computation, or QAT. See [recipe hardware requirements](docs/guides/models/precision-hardware.md).
- [Distributed execution](docs/guides/execution/runtime.md). Use JAX device meshes with data, tensor, pipeline, expert, and context parallelism, FSDP, DiLoCo, and [multi-node launches](docs/guides/execution/launch.md).
- [Optimization and memory](docs/guides/training/optimization.md). Configure AdamW or Muon, gradient accumulation, rematerialization, host offload, and [block-diffusion objectives](docs/guides/training/block-diffusion.md).
- [Model settings](docs/guides/models/family-settings.md). Configure expert routing, MoE kernels, multi-token prediction, and [attention and position encoding](docs/guides/models/attention.md) for supported architectures.
- [Datasets and processing](docs/guides/data/datasets.md). Load Hub or local data, process [text, images, video, and audio](docs/guides/data/processing.md), export [Parquet or ArrayRecord](docs/guides/data/export.md), and build [Megatron indexes](docs/guides/data/mmap-indexes.md).
- [Generation](docs/guides/inference/generation.md) and [serving](docs/guides/inference/serving.md). Sample text, batch requests, run continuous batching, and serve model endpoints.
- [Evaluation](docs/guides/inference/evaluation.md) and [prompt optimization](docs/guides/inference/prompt-optimization.md). Run task benchmarks, calculate [ROUGE](docs/guides/inference/text-metrics.md), and [score candidate tokens](docs/guides/inference/token-scores.md).
- [Training control](docs/guides/training/callbacks.md). Use callbacks, TensorBoard, Weights & Biases, Trackio, profiling, and [application diagnostics](docs/guides/execution/diagnostics.md).
- [Checkpoint workflows](docs/guides/training/checkpoints.md). Save and resume training, export weights, and [convert, inspect, or manage checkpoint files](docs/guides/checkpoints/checkpoint-utilities.md).
- [Advanced components](docs/guides/components/m3-components.md). Work with GLM components, [Wan diffusion components](docs/guides/components/wan-components.md), and [experimental Omni checkpoints](docs/guides/components/omni-checkpoints.md).
- [Benchmarking](docs/guides/benchmarks/benchmarks.md). Run cluster benchmarks, estimate compute and memory, and prepare [MLPerf inference workloads](docs/guides/benchmarks/mlperf-offline.md).

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
