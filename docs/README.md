# Lilax documentation

## Start here

1. [Set up your environment](getting-started/environment.md).
2. [Load, generate, and save](getting-started/quickstart.md).
3. Choose a guide below, or open the [complete API reference](reference/README.md) for signatures and options.

## Contents

<details>
<summary>Models</summary>

| Page | What's on it |
| --- | --- |
| [Load models from the Hub or disk](guides/models/loading.md) | Supported models and weight loading |
| [Configure a model](guides/models/configuration.md) | Model dimensions and focused settings |
| [Configure attention and position encoding](guides/models/attention.md) | Attention variants and positions |
| [Update model-family settings](guides/models/family-settings.md) | Family-specific model settings |
| [Models API](reference/models/models.md) | Factories and supported identities |
| [Configuration API](reference/models/configuration.md) | Model configuration objects |
| [Attention API](reference/models/attention.md) | Attention and position objects |

</details>

<details>
<summary>Precision and quantization</summary>

| Page | What's on it |
| --- | --- |
| [Choose precision and quantization](guides/models/quantization.md) | Dtypes and recipe selection |
| [Training recipe hardware requirements](guides/models/precision-hardware.md) | Recipe hardware targets |
| [Train with NVFP4](guides/models/nvfp4.md) | NVFP4 training with optional RHT |
| [Train with FP8](guides/models/fp8.md) | FP8 training recipes |
| [Run quantization-aware training](guides/models/qat.md) | Qwix QAT training |
| [Precision and quantization API](reference/models/quantization.md) | Precision and recipe objects |

</details>

<details>
<summary>Data and processing</summary>

| Page | What's on it |
| --- | --- |
| [Load and prepare datasets](guides/data/datasets.md) | Dataset loading and training records |
| [Tokenize text and process media](guides/data/processing.md) | Text, chat, and media inputs |
| [Prepare model-specific media prompts](guides/data/media-helpers.md) | Media formatting for model families |
| [Export Hub datasets to files](guides/data/export.md) | Parquet and ArrayRecord exports |
| [Build Megatron token indexes](guides/data/mmap-indexes.md) | Token indexes and blends |
| [Prepare specialized training corpora](guides/data/preparation.md) | Video, distillation, and C4 preparation |
| [Prepare teacher generation requests](guides/data/distillation-requests.md) | Teacher response generation |
| [Datasets and processing API](reference/data/data.md) | Dataset and processor contracts |
| [Media helpers API](reference/data/media-helpers.md) | Media records and formatting |
| [File exports API](reference/data/data-export.md) | Dataset export functions |
| [Megatron indexes API](reference/data/mmap-indexes.md) | Index and blend builders |
| [Corpus preparation API](reference/data/data-preparation.md) | Corpus preparation functions |
| [Teacher requests API](reference/data/distillation-requests.md) | Teacher requests and generation |

</details>

<details>
<summary>Training</summary>

| Page | What's on it |
| --- | --- |
| [Train a causal language model](guides/training/pretraining.md) | Pretraining with Trainer |
| [Supervised fine-tuning](guides/training/sft.md) | SFT on text and conversations |
| [Train LoRA adapters](guides/training/lora.md) | Adapter training and persistence |
| [Train on preferences with DPO or ORPO](guides/training/dpo.md) | Preference-pair training |
| [Distill a teacher model](guides/training/distillation.md) | Teacher-guided training |
| [Train with GRPO](guides/training/grpo.md) | Policy training with rewards |
| [Bind a synchronous OpenEnv environment](guides/training/environments.md) | Environment-driven rollouts |
| [Configure optimization and memory](guides/training/optimization.md) | Optimizers and memory controls |
| [Prepare block-diffusion tokens](guides/training/block-diffusion.md) | Block-diffusion training arrays |
| [Trainer API](reference/training/trainers.md) | Trainer constructors and methods |
| [Training arguments API](reference/training/training-arguments.md) | Common run settings |
| [Adapters API](reference/training/adapters.md) | LoRA objects and persistence |
| [Rollouts and rewards API](reference/training/rollouts.md) | Rollout engines and reward contracts |
| [Optimization API](reference/training/optimization.md) | Optimizer and objective objects |
| [Block-diffusion API](reference/training/block-diffusion.md) | Token transformation functions |

</details>

<details>
<summary>Callbacks, profiling, and checkpoints</summary>

| Page | What's on it |
| --- | --- |
| [Log metrics and control training](guides/training/callbacks.md) | Reporting, callbacks, and profiling |
| [Save and resume training](guides/training/checkpoints.md) | Weight export and training resume |
| [Callbacks and profiling API](reference/training/callbacks.md) | Callback events and profiler settings |
| [Checkpoints API](reference/training/checkpoints.md) | Saving, restoration, and conversion |

</details>

<details>
<summary>Inference and evaluation</summary>

| Page | What's on it |
| --- | --- |
| [Generate text and batch requests](guides/inference/generation.md) | Text generation and batching |
| [Serve a model](guides/inference/serving.md) | Model serving workflows |
| [Evaluate a model](guides/inference/evaluation.md) | Task evaluation through a serving endpoint |
| [Evaluate task datasets directly](guides/inference/task-evaluation.md) | Standalone task evaluation |
| [Optimize a math prompt](guides/inference/prompt-optimization.md) | Math prompt search |
| [Score generated text with ROUGE](guides/inference/text-metrics.md) | Text overlap metrics |
| [Score candidate next tokens](guides/inference/token-scores.md) | Candidate token probabilities |
| [Construct completion request records](guides/inference/request-models.md) | Completion requests and responses |
| [Generation API](reference/inference/generation.md) | Generation settings and methods |
| [Serving API](reference/inference/serving.md) | Serving configuration and entry points |
| [Evaluation API](reference/inference/evaluation.md) | Evaluation tasks and results |
| [Direct task evaluation API](reference/inference/task-evaluation.md) | Standalone evaluators |
| [Prompt optimization API](reference/inference/prompt-optimization.md) | Prompt search functions |
| [Text metrics API](reference/inference/text-metrics.md) | ROUGE functions |
| [Token scores API](reference/inference/token-scores.md) | vLLM token scoring |
| [Completion records API](reference/inference/request-models.md) | Request and response models |

</details>

<details>
<summary>Runtime and execution</summary>

| Page | What's on it |
| --- | --- |
| [Work with runtime and distributed controls](guides/execution/runtime.md) | Device meshes and distributed setup |
| [Launch a Python training script](guides/execution/launch.md) | Execution profiles for Python jobs |
| [Record application diagnostics](guides/execution/diagnostics.md) | Application metrics and diagnostics |
| [Drive a training engine](guides/training/engine-extensions.md) | Caller-owned training loops |
| [Runtime API](reference/execution/runtime.md) | Runtime initialization and contexts |
| [Execution profiles API](reference/execution/execution-profile.md) | Launch and execution settings |
| [Diagnostics API](reference/execution/diagnostics.md) | Diagnostic writers and reports |
| [Training engine API](reference/training/engine-extensions.md) | Engine protocol and payloads |

</details>

<details>
<summary>Advanced model components</summary>

| Page | What's on it |
| --- | --- |
| [Cache compiled inference functions](guides/components/aot-cache.md) | Reusable compiled functions |
| [Construct M3 model components](guides/components/m3-components.md) | GLM components and layer scanning |
| [Configure M3 component domains](guides/components/m3-settings.md) | Execution, training, and diffusion settings |
| [Work with Wan diffusion components](guides/components/wan-components.md) | Wan model assembly |
| [Evaluate a stitched Omni checkpoint](guides/components/omni-checkpoints.md) | Experimental Omni evaluation |
| [Decode with an existing runtime](guides/components/configured-decoding.md) | Configured decoding and scoring |
| [Run the benchmark completion server](guides/components/benchmark-server.md) | Batch generation and HTTP serving |
| [Executable cache API](reference/components/aot-cache.md) | Cache lifecycle functions |
| [M3 construction API](reference/components/m3-components.md) | Component construction and configuration |
| [M3 execution settings API](reference/components/m3-execution-settings.md) | Device, data, and checkpoint settings |
| [M3 training settings API](reference/components/m3-training-settings.md) | Component optimizer and trainer settings |
| [Diffusion settings API](reference/components/diffusion-settings.md) | Diffusion configuration domains |
| [Wan components API](reference/components/wan-components.md) | Wan component contracts |
| [Omni checkpoints API](reference/components/omni-checkpoints.md) | Stitched checkpoint operations |
| [Configured decoding API](reference/components/configured-decoding.md) | Decoding and scoring functions |
| [Benchmark service API](reference/components/benchmark-server.md) | Batch generation and HTTP API |

</details>

<details>
<summary>Checkpoint tools</summary>

| Page | What's on it |
| --- | --- |
| [Convert a family checkpoint](guides/checkpoints/family-checkpoints.md) | Family-specific weight conversion |
| [Save and convert GLM component checkpoints](guides/checkpoints/m3-checkpoints.md) | GLM component checkpoint operations |
| [Inspect and manage checkpoint files](guides/checkpoints/checkpoint-utilities.md) | Inspection, transforms, and storage |
| [Generate conversion code with an experimental agent](guides/checkpoints/conversion-agent.md) | Service-generated conversion code |
| [Checkpoint inspection API](reference/checkpoints/checkpoint-utilities.md) | Inspection and transformation functions |
| [Checkpoint storage API](reference/checkpoints/checkpoint-storage.md) | Managers and storage operations |
| [Qwen conversion API](reference/checkpoints/qwen-checkpoints.md) | Qwen conversion functions |
| [Llama and Gemma conversion API](reference/checkpoints/llama-gemma-checkpoints.md) | Llama and Gemma converters |
| [DeepSeek and GPT conversion API](reference/checkpoints/deepseek-gpt-checkpoints.md) | DeepSeek and GPT converters |
| [M3 checkpoint API](reference/checkpoints/m3-checkpoints.md) | GLM component checkpoint functions |
| [Conversion agents API](reference/checkpoints/conversion-agent.md) | Experimental conversion agents |

</details>

<details>
<summary>Benchmark workflows</summary>

| Page | What's on it |
| --- | --- |
| [Run specialized benchmarks](guides/benchmarks/benchmarks.md) | Specialized benchmark workflows |
| [Run cluster benchmark variants](guides/benchmarks/cluster-workflows.md) | Cluster runs and disruptions |
| [Measure or estimate an inference workload](guides/benchmarks/inference-benchmarks.md) | Inference timing and estimates |
| [Prepare convergence recipe settings](guides/benchmarks/convergence-recipes.md) | Convergence run configuration |
| [Score an MLPerf accuracy log](guides/benchmarks/mlperf-accuracy.md) | MLPerf accuracy scoring |
| [Configure an MLPerf offline run](guides/benchmarks/mlperf-offline.md) | MLPerf offline inference setup |
| [Specialized benchmarks API](reference/benchmarks/benchmarks.md) | Benchmark entry points |
| [Cluster operations API](reference/benchmarks/cluster-workflows.md) | Cluster benchmark operations |
| [Inference benchmarks API](reference/benchmarks/inference-benchmarks.md) | Timing and estimation functions |
| [Convergence recipes API](reference/benchmarks/convergence-recipes.md) | Convergence settings |
| [MLPerf accuracy API](reference/benchmarks/mlperf-accuracy.md) | Accuracy log scoring functions |
| [MLPerf offline API](reference/benchmarks/mlperf-offline.md) | Offline inference configuration |

</details>

The [complete API reference](reference/README.md) also groups model-family settings, component settings, and execution fields for lookup.
