# Public API reference

[Documentation home](../README.md) | [Quickstart](../getting-started/quickstart.md)

Choose a category for API signatures, arguments, return values, and errors.
For workflows and examples, use the [task guides](../README.md).

<details>
<summary>Models</summary>

| Page | What's on it |
| --- | --- |
| [Models and Hub loading](models/models.md) | Factories and supported identities |
| [Model configuration](models/configuration.md) | Model configuration objects |
| [Configuration enum values](models/configuration-enums.md) | Accepted enum values |
| [Attention and position settings](models/attention.md) | Attention and position objects |
| [Qwen family settings](models/qwen.md) | Qwen-specific configuration |
| [Vision and audio model settings](models/media-models.md) | Media encoder configuration |
| [Expert routing and kernels](models/moe.md) | Expert routing configuration |
| [Logits and multi-token prediction](models/model-loss.md) | Output and prediction settings |
| [Engram](models/engram.md) | Engram configuration |
| [Ming](models/ming.md) | Ming audio configuration |
| [Hyperconnections](models/hyperconnections.md) | Hyperconnection configuration |
| [Precision and quantization](models/quantization.md) | Precision and recipe objects |

</details>

<details>
<summary>Data and processing</summary>

| Page | What's on it |
| --- | --- |
| [Datasets and processing](data/data.md) | Dataset and processor contracts |
| [Multimodal helper records and formatting](data/media-helpers.md) | Media records and formatting |
| [File exports](data/data-export.md) | Dataset export functions |
| [Megatron index builders](data/mmap-indexes.md) | Index and blend builders |
| [Corpus preparation](data/data-preparation.md) | Corpus preparation functions |
| [Teacher generation requests](data/distillation-requests.md) | Teacher requests and generation |
| [Standalone batch loading](data/standalone-data.md) | Standalone data iteration |

</details>

<details>
<summary>Training</summary>

| Page | What's on it |
| --- | --- |
| [Trainer constructors and methods](training/trainers.md) | Trainers and method-specific settings |
| [Public SFT demo](training/sft-demo.md) | SFT demo entry point |
| [Common training arguments](training/training-arguments.md) | Common run settings |
| [Optimization and objectives](training/optimization.md) | Optimizer and objective objects |
| [Block-diffusion arrays](training/block-diffusion.md) | Token transformation functions |
| [LoRA adapters](training/adapters.md) | LoRA objects and persistence |
| [Rollouts and rewards](training/rollouts.md) | Rollout engines and reward contracts |
| [Callbacks and profiling](training/callbacks.md) | Callback events and profiler settings |
| [Checkpoints and conversion](training/checkpoints.md) | Saving, restoration, and conversion |
| [Training engine extensions](training/engine-extensions.md) | Engine protocol and payloads |

</details>

<details>
<summary>Inference and evaluation</summary>

| Page | What's on it |
| --- | --- |
| [Generation](inference/generation.md) | Generation settings and methods |
| [Serving](inference/serving.md) | Serving configuration and entry points |
| [Completion request and response models](inference/request-models.md) | Request and response records |
| [Evaluation](inference/evaluation.md) | Evaluation tasks and results |
| [Direct task evaluation](inference/task-evaluation.md) | Standalone evaluators |
| [Math prompt optimization](inference/prompt-optimization.md) | Prompt search functions |
| [Text metrics](inference/text-metrics.md) | ROUGE functions |
| [vLLM token scores](inference/token-scores.md) | Candidate token probabilities |

</details>

<details>
<summary>Runtime and execution</summary>

| Page | What's on it |
| --- | --- |
| [Runtime and distributed controls](execution/runtime.md) | Runtime initialization and contexts |
| [Execution profiles and launch options](execution/execution-profile.md) | Launch and execution settings |
| [Runtime-source fields for configured workflows](execution/runtime-source-fields.md) | Configured workflow fields |
| [Diagnostic writers and reports](execution/diagnostics.md) | Application diagnostics |

</details>

<details>
<summary>Advanced model components</summary>

| Page | What's on it |
| --- | --- |
| [AOT executable cache](components/aot-cache.md) | Cache lifecycle functions |
| [M3 construction](components/m3-components.md) | Component construction and configuration |
| [M3 execution and data settings](components/m3-execution-settings.md) | Device, data, and checkpoint settings |
| [M3 training settings](components/m3-training-settings.md) | Component optimizer and trainer settings |
| [GLM component settings](components/glm-settings.md) | GLM component configuration |
| [Diffusion component settings](components/diffusion-settings.md) | Diffusion configuration domains |
| [Wan components](components/wan-components.md) | Wan component contracts |
| [Diffusion schedulers](components/diffusion-schedulers.md) | Diffusion timestep operations |
| [Experimental Omni checkpoints](components/omni-checkpoints.md) | Stitched checkpoint operations |
| [Configured decoding and scoring](components/configured-decoding.md) | Decoding and scoring functions |
| [Benchmark generation and HTTP service](components/benchmark-server.md) | Batch generation and HTTP API |

</details>

<details>
<summary>Checkpoint tools</summary>

| Page | What's on it |
| --- | --- |
| [Checkpoint inspection and transforms](checkpoints/checkpoint-utilities.md) | Inspection and transformation functions |
| [Checkpoint storage operations](checkpoints/checkpoint-storage.md) | Managers and storage operations |
| [Qwen checkpoint conversion](checkpoints/qwen-checkpoints.md) | Qwen conversion functions |
| [Llama and Gemma checkpoint conversion](checkpoints/llama-gemma-checkpoints.md) | Llama and Gemma converters |
| [DeepSeek and GPT checkpoint conversion](checkpoints/deepseek-gpt-checkpoints.md) | DeepSeek and GPT converters |
| [M3 checkpoint operations](checkpoints/m3-checkpoints.md) | GLM component checkpoint functions |
| [Experimental conversion agents](checkpoints/conversion-agent.md) | Service-generated conversion code |

</details>

<details>
<summary>Benchmarks</summary>

| Page | What's on it |
| --- | --- |
| [Specialized benchmarks](benchmarks/benchmarks.md) | Benchmark entry points |
| [Cluster operations and disruptions](benchmarks/cluster-workflows.md) | Cluster benchmark operations |
| [Cluster metric summaries](benchmarks/cluster-metrics.md) | Cluster metric aggregation |
| [Inference benchmarks and estimates](benchmarks/inference-benchmarks.md) | Timing and estimation functions |
| [Convergence recipe settings](benchmarks/convergence-recipes.md) | Convergence settings |
| [MLPerf accuracy scoring](benchmarks/mlperf-accuracy.md) | Accuracy log scoring functions |
| [MLPerf offline inference](benchmarks/mlperf-offline.md) | Offline inference configuration |

</details>

Return to the [documentation home](../README.md).
