# Runtime and distributed controls

See [execution guide](../../guides/execution/runtime.md).

## Configuration objects

The runtime-source configuration is `model.runtime.public_config`. It contains the model's projected execution settings. The resolved configuration is `runtime.config` after `runtime.resolve_config()` or `runtime.initialize()`.

```python
from lilax import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
runtime = model.runtime
source = runtime.public_config
runtime.resolve_config()
resolved = runtime.config
```

Resolution initializes JAX but does not construct the training mesh. Before resolution, `runtime.config` is the same object as `runtime.public_config`.

The root model configuration, `model.config`, is a different object. The public `lilax.MaxTextConfig` and the runtime-source class named `MaxTextConfig` have different identities and are not interchangeable. Pass the produced objects above to APIs that require runtime-source or resolved configuration. GRPO prepares an RL runtime-source configuration for `Runtime.for_rl`; a root model configuration is not accepted there.

## Model runtime

Import `Runtime` from `lilax.runtime`. `model.runtime` is the runtime already bound to a model.

```python
Runtime(config, devices=None)
```

Its configuration is the runtime-source object produced for a public model, not the root HF-style model configuration. It stores that object as `public_config` and resolves it into `config`.

See [runtime-source fields](runtime-source-fields.md) for checkpoint, tokenizer, prompt, length, scoring, and benchmark settings consumed by configured workflows.

<details>
<summary>Parameters</summary>

- `config` (`object`). Runtime-source configuration from `model.runtime.public_config`, as described under [configuration objects](#configuration-objects).
- `devices` (`list[jax.Device] | None`, optional, defaults to `None`). Devices used to construct the mesh.

</details>

### resolve_config

```python
runtime.resolve_config() -> "Runtime"
```

Resolves configuration without constructing a mesh.

Returns the runtime.

### initialize

```python
runtime.initialize(elastic_retry_cache=None, *, overrides=None,
	train_dataset_provided=False, eval_dataset_provided=False)
```

Resolves configuration and builds or reuses the training mesh. Once initialized, incompatible mesh or sharding changes raise `ValueError`.

Returns the runtime.

<details>
<summary>Parameters</summary>

- `elastic_retry_cache` (`object | None`, optional, defaults to `None`). Retry cache supplied by the training lifecycle for mesh reuse across attempts.
- `overrides` (`dict[str, Any] | None`, optional, defaults to `None`). Runtime resolution overrides.
- `train_dataset_provided` (`bool`, optional, defaults to `False`). Presence of caller-supplied training data.
- `eval_dataset_provided` (`bool`, optional, defaults to `False`). Presence of caller-supplied evaluation data.

</details>

### context

```python
runtime.context(*, evaluation=False)
```

Activates the mesh and logical axis rules. Evaluation selects evaluation rules. Materialize the model before a direct forward-pass context.

Returns a context manager.

<details>
<summary>Parameters</summary>

- `evaluation` (`bool`, optional, defaults to `False`). Selection of evaluation logical-axis rules.

</details>

### resolve_runtime_config

Import from `lilax.generation.m3`.

```python
resolve_runtime_config(public_config, *, overrides=None,
	train_dataset_provided=False, eval_dataset_provided=False)
```

Resolves a runtime-source configuration. A root HF-style model configuration or an object of another class raises `TypeError`.

Returns the resolved runtime configuration.

<details>
<summary>Parameters</summary>

- `public_config` (`object`). Runtime-source configuration from `model.runtime.public_config`, as described under [configuration objects](#configuration-objects).
- `overrides` (`dict[str, Any] | None`, optional, defaults to `None`). Resolution overrides.
- `train_dataset_provided` (`bool`, optional, defaults to `False`). Presence of caller-supplied training data.
- `eval_dataset_provided` (`bool`, optional, defaults to `False`). Presence of caller-supplied evaluation data.

</details>

### for_rl

```python
Runtime.for_rl(config, *, devices=None)
```

Requires a runtime-source RL configuration, partitions trainer and sampler devices, and initializes the trainer runtime. A root model configuration raises `TypeError`. Normal GRPO users let the trainer prepare this configuration.

Returns `(runtime, sampler_config, sampler_devices)`.

<details>
<summary>Parameters</summary>

- `config` (`object`). RL runtime-source configuration prepared by the GRPO trainer. A non-RL runtime source or root model configuration raises `TypeError`.
- `devices` (`list[jax.Device] | None`, optional, defaults to `None`). Devices to partition between trainer and sampler.

</details>

## ParallelismConfig

Import the following Pydantic configuration classes from `lilax.distributed`. They reject extra fields. Pass them through trainer arguments rather than model configuration.

`ParallelismConfig` composes `MeshConfig` and `ShardingConfig`.

<details>
<summary>Parameters</summary>

- `mesh` (`MeshConfig | None`, optional, defaults to `None`). Mesh semantics.
- `sharding` (`ShardingConfig | None`, optional, defaults to `None`). Optimizer and weight sharding.

</details>

### MeshConfig

Import from `lilax.distributed`. Pydantic model with keyword fields. Extra fields are forbidden.

Controls sharding semantics and context exchange. Runtime resolution lowercases the context strategy and rejects unknown values with `ValueError`.

Import `ShardMode` from `lilax.distributed`. `ShardMode.AUTO` has value `"auto"` and lets JAX infer sharding through operations. `ShardMode.EXPLICIT` has value `"explicit"` and uses explicit output shardings in supported operations.

Context strategy selection does not establish compatibility with every attention configuration. `ulysses` and `usp` require TPU, flash attention with Tokamax Splash, explicit positive context sizes, and compatible sequence and head divisibility. They reject DCN context parallelism. TPU `ring` requires more than one context device, `context_sharding="context"`, flash attention with Tokamax Splash, and sequence length divisible by the square of the context size. Unsupported combinations raise `ValueError` during runtime resolution. Physical context sizes and attention settings belong to the [execution profile](execution-profile.md) and [runtime-source fields](runtime-source-fields.md).

<details>
<summary>Parameters</summary>

- `shard_mode` (`ShardMode`, optional, defaults to `ShardMode.AUTO`). `auto` or `explicit`.
- `context_parallel_strategy` (`str`, optional, defaults to `"all_gather"`). `all_gather`, `ring`, `ulysses`, or `usp`.

</details>

### ShardingConfig

Import from `lilax.distributed`. Pydantic model with keyword fields. Extra fields are forbidden.

Runtime validation rejects `save_fsdp_gathered_weights=True` with a nonempty quantization mode, FP8 weight dtype, or parameter host offload, with `ValueError`. Here, "save" means retaining kernels during a step, not saving a checkpoint. Prefetch requires `scan_layers=True` and has no effect without a multi-device FSDP mesh axis.

<details>
<summary>Parameters</summary>

- `shard_optimizer_over_data` (`bool`, optional, defaults to `False`). ZeRO-1 optimizer-state sharding over data.
- `save_fsdp_gathered_weights` (`bool`, optional, defaults to `False`). Gathered-kernel retention in activation dtype as rematerialization residuals to avoid backward gathers. Increases device-memory use.
- `prefetch_fsdp_weights` (`bool`, optional, defaults to `False`). Next-layer weight gathering during current-layer forward and backward computation for scanned layers.
- `prefetch_fsdp_defer_grad_reduce_scatter` (`bool`, optional, defaults to `False`). Gradient reduce-scatter deferral to the next backward step with prefetch and explicit sharding. Keeps an extra layer of unreduced gradients in memory.

</details>

## DiLoCoConfig

Import from `lilax.distributed`. Pydantic model with keyword fields. Extra fields are forbidden.

Controls outer synchronization between groups of training devices.

<details>
<summary>Parameters</summary>

- `enable_diloco` (`bool`, optional, defaults to `False`). DiLoCo synchronization.
- `diloco_sync_period` (`int`, optional, defaults to `36`). Local steps between synchronizations.
- `diloco_outer_lr` (`float`, optional, defaults to `0.3`). Outer optimizer learning rate.
- `diloco_outer_momentum` (`float`, optional, defaults to `0.9`). Outer optimizer momentum.

</details>
