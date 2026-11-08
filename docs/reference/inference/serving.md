# Serving

Import from `lilax.serving`. See [serving guide](../../guides/inference/serving.md).

## Server lifecycle

`serve` and `start_server` set the process-wide `jax_default_prng_impl` to `"unsafe_rbg"` and the environment variable `TF_CPP_MIN_LOG_LEVEL` to `"0"` before server construction. Neither setting is restored on exit or construction failure.

`serve` starts JetStream and blocks until termination. `start_server` returns its server object.

JetStream serves the `Orchestrator.Decode` streaming gRPC API on port `9000`. The port is fixed in Lilax and has no `JetStreamConfig` override. `prometheus_port` configures a separate metrics service. The [pinned JetStream protocol](https://github.com/AI-Hypercomputer/JetStream/blob/29329e8e73820993f77cfc8efe34eb2a73f5de98/jetstream/core/proto/jetstream.proto) accepts text or token IDs and returns stream samples containing text and token IDs. See the guide for a client request.

For materialized models, all serving entry points save a full checkpoint copy in a temporary directory under `~/tmp/lilax/serving/`. Unmaterialized pretrained models resolve their existing checkpoint instead. An unmaterialized model without a pretrained checkpoint raises `ValueError`.

`serve` explicitly cleans its temporary directory in `finally` after `wait_for_termination`. `ServingEndpoint` cleans its directory on context exit, including startup failure. `start_server` and `create_vllm_llm` attach the `TemporaryDirectory` to the returned object as `_lilax_checkpoint_directory`; neither installs a Lilax stop-time cleanup hook. The object retains the copy until temporary-directory cleanup or object finalization. All construction paths explicitly clean temporary storage on failure.

`create_vllm_llm` sets the process-wide JAX default PRNG implementation to `"unsafe_rbg"`. It sets `SKIP_JAX_PRECOMPILE="1"`, `NEW_MODEL_DESIGN="1"`, and `TF_CPP_MIN_LOG_LEVEL="0"`, and adds `--xla_tpu_spmd_rng_bit_generator_unsafe=true` to `LIBTPU_INIT_ARGS` if absent. These settings are not restored.

### serve

```python
serve(model, *, server_config: JetStreamConfig | None = None) -> None
```

Starts JetStream and waits for termination. The lifecycle effects and checkpoint ownership above apply.

Returns `None` after termination.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Pretrained or materialized model with weights to serve.
- `server_config` (`JetStreamConfig | None`, optional, defaults to `None`). Uses `JetStreamConfig()` when omitted.

</details>

### start_server

```python
start_server(model, *, server_config: JetStreamConfig | None = None) -> server_lib.JetStreamServer
```

Starts JetStream with the lifecycle effects and checkpoint ownership above.

Returns a JetStream server for caller-managed stopping or waiting.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Pretrained or materialized model with weights to serve.
- `server_config` (`JetStreamConfig | None`, optional, defaults to `None`). Uses `JetStreamConfig()` when omitted.

</details>

### create_vllm_llm

```python
create_vllm_llm(model, *, serving_config: VLLMConfig | None = None)
```

Creates a vLLM engine with the process-wide settings and checkpoint ownership above.

Returns the vLLM engine.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Pretrained or materialized model with weights to serve.
- `serving_config` (`VLLMConfig | None`, optional, defaults to `None`). Uses `VLLMConfig()` when omitted.

</details>

### ServingEndpoint

Import from `lilax.serving`. Class with the explicit constructor below.

```python
ServingEndpoint(model, *, config: EvaluationServerConfig, processing_class=None)
```

An evaluation-server context manager. `base_url` and `concurrency` require an entered context. Re-entering a running endpoint raises `RuntimeError`. Exiting stops the service and releases temporary checkpoint storage.

Its `base_url` is `http://{host}:{port}`. The service exposes `GET /health` and `POST /v1/completions`. It also exposes `POST /v1/chat/completions` when `EvaluationServerConfig(enable_chat_api=True)` is selected.

Returns a context manager whose entered value is the endpoint itself.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Pretrained or materialized model with weights to serve.
- `config` (`EvaluationServerConfig`). Evaluation server placement and request capacity.
- `processing_class` (`PreTrainedTokenizerBase | ProcessorMixin | None`, optional, defaults to `None`). Compatible processing class retained for evaluation.

</details>

## JetStreamConfig

Import from `lilax.serving`. Pydantic settings for JetStream. Unknown fields are rejected. The gRPC port remains fixed at `9000`.

<details>
<summary>Parameters</summary>

- `inference_server` (`str`, optional, defaults to `"MaxtextInterleavedServer"`). Server to start.
- `prefill_slice` (`str`, optional, defaults to `"v5e-16"`). Device slice for disaggregated prefill.
- `generate_slice` (`str`, optional, defaults to `"v5e-16"`). Device slice for disaggregated decode.
- `use_chunked_prefill` (`bool`, optional, defaults to `False`). Enable chunked prefilling.
- `prefill_chunk_size` (`int`, optional, defaults to `256`). Prefill chunk size.
- `enable_prefix_caching` (`bool`, optional, defaults to `False`). Enable prefix caching.
- `prefix_caching_hbm_byte` (`int`, optional, defaults to `10000000000`). HBM bytes for prefix caching.
- `prefix_caching_dram_byte` (`int`, optional, defaults to `100000000000`). DRAM bytes for prefix caching.
- `enable_model_warmup` (`bool`, optional, defaults to `False`). Run warmup before serving.
- `lora_input_adapters_path` (`str`, optional, defaults to `""`). Input LoRA adapter directory.
- `multi_sampling` (`bool`, optional, defaults to `False`). Enable multiple sampling configurations.
- `prometheus_port` (`int`, optional, defaults to `0`). Separate Prometheus service port.
- `parallelism_config` (`ParallelismConfig | None`, optional, defaults to `None`). Mesh and parameter sharding.
- `jax_distributed_initialization_timeout` (`int`, optional, defaults to `300`). Distributed initialization timeout in seconds.
- `cache_config` (`InferenceCacheConfig | None`, optional, defaults to `None`). Cache layout and saved prefill loading.

</details>

### InferenceCacheConfig

Import from `lilax.serving`. Pydantic cache settings passed as `JetStreamConfig.cache_config`. Unknown fields are rejected.

<details>
<summary>Parameters</summary>

- `load_from_prefill_dir` (`bool`, optional, defaults to `False`). Read saved prefill cache instead of computing it.
- `prefill_cache_dir` (`str`, optional, defaults to `""`). Prefill cache directory.
- `stack_prefill_result_cache` (`bool`, optional, defaults to `False`). Stack cache entries across unscanned layers. Scanned layers retain their existing layout.
- `prefill_cache_axis_order` (`str`, optional, defaults to `"1,2,0,3"`). Prefill KV-cache axis order.
- `ar_cache_axis_order` (`str`, optional, defaults to `"1,2,0,3"`). Autoregressive KV-cache axis order.
- `compute_axis_order` (`str`, optional, defaults to `"0,1,2,3"`). Compute axis order.
- `reshape_q` (`bool`, optional, defaults to `False`). Reshape Q in attention.

</details>

## VLLMConfig

Import from `lilax.serving`. Pydantic settings for vLLM. Unknown fields are rejected.

<details>
<summary>Parameters</summary>

- `hbm_utilization_vllm` (`float`, optional, defaults to `0.72`). Target HBM utilization.
- `enable_dp_attention` (`bool`, optional, defaults to `False`). Enable the attn_dp mesh axis.
- `async_scheduling` (`bool`, optional, defaults to `False`). Enable asynchronous scheduling.
- `max_num_batched_tokens` (`int | None`, optional, defaults to `None`). Batch token cap.
- `max_num_seqs` (`int | None`, optional, defaults to `None`). Batch sequence cap.
- `vllm_hf_config_path` (`str`, optional, defaults to `""`). Passed as `hf_config_path` to select the HF configuration.
- `hf_overrides` (`dict[str, Any]`, optional, defaults to `{}`). Overrides for the engine's HF configuration. A field factory creates a new dictionary.
- `additional_config` (`dict[str, Any]`, optional, defaults to `{}`). Additional engine options. A field factory creates a new dictionary.
- `parallelism_config` (`ParallelismConfig | None`, optional, defaults to `None`). Mesh and parameter sharding.
- `jax_distributed_initialization_timeout` (`int`, optional, defaults to `300`). Initialization timeout in seconds.

</details>

## EvaluationServerConfig

Import from `lilax.serving`. Pydantic settings for the evaluation server. Unknown fields are rejected. Chat evaluation requires `enable_chat_api=True`.

<details>
<summary>Parameters</summary>

- `max_model_len` (`int`). Total sequence-length limit.
- `host` (`str`, optional, defaults to `"localhost"`). Listening hostname or address.
- `port` (`int`, optional, defaults to `8000`). Listening TCP port.
- `tensor_parallel_size` (`int`, optional, defaults to `4`). Tensor-parallel partitions.
- `expert_parallel_size` (`int`, optional, defaults to `1`). Expert-parallel partitions.
- `data_parallel_size` (`int`, optional, defaults to `1`). Data-parallel replicas.
- `max_num_batched_tokens` (`int | None`, optional, defaults to `None`). Batch token cap.
- `max_num_seqs` (`int | None`, optional, defaults to `None`). Batch sequence cap.
- `hbm_memory_utilization` (`float`, optional, defaults to `0.3`). Fraction of device HBM for the server.
- `concurrency` (`int | None`, optional, defaults to `None`). Concurrent evaluation requests. Unset uses host and accelerator capacity, capped by `max_num_seqs` when set.
- `enable_chat_api` (`bool`, optional, defaults to `False`). Expose chat completions.

</details>
