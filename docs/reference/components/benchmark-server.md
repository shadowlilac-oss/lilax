# Benchmark generation and HTTP service

See [benchmark server guide](../../guides/components/benchmark-server.md).

`server.run()` starts HTTP on `0.0.0.0:8000` on process zero and enters the multihost processing loop on every process. Host and port are fixed by this API. It blocks until shutdown.

These constructors accept `model.runtime`, runtime-source configuration (`model.runtime.public_config`), or resolved configuration (`runtime.config`). See [configuration objects](../execution/runtime.md#configuration-objects). Root `lilax.MaxTextConfig` is not accepted directly.

## MaxTextGenerator

Import from `lilax.serving.generator`. Class.

```python
MaxTextGenerator(config)
```

Initializes a separate engine, loads parameters, builds a tokenizer, and allocates decode state.
Full training-state checkpoints are rejected. Changes JAX PRNG settings and reports system information.

<details>
<summary>Parameters</summary>

- `config` (`object`). Compatible `Runtime`, runtime-source configuration, or resolved configuration as described above.

</details>

### generate_batch

```python
generator.generate_batch(
	prompts, image_paths=None, max_tokens=None, logprobs=None, echo=False, stop=None,
	temperature=None, seed=None, top_k=None, top_p=None,
)
```

Chunks requests by engine batch size. Supplied prompt and image path counts must match, otherwise raises `ValueError`.

Returns `Completion` records.

<details>
<summary>Parameters</summary>

- `prompts` (`Sequence[str]`). The sequence of prompt strings.
- `image_paths` (`Sequence[str | None] | None`, optional, defaults to `None`). One image path per prompt. Omission uses one `None` per prompt.
- `max_tokens` (`int | None`, optional, defaults to `None`). Limit on completion tokens within configured capacity.
- `logprobs` (`int | None`, optional, defaults to `None`). Requested token log-probability count.
- `echo` (`bool`, optional, defaults to `False`). Flag for inclusion of prompt text and tokens in results.
- `stop` (`str | Sequence[str] | None`, optional, defaults to `None`). Stopping strings.
- `temperature` (`float | None`, optional, defaults to `None`). Sampling temperature.
- `seed` (`int | None`, optional, defaults to `None`). Request RNG seed override.
- `top_k` (`int | None`, optional, defaults to `None`). Top-k sampling limit.
- `top_p` (`float | None`, optional, defaults to `None`). Nucleus sampling probability.

</details>

### Completion

Import from `lilax.serving.generator`. Dataclass.

```python
Completion(
	index: int, text: str, tokens: list[int], logprobs: LogProbs | None,
	finish_reason: str = "stop", prompt_token_count: int = 0, completion_token_count: int = 0,
)
```

Stores a completion result. This record differs from `CompletionOutput` used by model batch generation.

<details>
<summary>Parameters</summary>

- `index` (`int`). Prompt index within its generated chunk.
- `text` (`str`). Decoded output.
- `tokens` (`list[int]`). Returned token IDs.
- `logprobs` (`LogProbs | None`). Requested token probabilities.
- `finish_reason` (`str`, optional, defaults to `"stop"`). Completion termination.
- `prompt_token_count` (`int`, optional, defaults to `0`). Prompt length.
- `completion_token_count` (`int`, optional, defaults to `0`). Completion length.

</details>

### LogProbs

Import from `lilax.serving.generator`. Dataclass.

```python
@dataclass
class LogProbs:
	tokens: list[int]
	token_logprobs: list[float]
	top_logprobs: list[None] | None = None
	text_offset: list[int] = field(default_factory=list)
```

Stores token probability data.

<details>
<summary>Parameters</summary>

- `tokens` (`list[int]`). Token IDs.
- `token_logprobs` (`list[float]`). Corresponding log probabilities.
- `top_logprobs` (`list[None] | None`, optional, defaults to `None`). Optional top-probability entries.
- `text_offset` (`list[int]`, optional, defaults to `[]`). Token text offsets. Each instance receives a fresh value.

</details>

## MaxTextServer

Import from `lilax.serving.server`. Class.

```python
MaxTextServer(config)
```

Constructs the generator and web application. Exposes `/v1/completions`, `/v1/chat/completions`, and `/`.
Requests that exceed the wait timeout receive HTTP 504. Generation errors receive HTTP 500.

Requests collected into one HTTP batch share `max_tokens`, `logprobs`, `echo`, `stop`, `temperature`, `seed`, `top_k`, and `top_p` from the first queued request. For a first chat request, `echo` is `False`, and the log-probability count is `None` when `logprobs=False`, otherwise `top_logprobs` or `1` when unset. The server does not group requests by these settings. Later requests' values do not control generation, so callers cannot rely on different per-request settings within a batch.

The request models accept `stream`, but this server ignores it and returns one completed JSON response. This differs from the separate [JetStream streaming gRPC service](../inference/serving.md#server-lifecycle).

This server loads its own configured weights. A pending Hub checkpoint in a separate model object is not sufficient setup for this constructor.

<details>
<summary>Parameters</summary>

- `config` (`object`). `Runtime`, runtime-source configuration, or resolved configuration accepted by `MaxTextGenerator`.

</details>

### run

```python
server.run()
```

Starts HTTP on `0.0.0.0:8000` on process zero and enters the inference loop on every process.
Host and port are fixed. Blocks until shutdown.

Returns `None`.

### main_loop

```python
server.main_loop()
```

Continuously processes and broadcasts queued batches across hosts. Blocks while the processing loop runs.

Returns no result during normal operation.

### health_check

```python
server.health_check()
```

Returns the status and message dictionary for the health endpoint.

### create_completion

```python
server.create_completion(request: CompletionRequest)
```

Asynchronously queues a typed completion request. Timeout and generation failures produce HTTP 504 and 500 respectively.

Returns a `CompletionResponse` when awaited.

<details>
<summary>Parameters</summary>

- `request` (`CompletionRequest`). The completion request from `lilax.serving.server_models`.

</details>

### create_chat_completion

```python
server.create_chat_completion(request: ChatCompletionRequest)
```

Asynchronously queues a typed chat request. Timeout and generation failures produce HTTP 504 and 500 respectively.

Returns a `ChatCompletionResponse` when awaited.

<details>
<summary>Parameters</summary>

- `request` (`ChatCompletionRequest`). The chat request from `lilax.serving.server_models`.

</details>
