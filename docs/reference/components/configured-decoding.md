# Configured decoding and scoring

See [runtime decoding guide](../../guides/components/configured-decoding.md).

Runtime-source configuration (`model.runtime.public_config`) and resolved configuration (`runtime.config`) are distinct from the root `lilax.MaxTextConfig`. See [configuration objects](../execution/runtime.md#configuration-objects).

## decode and decode_multi

### decode

Import from `lilax.generation.decode`.

```python
decode(config)
```

Loads parameters through its own engine. It does not reuse the weights of a `MaxTextModel` supplied elsewhere.
Root `MaxTextConfig` is not the execution configuration contract.

Rejects full training-state checkpoints, overlong prompts, and a global batch size smaller than its one stream. It changes JAX PRNG/partitioner configuration, configures logging, prints system information, and can run the configured profiler.

With `quantization="fp8"`, raises `AssertionError` with `"fp8 on NVIDIA GPUs is not supported in decode.py yet"`. With `quantization="nanoo_fp8"`, raises `AssertionError` with `"NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"`. These checks use the recipe string rather than detected device hardware.

Returns a decoded string.

<details>
<summary>Parameters</summary>

- `config` (`object`). Compatible `Runtime`, resolved configuration, or runtime-source configuration as described above.

</details>

### decode_multi

Import from `lilax.generation.decode_multi`.

```python
decode_multi(config)
```

Rejects full training states. Uses five streams and two initial prefill streams, requiring global batch size at least five.
Streams share the configured prompt. Releases pages when a stream reaches its length limit.

Returns a dictionary keyed by stream number.

The model factory's deferred checkpoint source does not automatically configure a separate decoding engine. A caller cannot assume that passing `model.runtime` to another engine loads the wrapper's pending Hub weights. Use a runtime with its own complete checkpoint selection, or `model.generate` for that model's weights.

<details>
<summary>Parameters</summary>

- `config` (`object`). `Runtime`, resolved configuration, or runtime-source configuration accepted by `decode`.

</details>

## build_chat_messages

Import from `lilax.serving.vllm`.

```python
build_chat_messages(config) -> list[dict[str, Any]]
```

Returns message dictionaries from resolved prompt settings. A nonempty system prompt becomes the first message.
Configured images produce image placeholders before user text. Root model configuration does not replace prompt and media execution settings.

<details>
<summary>Parameters</summary>

- `config` (`Any`). Object with `system_prompt`, `prompt`, `use_multimodal`, and `image_path` attributes, such as resolved configuration (`runtime.config`).

</details>

## score

Import from `lilax.serving.vllm_score`.

```python
score(config, **options)
```

Resolves a supported runtime-source configuration, initializes inference environment settings, and loads a tokenizer and vLLM engine.
Returns request-result dictionaries.

Top-k and repeat must be positive. Methods are `auto`, `top_k`, and `exact`. Output files use JSON Lines. For an engine already created from a loaded model, use [VllmScorer](../inference/token-scores.md).

Pass runtime-source configuration (`model.runtime.public_config`). The runtime itself is not accepted. Checkpoint loading follows `load_parameters_path`. An empty path selects dummy loading in the separate vLLM engine.

<details>
<summary>Parameters</summary>

- `config` (`object`). Supported runtime-source configuration (`model.runtime.public_config`).
- `**options` (`dict[str, Any]`). Scoring options listed below.
- `candidates` (`Any`, optional, defaults to `""`). Candidate values to score.
- `prompts_file` (`str`, optional, defaults to `""`). A JSONL prompt batch instead of the configured prompt.
- `output_file` (`str`, optional, defaults to `""`). Result JSONL output.
- `top_k` (`int`, optional, defaults to `20`). Limit on next-token log probabilities.
- `method` (`str`, optional, defaults to `"auto"`). Candidate scoring method, `auto`, `top_k`, or `exact`.
- `add_special_tokens` (`bool`, optional, defaults to `False`). Flag for special tokens during prompt tokenization.
- `enable_prefix_caching` (`bool`, optional, defaults to `False`). Flag for engine prefix caching.
- `repeat` (`int`, optional, defaults to `1`). Scoring repetition count. Only the final results are returned.
- `enable_thinking` (`bool`, optional, defaults to `False`). Flag for chat-template thinking mode.

</details>
