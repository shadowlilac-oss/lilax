# Decode with an existing runtime

Use configured decoding when an application already owns a compatible inference runtime. For supported Hub model loading, use [model generation](../inference/generation.md).

## Decode a configured checkpoint

Build the runtime-source configuration from a supported public model. Replace the parameter path with a matching checkpoint:

```python
from lilax.generation.decode import decode
from lilax import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_config(config)
source = model.runtime.public_config.model_copy(update={
	"load_parameters_path": "/checkpoints/qwen3-8b/0/items", "prompt": "Explain checkpoints.",
})
text = decode(source)
print(text)
```

The prompt, media paths, tokenizer, lengths, and checkpoint selection come from that configuration. `decode` creates a separate engine and loads the configured checkpoint, rather than the wrapper's live parameters. It returns decoded text and also prints its input and output. For stitched Omni checkpoints, use the [Omni runtime loader](omni-checkpoints.md).

See [runtime-source fields](../../reference/execution/runtime-source-fields.md) for the exact checkpoint, tokenizer, prompt, media, batch, and sampling keys.

## Run interleaved streams

`decode_multi(runtime)` uses five streams, with two initial prefill streams. It requires a compatible runtime whose global batch size is at least five.

This API does not expose stream-count or prompt keyword arguments. Prepare the compatible deployment runtime before calling it. The function returns stream IDs mapped to decoded text.

Start with the batch size from the public runtime configuration. Adjust it for your deployment if it does not yield at least five global slots or fit your model and devices.

```python
from lilax.generation.decode_multi import decode_multi

batch_per_device = source.per_device_batch_size
stream_source = source.model_copy(update={"per_device_batch_size": batch_per_device})
streams = decode_multi(stream_source)
print(streams)
```

This continues the checkpoint setup above. Select a batch size that yields at least five global slots and fits the model and devices.

## Build chat messages from configured input

```python
from lilax.serving.vllm import build_chat_messages
from lilax.runtime import Runtime

runtime = Runtime(source)
runtime.resolve_config()
messages = build_chat_messages(runtime.config)
print(messages)
```

This consumes configured prompt and media settings. It does not apply the tokenizer's chat template or load image bytes.

See [configured decoding reference](../../reference/components/configured-decoding.md) for setup boundaries and errors.

## Score a configured parameter checkpoint

The scoring wrapper takes the runtime-source configuration. It reuses `source` from the checkpoint setup above. Scoring initializes vLLM and registers the Lilax adapter before loading and scoring that checkpoint.

```python
from lilax.serving.vllm_score import score

scores = score(source)
```

The checkpoint must match this model's parameter layout. Without a parameter path the engine uses dummy loading; it does not load a wrapper's deferred Hub weights.
