# Serve a model

Choose JetStream for a server process, a vLLM engine for direct inference, or an evaluation endpoint for benchmark requests.

## Run JetStream

Start with the [TPU container setup](../../getting-started/environment.md#set-up-a-tpu-container), add the `serving` extra, and use a device layout supported by JetStream. The default slice settings target `v5e-16`. Set the slices and parallelism for your deployment before starting the server.

For an already materialized model, `serve`, `start_server`, `create_vllm_llm`, and entry into `ServingEndpoint` first save a full weight copy under `~/tmp/lilax/serving/`. Reserve disk space for this copy. `serve` cleans its directory when its wait exits. `ServingEndpoint` cleans its directory on context exit. `start_server` and `create_vllm_llm` retain temporary storage on the returned object without an explicit Lilax stop-time cleanup hook. All four paths clean their temporary directory if construction fails.

`serve` and `start_server` change the process-wide JAX default PRNG implementation to `unsafe_rbg` and set `TF_CPP_MIN_LOG_LEVEL="0"`. They do not restore either setting when the server stops. Use a separate process if other work needs different settings.

```python
from lilax import AutoModelForCausalLM
from lilax.serving import JetStreamConfig, serve

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
serve(model, server_config=JetStreamConfig())
```

`serve` blocks until termination. Configure prefill and generation slices for your deployment. `start_server` returns a server object when your application owns its wait and stop lifecycle.

JetStream listens on gRPC port `9000`. `JetStreamConfig` has no request-port setting. Once the server is ready, send a request from another process with the same pinned JetStream package installed. The following local client uses server-side tokenization and prints the streamed text for one sample. The server must have the model's tokenizer available. Keep `max_tokens` within the configured decode budget.

```python
import grpc
from jetstream.core.proto import jetstream_pb2, jetstream_pb2_grpc

request = jetstream_pb2.DecodeRequest(
	text_content=jetstream_pb2.DecodeRequest.TextContent(text="My dog is cute"),
	max_tokens=3, num_samples=1,
)
with grpc.insecure_channel("localhost:9000") as channel:
	grpc.channel_ready_future(channel).result()
	stub = jetstream_pb2_grpc.OrchestratorStub(channel)
	for response in stub.Decode(request):
		for sample in response.stream_content.samples:
			print(sample.text, end="", flush=True)
```

This follows the [official requester at Lilax's pinned JetStream revision](https://github.com/AI-Hypercomputer/JetStream/blob/29329e8e73820993f77cfc8efe34eb2a73f5de98/jetstream/tools/requester.py). The [protocol](https://github.com/AI-Hypercomputer/JetStream/blob/29329e8e73820993f77cfc8efe34eb2a73f5de98/jetstream/core/proto/jetstream.proto) defines `DecodeRequest` and streamed `DecodeResponse` messages. Each stream sample has `text` and `token_ids`. The insecure channel here is for local use, as in the official requester.

## Create a vLLM engine

Complete the [TPU-vLLM setup](../../getting-started/environment.md#tpu-vllm-workflows) before creating the engine.

`create_vllm_llm` also changes the process-wide JAX default PRNG to `unsafe_rbg`. It sets `SKIP_JAX_PRECOMPILE="1"`, `NEW_MODEL_DESIGN="1"`, and `TF_CPP_MIN_LOG_LEVEL="0"`, and adds the unsafe TPU RNG flag to `LIBTPU_INIT_ARGS`. These changes persist after the call. See the [serving reference](../../reference/inference/serving.md#server-lifecycle) for the exact settings.

```python
from lilax import AutoModelForCausalLM
from lilax.serving import VLLMConfig, create_vllm_llm
from vllm import SamplingParams

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
llm = create_vllm_llm(model, serving_config=VLLMConfig())
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=16))
print(outputs[0].outputs[0].text)
```

The returned object is a vLLM `LLM`. `generate` returns a list of request outputs, and `outputs[0].outputs[0].text` contains the first generated completion. The token limit above is an example request budget. See [vLLM offline inference](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference). Model registration and serving checkpoint preparation happen during construction.

## Own an evaluation endpoint

Use the `evaluation` extra and the [TPU-vLLM setup](../../getting-started/environment.md#tpu-vllm-workflows) for this endpoint. Entry starts the service and may write the temporary weight copy described above. Pass the model loaded above. To use a tokenizer already loaded by your application, pass it as `processing_class`.

```python
from lilax.serving import EvaluationServerConfig, ServingEndpoint

with ServingEndpoint(model, config=EvaluationServerConfig(max_model_len=2048, enable_chat_api=True)) as endpoint:
	print(endpoint.base_url)
	print(endpoint.concurrency)
```

The context starts and stops the evaluation service. Accessing `base_url` or `concurrency` outside an active context raises `RuntimeError`.

Continue with [evaluation](evaluation.md). See [serving reference](../../reference/inference/serving.md) for cache and capacity controls.
