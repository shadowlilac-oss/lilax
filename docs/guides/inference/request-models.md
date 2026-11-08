# Construct completion request records

Use typed request records when integrating with the benchmark API server's completion contract.

```python
from lilax.serving.server_models import ChatCompletionRequest, ChatMessage, CompletionRequest

completion = CompletionRequest(model="Qwen/Qwen3-8B", prompt="Explain checkpoints.", max_tokens=32)
chat = ChatCompletionRequest(
	model="Qwen/Qwen3-8B", messages=[ChatMessage(role="user", content="Explain checkpoints.")], max_tokens=32,
)
```

These objects describe requests. Construction does not load a model or start a service. Use [serving](serving.md) for service lifecycles.

For the benchmark server, send `completion.model_dump_json()` to `/v1/completions` and read `choices[0].text`. The [HTTP request example](../components/benchmark-server.md#request-a-completion) shows the separate client process.

See [request and response reference](../../reference/inference/request-models.md).
