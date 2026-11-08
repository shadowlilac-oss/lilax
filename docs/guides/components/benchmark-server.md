# Generate batches for the benchmark API

Use the benchmark generator with a runtime that already specifies its checkpoint, tokenizer, prompt lengths, and deployment settings.

```python
from lilax import AutoConfig, AutoModelForCausalLM
from lilax.serving.generator import MaxTextGenerator

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_config(config)
source = model.runtime.public_config.model_copy(update={
	"load_parameters_path": "/checkpoints/qwen3-8b/0/items",
})
generator = MaxTextGenerator(source)
completions = generator.generate_batch(["Explain checkpoints."], max_tokens=32)
print(completions[0].text)
```

Replace the path with a parameter checkpoint matching this model. This constructs a separate engine and loads its configured checkpoint. It does not adopt another wrapper's live parameters. The result is a list of completion records with `text`, `tokens`, `logprobs`, and token counts. For stitched Omni checkpoints, use the [Omni runtime loader](omni-checkpoints.md) instead.

## Run the HTTP service

Construction loads another engine from the configured checkpoint. `run()` listens on `0.0.0.0:8000` on process zero and blocks until shutdown. Run this in the server process with the `source` configuration above.

```python
from lilax.serving.server import MaxTextServer

server = MaxTextServer(source)
server.run()
```

The server owns its generator, request queue, and inference loop.

See [runtime-source fields](../../reference/execution/runtime-source-fields.md) for the checkpoint, tokenizer, capacity, and decode settings used to construct its engine.

## Request a completion

Run this from a separate client while the server is running. Use the process-zero host address if the client is on another machine.

```python
import json
from urllib.request import Request, urlopen
from lilax.serving.server_models import CompletionRequest

payload = CompletionRequest(model="Qwen/Qwen3-8B", prompt="Explain checkpoints.", max_tokens=32)
request = Request(
	"http://localhost:8000/v1/completions", data=payload.model_dump_json().encode("utf-8"),
	headers={"Content-Type": "application/json"}, method="POST",
)
with urlopen(request) as response:
	completion = json.load(response)
print(completion["choices"][0]["text"])
```

The printed value is generated text. The response contains a `choices` list and a `usage` object with `prompt_tokens`, `completion_tokens`, and `total_tokens`. The request's `model` label does not load or switch models. See [typed requests](../inference/request-models.md).

See [benchmark server reference](../../reference/components/benchmark-server.md).
