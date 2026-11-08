# Lilax

Train and generate with supported language models through a Hugging Face-style Python API on JAX.

[Documentation home](docs/README.md) | [Quickstart](docs/getting-started/quickstart.md) | [API reference](docs/reference/README.md)

Prepare your accelerator with [environment setup](docs/getting-started/environment.md) before running the example.

```python
from lilax import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
inputs = tokenizer("Explain gradient accumulation.", return_tensors="np", return_token_type_ids=False)
tokens = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

Start with the [quickstart](docs/getting-started/quickstart.md), then choose a task in the [documentation](docs/README.md).

- [Load and configure models](docs/guides/models/loading.md).
- [Prepare datasets](docs/guides/data/datasets.md).
- [Fine-tune with SFT](docs/guides/training/sft.md), [LoRA](docs/guides/training/lora.md), or [preferences](docs/guides/training/dpo.md).
- [Generate text](docs/guides/inference/generation.md) and [serve a model](docs/guides/inference/serving.md).
- [Look up the public API](docs/reference/README.md).
