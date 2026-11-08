# Load, generate, and save

Generate a continuation with Qwen3-8B and save its weights locally. Use an accelerator environment with enough memory for the model.

Complete [environment setup](environment.md) before running the code below.

## Load the model and tokenizer

```python
from lilax import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

Weight construction or restoration happens when an operation needs the weights.
The first Hub-weight use converts on CPU and stores a full additional weight copy. Plan disk space and shared multihost staging as described in [first-use conversion](../guides/models/loading.md#allow-space-for-first-use-conversion).

## Generate a continuation

```python
inputs = tokenizer("What is a learning rate?", return_tensors="np", return_token_type_ids=False)
sequences = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(sequences[0], skip_special_tokens=True))
```

The output contains the prompt and its continuation. Numerical choices throughout these examples illustrate the API; they are not tuning recommendations.

## Save and reload

Repeated native saves replace Lilax-owned weights under `_lilax_model` and update configuration files while preserving unrelated destination files. Choose a new path to retain an earlier model's weights.

```python
model.save_pretrained("./qwen-checkpoint")
tokenizer.save_pretrained("./qwen-checkpoint")
reloaded = AutoModelForCausalLM.from_pretrained("./qwen-checkpoint", local_files_only=True)
```

For portable Hugging Face weights, see [checkpoint export](../guides/training/checkpoints.md). Continue training with [SFT](../guides/training/sft.md).
