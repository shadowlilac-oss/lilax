# Generate text and batch requests

## Generate token sequences

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("Explain checkpoints.", return_tensors="np", return_token_type_ids=False)
options = GenerationConfig(max_new_tokens=32, eos_token_id=tokenizer.eos_token_id)
sequences = model.generate(**inputs, generation_config=options)
print(tokenizer.decode(sequences[0], skip_special_tokens=True))
```

The returned array includes prompt tokens and completion tokens. For padded input, supply an attention mask with the same shape.
The mask selects the engine prompt, but the returned row retains the entire original `input_ids` row, including its padding, before appending completion IDs. Slice `sequences[:, inputs["input_ids"].shape[1]:]` for completion IDs. Shorter output rows receive trailing fill tokens using `model.config.pad_token_id`, otherwise the first configured generation EOS ID, otherwise `0`.

Set `max_new_tokens` explicitly to bound the completion. When it is `None`, generation uses the engine's remaining token budget. A negative limit or a limit above that budget raises `ValueError`. See [generation limits](../../reference/inference/generation.md) for the budget calculation.

`model.generate` computes this full budget before trimming the returned completion through the first configured EOS token, including that token. EOS alone does not bound computation. Set `max_new_tokens` to bound generation work even for short returned answers. The separate `generate_batch` path below passes EOS IDs to its batching engine to mark completed requests during generation.

## Sample completions

```python
import jax

rng = jax.random.PRNGKey(0)
rng, sample_rng = jax.random.split(rng)
sequences = model.generate(
	**inputs, max_new_tokens=32, do_sample=True, top_p=0.9, temperature=0.8, rng=sample_rng,
)
```

Call-level sampling arguments override the generation configuration. Split `rng` again for each new sample. With `rng=None`, each prompt uses key `0` again, so repeated calls reuse the sampling randomness. An explicit key is folded with each row's index.

## Batch prompts of different lengths

```python
from lilax.generation import ContinuousBatchingConfig

prompts = ["Explain checkpoints.", "What is gradient accumulation?"]
tokenized = [tokenizer.encode(prompt) for prompt in prompts]
outputs = model.generate_batch(
	tokenized, generation_config=options,
	continuous_batching_config=ContinuousBatchingConfig(enable_batch_prefill=True),
)
for output in outputs:
	print(tokenizer.decode(output.token_ids[output.prompt_length:], skip_special_tokens=True))
```

`generate_batch` accepts nonempty, unpadded, one-dimensional token sequences and returns an ordered list. Batching controls are separate from sampling controls.

Each record has this illustrative shape. Array contents and lengths depend on the prompt and generated completion, not fixed outputs:

```text
CompletionOutput(index='0', token_ids=array([...]), logprobs=array([...]), prompt_length=<int>)
```

`index` is the string form of the input position. `token_ids` contains prompt and completion tokens. Slice at `prompt_length` to obtain only the completion.

See [generation reference](../../reference/inference/generation.md) for result fields, defaults, and scoring helpers.
