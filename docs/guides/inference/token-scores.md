# Score candidate next tokens

Use a vLLM engine to compare candidate tokens for each prompt.

Complete the [TPU-vLLM setup](../../getting-started/environment.md#tpu-vllm-workflows) before creating the engine.

```python
from lilax import AutoModelForCausalLM
from lilax.serving import create_vllm_llm
from lilax.serving.vllm_score import VllmScorer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
scorer = VllmScorer(create_vllm_llm(model))
prompts = ["The answer is"]
next_tokens = scorer.greedy_next_token(prompts)
logprobs = scorer.exact_candidate_logprobs(prompts, [[next_tokens[0]]])
print(logprobs)
```

Candidates are token IDs, not arbitrary strings. The example scores the greedy token so the candidate ID comes from the same tokenizer and model.

## Choose exact or top-k scoring

`candidate_logprobs(..., method="auto")` first uses top-k scores and obtains exact scores when candidates are missing. `"top_k"` raises an error for missing candidates. `"exact"` scores every supplied candidate through prompt log probabilities.

`candidate_probabilities` normalizes scores over the supplied candidates. The result describes that candidate set, not the full vocabulary.

See [token scoring reference](../../reference/inference/token-scores.md).
