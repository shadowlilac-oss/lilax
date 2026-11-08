# vLLM token scores

Import `VllmScorer` from `lilax.serving.vllm_score`. See [scoring guide](../../guides/inference/token-scores.md).

The signatures use source aliases `TokenIds = Sequence[int]` and `PromptLike = str | TokenIds`.

Prompt and candidate lists must have equal lengths. Candidate IDs must be valid for the engine tokenizer. Invalid methods, invalid top-k values, and mismatched input lengths raise `ValueError`. Missing vLLM log probabilities or candidates absent under `method="top_k"` raise `RuntimeError`. Scoring methods execute vLLM generation requests.

## VllmScorer

Import from `lilax.serving.vllm_score`. Class with the explicit constructor below.

```python
VllmScorer(llm: LLM, default_top_k: int = 20)
```

Wraps an existing vLLM engine. The default is clamped to the engine log-probability limit. Values below one raise `ValueError`.

<details>
<summary>Parameters</summary>

- `llm` (`vllm.LLM`). Existing engine with tokenizer and configured log-probability capacity.
- `default_top_k` (`int`, optional, defaults to `20`). Default top-k log-probability count.

</details>

### to_token_ids

```python
scorer.to_token_ids(prompt: PromptLike, add_special_tokens: bool = False) -> list[int]
```

Returns IDs for text or an existing token sequence.

<details>
<summary>Parameters</summary>

- `prompt` (`PromptLike`). Text or token IDs.
- `add_special_tokens` (`bool`, optional, defaults to `False`). Include tokenizer special tokens for text.

</details>

### next_token_logprobs

```python
scorer.next_token_logprobs(prompts: Sequence[PromptLike], top_k: int | None = None,
	add_special_tokens: bool = False) -> list[dict[int, float]]
```

Returns one token-ID-to-log-probability dictionary per prompt. The requested top-k must be positive and fit the engine limit.

<details>
<summary>Parameters</summary>

- `prompts` (`Sequence[PromptLike]`). Text or token-ID prompts.
- `top_k` (`int | None`, optional, defaults to `None`). Uses the scorer's default when unset.
- `add_special_tokens` (`bool`, optional, defaults to `False`). Include special tokens for text.

</details>

### greedy_next_token

```python
scorer.greedy_next_token(prompts: Sequence[PromptLike], add_special_tokens: bool = False) -> list[int]
```

Returns one greedy token ID per prompt.

<details>
<summary>Parameters</summary>

- `prompts` (`Sequence[PromptLike]`). Text or token-ID prompts.
- `add_special_tokens` (`bool`, optional, defaults to `False`). Include special tokens for text.

</details>

### exact_candidate_logprobs

```python
scorer.exact_candidate_logprobs(prompts: Sequence[PromptLike],
	candidate_token_ids: Sequence[TokenIds], add_special_tokens: bool = False) -> list[np.ndarray]
```

Returns score arrays in candidate order using prompt log probabilities for each appended candidate.

<details>
<summary>Parameters</summary>

- `prompts` (`Sequence[PromptLike]`). Text or token-ID prompts.
- `candidate_token_ids` (`Sequence[TokenIds]`). Valid vocabulary IDs for each prompt.
- `add_special_tokens` (`bool`, optional, defaults to `False`). Include special tokens for text.

</details>

### candidate_logprobs

```python
scorer.candidate_logprobs(prompts: Sequence[PromptLike], candidate_token_ids: Sequence[TokenIds],
	method: str = "auto", top_k: int | None = None, add_special_tokens: bool = False, *,
	top_k_logprobs: Sequence[dict[int, float]] | None = None) -> list[np.ndarray]
```

Returns candidate log-probability arrays. `auto` uses top-k scores and recomputes missing candidates through exact scoring. `top_k` raises when candidates are missing. `exact` uses prompt log probabilities directly.

<details>
<summary>Parameters</summary>

- `prompts` (`Sequence[PromptLike]`). Text or token-ID prompts.
- `candidate_token_ids` (`Sequence[TokenIds]`). Candidate IDs in desired score order per prompt.
- `method` (`str`, optional, defaults to `"auto"`). `auto`, `top_k`, or `exact`.
- `top_k` (`int | None`, optional, defaults to `None`). Top-k request count. Unset uses the scorer's default.
- `add_special_tokens` (`bool`, optional, defaults to `False`). Include special tokens for text.
- `top_k_logprobs` (`Sequence[dict[int, float]] | None`, optional, defaults to `None`). Reuse supplied distributions instead of requesting top-k scores. Must match the prompt count.

</details>

### candidate_probabilities

```python
scorer.candidate_probabilities(prompts: Sequence[PromptLike],
	candidate_token_ids: Sequence[TokenIds], **kwargs: Any) -> list[np.ndarray]
```

Returns normalized probabilities over the selected candidates, in candidate order.

<details>
<summary>Parameters</summary>

- `prompts` (`Sequence[PromptLike]`). Text or token-ID prompts.
- `candidate_token_ids` (`Sequence[TokenIds]`). Candidate IDs for each prompt.
- `**kwargs` (`Any`). Variadic keyword options accepted by [candidate_logprobs](#candidate_logprobs).

</details>
