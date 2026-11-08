# Generation

See [generation guide](../../guides/inference/generation.md).

## GenerationConfig

Import from `lilax` or `lilax.generation`.

Pydantic configuration for token selection. Extra fields are forbidden. Greedy selection applies when `do_sample=False`.
With `do_sample=True`, `top_k <= 0`, and `top_p < 0`, the algorithm is weighted sampling without top-k or nucleus filtering.

<details>
<summary>Parameters</summary>

- `max_new_tokens` (`int | None`, optional, defaults to `None`). Completion limit. `None` uses the remaining engine budget for `generate`. Zero skips generation.
- `do_sample` (`bool`, optional, defaults to `False`). Sample instead of greedy selection.
- `top_k` (`int`, optional, defaults to `0`). Top-k sampling when positive.
- `top_p` (`float`, optional, defaults to `-1.0`). With sampling enabled, nucleus sampling applies for `top_k <= 0` and `top_p >= 0`. Positive `top_k` and `top_p` combine both filters. Otherwise positive `top_k` selects top-k alone.
- `temperature` (`float`, optional, defaults to `1.0`). Sampling temperature.
- `eos_token_id` (`int | list[int] | None`, optional, defaults to `None`). EOS IDs used to trim `generate` completions after computation.

</details>

### sampling_options

```python
config.sampling_options()
```

Returns a dictionary with `algorithm`, `topk`, `nucleus_topp`, and `temperature` for the selected sampling mode.

## model.generate

Bound method of `MaxTextModel`, imported from `lilax`.

```python
model.generate(
	input_ids, generation_config=None, *, attention_mask=None, max_new_tokens=None, do_sample=None,
	top_k=None, top_p=None, temperature=None, rng=None, position_ids=None, mrope_deltas=None,
	images=None, image_masks=None, videos=None, video_masks=None, video_grid_thw=None,
	audio_values=None, audio_masks=None,
)
```

Returns a JAX array of prompt plus completion IDs. A one-dimensional input becomes a batch of one. Non-`None` call-level options override the selected configuration. Passing `None` keeps the configuration's value. Shorter outputs are padded to the maximum returned width.

The attention mask selects tokens sent to the engine. Each returned row starts with the unmodified input row, including its pad IDs, then appends completion IDs. Completion slicing starts at `input_ids.shape[1]` for batched input. Trailing batch padding uses `model.config.pad_token_id`, otherwise the first EOS ID from the selected generation configuration, otherwise `0`.

The engine's remaining budget is `max_target_length - max_prefill_predict_length + 1`. The prefill length is increased when the longest unmasked prompt exceeds the configured prefill length. When the selected configuration leaves `max_new_tokens=None`, generation uses that budget. Negative values and values above the budget raise `ValueError`.

`generate` computes the full selected token budget before trimming each returned completion through its first configured EOS token, including that token. EOS does not stop this computation early. An explicit `max_new_tokens` bounds generation work even when the returned completion is shorter.

With `rng=None`, each prompt starts with `jax.random.PRNGKey(0)`. Repeated sampling calls reuse this key. An explicit `rng` is folded with each row's index before generation.

Invalid input rank, an empty batch, or an attention-mask shape mismatch raise `ValueError`, including with zero new tokens. With `max_new_tokens=0`, the method returns the input as a JAX array after promoting a one-dimensional input to a batch. It bypasses unmasked-prompt checks, context-length checks, model materialization, and engine creation. For nonzero generation, empty unmasked prompts or prompts beyond the model context length raise `ValueError`.

<details>
<summary>Parameters</summary>

- `input_ids` (`numpy.typing.ArrayLike`). Integer IDs for one sequence or a batch of sequences.
- `generation_config` (`GenerationConfig | None`, optional, defaults to `None`). Uses `model.generation_config` when omitted.
- `attention_mask` (`numpy.typing.ArrayLike | None`, optional, defaults to `None`). Nonzero entries select engine input tokens. Omission selects all input tokens.
- `max_new_tokens` (`int | None`, optional, defaults to `None`). Overrides the completion limit when supplied.
- `do_sample` (`bool | None`, optional, defaults to `None`). Overrides greedy or sampled selection when supplied.
- `top_k` (`int | None`, optional, defaults to `None`). Overrides the top-k filter when supplied.
- `top_p` (`float | None`, optional, defaults to `None`). Overrides the nucleus filter when supplied.
- `temperature` (`float | None`, optional, defaults to `None`). Overrides sampling temperature when supplied.
- `rng` (`jax.Array | None`, optional, defaults to `None`). JAX PRNG key for row-specific randomness.
- `position_ids` (`numpy.typing.ArrayLike | None`, optional, defaults to `None`). Per-row positions selected by the attention mask and padded for prefill.
- `mrope_deltas` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Per-row MRoPE deltas. Compatible positions can supply derived deltas when omitted.
- `images` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Model-compatible image inputs.
- `image_masks` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Associated image masks.
- `videos` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Model-compatible video inputs.
- `video_masks` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Associated video masks.
- `video_grid_thw` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Video grid metadata.
- `audio_values` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Model-compatible audio inputs.
- `audio_masks` (`jax.Array | np.ndarray | None`, optional, defaults to `None`). Associated audio masks.

</details>

## model.generate_batch

Bound method of `MaxTextModel`, imported from `lilax`.

```python
model.generate_batch(inputs, generation_config=None, *, continuous_batching_config=None, rng=None)
```

Returns an ordered list of [CompletionOutput](#completionoutput) records. Empty input returns an empty list. Each token sequence must be one-dimensional, nonempty, and unpadded.

This method passes EOS IDs to the offline batching engine, which marks a request complete when it emits EOS. It does not use `generate`'s full-budget-then-trim path.

`CompletionOutput` has `index`, `token_ids`, `logprobs`, and `prompt_length`. `token_ids` contains prompt and generated tokens. Slice at `prompt_length` for the completion.

<details>
<summary>Parameters</summary>

- `inputs` (`list[numpy.typing.ArrayLike]`). Unpadded token sequences in output order.
- `generation_config` (`GenerationConfig | None`, optional, defaults to `None`). Uses `model.generation_config` when omitted.
- `continuous_batching_config` (`ContinuousBatchingConfig | None`, optional, defaults to `None`). Uses a new default scheduling configuration when omitted.
- `rng` (`jax.Array | None`, optional, defaults to `None`). JAX PRNG key for the offline engine.

</details>

## CompletionOutput

```python
from lilax.generation import CompletionOutput

CompletionOutput(index: str, token_ids: np.ndarray, logprobs: np.ndarray, prompt_length: int)
```

Dataclass returned by `model.generate_batch`. All four fields are required. The index is a string, including when it represents the input's numeric position.

`token_ids` contains prompt and generated tokens. Slice at `prompt_length` for the completion. The engine concatenates prompt and completion log probabilities. Zero-token generation returns the prompt and an empty `logprobs` array.

See the [existing batch-generation example](../../guides/inference/generation.md#batch-prompts-of-different-lengths).

<details>
<summary>Parameters</summary>

- `index` (`str`). Input identifier. Public batch generation uses the input position converted to a string.
- `token_ids` (`np.ndarray`). Prompt followed by generated token IDs.
- `logprobs` (`np.ndarray`). Prompt and generated-token log probabilities, or an empty array when no tokens are generated.
- `prompt_length` (`int`). Number of prompt tokens before the completion.

</details>

## ContinuousBatchingConfig

Import from `lilax.generation`. This Pydantic model controls scheduling separately from `GenerationConfig` and rejects extra fields.

<details>
<summary>Parameters</summary>

- `enable_batch_prefill` (`bool`, optional, defaults to `False`). Batches prefill requests. Requests in the largest prefill bucket still use single-request prefill.
- `min_decode_steps` (`int`, optional, defaults to `10`). Decode steps submitted per iteration before the final step waits for device readiness.
- `prefill_lengths` (`list[int] | Literal["auto"]`, optional, defaults to `"auto"`). Prompt-padding buckets. Auto uses powers of two starting at 64 strictly below the effective prefill limit, then appends that limit.
- `batch_prefill_max_batch_size` (`int`, optional, defaults to `16`). Maximum prefill batch size with batch prefill enabled.
- `debug` (`bool`, optional, defaults to `False`). Log extra decode-state initialization and detokenization timings.

</details>

## Token scoring

Import these helpers from `lilax.generation`:

### log_prob_of_chosen_token

```python
log_prob_of_chosen_token(logits, chosen_index)
```

Returns selected-token log probabilities along the vocabulary axis.

For an existing forward-pass `logits` array and matching integer `chosen_ids` array, scoring is `log_prob_of_chosen_token(logits, chosen_ids)`. The index array has the logits shape without the vocabulary dimension.

<details>
<summary>Parameters</summary>

- `logits` (`jax.Array | np.ndarray`). Logits with vocabulary on the last axis.
- `chosen_index` (`jax.Array | np.ndarray`). Integer IDs with the logits shape excluding vocabulary.

</details>

### prompt_logprobs_from_prefill

```python
prompt_logprobs_from_prefill(logits, input_tokens, true_length)
```

Returns shifted prompt log probabilities. First tokens and padding receive NaN.

<details>
<summary>Parameters</summary>

- `logits` (`jax.Array | np.ndarray`). Shape `(batch, sequence, vocabulary)`.
- `input_tokens` (`jax.Array | np.ndarray`). Integer IDs with shape `(batch, sequence)`.
- `true_length` (`int | jax.Array | np.ndarray`). Unpadded prompt length, shared or per row.

</details>

### prompt_logprobs_from_packed_prefill

```python
prompt_logprobs_from_packed_prefill(logits, input_tokens, decoder_positions, decoder_segment_ids, true_lengths)
```

Returns shifted prompt log probabilities using segment-specific lengths. First tokens and padding receive NaN.

<details>
<summary>Parameters</summary>

- `logits` (`jax.Array | np.ndarray`). Shape `(batch, sequence, vocabulary)`.
- `input_tokens` (`jax.Array | np.ndarray`). Integer IDs with shape `(batch, sequence)`.
- `decoder_positions` (`jax.Array | np.ndarray`). Integer positions within each segment.
- `decoder_segment_ids` (`jax.Array | np.ndarray`). Integer segment IDs selecting the corresponding length.
- `true_lengths` (`jax.Array | np.ndarray`). Length indexed by segment ID.

</details>

### sampling

```python
sampling(logits, rng, algorithm, topk=0, nucleus_topp=0, temperature=1.0)
```

Returns selected token indices. Unknown algorithms raise `ValueError`. Top-k requires a positive cutoff. Composite sampling also requires `0 < nucleus_topp <= 1`. Nucleus sampling rejects a negative threshold.

<details>
<summary>Parameters</summary>

- `logits` (`jax.Array | np.ndarray`). Token logits with vocabulary on the last axis.
- `rng` (`jax.Array`). JAX PRNG key, unused by greedy selection.
- `algorithm` (`str`). `greedy`, `weighted`, `nucleus`, `topk`, or `composite`.
- `topk` (`int`, optional, defaults to `0`). Cutoff for top-k and composite selection.
- `nucleus_topp` (`int | float`, optional, defaults to `0`). Probability threshold for nucleus and composite selection.
- `temperature` (`int | float`, optional, defaults to `1.0`). Sampling temperature.

</details>

## Full-forward decoding example

Import `decode_next_token` from `lilax.generation.examples`.

```python
decode_next_token(model, tokenizer, prompt)
```

Uses a loaded model and its runtime settings to prepare a padded full-forward batch, gather logits, and select one token.

Returns a dictionary with `input_ids`, `ids`, `decoder_segment_ids`, `decoder_positions`, `logits`, `token_id`, and `text`.

The prompt must fit the runtime target length. Selection uses the demonstration's fixed position near the end of the padded row, not the unpadded prompt boundary. Use `model.generate` for ordinary next-token generation. The tokenizer must expose `encode` and `decode`.

<details>
<summary>Parameters</summary>

- `model` (`MaxTextModel`). Loaded model supplying forward computation and runtime settings.
- `tokenizer` (`PreTrainedTokenizerBase`). Tokenizer with `encode` and `decode` that converts the prompt and selected token.
- `prompt` (`str`). Text that fits the runtime target length.

</details>
