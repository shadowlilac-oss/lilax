# Completion request and response models

Import from `lilax.serving.server_models`. See [request guide](../../guides/inference/request-models.md).

## Requests

The benchmark HTTP server accepts but ignores `stream`, including `True`, and returns one completed JSON response. Requests in the same batch inherit generation settings from the first queued request. See the [exact shared fields and chat log-probability mapping](../components/benchmark-server.md#maxtextserver). These restrictions concern the benchmark server, not the separate JetStream service.

These are Pydantic field references, not handwritten constructor signatures. Shared validation and serialization methods follow [BaseModel](https://docs.pydantic.dev/latest/api/base_model/).

### SamplingParams

Import from `lilax.serving.server_models`. Pydantic model with shared sampling fields inherited by both request models.

<details>
<summary>Parameters</summary>

- `max_tokens` (`Optional[int]`, optional, defaults to `None`). Completion token limit.
- `temperature` (`Optional[float]`, optional, defaults to `None`). Sampling temperature.
- `top_p` (`Optional[float]`, optional, defaults to `None`). Nucleus probability.
- `top_k` (`Optional[int]`, optional, defaults to `None`). Top-k cutoff.
- `stream` (`Optional[bool]`, optional, defaults to `False`). Streaming request flag, ignored by the benchmark server.
- `stop` (`Optional[Union[str, List[str]]]`, optional, defaults to `None`). Stop text.
- `seed` (`Optional[int]`, optional, defaults to `None`). Sampling seed.

</details>

### CompletionRequest

Import from `lilax.serving.server_models`. Pydantic text-completion request inheriting [SamplingParams](#samplingparams). Negative `logprobs` raises `ValueError`.

<details>
<summary>Parameters</summary>

- `model` (`str`). Requested model name.
- `prompt` (`Union[str, List[str], List[int], List[List[int]]]`). Text, text batch, token IDs, or token-sequence batch.
- `echo` (`Optional[bool]`, optional, defaults to `False`). Request prompt echo.
- `logprobs` (`Optional[int]`, optional, defaults to `None`). Nonnegative log-probability count.

</details>

### ChatMessage

Import from `lilax.serving.server_models`. Pydantic model for one chat message with optional reasoning text.

<details>
<summary>Parameters</summary>

- `role` (`str`). Message role.
- `content` (`str`). Message text.
- `reasoning_content` (`Optional[str]`, optional, defaults to `None`). Reasoning text.

</details>

### ChatCompletionRequest

Import from `lilax.serving.server_models`. Pydantic chat-completion request inheriting [SamplingParams](#samplingparams).

<details>
<summary>Parameters</summary>

- `model` (`str`). Requested model name.
- `messages` (`List[ChatMessage]`). Conversation messages.
- `logprobs` (`Optional[bool]`, optional, defaults to `False`). Request token log probabilities.
- `top_logprobs` (`Optional[int]`, optional, defaults to `None`). Top log-probability count.
- `thinking_mode` (`Optional[str]`, optional, defaults to `"thinking"`). Thinking-mode selection.

</details>

## Responses

### Usage

Import from `lilax.serving.server_models`. Pydantic model for token usage counters.

<details>
<summary>Parameters</summary>

- `prompt_tokens` (`int`). Prompt token count.
- `completion_tokens` (`int`). Generated token count.
- `total_tokens` (`int`). Combined token count.

</details>

### LogProbsPayload

Import from `lilax.serving.server_models`. Pydantic model for per-token text, probabilities, and offsets.

<details>
<summary>Parameters</summary>

- `tokens` (`List[str]`). Token text.
- `token_logprobs` (`List[Optional[float]]`). Corresponding log probabilities.
- `top_logprobs` (`Optional[List[Optional[Dict[str, float]]]]`, optional, defaults to `None`). Top alternatives per token.
- `text_offset` (`List[int]`). Token offsets in text.

</details>

### CompletionChoice

Import from `lilax.serving.server_models`. Pydantic model for one text-completion choice.

<details>
<summary>Parameters</summary>

- `text` (`str`). Generated text.
- `index` (`int`). Choice index.
- `logprobs` (`Optional[LogProbsPayload]`, optional, defaults to `None`). Token probabilities.
- `finish_reason` (`str`, optional, defaults to `"stop"`). Completion termination reason.

</details>

### ChatCompletionChoice

Import from `lilax.serving.server_models`. Pydantic model for one chat-completion choice.

<details>
<summary>Parameters</summary>

- `index` (`int`). Choice index.
- `message` (`ChatMessage`). Generated message.
- `finish_reason` (`str`, optional, defaults to `"stop"`). Completion termination reason.
- `logprobs` (`Optional[LogProbsPayload]`, optional, defaults to `None`). Token probabilities.

</details>

### BaseCompletionResponse

Import from `lilax.serving.server_models`. Generic Pydantic response schema `BaseCompletionResponse[ChoiceType]`. Concrete types specialize choices and supply ID and object defaults.

<details>
<summary>Parameters</summary>

- `id` (`str`). Response ID.
- `object` (`str`). Response object type.
- `created` (`int`, optional, defaults to `int(time.time())`). Creation time evaluated by a field factory for each response.
- `model` (`str`). Model name.
- `choices` (`List[ChoiceType]`). Typed completion choices.
- `usage` (`Usage`). Token counters.

</details>

### CompletionResponse

Import from `lilax.serving.server_models`. Pydantic text response specializing `BaseCompletionResponse[CompletionChoice]`. Construction creates a UUID-derived ID and a current Unix timestamp when omitted.

<details>
<summary>Parameters</summary>

- `id` (`str`, optional, defaults to `f"cmpl-{uuid.uuid4().hex}"`). Response ID evaluated by a field factory for each response.
- `object` (`str`, optional, defaults to `"text_completion"`). Object type.
- `created` (`int`, optional, defaults to `int(time.time())`). Creation time evaluated by a field factory for each response.
- `model` (`str`). Model name.
- `choices` (`List[CompletionChoice]`). Completion choices.
- `usage` (`Usage`). Token counters.

</details>

### ChatCompletionResponse

Import from `lilax.serving.server_models`. Pydantic chat response specializing `BaseCompletionResponse[ChatCompletionChoice]`. Construction creates a UUID-derived ID and a current Unix timestamp when omitted.

<details>
<summary>Parameters</summary>

- `id` (`str`, optional, defaults to `f"chatcmpl-{uuid.uuid4().hex}"`). Response ID evaluated by a field factory for each response.
- `object` (`str`, optional, defaults to `"chat.completion"`). Object type.
- `created` (`int`, optional, defaults to `int(time.time())`). Creation time evaluated by a field factory for each response.
- `model` (`str`). Model name.
- `choices` (`List[ChatCompletionChoice]`). Chat choices.
- `usage` (`Usage`). Token counters.

</details>
