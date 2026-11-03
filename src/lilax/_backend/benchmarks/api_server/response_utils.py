"""Response formatting helpers for the benchmark API server."""

import bisect
import logging
import math

from openai_harmony import Role

from lilax._backend.benchmarks.api_server import server_utils
from lilax._backend.benchmarks.api_server.encoding import encoding_dsv32
from lilax._backend.input_pipeline import encoding_dsv4
from lilax.serving.server_models import (
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    CompletionChoice, CompletionResponse, LogProbsPayload, Usage,
)

logger = logging.getLogger(__name__)


def decode_token_id(token_id, llm):
  return llm.tokenizer.decode([int(token_id)])


def finite_or_none(value):
  if value is None:
    return None
  value = float(value)
  return value if math.isfinite(value) else None


def to_openai_logprobs(lp_obj, llm, want_top=True):
  if lp_obj is None:
    return None
  token_strings = [decode_token_id(token_id, llm) for token_id in lp_obj.tokens]
  token_logprobs = [finite_or_none(value) for value in lp_obj.token_logprobs]
  text_offset = list(lp_obj.text_offset)
  min_len = min(len(token_strings), len(token_logprobs), len(text_offset))
  token_strings = token_strings[:min_len]
  token_logprobs = token_logprobs[:min_len]
  text_offset = text_offset[:min_len]
  top_logprobs = None
  if want_top:
    top_logprobs = [({token: value} if value is not None else None) for token, value in zip(token_strings, token_logprobs)]
  return LogProbsPayload(
      tokens=token_strings, token_logprobs=token_logprobs, top_logprobs=top_logprobs, text_offset=text_offset,
  )


def count_tokens(text, llm):
  try:
    ids = llm.tokenizer.tokenizer.encode(text, add_special_tokens=False)
    return len(ids)
  except (ValueError, TypeError) as error:
    logger.warning("Could not count tokens for string '%s...': %s", text[:50], error)
    return 0


def apply_stops_to_text_and_logprobs(text, logprobs_payload, stop):
  if not stop:
    return text, logprobs_payload, None
  stops = [stop] if isinstance(stop, str) else list(stop)
  first_stop_index = -1
  for sequence in stops:
    if not sequence:
      continue
    index = text.find(sequence)
    if index != -1:
      first_stop_index = index if first_stop_index == -1 else min(first_stop_index, index)
  if first_stop_index == -1:
    return text, logprobs_payload, None
  new_text = text[:first_stop_index]
  if logprobs_payload is not None:
    truncate_at_index = bisect.bisect_left(logprobs_payload.text_offset, first_stop_index)
    new_logprobs = LogProbsPayload(
        tokens=logprobs_payload.tokens[:truncate_at_index],
        token_logprobs=logprobs_payload.token_logprobs[:truncate_at_index],
        top_logprobs=logprobs_payload.top_logprobs[:truncate_at_index]
        if logprobs_payload.top_logprobs is not None else None,
        text_offset=logprobs_payload.text_offset[:truncate_at_index],
    )
    return new_text, new_logprobs, "stop"
  return new_text, logprobs_payload, "stop"


def build_chat_completion_response(request, completion_result, llm, harmony_enc):
  text_out = completion_result.text
  reasoning_out = None
  if "gpt-oss" in request.model and harmony_enc:
    try:
      parsed_messages = harmony_enc.parse_messages_from_completion_tokens(completion_result.tokens, role=Role.ASSISTANT)
      user_visible = "".join(part.text for message in parsed_messages if message.channel == "final" for part in message.content)
      if user_visible:
        text_out = user_visible
      else:
        logger.warning("Harmony parsing for gpt-oss did not yield content in the 'final' channel. Falling back to raw text.")
    except (ValueError, IndexError) as error:
      logger.error("Harmony parsing failed for gpt-oss: %s. Falling back to raw text.", error, exc_info=True)

  thinking_mode = getattr(request, "thinking_mode", "thinking")
  if server_utils.is_dsv32_encoding_enabled(request.model):
    try:
      parsed = encoding_dsv32.parse_message_from_completion_text(text_out, thinking_mode=thinking_mode)
      text_out = parsed.get("content", text_out)
      reasoning_out = parsed.get("reasoning_content")
    except (AssertionError, ValueError, IndexError) as error:
      logger.error("DeepSeek-V3.2 parsing failed: %s. Falling back to raw text.", error, exc_info=True)
  if server_utils.is_dsv4_encoding_enabled(request.model):
    try:
      parsed = encoding_dsv4.parse_message_from_completion_text(text_out, thinking_mode=thinking_mode)
      text_out = parsed.get("content", text_out)
      reasoning_out = parsed.get("reasoning_content")
    except (AssertionError, ValueError, IndexError) as error:
      logger.error("DeepSeek-V4 parsing failed: %s. Falling back to raw text.", error, exc_info=True)

  want_top = (request.top_logprobs or 0) > 0 if isinstance(request, ChatCompletionRequest) else (request.logprobs or 0) > 0
  lp_payload = to_openai_logprobs(getattr(completion_result, "logprobs", None), llm, want_top=want_top)
  text_out, lp_payload, finish_reason = apply_stops_to_text_and_logprobs(text_out, lp_payload, request.stop)
  if finish_reason is None:
    finish_reason = completion_result.finish_reason
  usage = Usage(
      prompt_tokens=completion_result.prompt_token_count,
      completion_tokens=completion_result.completion_token_count,
      total_tokens=completion_result.prompt_token_count + completion_result.completion_token_count,
  )
  return ChatCompletionResponse(
      model=request.model,
      choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=text_out,
                                                                  reasoning_content=reasoning_out),
                                    finish_reason=finish_reason, logprobs=lp_payload)],
      usage=usage,
  )


def build_completion_response(request, completions, prompts, llm):
  choices = []
  prompt_tokens_total = 0
  completion_tokens_total = 0
  for index, _ in enumerate(prompts):
    item = completions[index]
    lp_payload = to_openai_logprobs(getattr(item, "logprobs", None), llm, want_top=(request.logprobs or 0) > 0)
    finish_reason = getattr(item, "finish_reason", "stop")
    text_out, lp_payload, stop_reason = apply_stops_to_text_and_logprobs(item.text, lp_payload, request.stop)
    if stop_reason is not None:
      finish_reason = stop_reason
    prompt_tokens_total += item.prompt_token_count
    completion_tokens_total += item.completion_token_count
    choices.append(CompletionChoice(text=text_out, index=index, logprobs=lp_payload, finish_reason=finish_reason))
  usage = Usage(prompt_tokens=prompt_tokens_total, completion_tokens=completion_tokens_total,
                total_tokens=prompt_tokens_total + completion_tokens_total)
  return CompletionResponse(model=request.model, choices=choices, usage=usage)


def create_response(request, completions, prompts, is_chat, llm, harmony_enc):
  if is_chat:
    return build_chat_completion_response(request, completions[0], llm, harmony_enc)
  return build_completion_response(request, completions, prompts, llm)
