"""DeepSeek-V4 conversation formatting into prompt and completion segments."""

import re

from lilax._backend.input_pipeline import encoding_dsv4

_DS4_TURN_RE = re.compile(
    re.escape(encoding_dsv4.ASSISTANT_SP_TOKEN)
    + f"({re.escape(encoding_dsv4.thinking_start_token)}|"
    + f"{re.escape(encoding_dsv4.thinking_end_token)})"
    + f"(?:{re.escape(encoding_dsv4.DS_TASK_SP_TOKENS['action'])})?"
    + f"(.*?{re.escape(encoding_dsv4.eos_token)})",
    re.DOTALL,
)


def _is_deepseek_v4(tokenizer_model, model_name=None) -> bool:
  """Detects DeepSeek-V4 by model or tokenizer name.

  Vocabulary checks are not used: the DeepSeek-V3.2 tokenizer has the same special tokens and
  also ships without a chat template, but uses a different encoding.
  """
  for name in (model_name, getattr(tokenizer_model, "model_name", None), getattr(tokenizer_model, "name_or_path", None)):
    if name and any(x in str(name).lower() for x in ("deepseek4", "deepseek-v4", "deepseek_v4")):
      return True
  return False


def _format_conversations_dsv4(example, data_column_name):
  """Formats DeepSeek-V4 conversational data using encoding_dsv4.encode_messages."""
  raw_messages = example[data_column_name]
  for idx, message in enumerate(raw_messages):
    if message.get("role") == "system" and idx != 0:
      raise ValueError(f"System message found at index {idx}. System messages must be at index 0.")

  assistants = [m for m in raw_messages if m.get("role") == "assistant"]
  if not assistants:
    example["is_prompt"] = []
    example[data_column_name] = []
    return example

  last_assistant_idx = max(i for i, m in enumerate(raw_messages) if m.get("role") == "assistant")
  messages_to_encode = list(raw_messages[: last_assistant_idx + 1])
  if example.get("tools") and not any(m.get("tools") for m in messages_to_encode):
    target_idx = next(
        (i for i, m in enumerate(messages_to_encode) if m.get("role") in ("system", "developer")),
        None,
    )
    if target_idx is not None:
      messages_to_encode[target_idx] = {**messages_to_encode[target_idx], "tools": example["tools"]}
    else:
      messages_to_encode.insert(0, {"role": "system", "content": "", "tools": example["tools"]})
  thinking_mode = "thinking" if any(m.get("reasoning_content") for m in messages_to_encode) else "chat"
  has_tools = bool(example.get("tools")) or any(
      m.get("tools") or m.get("tool_calls") or m.get("role") == "tool" for m in messages_to_encode
  )
  text = encoding_dsv4.encode_messages(messages_to_encode, thinking_mode=thinking_mode, drop_thinking=not has_tools)
  matches = list(_DS4_TURN_RE.finditer(text))
  if len(matches) != len(assistants):
    raise ValueError(
        f"Mismatch between number of assistant messages ({len(assistants)}) "
        f"and encoded turns ({len(matches)}). Encoded text: {text!r}"
    )

  messages = []
  is_prompt = []
  cursor = 0
  for match in matches:
    if thinking_mode == "thinking" and match.group(1) == encoding_dsv4.thinking_end_token:
      continue
    messages.extend([text[cursor : match.start(2)], match.group(2)])
    is_prompt.extend([True, False])
    cursor = match.end(2)

  if cursor != len(text):
    raise ValueError(
        f"Encoded text length ({len(text)}) does not match final completion boundary ({cursor}); "
        f"the final assistant turn was not matched as a trainable completion: {text[cursor:]!r}"
    )

  example["is_prompt"] = is_prompt
  example[data_column_name] = messages
  return example
