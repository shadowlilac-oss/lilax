"""Chat-template formatting of conversations into prompt and completion segments."""

from lilax._backend.input_pipeline.conversation import dsv4_chat_format
from lilax._backend.input_pipeline.transforms import hf_tokenization
from lilax._backend.utils import max_logging


def _split_turn_into_prompt_and_completion(tokenizer_model, round_msgs):
  """Splits a conversational round (system + user + assistant) into prompt and completion formatted strings.

  Uses the longest-common-prefix between the full conversation tokens and the
  generation-prompt tokens to locate where the prompt ends and completion begins.

  Args:
    tokenizer_model: The tokenizer instance.
    round_msgs: Messages for the current conversational turn including the assistant response.

  Returns:
    A tuple of (prompt_str, completion_str).
  """
  full_tokens = hf_tokenization.extract_token_ids(
      tokenizer_model.apply_chat_template(round_msgs, add_generation_prompt=False, tokenize=True)
  )
  prompt_tokens = hf_tokenization.extract_token_ids(
      tokenizer_model.apply_chat_template(round_msgs[:-1], add_generation_prompt=True, tokenize=True)
  )

  common_len = 0
  for full_id, prompt_id in zip(full_tokens, prompt_tokens):
    if full_id == prompt_id:
      common_len += 1
    else:
      break

  if common_len == 0:
    raise ValueError(
        "Chat template generation prompt mismatch: no common prefix tokens found.\n"
        f"Full conversation tokens: {full_tokens} ('{tokenizer_model.decode(full_tokens)}')\n"
        f"Generation prompt tokens: {prompt_tokens} ('{tokenizer_model.decode(prompt_tokens)}')\n"
        "Cannot determine completion boundary."
    )

  prompt_str = tokenizer_model.decode(full_tokens[:common_len], skip_special_tokens=False)
  completion_str = tokenizer_model.decode(full_tokens[common_len:], skip_special_tokens=False)
  return prompt_str, completion_str


def _get_completion_in_chat_template(tokenizer_model, round_msgs):
  """Calculates the completion part of a conversation turn formatted with a chat template."""
  _, completion_str = _split_turn_into_prompt_and_completion(tokenizer_model, round_msgs)
  return completion_str


def apply_chat_template(example, tokenizer_model, data_column_name, model_name=None):
  """Formats conversational data by applying the tokenizer's chat template
  and identifying prompt/completion segments for SFT masking.

  Args:
    example: A dictionary containing conversational data. It is expected to have a key
      specified by `data_column_name` that holds a list of messages.
    tokenizer_model: The tokenizer instance associated with the language model,
      which contains the specific chat template.
    data_column_name: The name of the column in the `example` dictionary
      that contains the list of messages.
    model_name: Optional model name used as a fallback to identify models
      that use a custom chat encoder when `tokenizer_model.chat_template` is None.

  Returns:
    The modified `example` dictionary.
      - The `data_column_name` column will be updated to a list of
        messages, each formatted according to the tokenizer's chat template.
      - A new column "is_prompt" is added, where `True` indicates the
        tokens contain the system message, user message, and generation
        prompt (if applicable). `False` indicates the expected LLM
        completion, excluding the assistant's start tokens.
  """
  messages = []
  is_prompt = []
  round_msgs = []
  try:
    if getattr(tokenizer_model, "chat_template", None) is None and dsv4_chat_format._is_deepseek_v4(
        tokenizer_model, model_name
    ):
      return dsv4_chat_format._format_conversations_dsv4(example, data_column_name)
    for idx, message in enumerate(example[data_column_name]):
      if message["role"] == "system":
        if idx != 0:
          raise ValueError(f"System message found at index {idx}. System messages must be at index 0.")
        round_msgs.append(message)
      elif message["role"] == "user":
        round_msgs.append(message)
      elif message["role"] == "assistant":
        round_msgs.append(message)
        prompt_str, completion_str = _split_turn_into_prompt_and_completion(tokenizer_model, round_msgs)
        messages.extend([prompt_str, completion_str])
        is_prompt.extend([True, False])
        round_msgs.clear()
  except ValueError as e:
    max_logging.log(f"Unable to apply chat template: {e}")
    raise e
  example["is_prompt"] = is_prompt
  example[data_column_name] = messages
  return example
